from __future__ import annotations

import collections
import functools
import heapq
import itertools
import json
import os
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set, NamedTuple, Generator, Any
import types
from ..stats import Stats

import _sep1 as ffi
from tqdm import tqdm

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
# from python.gss_tester.implementations.leveled_per_acc_impl import LeveledPerAccGSS as GSS
# from python.gss_tester.implementations.leveled_per_acc_standalone_impl import LeveledPerAccGSS as GSS
from ..common_interface import GraphProvider
from ..range_set import FFIRangeSet as RangeSet
from ..range_set import SetRangeSet as RangeSetOut
from ..range_set import SetRangeSet as RangeSetStates

# Type Aliases
NodeID = int
LLMTokenSet = RangeSet
StateIDSet = RangeSetStates
TerminalIdSet = RangeSet

# --- Monkey-patch RangeSet to collect stats on union/intersection ---
# This is to fulfill the request of tracking ffi.Bitset.union and intersection calls.
# Since the code was refactored to use a pure Python RangeSet, we track its methods instead.
_original_rangeset_union = RangeSet.union
_original_rangeset_intersection = RangeSet.intersection

def _patched_union(self, other: "RangeSet") -> "RangeSet":
    """Patched version of RangeSet.union that increments a stats counter."""
    stats = Stats.get()
    stats.inc('bitset.union.calls')
    stats.start('bitset.union.time')
    result = _original_rangeset_union(self, other)
    stats.stop('bitset.union.time')
    return result

def _patched_intersection(self, other: "RangeSet") -> "RangeSet":
    """Patched version of RangeSet.intersection that increments a stats counter."""
    stats = Stats.get()
    stats.inc('bitset.intersection.calls')
    stats.start('bitset.intersection.time')
    result = _original_rangeset_intersection(self, other)
    stats.stop('bitset.intersection.time')
    return result

# Apply the patches
RangeSet.union = _patched_union
RangeSet.intersection = _patched_intersection
# --- End of monkey-patch ---

def _acc_memoize(stats_prefix: Optional[str] = None, use_value_cache: bool = True):
    """Per-invocation memoization for PyAcc transformers."""
    def decorator(fn):
        id_memo = {}
        val_memo = {}
        def wrapper(acc):
            acc_id = id(acc)
            if acc_id in id_memo:
                if stats_prefix:
                    Stats.get().inc(f'{stats_prefix}.memo_hits')
                return id_memo[acc_id]

            if use_value_cache:
                cached = val_memo.get(acc)
                if cached is not None:
                    id_memo[acc_id] = cached
                    if stats_prefix:
                        Stats.get().inc(f'{stats_prefix}.memo_hits')
                    return cached

            result = fn(acc)
            id_memo[acc_id] = result
            if use_value_cache and result is not None:
                val_memo[acc] = result
            return result
        wrapper._acc_memo_size = lambda: len(id_memo)
        return wrapper
    return decorator

@dataclass(frozen=True)
class DFAState:
    transitions: Dict[int, int]
    finalizers: Set[int]
    possible_future_group_ids: Set[int]


@dataclass(frozen=True)
class PyTokenizer:
    states: List[DFAState]
    start_state: int
    non_greedy_finalizers: Set[int]

    def execute_from_state(self, text: bytes, state_id: int) -> Tuple[Optional[int], List[Tuple[int, int]]]:
        current_state = state_id
        matches = {}
        done = False

        # Check for initial matches (epsilon)
        for group_id in self.states[current_state].finalizers:
            if group_id in self.non_greedy_finalizers:
                matches.setdefault(group_id, 0)
            else:
                matches[group_id] = 0

        for i, byte in enumerate(text):
            state_data = self.states[current_state]
            next_state = state_data.transitions.get(byte)

            if next_state is None:
                done = True
                break

            current_state = next_state
            for group_id in self.states[current_state].finalizers:
                if group_id in self.non_greedy_finalizers:
                    matches.setdefault(group_id, i + 1)
                else:
                    matches[group_id] = i + 1

        end_state = None if done else current_state
        return end_state, [(gid, width) for gid, width in matches.items() if width > 0]

    def tokens_accessible_from_state(self, state_id: int) -> List[int]:
        return list(self.states[state_id].possible_future_group_ids)

    def initial_state_id(self) -> int:
        return self.start_state

    def max_state(self) -> int:
        return len(self.states)

@dataclass(frozen=True)
class Reduce:
    nonterminal_id: int
    len: int
    production_ids: Tuple[int, ...]

@dataclass(frozen=True)
class Split:
    shift: Optional[int]
    reduces: Dict[int, Dict[int, Tuple[int, ...]]]

Action = Union[int, Reduce, Split]

@dataclass
class Row:
    actions: Dict[int, Action] = field(default_factory=dict)
    gotos: Dict[int, int] = field(default_factory=dict)

@dataclass
class ParserTable:
    start_state_id: int
    table: Dict[int, Row]

@dataclass
class ArenaEdgeDest:
    dest_idx: NodeID
    state_bv: StateIDSet

@dataclass
class LoadedArenaEdgeDest:
    dest_idx: NodeID
    state_bv: StateIDSet

@dataclass
class LoadedArenaEdge:
    pop: int
    llm_bv: LLMTokenSet
    dests: List[LoadedArenaEdgeDest]

@dataclass
class LoadedArenaNode:
    children: List[LoadedArenaEdge]
    clean_end: bool

@dataclass
class IntermediateArenaEdgeDest:
    dest_idx: NodeID
    state_bv: StateIDSet

@dataclass
class IntermediateArenaEdge:
    pop: int
    llm_bv: LLMTokenSet
    dests: IntermediateArenaEdgeDest

@dataclass
class IntermediateArenaNode:
    children: List[IntermediateArenaEdge]
    clean_end: bool

@dataclass
class ArenaEdge:
    pop: int
    llm_bv: LLMTokenSet
    dests: List[ArenaEdgeDest] = field(default_factory=list)
    dest_states_union: StateIDSet = field(default_factory=RangeSetStates.empty)
    llm_bv_not: Optional[LLMTokenSet] = None
    state_to_dest: Optional[Dict[int, List[int]]] = None

    def ensure_index(self) -> None:
        if self.state_to_dest is not None:
            return

        mapping: Dict[int, List[int]] = collections.defaultdict(list)
        for j, dest in enumerate(self.dests):
            for sid in dest.state_bv.iter_indices():
                mapping[sid].append(j)
        self.state_to_dest = dict(mapping)

@dataclass
class ArenaNode:
    children: List[ArenaEdge] = field(default_factory=list)
    llm_bv_union: LLMTokenSet = field(default_factory=RangeSet.empty)
    clean_end: bool = False
    # New accelerators
    llm_bv_descendant: LLMTokenSet = field(default_factory=RangeSet.empty)
    pop_to_state_union: Dict[int, StateIDSet] = field(default_factory=dict)
    pop_to_llm_union: Dict[int, LLMTokenSet] = field(default_factory=dict)

def _optimize_intermediate_arena(intermediate_arena: Dict[NodeID, IntermediateArenaNode], max_depth: Dict[NodeID, int]):
    for node in tqdm(intermediate_arena.values(), desc="Optimizing intermediate arena"):
        if not node.children:
            continue
        # Sort edges by destination depth (desc) and then pop count (asc)
        node.children.sort(key=lambda e: (-max_depth.get(int(e.dests.dest_idx), 0), e.pop))

def _load_and_flatten_arena(loaded_arena: Dict[NodeID, LoadedArenaNode]) -> Dict[NodeID, IntermediateArenaNode]:
    """Stage 1: Convert from the loaded format to a flattened intermediate format."""
    intermediate_arena: Dict[NodeID, IntermediateArenaNode] = {}
    for uid, loaded_node in tqdm(loaded_arena.items(), desc="Stage 1: Loading and flattening arena"):
        intermediate_children: List[IntermediateArenaEdge] = []
        for loaded_edge in loaded_node.children:
            for d in loaded_edge.dests:
                intermediate_children.append(IntermediateArenaEdge(
                    pop=loaded_edge.pop,
                    llm_bv=loaded_edge.llm_bv,
                    dests=IntermediateArenaEdgeDest(d.dest_idx, d.state_bv),
                ))
        intermediate_arena[uid] = IntermediateArenaNode(
            children=intermediate_children,
            clean_end=loaded_node.clean_end,
        )
    return intermediate_arena

def _merge_and_finalize_arena(intermediate_arena: Dict[NodeID, IntermediateArenaNode]) -> Dict[NodeID, ArenaNode]:
    """Stage 2: Merge flattened edges back into the final ArenaNode structure."""
    arena: Dict[NodeID, ArenaNode] = {}
    for uid, intermediate_node in tqdm(intermediate_arena.items(), desc="Stage 2: Merging and converting arena"):
        new_children: List[ArenaEdge] = []
        llm_bv_union = RangeSet.empty()

        if not intermediate_node.children:
            arena[uid] = ArenaNode(children=[], llm_bv_union=llm_bv_union, clean_end=intermediate_node.clean_end)
            continue

        children_it = iter(intermediate_node.children)
        first = next(children_it)
        edge_dests = []
        prev_pop = first.pop
        prev_llm_bv = first.llm_bv
        edge_dests.append(ArenaEdgeDest(first.dests.dest_idx, first.dests.state_bv))
        def flush() -> None:
            nonlocal edge_dests, prev_pop, prev_llm_bv, new_children, llm_bv_union
            dest_states_union = RangeSetStates.empty()
            for d in edge_dests:
                dest_states_union |= d.state_bv
            llm_bv_union |= prev_llm_bv
            new_children.append(ArenaEdge(
                pop=prev_pop,
                llm_bv=prev_llm_bv,
                dests=edge_dests,
                dest_states_union=dest_states_union,
            ))
        for edge in children_it:
            if not (edge.pop == prev_pop and edge.llm_bv == prev_llm_bv):
                flush()
                prev_pop = edge.pop
                prev_llm_bv = edge.llm_bv
                edge_dests = []
            edge_dests.append(ArenaEdgeDest(edge.dests.dest_idx, edge.dests.state_bv))
        flush()

        arena[uid] = ArenaNode(
            children=new_children,
            llm_bv_union=llm_bv_union,
            clean_end=intermediate_node.clean_end,
        )
    return arena

def _convert_arena(loaded_arena: Dict[NodeID, LoadedArenaNode], max_depth: Dict[NodeID, int]) -> Dict[NodeID, ArenaNode]:
    """Orchestrates the full conversion from loaded data to the final arena format."""
    intermediate_arena = _load_and_flatten_arena(loaded_arena)
    _optimize_intermediate_arena(intermediate_arena, max_depth)
    final_arena = _merge_and_finalize_arena(intermediate_arena)
    return final_arena

@dataclass(frozen=True, eq=False)
class PyAcc:
    terminals_union: Dict[int, TerminalIdSet]
    llm_mask: LLMTokenSet

    def __eq__(self, other):
        if not isinstance(other, PyAcc):
            return NotImplemented
        return self.llm_mask == other.llm_mask and self.terminals_union == other.terminals_union

    def __hash__(self):
        # Correctly hash the dictionary content for memoization
        return hash((len(self.terminals_union), self.llm_mask))

    def merge(self, other: "PyAcc") -> "PyAcc":
        new_terminals_union = self.terminals_union.copy()
        for k, v in other.terminals_union.items():
            if k in new_terminals_union:
                new_terminals_union[k] = new_terminals_union[k].union(v)
            else:
                new_terminals_union[k] = v
        return PyAcc(
            terminals_union=new_terminals_union,
            llm_mask=self.llm_mask.union(other.llm_mask),
        )

    def is_empty(self):
        return self.llm_mask.is_empty()

@dataclass
class Enqueue:
    node_id: NodeID
    gss: GSS
    depth: int

@dataclass
class Suspend:
    node_id: NodeID
    priority: Any
    depth: int

@dataclass
class WorkItemNew:
    node_id: NodeID
    gss: GSS
    depth: int

@dataclass
class WorkItemSuspended:
    generator: Generator
    llm_mask: LLMTokenSet
    depth: int


@dataclass
class Model(GraphProvider):
    stats = Stats.get()
    stats.add_group('get_mask')
    stats.add_group('commit')

    arena: Dict[NodeID, ArenaNode]
    roots_map: Dict[int, NodeID]
    max_depth: Dict[NodeID, int]
    parser_table: ParserTable
    tokenizer: PyTokenizer
    tokenizer_initial_state: int
    possible_matches_cache: Dict[int, Dict[int, LLMTokenSet]]
    id_to_token: Dict[int, bytes]
    internal_to_original_map: Dict[int, RangeSetOut]
    all_internal_llm_tokens_bitset: LLMTokenSet
    ignore_terminal_id: Optional[int]
    state: Dict[int, GSS]
    # Runtime tunables and results
    gm_max_edges: int = 256
    gm_max_dests: int = 4096
    last_get_mask_cost: int = 0
    last_get_mask_metrics: Dict[str, float] = field(default_factory=dict)
    # Oracle analysis mode
    oracle_mode: bool = True
    oracle_exact_cover_threshold: int = 24
    oracle_debug: bool = True
    suppress_stats_report: bool = False
    # Advanced oracle planning tunables
    # Weight of pruning benefit (in units of "node cost") when ordering end nodes
    oracle_prune_weight: float = 1.0
    # Weight of token coverage when ordering end nodes
    oracle_token_weight: float = 0.01
    oracle_apply_weight: float = 2.0; oracle_isolate_weight: float = 1.0

    # Oracle search tunables
    oracle_trials: int = 8
    oracle_jitter: int = 7
    oracle_reward_prune: bool = True
    # New oracle controls
    oracle_dynamic_prioritization: bool = True
    # Stronger oracle search and planning controls
    oracle_beam_width: int = 4
    oracle_search_trials: int = 8
    oracle_edge_trials_per_node: int = 16
    oracle_meganode_top_k: int = 0
    oracle_plan_use_exact_end_masks: bool = True
    oracle_use_beam_search: bool = False
    oracle_debug_verbose: bool = False
    oracle_hotspot_report_top_k: int = 10
    # Try random permutations of the top-K largest end masks
    oracle_permute_top_ends: int = 8

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        Stats.get().reset()
        data = json.loads(s)

        # Arena
        roots_map = {int(s): int(r) for s, r in data["precomputed3"]}
        arena_dict = {int(k): v for k, v in data["trie3_god"].get("values", [])}
        max_depth: Dict[NodeID, int] = {}
        dumps, bs_from_json = json.dumps, ffi.Bitset.from_json_string

        loaded_arena: Dict[NodeID, LoadedArenaNode] = {}
        for uid, node_data in arena_dict.items():
            max_depth[uid] = int(node_data.get("max_depth", 0) or 0)
            children_data = node_data.get("children") or []

            loaded_children: List[LoadedArenaEdge] = []
            for (pop, llm_json), dest_map_json in children_data:
                llm_bv = RangeSet.from_ranges(bs_from_json(dumps(llm_json)).to_ranges())
                dests: List[ArenaEdgeDest] = []
                for dest_idx, state_json in dest_map_json:
                    state_bv = RangeSetStates.from_ranges(bs_from_json(dumps(state_json)).to_ranges())
                    dests.append(LoadedArenaEdgeDest(int(dest_idx), state_bv))
                loaded_children.append(LoadedArenaEdge(int(pop), llm_bv, dests))

            clean_end = node_data.get("value", {}).get("clean_end", False)
            loaded_arena[uid] = LoadedArenaNode(children=loaded_children, clean_end=clean_end)

        arena = _convert_arena(loaded_arena, max_depth)
        # Tokenizer
        dfa_data = data['tokenizer']['dfa']
        dfa_states = [DFAState(transitions={int(k): v for k, v in s['transitions'].get('data', {}).items()}, finalizers=set(s['finalizers']), possible_future_group_ids=set(s['possible_future_group_ids'])) for s in dfa_data['states']]
        tokenizer = PyTokenizer(dfa_states, dfa_data['start_state'], set(dfa_data['non_greedy_finalizers']))

        # Parser Table
        parser_data = data['parser']
        py_table: Dict[int, Row] = {}
        for state_id_str, row_data in parser_data['stage_7_table']:
            state_id, py_row = int(state_id_str), Row()
            for term_id_str, action_data in row_data['shifts_and_reduces_full']:
                term_id, variant = int(term_id_str), action_data['variant']
                if variant == 'Shift': py_row.actions[term_id] = action_data['state_id']
                elif variant == 'Reduce': py_row.actions[term_id] = Reduce(action_data['nonterminal_id'], action_data['len'], tuple(sorted(action_data['production_ids'])))
                elif variant == 'Split':
                    reduces = {int(l): {int(n): tuple(sorted(p)) for n, p in nd} for l, nd in action_data['reduces']}
                    py_row.actions[term_id] = Split(action_data['shift'], reduces)
            py_row.gotos = {int(nt): goto['state_id'] for nt, goto in row_data['gotos'] if goto['state_id'] is not None}
            py_table[state_id] = py_row
        parser_table = ParserTable(parser_data['start_state_id'], py_table)

        # Misc data (some still requires FFI for loading)
        constraint = ffi.GrammarConstraint.from_json_string(s)
        pmc_ffi = constraint.possible_matches()
        possible_matches_cache = {int(t): {int(i): RangeSet.from_ranges(b.to_ranges()) for i, b in inner.items()} for t, inner in pmc_ffi.items()}
        vocab = data['precompute3_vocab']
        all_internal_llm_tokens_bitset = RangeSet.from_ranges([(0, vocab['internal_max_llm_token'])])

        # Initial state
        initial_acc = PyAcc({}, all_internal_llm_tokens_bitset)
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(parser_table.start_state_id)

        model = Model(
            arena=arena, roots_map=roots_map, max_depth=max_depth, parser_table=parser_table,
            tokenizer=tokenizer, tokenizer_initial_state=tokenizer.initial_state_id(),
            possible_matches_cache=possible_matches_cache,
            id_to_token={v: bytes(k) for k, v in data['llm_token_map']},
            internal_to_original_map={int(k): RangeSetOut.from_indices(v) for k, v in dict(vocab['internal_to_original']).items()},
            all_internal_llm_tokens_bitset=all_internal_llm_tokens_bitset,
            ignore_terminal_id=constraint.glr_parser().ignore_terminal_id,
            state={tokenizer.initial_state_id(): initial_gss},
        )
        model._compute_edge_accelerators()
        # New: per-pop unions and descendant closures for stronger pruning
        model._compute_bucket_unions()
        model._compute_descendant_llm_closure()
        model.optimize_traversal()
        model._compute_and_print_stats()
        return model

    def _compute_edge_accelerators(self) -> None:
        all_ones = self.all_internal_llm_tokens_bitset
        for node in self.arena.values():
            for edge in node.children:
                edge.llm_bv_not = all_ones.difference(edge.llm_bv)

    def _compute_bucket_unions(self) -> None:
        """Per node, per-pop unions for early bucket pruning."""
        for node in self.arena.values():
            if not node.children:
                node.pop_to_state_union = {}
                node.pop_to_llm_union = {}
                continue
            state_union_by_pop: Dict[int, StateIDSet] = collections.defaultdict(RangeSetStates.empty)
            llm_union_by_pop: Dict[int, LLMTokenSet] = collections.defaultdict(RangeSet.empty)
            for edge in node.children:
                state_union_by_pop[edge.pop] |= edge.dest_states_union
                llm_union_by_pop[edge.pop] |= edge.llm_bv
            node.pop_to_state_union = dict(state_union_by_pop)
            node.pop_to_llm_union = dict(llm_union_by_pop)

    def _compute_descendant_llm_closure(self) -> None:
        """Compute an upper-bound mask of tokens reachable from each node to a clean_end.
        We use a monotone fixed-point iteration to be correct even when max_depth does
        not strictly decrease along edges (e.g., equal depths or cycles).
        """
        all_ones = self.all_internal_llm_tokens_bitset
        # Initialize: end nodes are all-ones; others start empty.
        for nid, node in self.arena.items():
            node.llm_bv_descendant = all_ones if node.clean_end else RangeSet.empty()

        # Iteratively grow descendant closures until a fixed point is reached.
        changed = True
        while changed:
            changed = False
            for nid, node in self.arena.items():
                if node.clean_end:
                    # Already saturated
                    continue
                if not node.children:
                    # Leaf without clean_end stays empty unless it was already grown previously
                    continue
                new_accum = RangeSet.empty()
                for edge in node.children:
                    dest_union_closure = RangeSet.empty()
                    for d in edge.dests:
                        dest_union_closure |= self.arena[int(d.dest_idx)].llm_bv_descendant
                    if not dest_union_closure.is_empty():
                        new_accum |= edge.llm_bv.intersection(dest_union_closure)
                # Monotone update: only grow the closure.
                updated = node.llm_bv_descendant.union(new_accum)
                if updated != node.llm_bv_descendant:
                    node.llm_bv_descendant = updated
                    changed = True

    def optimize_traversal(self) -> None:
        for node in self.arena.values():
            for edge in node.children:
                edge.ensure_index()

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        term_rs = RangeSet.from_indices([terminal_id])
        @_acc_memoize(use_value_cache=False)
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current = acc.terminals_union.get(state_id, RangeSet.empty())
            if current.contains(terminal_id): return acc
            new_map = acc.terminals_union.copy()
            new_map[state_id] = current.union(term_rs)
            return PyAcc(new_map, acc.llm_mask)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> NodeID: return self.roots_map[int(state_id)]
    def is_end(self, node: NodeID) -> bool: return self.arena[node].clean_end

    def iter_edges(self, node: NodeID, token: int):
        a_node = self.arena.get(node)
        if not a_node: return
        for edge in a_node.children:
            if edge.llm_bv.contains(token):
                for dest in edge.dests:
                    for start, end in dest.state_bv.to_ranges():
                        for sid in range(start, end + 1):
                            yield (edge.pop, sid, dest.dest_idx)

    def commit(self, token_id: int):
        token_bytes = self.id_to_token[token_id]
        terminals_map, state_map = {}, {}
        for tsid in self.state:
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tsid)
            if end_state is not None: state_map[tsid] = end_state
            terminals_map[tsid] = RangeSet.from_indices([m[0] for m in matches])

        @_acc_memoize()
        def mutator(acc: PyAcc) -> Optional[PyAcc]:
            for tsid, matched in terminals_map.items():
                if acc.terminals_union.get(tsid, RangeSet.empty()).intersects(matched): return None
            new_bvs = collections.defaultdict(RangeSet.empty)
            for old, new in state_map.items():
                if old in acc.terminals_union: new_bvs[new] |= acc.terminals_union[old]
            return PyAcc(dict(new_bvs), acc.llm_mask)

        # Share a memo cache across GSSes for this transformation to avoid redundant work
        cache = {}
        current = {tsid: g.apply_and_prune(mutator, cache) for tsid, g in self.state.items()}
        current = {tsid: g for tsid, g in current.items() if not g.is_empty()}

        new_states = collections.defaultdict(list)
        work = {(0, tsid): gss for tsid, gss in current.items()}
        q = collections.deque(work.keys())

        while q:
            offset, tsid = q.popleft()
            gss = work.pop((offset, tsid))
            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tsid)

            for term_id, width in matches:
                proc_gss = self._process_token(gss, term_id)
                if end_state is not None and term_id in self.tokenizer.states[end_state].possible_future_group_ids:
                    proc_gss = self._disallow_terminal_in_state(proc_gss, end_state, term_id)
                if not proc_gss.is_empty():
                    new_offset, next_tsid = offset + width, self.tokenizer.initial_state_id()
                    if new_offset == len(token_bytes): new_states[next_tsid].append(proc_gss)
                    else:
                        key = (new_offset, next_tsid)
                        if key in work: work[key] = work[key].merge(proc_gss)
                        else:
                            work[key] = proc_gss
                            q.append(key)
            if end_state is not None: new_states[end_state].append(gss)

        merged = {sid: GSS.merge_many(gssl) for sid, gssl in new_states.items() if gssl}
        self.state = {sid: g for sid, g in merged.items() if not g.is_empty()}

    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        if self.ignore_terminal_id == terminal_id: return gss

        heads_by_state = collections.defaultdict(list)
        for state_id in gss.peek(): heads_by_state[state_id].append(gss.isolate(state_id))

        shifted = []
        while heads_by_state:
            state_id, gss_list = heads_by_state.popitem()
            state_gss = GSS.merge_many(gss_list)
            row = self.parser_table.table.get(state_id)
            if not row: continue
            action = row.actions.get(terminal_id)
            if action is None: continue

            if isinstance(action, int): shifted.append(state_gss.push(action))
            elif isinstance(action, Reduce):
                popped = state_gss.popn(action.len)
                for from_id in popped.peek():
                    goto_id = self.parser_table.table[from_id].gotos[action.nonterminal_id]
                    heads_by_state[goto_id].append(popped.isolate(from_id).push(goto_id))
            elif isinstance(action, Split):
                if action.shift is not None: shifted.append(state_gss.push(action.shift))
                for length, nts in action.reduces.items():
                    popped = state_gss.popn(length)
                    for nt_id in nts:
                        for from_id in popped.peek():
                            goto_id = self.parser_table.table[from_id].gotos[nt_id]
                            heads_by_state[goto_id].append(popped.isolate(from_id).push(goto_id))
        return GSS.merge_many(shifted)

    def _compute_and_print_stats(self):
        import collections
        try:
            import numpy as np
        except ImportError:
            print("Numpy not found, cannot print detailed distribution stats.")
            np = None

        stats = collections.defaultdict(list)
        stats['edges_per_pop_bucket'] = []
        stats['state_coverage_per_pop_bucket'] = []
        stats['avg_edges_per_state_per_pop_bucket'] = []
        stats['unique_llm_bvs_per_pop_bucket'] = []
        stats['unique_dest_states_unions_per_pop_bucket'] = []
        pop_counts = collections.Counter()
        num_nodes = len(self.arena)
        num_clean_end_nodes = 0
        total_llm_bv_cardinality = 0
        total_state_bv_cardinality = 0

        for node in self.arena.values():
            if node.clean_end:
                num_clean_end_nodes += 1
            stats['edges_per_node'].append(len(node.children))

            if node.children:
                unique_dests_in_node = set(d.dest_idx for edge in node.children for d in edge.dests)
                stats['unique_dests_per_node'].append(len(unique_dests_in_node))

                sum_llm_bv_len_for_node = sum(len(edge.llm_bv) for edge in node.children)
                if node.llm_bv_union and len(node.llm_bv_union) > 0:
                    overlap_factor = sum_llm_bv_len_for_node / len(node.llm_bv_union)
                    stats['llm_bv_overlap_factor'].append(overlap_factor)

            for edge in node.children:
                stats['dests_per_edge'].append(len(edge.dests))
                if edge.dests:
                    stats['unique_dests_per_edge'].append(len({d.dest_idx for d in edge.dests}))
                pop_counts[edge.pop] += 1

                llm_bv_len = len(edge.llm_bv)
                stats['llm_bv_cardinality'].append(llm_bv_len)
                total_llm_bv_cardinality += llm_bv_len

                stats['dest_states_union_cardinality'].append(len(edge.dest_states_union))

                for dest in edge.dests:
                    state_bv_len = len(dest.state_bv)
                    stats['dest_state_bv_cardinality'].append(state_bv_len)
                    total_state_bv_cardinality += state_bv_len

            # --- New pop-bucket stats for the current node ---
            pop_buckets = collections.defaultdict(list)
            for edge in node.children:
                pop_buckets[edge.pop].append(edge)

            for pop, edges_in_bucket in pop_buckets.items():
                # Edges per pop-bucket
                stats['edges_per_pop_bucket'].append(len(edges_in_bucket))

                # State coverage and avg edges per state
                pop_state_union = RangeSetStates.empty()
                sum_dest_states_union_cardinality = 0
                for edge in edges_in_bucket:
                    pop_state_union |= edge.dest_states_union
                    sum_dest_states_union_cardinality += len(edge.dest_states_union)

                state_coverage = len(pop_state_union)
                stats['state_coverage_per_pop_bucket'].append(state_coverage)

                if state_coverage > 0:
                    avg_edges = sum_dest_states_union_cardinality / state_coverage
                    stats['avg_edges_per_state_per_pop_bucket'].append(avg_edges)

                # B_p(u): Unique LLM BVs per Pop-Bucket. This predicts the number of
                # apply_and_prune calls needed if we group by llm_bv.
                unique_llm_bvs = {edge.llm_bv for edge in edges_in_bucket}
                stats['unique_llm_bvs_per_pop_bucket'].append(len(unique_llm_bvs))

                # Proxy for s-list duplication: Unique dest_states_union per Pop-Bucket.
                # If this is low, it suggests high potential for caching isolate_many results.
                unique_dest_states = {edge.dest_states_union for edge in edges_in_bucket}
                stats['unique_dest_states_unions_per_pop_bucket'].append(len(unique_dest_states))


        num_edges = sum(stats['edges_per_node']) if stats['edges_per_node'] else 0
        num_dests = sum(stats['dests_per_edge']) if stats['dests_per_edge'] else 0

        print("\nCalculating inverted index stats for (Pop, StateID) ambiguity (this may take a moment)...")
        llm_bvs_per_pop_state = []
        dests_per_pop_state = []

        for node in tqdm(self.arena.values(), desc="Analyzing state ambiguity"):
            per_node_inverted_map = collections.defaultdict(lambda: collections.defaultdict(lambda: {'llm_bvs': set(), 'dests': set()}))
            for edge in node.children:
                edge.ensure_index()
                for sid, dest_j_list in edge.state_to_dest.items():
                    state_entry = per_node_inverted_map[edge.pop][sid]
                    state_entry['llm_bvs'].add(edge.llm_bv)
                    for dest_j in dest_j_list:
                        state_entry['dests'].add(edge.dests[dest_j].dest_idx)

            for pop_map in per_node_inverted_map.values():
                for state_map_val in pop_map.values():
                    llm_bvs_per_pop_state.append(len(state_map_val['llm_bvs']))
                    dests_per_pop_state.append(len(state_map_val['dests']))


        print("\n--- Arena Stats ---")
        print(f"Total nodes: {num_nodes:,}")
        print(f"Clean end nodes: {num_clean_end_nodes:,}")
        print(f"Total edges: {num_edges:,}")
        print(f"Total destinations: {num_dests:,}")
        print(f"Total LLM token cardinality (sum over edges): {total_llm_bv_cardinality:,}")
        print(f"Total state ID cardinality (sum over dests): {total_state_bv_cardinality:,}")

        def print_dist_stats(name, data):
            if not data or not np:
                print(f"\n--- {name} Distribution ---")
                print(f"  (No data or numpy not available)")
                return
            arr = np.array(data)
            # Check if original data was integer to format min/max appropriately
            is_int_data = all(isinstance(x, int) for x in data)
            fmt = "," if is_int_data else ",.2f"

            print(f"\n--- {name} Distribution ---")
            print(f"  Min: {np.min(arr):{fmt}}")
            print(f"  Max: {np.max(arr):{fmt}}")
            print(f"  Mean: {np.mean(arr):,.2f}")
            print(f"  Median: {np.median(arr):,.2f}") # Median can be float for int arrays
            print(f"  Std Dev: {np.std(arr):,.2f}")
            percentiles = np.percentile(arr, [25, 50, 75, 90, 99])
            print(f"  Percentiles (25, 50, 75, 90, 99): [{', '.join(f'{p:,.2f}' for p in percentiles)}]")

        print_dist_stats("Edges per Node", stats['edges_per_node'])
        print_dist_stats("Unique Destinations per Node", stats['unique_dests_per_node'])
        print_dist_stats("Destinations per Edge", stats['dests_per_edge'])
        print_dist_stats("Unique Destinations per Edge", stats['unique_dests_per_edge'])
        print_dist_stats("LLM TokenSet Cardinality per Edge", stats['llm_bv_cardinality'])
        print_dist_stats("LLM BV Overlap Factor per Node", stats['llm_bv_overlap_factor'])
        print_dist_stats("StateIDSet Union Cardinality per Edge", stats['dest_states_union_cardinality'])
        print_dist_stats("StateIDSet Cardinality per Destination", stats['dest_state_bv_cardinality'])
        print_dist_stats("Unique LLM BVs per (Node, Pop, StateID)", llm_bvs_per_pop_state)
        print_dist_stats("Unique Destinations per (Node, Pop, StateID)", dests_per_pop_state)
        print_dist_stats("Edges per Pop-Bucket", stats['edges_per_pop_bucket'])
        print_dist_stats("Unique LLM BVs per Pop-Bucket", stats['unique_llm_bvs_per_pop_bucket'])
        print_dist_stats("Unique Dest State Unions per Pop-Bucket", stats['unique_dest_states_unions_per_pop_bucket'])
        print_dist_stats("State Coverage per Pop-Bucket", stats['state_coverage_per_pop_bucket'])
        print_dist_stats("Avg Edges per State per Pop-Bucket", stats['avg_edges_per_state_per_pop_bucket'])
        if self.max_depth:
            print_dist_stats("Max Depth per Node", list(self.max_depth.values()))

        print("\n--- Pop Counts ---")
        if num_edges > 0:
            for pop_val, count in sorted(pop_counts.items()):
                print(f"  Pop {pop_val}: {count:,} edges ({count/num_edges*100:.2f}%)")
        else:
            print("  (No edges)")

        print("-------------------\n")

    def _process_internal_node_gen(
        self,
        node_id: NodeID,
        gss_node: GSS,
        remaining_mask: LLMTokenSet,
        gss_mask: LLMTokenSet,
        depth: int,
        global_pop_cache: Optional[Dict] = None,
        apply_cache: Optional[Dict] = None,
        isolate_cache: Optional[Dict] = None,
        edge_order_override_map: Optional[Dict[int, List[int]]] = None,
        dest_scores: Optional[Dict[int, int]] = None,
        oracle_counters: Optional[Dict[int, Dict[str, int]]] = None,
        node_allowed_mask: Optional[LLMTokenSet] = None,
    ) -> Generator[Union[Enqueue, Suspend], None, None]:
        stats = Stats.get()
        a_node = self.arena.get(node_id)
        if not a_node:
            return

        # max_edges, max_dests = (8, 2048) if is_final_mask_empty else (16, 4096)
        # Strengthen remaining-mask with node-specific allowance (oracle reward) when provided.
        active_remaining_mask = remaining_mask if node_allowed_mask is None else remaining_mask.intersection(node_allowed_mask)

        max_edges, max_dests = (self.gm_max_edges, self.gm_max_dests)
        edges_proc, dests_proc = 0, 0
        peek0_rs = None
        pop_cache: Dict[int, Tuple[Any, Any, List[int], StateIDSet]] = {}
        skip_pops: Set[int] = set()
        checked_pops: Set[int] = set()
        # Local per-(popped, llm_bv) apply cache to amortize within this node
        local_apply_cache: Dict[Tuple[int, int], GSS] = {} if apply_cache is None else apply_cache

        local_ctr = None
        # High-resolution timers per-node for hotspot analysis
        _perf_now = time.perf_counter
        if oracle_counters is not None:
            local_ctr = oracle_counters.setdefault(int(node_id), {"edges": 0, "apply": 0, "isolate": 0})

        edge_iterator = None
        ordered_indices = edge_order_override_map.get(node_id) if edge_order_override_map else None
        if ordered_indices:
            # Oracle-guided order. We still need to iterate non-guided edges if any.
            all_indices = set(range(len(a_node.children)))
            other_indices = sorted(list(all_indices - set(ordered_indices)))
            edge_iterator = ((i, a_node.children[i]) for i in ordered_indices + other_indices)
        else:
            edge_iterator = enumerate(a_node.children)
        for edge_i, edge in edge_iterator:
            if edge.pop in skip_pops:
                continue
            if edge.llm_bv.isdisjoint(active_remaining_mask):
                stats.inc('get_mask.traversal.edge.skipped_no_new_tokens')
                continue
            if edge.llm_bv.isdisjoint(gss_mask):
                stats.inc('get_mask.main_loop.edge.pre_gss_disjoint_skips')
                continue
            if edge.pop == 0:
                if peek0_rs is None: peek0_rs = RangeSetStates.from_indices(gss_node.peek())
                if edge.dest_states_union.isdisjoint(peek0_rs):
                    stats.inc('get_mask.main_loop.edge.dest_union_pruned_pop0')
                    continue

            stats.inc('get_mask.traversal.edges_traversed')
            stats.inc(f'get_mask.traversal.edge_pop_val.{edge.pop}')
            if oracle_counters is not None and local_ctr is not None: local_ctr["edges"] += 1

            if edge.pop in pop_cache:
                popped, popped_acc, peeked, peek_rs = pop_cache[edge.pop]
                stats.inc('get_mask.main_loop.edge.pop_cache_hits')
            else:
                # Try global cache first
                popped = None
                if global_pop_cache is not None:
                    key = (id(gss_node), edge.pop)
                    cached = global_pop_cache.get(key)
                    if cached is not None:
                        popped, popped_acc, peeked, peek_rs = cached
                        pop_cache[edge.pop] = cached
                        stats.inc('get_mask.global_pop_cache_hits')
                if popped is None:
                    stats.start('get_mask.main_loop.edge.popn')
                    _t0 = _perf_now()
                    popped = gss_node.popn(edge.pop)
                    _t1 = _perf_now()
                    stats.stop('get_mask.main_loop.edge.popn')
                    if oracle_counters is not None and local_ctr is not None:
                        local_ctr["popn_calls"] = local_ctr.get("popn_calls", 0) + 1
                        local_ctr["popn_time_ms"] = local_ctr.get("popn_time_ms", 0.0) + ((_t1 - _t0) * 1000.0)
                if popped.is_empty():
                    stats.inc('get_mask.traversal.edge.popped_empty')
                    pop_cache[edge.pop] = (popped, None, [], RangeSetStates.empty())
                    continue
                stats.start('get_mask.main_loop.edge.popped.reduce_acc')
                popped_acc = popped.reduce_acc()
                stats.stop('get_mask.main_loop.edge.popped.reduce_acc')
                if not popped_acc or popped_acc.llm_mask.is_empty():
                    pop_cache[edge.pop] = (GSS.empty(), None, [], RangeSetStates.empty())
                    continue
                # Avoid double-peek: call once and reuse both list and RangeSetStates
                peeked = popped.peek()
                peek_rs = RangeSetStates.from_indices(peeked)
                pop_cache[edge.pop] = (popped, popped_acc, peeked, peek_rs)
                if global_pop_cache is not None:
                    global_pop_cache[(id(gss_node), edge.pop)] = (popped, popped_acc, peeked, peek_rs)

            # One-time per-pop bucket pruning: states and llm masks
            if edge.pop not in checked_pops:
                checked_pops.add(edge.pop)
                pop_states_union = a_node.pop_to_state_union.get(edge.pop)
                if pop_states_union is not None and pop_states_union.isdisjoint(peek_rs):
                    stats.inc('get_mask.main_loop.bucket_pruned_state')
                    skip_pops.add(edge.pop)
                    continue
                pop_llm_union = a_node.pop_to_llm_union.get(edge.pop)
                # Extra prune when node_allowed_mask present: if the bucket's llm union has no overlap with
                # active_remaining_mask, skip the whole pop bucket.
                if pop_llm_union is not None and (
                    (popped_acc and popped_acc.llm_mask.isdisjoint(pop_llm_union)) or pop_llm_union.isdisjoint(active_remaining_mask)):
                    stats.inc('get_mask.main_loop.bucket_pruned_llm')
                    skip_pops.add(edge.pop)
                    continue

            if not popped_acc or edge.dest_states_union.isdisjoint(peek_rs):
                if popped_acc: stats.inc('get_mask.main_loop.edge.dest_union_pruned_after_pop')
                continue

            # Apply-and-prune caching across edges sharing the same llm_bv
            source_after_apply = popped
            if not (edge.llm_bv_not and popped_acc.llm_mask.isdisjoint(edge.llm_bv_not)):
                if popped_acc.llm_mask.isdisjoint(edge.llm_bv):
                    continue
                key_apply = (id(popped), id(edge.llm_bv))
                cached_apply = local_apply_cache.get(key_apply)
                if cached_apply is not None:
                    source_after_apply = cached_apply
                    stats.inc('get_mask.apply.cache_hits')
                else:
                    @_acc_memoize(use_value_cache=False)
                    def intersect(acc: PyAcc):
                        new_mask = acc.llm_mask.intersection(edge.llm_bv)
                        return None if new_mask.is_empty() else PyAcc(acc.terminals_union, new_mask)
                    if oracle_counters is not None and local_ctr is not None: local_ctr["apply"] += 1
                    stats.start('get_mask.main_loop.edge.apply_and_prune')
                    _t0 = _perf_now()
                    tmp = popped.apply_and_prune(intersect)
                    _t1 = _perf_now()
                    stats.stop('get_mask.main_loop.edge.apply_and_prune')
                    if oracle_counters is not None and local_ctr is not None:
                        local_ctr["apply_time_ms"] = local_ctr.get("apply_time_ms", 0.0) + ((_t1 - _t0) * 1000.0)
                    if tmp.is_empty():
                        continue
                    source_after_apply = tmp
                    local_apply_cache[key_apply] = tmp
            if not peeked: continue

            grouped: Dict[int, List[int]] = {}
            m = edge.state_to_dest
            for sid in peeked:
                dest_list = m.get(sid)
                if not dest_list: continue
                for dest_j in dest_list:
                    lst = grouped.get(dest_j)
                    if lst is None: grouped[dest_j] = [sid]
                    else: lst.append(sid)

            dest_keys = list(grouped.keys())
            if dest_scores:
                # Sort by destination node score (desc), then by original index (asc) for stability
                dest_keys.sort(key=lambda j: (-dest_scores.get(int(edge.dests[j].dest_idx), 0), j))
            else:
                # Iterate grouped dests in ascending order for locality
                dest_keys.sort()
            for dest_j in dest_keys:
                if dests_proc >= max_dests:
                    priority = (-self.max_depth.get(node_id, 0), edge_i, dest_j)
                    yield Suspend(priority, depth)
                    dests_proc = 0
                dest = edge.dests[dest_j]
                values_to_keep = grouped[dest_j]
                # If all heads survive, reuse popped directly
                if len(values_to_keep) == len(peeked):
                    child_gss = source_after_apply
                else:
                    # Global/local cache for isolate_many
                    key_isolate = (id(source_after_apply), tuple(values_to_keep))
                    if isolate_cache is not None and key_isolate in isolate_cache:
                        stats.inc('get_mask.isolate_many.cache_hits')
                        child_gss = isolate_cache[key_isolate]
                    else:
                        stats.start('get_mask.main_loop.edge.isolate_many')
                        _t0 = _perf_now()
                        child_gss = source_after_apply.isolate_many(values_to_keep)
                        _t1 = _perf_now()
                        stats.stop('get_mask.main_loop.edge.isolate_many')
                        if oracle_counters is not None and local_ctr is not None: local_ctr["isolate"] += 1
                        if oracle_counters is not None and local_ctr is not None:
                            local_ctr["isolate_calls"] = local_ctr.get("isolate_calls", 0) + 1
                            local_ctr["isolate_time_ms"] = local_ctr.get("isolate_time_ms", 0.0) + ((_t1 - _t0) * 1000.0)
                        if isolate_cache is not None:
                            isolate_cache[key_isolate] = child_gss
                if child_gss.is_empty(): continue
                d: NodeID = int(dest.dest_idx)
                yield Enqueue(d, child_gss, depth + 1)
                dests_proc += 1
                if oracle_counters is not None and local_ctr is not None:
                    local_ctr["dests"] = local_ctr.get("dests", 0) + 1

            if edges_proc >= max_edges:
                # Suspend work after a batch to allow priority reordering at the heap level
                priority = (-self.max_depth.get(node_id, 0), edge_i + 1, 0)
                yield Suspend(int(node_id), priority, depth)
                edges_proc = 0
            edges_proc += 1
        # When max_dests suspend triggers
        # (Handled inside the loop above with proper node_id propagation)

    def _ensure_reverse_adjacency(self) -> Dict[int, Set[int]]:
        """Build and cache reverse adjacency: dest_node -> set(parent_nodes)."""
        if getattr(self, 'reverse_adj', None) is not None:
            return self.reverse_adj
        parents: Dict[int, Set[int]] = collections.defaultdict(set)
        for u, node in self.arena.items():
            for edge in node.children:
                for dest in edge.dests:
                    parents[int(dest.dest_idx)].add(int(u))
        self.reverse_adj = dict(parents)
        return self.reverse_adj

    def _oracle_compute_reward_masks(self, end_events: List[Tuple[int, LLMTokenSet]]) -> Dict[int, LLMTokenSet]:
        """Compute target-specific reward masks per node:
        For each node u, reward[u] is the set of tokens that appear in some end_event and
        can flow from u to that end through edges whose llm_bv admit those tokens.
        Monotone fixed-point: initialized at end nodes with their delta masks, then
        propagated upstream via edge.llm_bv intersections.
        """
        reward: Dict[int, LLMTokenSet] = {}
        # Seed from end events: coalesce by end-node id
        for nid, delta in end_events:
            if delta is None or delta.is_empty():
                continue
            k = int(nid)
            prev = reward.get(k)
            reward[k] = delta if prev is None else prev.union(delta)

        if not reward:
            return {}

        # Fixed-point propagation upstream
        changed = True
        while changed:
            changed = False
            for u, node in self.arena.items():
                if not node.children:
                    continue
                accum = RangeSet.empty()
                for edge in node.children:
                    # Union of reward at all children along this edge
                    child_union = RangeSet.empty()
                    for d in edge.dests:
                        cm = reward.get(int(d.dest_idx))
                        if cm is not None and not cm.is_empty():
                            child_union |= cm
                    if not child_union.is_empty():
                        thru = child_union.intersection(edge.llm_bv)
                        if not thru.is_empty():
                            accum |= thru
                if accum.is_empty():
                    continue
                prev = reward.get(int(u))
                updated = accum if prev is None else prev.union(accum)
                if prev is None or prev != updated:
                    reward[int(u)] = updated
                    changed = True
        return reward

    def _oracle_build_guidance_advanced(
        self,
        plan: Dict,
        reward_masks: Dict[int, LLMTokenSet],
        target_mask: LLMTokenSet,
        jitter: int = 0,
    ) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """Target-aware guidance (advanced).
        - Node scores ~ (reward tokens at node intersect target)/(estimated node cost).
        - Edge order per node is by the sum of child reward tokens (intersected with target).
        - Optional jitter adds small random noise to break ties differently across trials.
        """
        node_costs_raw: Dict[int, Dict[str, int]] = {int(k): v for k, v in plan.get('node_costs', {}).items()}
        def node_cost(u: int) -> float:
            c = node_costs_raw.get(int(u), {"edges": 0, "apply": 0, "isolate": 0})
            return float(c.get("edges", 0)) + self.oracle_apply_weight * float(c.get("apply", 0)) + self.oracle_isolate_weight * float(c.get("isolate", 0)) + 1.0

        scores: Dict[int, int] = {}
        # Node scores: tokens per cost (scaled)
        for u, m in reward_masks.items():
            if m is None or m.is_empty():
                continue
            tok = len(m.intersection(target_mask))
            if tok <= 0:
                continue
            base = int(1000.0 * float(tok) / node_cost(int(u)))
            if jitter > 0:
                base += random.randint(0, int(jitter))
            if base > 0:
                scores[int(u)] = base

        # Edge ordering: by sum of child reward tokens (intersected with target)
        edge_order_map: Dict[int, List[int]] = {}
        for u, node in self.arena.items():
            if not node.children:
                continue
            weighted: List[Tuple[int, int]] = []
            for idx, edge in enumerate(node.children):
                s = 0
                for d in edge.dests:
                    cm = reward_masks.get(int(d.dest_idx))
                    if cm is not None and not cm.is_empty():
                        s += int(len(cm.intersection(target_mask)))
                if jitter > 0 and s > 0:
                    s += random.randint(0, int(jitter))
                weighted.append((idx, s))
            if any(w > 0 for _, w in weighted):
                weighted.sort(key=lambda t: t[1], reverse=True)
                edge_order_map[int(u)] = [idx for (idx, _) in weighted]
        return scores, edge_order_map

    def _oracle_compute_reward_masks_from_end_masks(self, end_masks: Dict[int, LLMTokenSet]) -> Dict[int, LLMTokenSet]:
        """Exact reward propagation: seed end nodes with their exact end masks (union across all visits),
        then propagate upstream via edge.llm_bv intersections until a fixed point.
        This avoids order-dependent artifacts of delta-based end events."""
        reward: Dict[int, LLMTokenSet] = {}
        # Seed end nodes
        for nid, m in (end_masks or {}).items():
            if m is None or m.is_empty():
                continue
            reward[int(nid)] = m

        if not reward:
            return {}

        changed = True
        while changed:
            changed = False
            for u, node in self.arena.items():
                if not node.children:
                    continue
                accum = RangeSet.empty()
                for edge in node.children:
                    child_union = RangeSet.empty()
                    for d in edge.dests:
                        cm = reward.get(int(d.dest_idx))
                        if cm is not None and not cm.is_empty():
                            child_union |= cm
                    if not child_union.is_empty():
                        thru = child_union.intersection(edge.llm_bv)
                        if not thru.is_empty():
                            accum |= thru
                if accum.is_empty():
                    continue
                prev = reward.get(int(u))
                updated = accum if prev is None else prev.union(accum)
                if prev is None or prev != updated:
                    reward[int(u)] = updated
                    changed = True
        return reward

    def _oracle_estimate_end_cost(self, end_node_id: int, analysis_node_costs: Optional[Dict[int, Dict[str, int]]] = None) -> float:
        """Estimate minimal cost from any root to the given end node by reverse Dijkstra over parents.
        Edge costs are modeled by node costs (sum along a path). This is a heuristic but helps the oracle order ends."""
        rev = self._ensure_reverse_adjacency()
        roots: Set[int] = set(int(r) for r in self.roots_map.values())
        # If end is itself a root
        if int(end_node_id) in roots:
            return self._oracle_node_cost(int(end_node_id), analysis_node_costs)
        # Dijkstra from end toward roots via parents (reverse edges)
        import heapq as _hq
        INF = float('inf')
        dist: Dict[int, float] = {}
        pq: List[Tuple[float, int]] = []
        start = int(end_node_id)
        dist[start] = self._oracle_node_cost(start, analysis_node_costs)
        _hq.heappush(pq, (dist[start], start))
        best_to_any_root = INF
        while pq:
            cd, u = _hq.heappop(pq)
            if cd > dist.get(u, INF):
                continue
            if u in roots:
                best_to_any_root = min(best_to_any_root, cd)
                break
            for parent in rev.get(u, ()):
                v = int(parent)
                nd = cd + self._oracle_node_cost(v, analysis_node_costs)
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    _hq.heappush(pq, (nd, v))
        if best_to_any_root == INF:
            # Fallback: just return cost of this node
            return self._oracle_node_cost(int(end_node_id), analysis_node_costs)
        return best_to_any_root

    def _oracle_beam_search_end_sequence(
        self,
        end_masks: Dict[int, LLMTokenSet],
        target_mask: LLMTokenSet,
        analysis_node_costs: Optional[Dict[int, Dict[str, int]]] = None,
        trials: int = 8,
        beam_width: int = 8,
        jitter: int = 0,
    ) -> List[int]:
        """Cost-aware beam search over end-node sequences to maximize token gain per cost.
        Returns one promising end-node ordering."""
        if not end_masks:
            return []
        # Precompute costs and sizes
        ends = list(int(k) for k in end_masks.keys())
        end_cost: Dict[int, float] = {e: self._oracle_estimate_end_cost(e, analysis_node_costs) for e in ends}
        # Precompute mask sizes intersected with target
        end_target_sizes: Dict[int, int] = {}
        for e in ends:
            m = end_masks.get(int(e))
            end_target_sizes[int(e)] = int(len((m or RangeSet.empty()).intersection(target_mask)))
        rng = random.Random(31337)
        best_seq: List[int] = []
        best_score = -1.0
        # Beam state: (covered_mask, sequence, score)
        for _ in range(max(1, int(trials))):
            beam: List[Tuple[LLMTokenSet, List[int], float]] = [(RangeSet.empty(), [], 0.0)]
            visited_seq: Set[Tuple[int, ...]] = set()
            while beam:
                # Goal check
                beam.sort(key=lambda t: (-len(t[0]), -t[2]))
                if not beam:
                    break
                new_beam: List[Tuple[LLMTokenSet, List[int], float]] = []
                for covered, seq, sc in beam[:max(1, int(beam_width))]:
                    remaining = target_mask.difference(covered)
                    if remaining.is_empty():
                        if sc > best_score:
                            best_score = sc
                            best_seq = seq
                        continue
                    # Rank candidates by gain/cost with jitter
                    candidates: List[Tuple[float, int, int]] = []
                    for e in ends:
                        if e in seq:
                            continue
                        m = end_masks.get(int(e), RangeSet.empty())
                        gain = len(remaining.intersection(m))
                        if gain <= 0:
                            continue
                        cost = end_cost.get(int(e), 1.0)
                        score = float(gain) / max(1.0, cost)
                        if jitter > 0:
                            score *= (1.0 + rng.uniform(-float(jitter)/100.0, float(jitter)/100.0))
                        candidates.append((score, int(e), int(gain)))
                    candidates.sort(key=lambda t: (-t[0], -t[2]))
                    if not candidates:
                        # dead end
                        continue
                    # Expand top-k candidates from this beam state
                    k = min(len(candidates), max(1, int(beam_width)))
                    for i in range(k):
                        _, e, _ = candidates[i]
                        m = end_masks[int(e)]
                        new_cov = covered.union(m)
                        new_seq = seq + [int(e)]
                        key = tuple(new_seq)
                        if key in visited_seq:
                            continue
                        visited_seq.add(key)
                        # New state score = covered tokens cardinality / sum end costs
                        tot_cost = sum(end_cost.get(int(x), 1.0) for x in new_seq)
                        new_score = float(len(new_cov.intersection(target_mask))) / max(1.0, tot_cost)
                        new_beam.append((new_cov, new_seq, new_score))
                beam = new_beam
            # If we didn't complete coverage in this trial, still accept best partial state
            if not best_seq and beam:
                beam.sort(key=lambda t: (-len(t[0]), -t[2]))
                best_seq = beam[0][1]
        return best_seq

    def _oracle_local_edge_order_search(
        self,
        reward_masks: Dict[int, LLMTokenSet],
        target_mask: LLMTokenSet,
        analysis_node_costs: Optional[Dict[int, Dict[str, int]]] = None,
        top_k: int = 10,
        trials_per_node: int = 32,
        jitter: int = 0,
    ) -> Dict[int, List[int]]:
        """Randomized local search for edge ordering at top 'mega nodes'.
        For each selected node u, sample several edge orders biased by child reward tokens and pick the best.
        Returns a partial edge_order_map (node_id -> list of edge indices)."""
        node_costs = analysis_node_costs or {}
        # Rank nodes by "hotness"
        ranked: List[Tuple[float, int]] = []
        for nid_str, cnt in node_costs.items():
            try:
                nid = int(nid_str)
            except Exception:
                nid = int(nid_str)
            edges = float(cnt.get("edges", 0))
            apply_calls = float(cnt.get("apply", 0))
            isolate_calls = float(cnt.get("isolate", 0))
            score = edges + self.oracle_apply_weight * apply_calls + self.oracle_isolate_weight * isolate_calls
            ranked.append((score, nid))
        ranked.sort(key=lambda t: t[0], reverse=True)
        selected = [nid for (_, nid) in ranked[:max(1, int(top_k))]]
        rng = random.Random(20240518)
        result: Dict[int, List[int]] = {}
        for u in selected:
            node = self.arena.get(int(u))
            if not node or not node.children:
                continue
            # Compute deterministic edge weights from child rewards
            base_weights: List[Tuple[int, int]] = []
            for idx, edge in enumerate(node.children):
                s = 0
                for d in edge.dests:
                    cm = reward_masks.get(int(d.dest_idx))
                    if cm is not None and not cm.is_empty():
                        s += int(len(cm.intersection(target_mask)))
                base_weights.append((idx, s))
            if not any(w > 0 for (_, w) in base_weights):
                # Nothing to bias on; skip
                continue
            # Sample permutations with softmax-like bias
            best_order: List[int] = []
            best_score: int = -1
            for _ in range(max(1, int(trials_per_node))):
                # Shuffle with bias: larger weight edges tend to go first
                items = base_weights[:]
                perm: List[int] = []
                while items:
                    # compute probabilities proportional to w + epsilon + jitter
                    weights = []
                    for (_, w) in items:
                        val = max(1, w)
                        if jitter > 0:
                            val = int(val * (1.0 + rng.uniform(-float(jitter)/100.0, float(jitter)/100.0)))
                            val = max(1, val)
                        weights.append(val)
                    total = float(sum(weights))
                    r = rng.uniform(0.0, total)
                    acc = 0.0
                    pick = 0
                    for i, val in enumerate(weights):
                        acc += float(val)
                        if r <= acc:
                            pick = i
                            break
                    idx, w = items.pop(pick)
                    perm.append(idx)
                # Score permutation by cumulative reward if taken in this order
                # Approximate: sum of weights; better if high weights front-loaded
                cum = 0
                for rank, idx in enumerate(perm):
                    w = 0
                    for (j, ww) in base_weights:
                        if j == idx:
                            w = ww
                            break
                    # Discount later edges (prefer early gain)
                    cum += int(w / (1 + rank))
                if cum > best_score:
                    best_score = cum
                    best_order = perm
            if best_order:
                result[int(u)] = best_order
        return result

    def _oracle_build_guidance_from_plan(self, plan: Dict, jitter: int = 0) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """Build a stronger oracle guidance from planning analysis:
        - plan['end_events']: list[(end_node_id, delta_mask)]
        - plan['frontiers']: node_id -> RangeSet (descendant closure ∩ gss_acc at visit)
        - plan['node_costs']: node_id -> {edges:int, apply:int, isolate:int}
        Returns (guided_scores, edge_order_map).
        """
        # 1) Coalesce end masks
        items_by_end: Dict[int, LLMTokenSet] = {}
        universe = RangeSet.empty()
        for nid, delta in plan.get('end_events', []):
            if delta is None or delta.is_empty():
                continue
            prev = items_by_end.get(int(nid))
            if prev is None:
                items_by_end[int(nid)] = delta
            else:
                items_by_end[int(nid)] = prev.union(delta)
        for v in items_by_end.values():
            universe |= v
        end_items: List[Tuple[int, LLMTokenSet]] = list(items_by_end.items())
        frontiers: Dict[int, LLMTokenSet] = {int(k): v for k, v in plan.get('frontiers', {}).items()}
        node_costs_raw: Dict[int, Dict[str, int]] = {int(k): v for k, v in plan.get('node_costs', {}).items()}
        # 2) Build node cost weights
        def node_weight(u: int) -> float:
            c = node_costs_raw.get(int(u), {"edges": 0, "apply": 0, "isolate": 0})
            return float(c.get("edges", 0)) + self.oracle_apply_weight * float(c.get("apply", 0)) + self.oracle_isolate_weight * float(c.get("isolate", 0))
        # 3) Greedy end sequence maximizing pruning benefit + token coverage
        selected_order: List[int] = []
        if not end_items or universe.is_empty():
            return self._oracle_prepare_guidance([])

        X = RangeSet.empty()
        covered_nodes: Set[int] = set()
        candidate_pool = list(end_items)

        # Pre-calculate nodes covered by an empty mask
        for u, fmask in frontiers.items():
            if fmask.is_empty():
                covered_nodes.add(u)

        while candidate_pool:
            remaining_tok = universe.difference(X)
            best_id, best_mask, best_idx, best_score = None, None, -1, -1.0

            for i, (nid, mask) in enumerate(candidate_pool):
                tok_gain = len(remaining_tok.intersection(mask))
                prune_gain_weight = 0.0
                if frontiers:
                    newX = X.union(mask)
                    for u, fmask in frontiers.items():
                        if u not in covered_nodes and fmask.issubset(newX):
                            prune_gain_weight += node_weight(u)
                if jitter > 0 and prune_gain_weight > 0:
                    prune_gain_weight *= (1.0 + random.uniform(-jitter / 100.0, jitter / 100.0))

                score = self.oracle_prune_weight * prune_gain_weight + self.oracle_token_weight * float(tok_gain)
                if score > best_score:
                    best_score, best_id, best_mask, best_idx = score, nid, mask, i

            if best_id is None:
                break

            selected_order.append(best_id)
            candidate_pool.pop(best_idx)
            
            X = X.union(best_mask)
            newly_covered = set()
            for u, fmask in frontiers.items():
                if u not in covered_nodes and fmask.issubset(X):
                    newly_covered.add(u)
            covered_nodes.update(newly_covered)

            if X.issuperset(universe):
                break
        
        for nid, _ in candidate_pool:
            selected_order.append(nid)
        # 4) Convert sequence into per-node "prune step"
        # When does each node u become prunable (frontier covered)?
        prune_step: Dict[int, int] = {}
        acc = RangeSet.empty()
        # Build lookup of end id -> mask
        end_mask_map: Dict[int, LLMTokenSet] = dict(end_items)
        for idx, eid in enumerate(selected_order, start=1):
            m = end_mask_map.get(eid)
            if m is None:
                continue
            acc = acc.union(m)
            for u, fmask in frontiers.items():
                if u in prune_step:
                    continue
                if fmask.difference(acc).is_empty():
                    prune_step[u] = idx
        # Nodes never covered get a very late step value
        default_step = len(selected_order) + 1 if selected_order else 1
        # 5) Build guided node scores (higher means schedule earlier)
        scores: Dict[int, int] = {}
        for u in self.arena.keys():
            step = prune_step.get(int(u), default_step)
            # Earlier step -> higher score. Use a large constant base to avoid negatives in priority.
            scores[int(u)] = max(0, (len(selected_order) + 1) - int(step))
        # 6) Edge ordering: sort each node's edges by sum of child destination scores (desc)
        edge_order_map: Dict[int, List[int]] = {}
        for u, node in self.arena.items():
            if not node.children:
                continue
            edge_scores: List[Tuple[int, int]] = []
            for idx, edge in enumerate(node.children):
                s = 0
                for d in edge.dests:
                    s += int(scores.get(int(d.dest_idx), 0))
                edge_scores.append((idx, s))
            if any(s > 0 for (_, s) in edge_scores):
                ordered = sorted(edge_scores, key=lambda t: t[1], reverse=True)
                edge_order_map[int(u)] = [idx for (idx, _) in ordered]
        return scores, edge_order_map

    def _oracle_report_hotspots(self, plan: Dict, top_k: int = 10) -> None:
        """Report top hotspots from planning counters to help identify 'mega nodes'."""
        node_costs = plan.get('node_costs', {}) or {}
        if not node_costs:
            print("[oracle] No per-node costs collected.")
            return
        # Compose a sortable score with time if available
        ranked = []
        for nid_str, cnt in node_costs.items():
            try:
                nid = int(nid_str) if isinstance(nid_str, str) else int(nid_str)
            except Exception:
                nid = int(nid_str)
            edges = int(cnt.get("edges", 0))
            apply_calls = int(cnt.get("apply", 0))
            isolate_calls = int(cnt.get("isolate", 0))
            popn_calls = int(cnt.get("popn_calls", 0))
            dests = int(cnt.get("dests", 0))
            # time fields are optional floats
            popn_ms = float(cnt.get("popn_time_ms", 0.0))
            apply_ms = float(cnt.get("apply_time_ms", 0.0))
            isolate_ms = float(cnt.get("isolate_time_ms", 0.0))
            total_ms = popn_ms + apply_ms + isolate_ms
            # Composite score: weight edges+apply+isolate + time
            score = float(edges) + self.oracle_apply_weight * float(apply_calls) + self.oracle_isolate_weight * float(isolate_calls) + 0.001 * total_ms
            ranked.append((score, nid, edges, apply_calls, isolate_calls, popn_calls, dests, total_ms))
        ranked.sort(key=lambda t: t[0], reverse=True)
        print("\n[oracle] Top hotspots (per-node costs from planning):")
        for i, item in enumerate(ranked[:max(1, int(top_k))], start=1):
            score, nid, edges, apply_calls, isolate_calls, popn_calls, dests, total_ms = item
            md = self.max_depth.get(int(nid), 0)
            ch = len(self.arena.get(int(nid), ArenaNode()).children) if int(nid) in self.arena else 0
            print(f"  {i:2d}) node={nid} depth={md} children={ch} score={score:.2f} total_ms={total_ms:.3f} "
                  f"| edges={edges} apply={apply_calls} isolate={isolate_calls} popn_calls={popn_calls} dests={dests}")
        print()

    def _oracle_node_cost(self, node_id: int, analysis_node_costs: Optional[Dict[int, Dict[str, int]]] = None) -> float:
        if analysis_node_costs is not None and int(node_id) in analysis_node_costs:
            c = analysis_node_costs[int(node_id)]
            return float(c.get("edges", 0)) + self.oracle_apply_weight * float(c.get("apply", 0)) + self.oracle_isolate_weight * float(c.get("isolate", 0)) + 1.0
        node = self.arena.get(int(node_id))
        if not node:
            return 1.0
        # Cheap fallback: edges + 0.25 * total dest count + 1
        total_dests = 0
        for e in node.children:
            total_dests += len(e.dests)
        return 1.0 + float(len(node.children)) + 0.25 * float(total_dests)

    def _compute_oracle_min_cover(self, end_union_events: List[Tuple[int, LLMTokenSet]]) -> List[int]:
        """Given a list of (end_node_id, delta_mask) events collected from a full planning run,
        compute a minimal set of end nodes whose union covers the final mask.
        If the number of distinct end nodes is <= oracle_exact_cover_threshold, run exact branch-and-bound;
        otherwise, run greedy maximum-coverage.
        Returns a list of selected end node IDs (ordering chosen greedily by marginal gain for replay)."""
        # Coalesce by end-node id
        items_by_end: Dict[int, LLMTokenSet] = {}
        universe = RangeSet.empty()
        for nid, delta in end_union_events:
            if delta is None or delta.is_empty():
                continue
            prev = items_by_end.get(nid)
            if prev is None:
                items_by_end[nid] = delta
            else:
                items_by_end[nid] = prev.union(delta)
        # Remove empties if any remained
        items_by_end = {k: v for k, v in items_by_end.items() if v is not None and not v.is_empty()}
        # Build universe
        for v in items_by_end.values():
            universe |= v
        items: List[Tuple[int, LLMTokenSet]] = list(items_by_end.items())
        # Nothing to cover => return empty plan
        if universe.is_empty() or not items:
            return []

        # Utility: greedy sequencing (also used to order final selected items)
        def greedy_sequence(cand_items: List[Tuple[int, LLMTokenSet]], target: LLMTokenSet) -> List[int]:
            remaining = target
            selected: List[int] = []
            # Copy items to avoid mutating caller
            pool = list(cand_items)
            while not remaining.is_empty() and pool:
                best_idx = None
                best_gain = 0
                for i, (nid, mask) in enumerate(pool):
                    gain = len(remaining.intersection(mask))
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx is None or best_gain == 0:
                    break
                nid, mask = pool.pop(best_idx)
                selected.append(nid)
                remaining = remaining.difference(mask)
            return selected

        # Exact cover via branch-and-bound if small
        if len(items) <= max(0, int(self.oracle_exact_cover_threshold)):
            # Pre-sort items by descending size to improve pruning
            items.sort(key=lambda it: len(it[1]), reverse=True)
            best_solution: Optional[List[int]] = None

            # Precompute masks list for quick access
            masks_only = [m for (_, m) in items]
            node_ids = [nid for (nid, _) in items]

            def dfs(remaining: LLMTokenSet, start_idx: int, chosen_idxs: List[int]) -> None:
                nonlocal best_solution
                if remaining.is_empty():
                    # Found a feasible cover
                    if best_solution is None or len(chosen_idxs) < len(best_solution):
                        best_solution = chosen_idxs.copy()
                    return
                if best_solution is not None and len(chosen_idxs) >= len(best_solution):
                    return
                # Lower bound: ceil(|remaining| / max possible coverage of any single item)
                max_gain = 0
                for i in range(start_idx, len(items)):
                    g = len(remaining.intersection(masks_only[i]))
                    if g > max_gain:
                        max_gain = g
                if max_gain == 0:
                    return  # impossible to cover remaining with available items
                lb = int(math.ceil(len(remaining) / max(1, max_gain)))
                if best_solution is not None and len(chosen_idxs) + lb >= len(best_solution):
                    return  # cannot beat current best
                # Heuristic: try the item with largest marginal first
                best_i = None
                best_i_gain = -1
                for i in range(start_idx, len(items)):
                    g = len(remaining.intersection(masks_only[i]))
                    if g > best_i_gain:
                        best_i_gain = g
                        best_i = i
                # Include branch
                if best_i is not None and best_i_gain > 0:
                    new_remaining = remaining.difference(masks_only[best_i])
                    dfs(new_remaining, best_i + 1, chosen_idxs + [best_i])
                # Exclude branch: skip best_i and continue
                # This helps ensure exactness if including best_i is not optimal
                next_start = start_idx
                for i in range(start_idx, len(items)):
                    if i == best_i:
                        continue
                    # Try excluding best_i by exploring others; we move sequentially to avoid combinatorial explosion
                    # This is still exponential but OK under small thresholds
                    dfs(remaining, i + 1, chosen_idxs)
                    break  # only move one step to keep branching factor in check

            dfs(universe, 0, [])
            if best_solution is None:
                # Fallback to greedy (shouldn't happen unless masks do not cover universe)
                return greedy_sequence(items, universe)
            # Order the chosen items greedily for replay
            chosen_items = [(node_ids[i], masks_only[i]) for i in best_solution]
            return greedy_sequence(chosen_items, universe)
        else:
            # Greedy coverage for larger sets
            return greedy_sequence(items, universe)

    def _oracle_prepare_guidance(self, selected_end_nodes: List[int]) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
        """Compute guided priorities from selected end nodes:
        - scores: node_id -> count of selected ends reachable downstream (via reverse edges)
        - edge_order_map: node_id -> list of edge indices sorted by descending score of destination
        """
        rev = self._ensure_reverse_adjacency()
        scores: Dict[int, int] = collections.defaultdict(int)
        # For each selected end, propagate +1 to all ancestors
        for end_nid in selected_end_nodes:
            seen: Set[int] = set()
            dq = collections.deque([end_nid])
            while dq:
                cur = dq.popleft()
                if cur in seen:
                    continue
                seen.add(cur)
                scores[cur] += 1
                for parent in rev.get(cur, ()):
                    if parent not in seen:
                        dq.append(parent)
        # Build per-node edge ordering by sum of destination scores
        edge_order_map: Dict[int, List[int]] = {}
        for u, node in self.arena.items():
            if not node.children:
                continue
            edge_scores: List[Tuple[int, int]] = []
            for idx, edge in enumerate(node.children):
                s = 0
                for d in edge.dests:
                    s += int(scores.get(int(d.dest_idx), 0))
                edge_scores.append((idx, s))
            # If all scores zero, skip ordering for that node
            if any(s > 0 for (_, s) in edge_scores):
                ordered = sorted(edge_scores, key=lambda t: t[1], reverse=True)
                edge_order_map[int(u)] = [idx for (idx, _) in ordered]
        return dict(scores), edge_order_map

    def _priority_for_node(
        self,
        node_id: int,
        depth: int,
        remaining_mask: LLMTokenSet,
        guided_scores: Optional[Dict[int, int]] = None,
        oracle_reward_mask: Optional[Dict[int, LLMTokenSet]] = None,
        analysis_node_costs: Optional[Dict[int, Dict[str, int]]] = None,
    ) -> Tuple:
        """Dynamic priority tuple for heap scheduling.
        Higher token gain per unit cost and larger gain are preferred.
        Tie-breakers: guided score (if any) and max_depth.
        """
        node_id_i = int(node_id)
        # Estimated token gain if we fully explore this node's subtree.
        if oracle_reward_mask is not None:
            m = oracle_reward_mask.get(node_id_i, RangeSet.empty())
        else:
            an = self.arena.get(node_id_i)
            m = an.llm_bv_descendant if an else RangeSet.empty()
        gain = 0
        if m is not None and not m.is_empty():
            gain = int(len(m.intersection(remaining_mask)))
        cost = self._oracle_node_cost(node_id_i, analysis_node_costs)
        gs = int(guided_scores.get(node_id_i, 0)) if guided_scores is not None else 0
        # Negative for min-heap; depth tiebreaker to encourage shallower early
        return (-float(gain) / max(1.0, float(cost)), -int(gain), -int(gs), -int(self.max_depth.get(node_id_i, 0)), int(depth))

    def _get_mask_run(
        self,
        guided_scores: Optional[Dict[int, int]] = None,
        edge_order_map: Optional[Dict[int, List[int]]] = None,
        record_end_unions: bool = False,
        oracle_target_mask: Optional[LLMTokenSet] = None,
        oracle_reward_mask: Optional[Dict[int, LLMTokenSet]] = None,
        oracle_node_costs: Optional[Dict[int, Dict[str, int]]] = None,
        ) -> Tuple[Dict, Dict]:

        """Internal engine behind get_mask.
        If guided_scores/edge_order_map are provided, they influence traversal priority and edge ordering.
        If record_end_unions is True:
          - Returns: (timed_output_dict, oracle_data) where oracle_data is a dict with:
              'end_events': list of (end_node_id, delta_mask),
              'frontiers': node_id -> RangeSet, 'node_costs': node_id -> dict(counts)
        Returns (timed_output_dict, oracle_data)."""
        stats = Stats.get()
        stats.start('get_mask')
        stats.counts['get_mask.traversal.max_depth'] = 0
        stats.inc('get_mask.initial_tokenizer_states', len(self.state))

        # Global caches across the whole traversal
        global_pop_cache: Dict[Tuple[int, int], Tuple[Any, Any, List[int], StateIDSet]] = {}
        apply_cache_by_bv: Dict[Tuple[int, int], GSS] = {}
        isolate_many_cache: Dict[Tuple[int, Tuple[int, ...]], GSS] = {}
        @dataclass
        class HeapItem:
            priority: Any
            item: Any

            def __lt__(self, other: 'HeapItem') -> bool:
                if not isinstance(other, HeapItem):
                    return NotImplemented

                return self.priority < other.priority
        # When oracle_target_mask is provided, restrict the universe to the exact final target discovered in planning.
        all_ones = oracle_target_mask if oracle_target_mask is not None else self.all_internal_llm_tokens_bitset
        final_mask = RangeSet.empty()
        work_heap = []
        end_union_events: List[Tuple[int, LLMTokenSet]] = []

        @_acc_memoize(use_value_cache=False)
        # Planning: remove impossible tokens implied by terminal disallows at the roots
        # and start with all remaining tokens allowed (all_ones minus disallowed).
        # This is exact and independent of traversal order.
        def initialize_acc(acc: PyAcc) -> PyAcc:
            disallowed = RangeSet.empty()
            for tsid, terms in acc.terminals_union.items():
                if tsid in self.possible_matches_cache:
                    term_map = self.possible_matches_cache[tsid]
                    for term_id in terms.iter_indices():
                        if term_id in term_map: disallowed |= term_map[term_id]
            return PyAcc({}, all_ones.difference(disallowed))

        stats.start('get_mask.seeding')
        init_cache = {}
        for sid, gss in self.state.items():
            r = self.roots_map[int(sid)]
            gss_init = gss.apply(initialize_acc, init_cache)
            if not gss_init.is_empty():
                base_pri = (-self.max_depth.get(r, 0), 0, 0)
                # Dynamic, target-aware seeding priority if reward masks present
                if oracle_reward_mask is not None and self.oracle_dynamic_prioritization:
                    priority = self._priority_for_node(
                        r, 0, RangeSet.empty() if oracle_target_mask is None else oracle_target_mask,
                        guided_scores=guided_scores,
                        oracle_reward_mask=oracle_reward_mask,
                        analysis_node_costs=oracle_node_costs,
                    )
                elif guided_scores is not None:
                    # Higher guided_scores -> earlier (we negate to get min-heap behavior)
                    priority = (-int(guided_scores.get(r, 0)),) + base_pri
                else:
                    priority = base_pri
                heapq.heappush(work_heap, HeapItem(priority, WorkItemNew(r, gss_init, 0)))
        stats.stop('get_mask.seeding')

        stats.start('get_mask.main_loop')
        remaining_mask = all_ones
        # Oracle planning accumulators
        analysis_frontiers: Dict[int, LLMTokenSet] = {} if record_end_unions else None
        analysis_node_costs: Dict[int, Dict[str, int]] = {} if record_end_unions else None
        analysis_end_masks: Dict[int, LLMTokenSet] = {} if record_end_unions else None
        while work_heap:
            if not record_end_unions and remaining_mask.is_empty():
                stats.inc('get_mask.early_exit_full_mask')
                break

            stats.inc('get_mask.traversal.depth_heap.pops')
            heap_item = heapq.heappop(work_heap)
            priority, work = heap_item.priority, heap_item.item

            if isinstance(work, WorkItemSuspended):
                gen, work_llm_mask, depth = work.generator, work.llm_mask, work.depth

                if work_llm_mask.isdisjoint(remaining_mask):
                    continue
            elif isinstance(work, WorkItemNew):
                stats.inc('get_mask.traversal.nodes_processed')
                node_id, gss_node, depth = work.node_id, work.gss, work.depth
                stats.counts['get_mask.traversal.max_depth'] = max(stats.counts.get('get_mask.traversal.max_depth', 0), depth)
                stats.inc('get_mask.gss.at_node.accs.sum', len(getattr(gss_node, 'get_all_accs', lambda: [])()))

                assert isinstance(node_id, int)
                assert isinstance(gss_node, GSS)
                stats.start('get_mask.main_loop.node.reduce_acc')
                gss_acc = gss_node.reduce_acc()
                stats.stop('get_mask.main_loop.node.reduce_acc')

                if self.is_end(node_id):
                    stats.inc('get_mask.traversal.end_nodes')
                    # Record delta for planning analysis (before union)
                    if not final_mask.issuperset(gss_acc.llm_mask):
                        delta = gss_acc.llm_mask.difference(final_mask)
                        if record_end_unions and not delta.is_empty():
                            end_union_events.append((int(node_id), delta))
                    # Record exact end mask (union across visits), independent of delta
                    if record_end_unions and analysis_end_masks is not None:
                        prev_mask = analysis_end_masks.get(int(node_id))
                        analysis_end_masks[int(node_id)] = gss_acc.llm_mask if prev_mask is None else prev_mask.union(gss_acc.llm_mask)
                    stats.start('get_mask.main_loop.end_node.final_mask_union')
                    final_mask |= gss_acc.llm_mask
                    stats.stop('get_mask.main_loop.end_node.final_mask_union')
                    remaining_mask = all_ones.difference(final_mask)
                    if remaining_mask.is_empty():
                        stats.inc('get_mask.early_exit_full_mask')
                        break

                a_node = self.arena.get(node_id)
                # Stronger upper bound than llm_bv_union: descendant closure
                if oracle_reward_mask is not None and a_node:
                    node_desc_mask = oracle_reward_mask.get(int(node_id), RangeSet.empty())
                else:
                    node_desc_mask = a_node.llm_bv_descendant if a_node and a_node.llm_bv_descendant is not None else RangeSet.empty()
                work_llm_mask = node_desc_mask.intersection(gss_acc.llm_mask) if a_node else RangeSet.empty()

                if not a_node or not a_node.children:
                    # Still record frontier mask for planning if requested
                    if record_end_unions and a_node:
                        if analysis_frontiers is not None:
                            if not work_llm_mask.is_empty():
                                prev = analysis_frontiers.get(int(node_id))
                                analysis_frontiers[int(node_id)] = work_llm_mask if prev is None else prev.union(work_llm_mask)
                    continue
                # Closure-based pruning
                if work_llm_mask.isdisjoint(remaining_mask):
                    stats.inc('get_mask.main_loop.node.closure_pruned')
                    continue

                gen = self._process_internal_node_gen(
                    node_id,
                    gss_node,
                    remaining_mask,
                    gss_acc.llm_mask,
                    depth,
                    global_pop_cache=global_pop_cache,
                    apply_cache=apply_cache_by_bv,
                    isolate_cache=isolate_many_cache,
                    edge_order_override_map=edge_order_map,
                    dest_scores=guided_scores,
                    oracle_counters=analysis_node_costs if record_end_unions else None,
                    node_allowed_mask=node_desc_mask if oracle_reward_mask is not None else None,
                )
                # Record frontier for planning if requested
                if record_end_unions and a_node:
                    if analysis_frontiers is not None and not work_llm_mask.is_empty():
                        prev = analysis_frontiers.get(int(node_id))
                        analysis_frontiers[int(node_id)] = work_llm_mask if prev is None else prev.union(work_llm_mask)
            else:
                raise ValueError(f'Unexpected work item: {work}')

            if gen:
                while True:
                    try:
                        yielded = next(gen)

                        if isinstance(yielded, Enqueue):
                            new_node_id, new_gss, new_depth = yielded.node_id, yielded.gss, yielded.depth
                            base_child_pri = (-self.max_depth.get(new_node_id, 0), 0, 0)
                            # Dynamic, target-aware child priority if enabled and reward masks present
                            if oracle_reward_mask is not None and self.oracle_dynamic_prioritization:
                                child_priority = self._priority_for_node(
                                    new_node_id,
                                    new_depth,
                                    remaining_mask,
                                    guided_scores=guided_scores,
                                    oracle_reward_mask=oracle_reward_mask,
                                    analysis_node_costs=oracle_node_costs,
                                )
                            elif guided_scores is not None:
                                child_priority = (-int(guided_scores.get(new_node_id, 0)),) + base_child_pri
                            else:
                                child_priority = base_child_pri
                            heapq.heappush(work_heap, HeapItem(child_priority, WorkItemNew(new_node_id, new_gss, new_depth)))
                        elif isinstance(yielded, Suspend):
                            # Recompute scheduling priority for the suspended work dynamically based on remaining_mask
                            if oracle_reward_mask is not None and self.oracle_dynamic_prioritization:
                                susp_pri = self._priority_for_node(
                                    yielded.node_id,
                                    yielded.depth,
                                    remaining_mask,
                                    guided_scores=guided_scores,
                                    oracle_reward_mask=oracle_reward_mask,
                                    analysis_node_costs=oracle_node_costs,
                                )
                            else:
                                susp_pri = yielded.priority
                            heapq.heappush(work_heap, HeapItem(susp_pri, WorkItemSuspended(gen, work_llm_mask, yielded.depth)))
                            break

                    except StopIteration:
                        break # Generator is done.
        stats.stop('get_mask.main_loop')

        stats.start('get_mask.final_conversion')
        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        stats.stop('get_mask.final_conversion')

        stats.stop('get_mask')

        # Capture metrics for coordinator/runner usage
        self.last_get_mask_cost = int(Stats.get().counts.get('get_mask.traversal.edges_traversed', 0))
        self.last_get_mask_metrics = {
            "edges_traversed": float(self.last_get_mask_cost),
            "nodes_processed": float(Stats.get().counts.get('get_mask.traversal.nodes_processed', 0)),
            "end_nodes": float(Stats.get().counts.get('get_mask.traversal.end_nodes', 0)),
            "main_loop_ms": float(Stats.get().times.get('get_mask.main_loop', 0.0) * 1000.0),
            "total_ms": float(Stats.get().times.get('get_mask', 0.0) * 1000.0),
            "max_depth": float(Stats.get().counts.get('get_mask.traversal.max_depth', 0)),
        }

        # Optional internal stats printout (suppressed by default)
        if not self.suppress_stats_report:
            if Stats.get().times['get_mask.main_loop']*1000 > 1:
                Stats.get().report()

        if record_end_unions:
            oracle_data = {
                "end_events": end_union_events,
                "frontiers": analysis_frontiers or {},
                "node_costs": analysis_node_costs or {},
                "end_masks": analysis_end_masks or {},
            }
        else:
            oracle_data = {}
        return {
            "type": "timed_output",
            "output": original_indices,
            "time_sec": Stats.get().times.get('get_mask.main_loop', 0.0),
        }, oracle_data

    def get_mask(self) -> Union[RangeSetOut, Dict]:
        # Oracle wrapper: plan, then guided run
        if not self.oracle_mode:
            out, _ = self._get_mask_run(guided_scores=None, edge_order_map=None, record_end_unions=False)
            return out

        # 1) Planning pass: collect detailed analysis (end-node masks, node frontiers, per-node costs)
        prev_suppress = self.suppress_stats_report
        try:
            self.suppress_stats_report = True
            _, plan = self._get_mask_run(guided_scores=None, edge_order_map=None, record_end_unions=True)
        finally:
            self.suppress_stats_report = prev_suppress

        if not isinstance(plan, dict):
            plan = {}

        end_events: List[Tuple[int, LLMTokenSet]] = plan.get("end_events", [])
        end_masks_map: Dict[int, LLMTokenSet] = plan.get("end_masks", {}) or {}
        # Prefer exact end masks to compute the per-call target, if available
        if self.oracle_plan_use_exact_end_masks and end_masks_map:
            target_mask = RangeSet.empty()
            for m in end_masks_map.values():
                if m is not None and not m.is_empty():
                    target_mask |= m
        else:
            # Fallback to union of deltas (order-dependent but fine as a fallback)
            target_mask = RangeSet.empty()
            for nid, delta in end_events:
                if delta is not None and not delta.is_empty():
                    target_mask |= delta

        # If nothing to cover, run a trivial guided pass
        if target_mask.is_empty():
            Stats.get().reset()
            out, _ = self._get_mask_run(guided_scores=None, edge_order_map=None, record_end_unions=False)
            return out

        # Compute target-aware reward masks (oracle pruning), prefer exact end masks
        if self.oracle_reward_prune:
            if self.oracle_plan_use_exact_end_masks and end_masks_map:
                reward_masks = self._oracle_compute_reward_masks_from_end_masks(end_masks_map)
            else:
                reward_masks = self._oracle_compute_reward_masks(end_events)
        else:
            reward_masks = {}

        # Optional: report hotspots to investigate potential "mega nodes"
        if self.oracle_debug:
            self._oracle_report_hotspots(plan, top_k=self.oracle_hotspot_report_top_k)

        # Prepare candidate end-node sequences:
        #  - minimal cover
        #  - token-mass order
        #  - random permutations of top-K end nodes by token mass
        end_items: List[Tuple[int, LLMTokenSet]] = []
        for nid, delta in end_events:
            if delta is not None and not delta.is_empty():
                end_items.append((int(nid), delta))
        end_items_unique = {}
        for nid, m in end_items:
            prev = end_items_unique.get(int(nid))
            end_items_unique[int(nid)] = m if prev is None else prev.union(m)
        end_items = [(nid, m) for nid, m in end_items_unique.items()]
        # If exact end masks available, override end_items with exacts (better for planning)
        if self.oracle_plan_use_exact_end_masks and end_masks_map:
            end_items = [(int(nid), m) for nid, m in end_masks_map.items() if m is not None and not m.is_empty()]
        # Minimal cover sequence (greedy/exact from recorded end masks)
        min_cover_seq = self._compute_oracle_min_cover(end_items)
        # Token-mass sorted sequence
        mass_sorted_seq = [nid for nid, _ in sorted(end_items, key=lambda t: len(t[1]), reverse=True)]
        # Beam-search cost-aware sequence over ends
        if self.oracle_use_beam_search and (end_masks_map or end_items):
            beam_seq = self._oracle_beam_search_end_sequence(
                end_masks=end_masks_map if (self.oracle_plan_use_exact_end_masks and end_masks_map) else dict(end_items),
                target_mask=target_mask,
                analysis_node_costs=plan.get("node_costs", {}),
                trials=max(1, int(self.oracle_search_trials)),
                beam_width=max(1, int(self.oracle_beam_width)),
                jitter=int(self.oracle_jitter),
            )
        else:
            beam_seq = []
        # Random permutations of top-K ends
        topK = min(max(1, int(self.oracle_permute_top_ends)), len(mass_sorted_seq))
        top_list = mass_sorted_seq[:topK]
        rng = random.Random(1337)
        trial_sequences: List[List[int]] = []
        # Always include baseline sequences
        if min_cover_seq:
            trial_sequences.append(min_cover_seq)
        if mass_sorted_seq:
            trial_sequences.append(mass_sorted_seq)
        # Also include a guidance derived from plan (as a baseline candidate)
        plan_guided_scores, plan_edge_order_map = self._oracle_build_guidance_from_plan(plan, jitter=0)
        # Also include the advanced reward/target-aware guidance
        adv_scores, adv_edge_order_map = self._oracle_build_guidance_advanced(plan, reward_masks, target_mask, jitter=int(self.oracle_jitter))
        # Include beam-search-based end sequence if any
        if beam_seq:
            trial_sequences.append(beam_seq)
        # Add random permutations
        for _ in range(max(0, int(self.oracle_trials))):
            perm = top_list[:]
            rng.shuffle(perm)
            # Fill remaining ends in mass order
            rest = [x for x in mass_sorted_seq if x not in perm]
            trial_sequences.append(perm + rest)
        # Deduplicate sequences
        dedup = []
        seen = set()
        for seq in trial_sequences:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                dedup.append(seq)
        trial_sequences = dedup

        # Evaluate candidates (including the plan-based and advanced-guided ones)
        best_scores = adv_scores if adv_scores else plan_guided_scores
        best_edge_map = adv_edge_order_map if adv_edge_order_map else plan_edge_order_map
        best_edges = None
        best_time_ms = None
        # First evaluate the best available base guidance (adv or plan)
        prev_suppress = self.suppress_stats_report
        try:
            self.suppress_stats_report = True
            Stats.get().reset()
            _out, _ = self._get_mask_run(
                guided_scores=best_scores,
                edge_order_map=best_edge_map,
                record_end_unions=False,
                oracle_target_mask=target_mask,
                oracle_reward_mask=reward_masks if self.oracle_reward_prune else None,
                oracle_node_costs=plan.get("node_costs", {}),
            )
            edges = int(Stats.get().counts.get('get_mask.traversal.edges_traversed', 0))
            time_ms = float(Stats.get().times.get('get_mask.main_loop', 0.0) * 1000.0)
            best_edges, best_time_ms = edges, time_ms
        finally:
            self.suppress_stats_report = prev_suppress

        # Local randomized edge-order search for mega nodes; merge into baseline edge map
        local_edge_map = self._oracle_local_edge_order_search(
            reward_masks=reward_masks if self.oracle_reward_prune else {},
            target_mask=target_mask,
            analysis_node_costs=plan.get("node_costs", {}),
            top_k=int(self.oracle_meganode_top_k),
            trials_per_node=int(self.oracle_edge_trials_per_node),
            jitter=int(self.oracle_jitter),
        )
        if local_edge_map:
            merged = dict(best_edge_map) if best_edge_map else {}
            merged.update(local_edge_map)
            best_edge_map = merged

        for seq in trial_sequences:
            trial_scores, trial_edge_map = self._oracle_prepare_guidance(seq)
            # Merge advanced edge map and local improvements
            merged_edge_map = dict(adv_edge_order_map) if adv_edge_order_map else {}
            merged_edge_map.update(trial_edge_map or {})
            if local_edge_map:
                merged_edge_map.update(local_edge_map)
            prev_suppress = self.suppress_stats_report
            try:
                self.suppress_stats_report = True
                Stats.get().reset()
                _out, _ = self._get_mask_run(
                    guided_scores=trial_scores,
                    edge_order_map=merged_edge_map,
                    record_end_unions=False,
                    oracle_target_mask=target_mask,
                    oracle_reward_mask=reward_masks if self.oracle_reward_prune else None,
                    oracle_node_costs=plan.get("node_costs", {}),
                )
                edges = int(Stats.get().counts.get('get_mask.traversal.edges_traversed', 0))
                time_ms = float(Stats.get().times.get('get_mask.main_loop', 0.0) * 1000.0)
            finally:
                self.suppress_stats_report = prev_suppress

            if best_edges is None or edges < best_edges or (edges == best_edges and (best_time_ms is None or time_ms < best_time_ms)):
                best_edges = edges
                best_time_ms = time_ms
                best_scores = trial_scores
                best_edge_map = trial_edge_map

        if self.oracle_debug:
            print(f"[oracle] candidates={len(trial_sequences)+1}, best_edges_traversed={best_edges}, best_main_loop_ms={best_time_ms:.3f}")

        # Final guided run with the best plan, stats reset as requested
        Stats.get().reset()
        out, _ = self._get_mask_run(
            guided_scores=best_scores,
            edge_order_map=best_edge_map,
            record_end_unions=False,
            oracle_target_mask=target_mask,
            oracle_reward_mask=reward_masks if self.oracle_reward_prune else None,
            oracle_node_costs=plan.get("node_costs", {}),
        )
        return out

    def finalize(self):
        """Called at the end of a benchmark run to perform any final actions, like printing stats."""
        print("\n--- Final Stats Report from Model ---")
        Stats.get().report()

