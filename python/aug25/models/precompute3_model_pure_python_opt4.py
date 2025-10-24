from __future__ import annotations

import collections
import heapq
import itertools
import json
import os
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
    # New fields for optimization
    pop_inv_index: Dict[int, Dict[int, Dict[LLMTokenSet, List[Tuple[int, int]]]]] = field(default_factory=dict)
    pop_state_union: Dict[int, StateIDSet] = field(default_factory=dict)
    pop_llm_union: Dict[int, LLMTokenSet] = field(default_factory=dict)

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
    gm_max_dests: int = 2048
    last_get_mask_cost: int = 0
    last_get_mask_metrics: Dict[str, float] = field(default_factory=dict)
    suppress_stats_report: bool = False

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
        model.optimize_traversal()
        model._build_inverted_indices()
        model._compute_and_print_stats()
        return model

    def _compute_edge_accelerators(self) -> None:
        all_ones = self.all_internal_llm_tokens_bitset
        for node in self.arena.values():
            for edge in node.children:
                edge.llm_bv_not = all_ones.difference(edge.llm_bv)

    def optimize_traversal(self) -> None:
        for node in self.arena.values():
            for edge in node.children:
                edge.ensure_index()

    def _build_inverted_indices(self) -> None:
        for node in tqdm(self.arena.values(), desc="Building inverted indices"):
            pop_inv_index: Dict[int, Dict[int, Dict[LLMTokenSet, List[Tuple[int, int]]]]] = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
            pop_state_union: Dict[int, StateIDSet] = collections.defaultdict(RangeSetStates.empty)
            pop_llm_union: Dict[int, LLMTokenSet] = collections.defaultdict(RangeSet.empty)

            for edge_i, edge in enumerate(node.children):
                edge.ensure_index()
                pop = edge.pop
                pop_llm_union[pop] |= edge.llm_bv
                pop_state_union[pop] |= edge.dest_states_union

                for state_id, dest_indices in edge.state_to_dest.items():
                    for dest_j in dest_indices:
                        pop_inv_index[pop][state_id][edge.llm_bv].append((edge_i, dest_j))
            
            node.pop_inv_index = {p: dict(s_map) for p, s_map in pop_inv_index.items()}
            node.pop_state_union = dict(pop_state_union)
            node.pop_llm_union = dict(pop_llm_union)

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

    def _process_internal_node_gen(self, node_id: NodeID, gss_node: GSS, remaining_mask: LLMTokenSet, gss_mask: LLMTokenSet, depth: int) -> Generator[Union[Enqueue, Suspend], None, None]:
        stats = Stats.get()
        a_node = self.arena.get(node_id)
        if not a_node:
            return
        pop_cache = {}

        # Iterate over pop values present at this node
        for pop, inv_index_for_pop in a_node.pop_inv_index.items():
            stats.inc(f'get_mask.traversal.pop_val.{pop}')
            # Early exit checks using precomputed unions
            if a_node.pop_llm_union[pop].isdisjoint(remaining_mask):
                stats.inc('get_mask.traversal.pop_bucket.skipped_no_new_tokens')
                continue
            if a_node.pop_llm_union[pop].isdisjoint(gss_mask):
                stats.inc('get_mask.traversal.pop_bucket.pre_gss_disjoint_skips')
                continue

            # Get popped GSS (from cache or compute)
            if pop in pop_cache:
                popped, popped_acc, peeked, peek_rs = pop_cache[pop]
                stats.inc('get_mask.main_loop.pop_cache_hits')
            else:
                stats.start('get_mask.main_loop.popn')
                popped = gss_node.popn(pop)
                stats.stop('get_mask.main_loop.popn')
                if popped.is_empty():
                    stats.inc('get_mask.traversal.popped_empty')
                    pop_cache[pop] = (popped, None, [], RangeSetStates.empty())
                    continue
                
                stats.start('get_mask.main_loop.popped.reduce_acc')
                popped_acc = popped.reduce_acc()
                stats.stop('get_mask.main_loop.popped.reduce_acc')
                if not popped_acc or popped_acc.llm_mask.is_empty():
                    pop_cache[pop] = (GSS.empty(), None, [], RangeSetStates.empty())
                    continue
                
                peeked = popped.peek()
                peek_rs = RangeSetStates.from_indices(peeked)
                pop_cache[pop] = (popped, popped_acc, peeked, peek_rs)

            if not popped_acc:
                continue

            # Intersect peeked states with states relevant for this pop bucket
            relevant_peek_rs = peek_rs.intersection(a_node.pop_state_union[pop])
            if relevant_peek_rs.is_empty():
                stats.inc('get_mask.traversal.pop_bucket.no_relevant_states')
                continue
            
            dests_by_llm_bv: Dict[LLMTokenSet, Dict[Tuple[int, int], List[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
            
            for state_id in relevant_peek_rs.iter_indices():
                if state_id in inv_index_for_pop:
                    for llm_bv, dest_tuples in inv_index_for_pop[state_id].items():
                        for edge_i, dest_j in dest_tuples:
                            dests_by_llm_bv[llm_bv][(edge_i, dest_j)].append(state_id)

            apply_prune_cache = {}
            
            for llm_bv, dest_groups in dests_by_llm_bv.items():
                if llm_bv.isdisjoint(remaining_mask) or llm_bv.isdisjoint(popped_acc.llm_mask):
                    continue
                
                if llm_bv in apply_prune_cache:
                    popped_masked = apply_prune_cache[llm_bv]
                    stats.inc('get_mask.main_loop.apply_prune_cache_hits')
                else:
                    all_ones = self.all_internal_llm_tokens_bitset
                    llm_bv_not = all_ones.difference(llm_bv)

                    if popped_acc.llm_mask.isdisjoint(llm_bv_not):
                        popped_masked = popped
                    else:
                        @_acc_memoize(use_value_cache=False)
                        def intersect(acc: PyAcc):
                            new_mask = acc.llm_mask.intersection(llm_bv)
                            return None if new_mask.is_empty() else PyAcc(acc.terminals_union, new_mask)
                        
                        stats.start('get_mask.main_loop.apply_and_prune')
                        popped_masked = popped.apply_and_prune(intersect)
                        stats.stop('get_mask.main_loop.apply_and_prune')
                    
                    apply_prune_cache[llm_bv] = popped_masked

                if popped_masked.is_empty():
                    continue

                isolate_cache = {}

                for (edge_i, dest_j), s_list in dest_groups.items():
                    s_list_tuple = tuple(sorted(s_list))
                    if s_list_tuple in isolate_cache:
                        child_gss = isolate_cache[s_list_tuple]
                        stats.inc('get_mask.main_loop.isolate_cache_hits')
                    else:
                        if len(s_list) == len(peeked):
                            child_gss = popped_masked
                        else:
                            stats.start('get_mask.main_loop.isolate_many')
                            child_gss = popped_masked.isolate_many(s_list)
                            stats.stop('get_mask.main_loop.isolate_many')
                        isolate_cache[s_list_tuple] = child_gss
                    
                    if child_gss.is_empty():
                        continue
                        
                    edge = a_node.children[edge_i]
                    dest = edge.dests[dest_j]
                    d: NodeID = int(dest.dest_idx)
                    yield Enqueue(d, child_gss, depth + 1)

    def get_mask(self) -> Union[RangeSetOut, Dict]:
        stats = Stats.get()
        stats.start('get_mask')
        stats.counts['get_mask.traversal.max_depth'] = 0
        stats.inc('get_mask.initial_tokenizer_states', len(self.state))

        @dataclass
        class HeapItem:
            priority: Any
            item: Any

            def __lt__(self, other: 'HeapItem') -> bool:
                if not isinstance(other, HeapItem):
                    return NotImplemented
                return self.priority < other.priority

        all_ones, final_mask = self.all_internal_llm_tokens_bitset, RangeSet.empty()
        work_heap = []

        @_acc_memoize(use_value_cache=False)
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
                priority = (-self.max_depth.get(r, 0), 0, 0)
                heapq.heappush(work_heap, HeapItem(priority, WorkItemNew(r, gss_init, 0)))
        stats.stop('get_mask.seeding')

        stats.start('get_mask.main_loop')
        remaining_mask = all_ones
        while work_heap:
            if remaining_mask.is_empty():
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
                    if not final_mask.issuperset(gss_acc.llm_mask):
                        stats.start('get_mask.main_loop.end_node.final_mask_union')
                        final_mask |= gss_acc.llm_mask
                        stats.stop('get_mask.main_loop.end_node.final_mask_union')
                        remaining_mask = all_ones.difference(final_mask)
                        if remaining_mask.is_empty():
                            stats.inc('get_mask.early_exit_full_mask')
                            break

                a_node = self.arena.get(node_id)
                work_llm_mask = a_node.llm_bv_union.intersection(gss_acc.llm_mask)

                if not a_node or not a_node.children or work_llm_mask.isdisjoint(remaining_mask):
                    continue

                gen = self._process_internal_node_gen(node_id, gss_node, remaining_mask, gss_acc.llm_mask, depth)
            else:
                raise ValueError(f'Unexpected work item: {work}')

            if gen:
                while True:
                    try:
                        yielded = next(gen)

                        if isinstance(yielded, Enqueue):
                            new_node_id, new_gss, new_depth = yielded.node_id, yielded.gss, yielded.depth
                            child_priority = (-self.max_depth.get(new_node_id, 0), 0, 0)
                            heapq.heappush(work_heap, HeapItem(child_priority, WorkItemNew(new_node_id, new_gss, new_depth)))
                        elif isinstance(yielded, Suspend):
                            heapq.heappush(work_heap, HeapItem(yielded.priority, WorkItemSuspended(gen, work_llm_mask, yielded.depth)))
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
            if stats.times['get_mask.main_loop']*1000 > 1:
                stats.report()

        return {
            "type": "timed_output",
            "output": original_indices,
            "time_sec": stats.times.get('get_mask.main_loop', 0.0),
        }

    def finalize(self):
        """Called at the end of a benchmark run to perform any final actions, like printing stats."""
        print("\n--- Final Stats Report from Model ---")
        Stats.get().report()
