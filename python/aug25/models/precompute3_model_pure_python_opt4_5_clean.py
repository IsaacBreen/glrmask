from __future__ import annotations

import collections
import functools
import heapq
import itertools
import json
import math
import types
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set, Generator, Any

import _sep1 as ffi

from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from ..common_interface import GraphProvider
from ..range_set import FFIRangeSet as RangeSet
from ..range_set import SetRangeSet as RangeSetOut
from ..range_set import SetRangeSet as RangeSetStates

# Type Aliases
NodeID = int
LLMTokenSet = RangeSet
StateIDSet = RangeSetStates
TerminalIdSet = RangeSet

def _acc_memoize(fn):
    """Per-invocation memoization for PyAcc transformers."""
    id_memo = {}
    val_memo = {}
    def wrapper(acc):
        acc_id = id(acc)
        if acc_id in id_memo:
            return id_memo[acc_id]

        cached = val_memo.get(acc)
        if cached is not None:
            id_memo[acc_id] = cached
            return cached

        result = fn(acc)
        id_memo[acc_id] = result
        if result is not None:
            val_memo[acc] = result
        return result
    return wrapper

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
    state_to_dest: Optional[Dict[int, List[int]]] = None
    priority: float = 0.0

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

def _load_and_flatten_arena(loaded_arena: Dict[NodeID, LoadedArenaNode]) -> Dict[NodeID, IntermediateArenaNode]:
    """Stage 1: Convert from the loaded format to a flattened intermediate format."""
    intermediate_arena: Dict[NodeID, IntermediateArenaNode] = {}
    for uid, loaded_node in loaded_arena.items():
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
    for uid, intermediate_node in intermediate_arena.items():
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

def _convert_arena(loaded_arena: Dict[NodeID, LoadedArenaNode]) -> Dict[NodeID, ArenaNode]:
    """Orchestrates the full conversion from loaded data to the final arena format."""
    intermediate_arena = _load_and_flatten_arena(loaded_arena)
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
        # Correctly hash the dictionary content for memoization.
        # Sort items by key for consistent hash value.
        # Assumes that RangeSet (the dict value) is hashable.
        sorted_items = tuple(sorted(self.terminals_union.items()))
        return hash((sorted_items, self.llm_mask))

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
    depth: int


@dataclass
class Model(GraphProvider):
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
    min_dist_to_end: Dict[NodeID, int] = field(default_factory=dict)

    @staticmethod
    def from_json_string(s: str) -> 'Model':
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

        arena = _convert_arena(loaded_arena)
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

        # Misc data
        pmc_json = data['possible_matches']
        possible_matches_cache = {}
        for tsid_json, term_map_json in pmc_json:
            tsid = int(tsid_json)
            term_map = {}
            for term_id_json, bv_json in term_map_json:
                term_id = int(term_id_json)
                bv = RangeSet.from_ranges(bs_from_json(dumps(bv_json)).to_ranges())
                term_map[term_id] = bv
            possible_matches_cache[tsid] = term_map

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
            ignore_terminal_id=parser_data.get('ignore_terminal_id'),
            state={tokenizer.initial_state_id(): initial_gss},
        )
        model._compute_min_dist_to_end()
        model._sort_edges_by_heuristic()
        model.optimize_traversal()
        return model

    def _ensure_reverse_adjacency(self) -> Dict[int, Set[int]]:
        """Build and cache reverse adjacency: dest_node -> set(parent_nodes)."""
        if hasattr(self, 'reverse_adj'):
            return self.reverse_adj
        parents: Dict[int, Set[int]] = collections.defaultdict(set)
        for u, node in self.arena.items():
            for edge in node.children:
                for dest in edge.dests:
                    parents[int(dest.dest_idx)].add(int(u))
        self.reverse_adj = dict(parents)
        return self.reverse_adj

    def _compute_min_dist_to_end(self) -> None:
        """Computes shortest distance from each node to a clean_end node via reverse BFS."""
        self.min_dist_to_end = {}
        q = collections.deque()
        for nid, node in self.arena.items():
            if node.clean_end:
                self.min_dist_to_end[nid] = 0
                q.append(nid)

        rev_adj = self._ensure_reverse_adjacency()

        while q:
            u = q.popleft()
            dist = self.min_dist_to_end[u]
            for v in rev_adj.get(u, []):
                if v not in self.min_dist_to_end:
                    self.min_dist_to_end[v] = dist + 1
                    q.append(v)

    def _sort_edges_by_heuristic(self) -> None:
        """Pre-computes edge priorities and sorts edges in each node."""
        infinity = float('inf')
        for node in self.arena.values():
            for edge in node.children:
                if not edge.dests:
                    edge.priority = infinity
                else:
                    # Priority is the minimum distance to an end node over all destinations of this edge.
                    edge.priority = min(self.min_dist_to_end.get(d.dest_idx, infinity) for d in edge.dests)
            # Sort edges by priority (lower is better).
            node.children.sort(key=lambda edge: edge.priority)

    def optimize_traversal(self) -> None:
        for node in self.arena.values():
            for edge in node.children:
                edge.ensure_index()

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        term_rs = RangeSet.from_indices([terminal_id])
        @_acc_memoize
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current = acc.terminals_union.get(state_id, RangeSet.empty())
            if current.contains(terminal_id): return acc
            new_map = acc.terminals_union.copy()
            new_map[state_id] = current.union(term_rs)
            return PyAcc(new_map, acc.llm_mask)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> NodeID: return self.roots_map[int(state_id)]
    def is_end(self, node: NodeID) -> bool: return self.arena[node].clean_end

    def commit(self, token_id: int):
        token_bytes = self.id_to_token[token_id]
        terminals_map, state_map = {}, {}
        for tsid in self.state:
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tsid)
            if end_state is not None: state_map[tsid] = end_state
            terminals_map[tsid] = RangeSet.from_indices([m[0] for m in matches])

        @_acc_memoize
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

    def _process_internal_node_gen(
        self,
        node_id: NodeID,
        gss_node: GSS,
        remaining_mask: LLMTokenSet,
        gss_mask: LLMTokenSet,
        depth: int,
        global_pop_cache: Dict,
        apply_cache: Dict,
        isolate_cache: Dict,
    ) -> Generator[Union[Enqueue, Suspend], Optional[LLMTokenSet], None]:
        a_node = self.arena.get(node_id)
        if not a_node:
            return

        peek0 = gss_node.peek()
        peek0_rs = RangeSetStates.from_indices(peek0) if peek0 else RangeSetStates.empty()

        if remaining_mask.is_empty():
            return

        max_edges, max_dests = 256, 4096
        edges_proc, dests_proc = 0, 0
        pop_cache: Dict[int, Tuple[Any, Any, List[int], StateIDSet]] = {}
        local_apply_cache: Dict[Tuple[int, int], GSS] = {} if apply_cache is None else apply_cache

        for edge_i, edge in enumerate(a_node.children):
            if edge.llm_bv.isdisjoint(remaining_mask):
                continue
            if edge.llm_bv.isdisjoint(gss_mask):
                continue

            if edge.pop in pop_cache:
                popped, popped_acc, peeked, peek_rs = pop_cache[edge.pop]
            else:
                if edge.pop == 0:
                    popped = gss_node
                    popped_acc = types.SimpleNamespace(llm_mask=gss_mask)
                    peeked = peek0
                    peek_rs = peek0_rs
                    pop_cache[edge.pop] = (popped, popped_acc, peeked, peek_rs)
                else:
                    popped = None
                    if global_pop_cache is not None:
                        key = (id(gss_node), edge.pop)
                        cached = global_pop_cache.get(key)
                        if cached is not None:
                            popped, popped_acc, peeked, peek_rs = cached
                            pop_cache[edge.pop] = cached
                    if popped is None:
                        popped = gss_node.popn(edge.pop)
                    if popped.is_empty():
                        pop_cache[edge.pop] = (popped, None, [], RangeSetStates.empty())
                        continue
                    popped_acc = popped.reduce_acc()
                    if not popped_acc or popped_acc.llm_mask.is_empty():
                        pop_cache[edge.pop] = (GSS.empty(), None, [], RangeSetStates.empty())
                        continue
                    peeked = popped.peek()
                    peek_rs = RangeSetStates.from_indices(peeked)
                    pop_cache[edge.pop] = (popped, popped_acc, peeked, peek_rs)
                    if global_pop_cache is not None:
                        global_pop_cache[(id(gss_node), edge.pop)] = (popped, popped_acc, peeked, peek_rs)

            if not popped_acc or edge.dest_states_union.isdisjoint(peek_rs):
                continue

            source_after_apply = popped
            if not popped_acc.llm_mask.isdisjoint(edge.llm_bv):
                key_apply = (id(popped), id(edge.llm_bv))
                cached_apply = local_apply_cache.get(key_apply)
                if cached_apply is not None:
                    source_after_apply = cached_apply
                else:
                    @_acc_memoize
                    def intersect(acc: PyAcc):
                        new_mask = acc.llm_mask.intersection(edge.llm_bv)
                        return None if new_mask.is_empty() else PyAcc(acc.terminals_union, new_mask)
                    tmp = popped.apply_and_prune(intersect)
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

            for dest_j in sorted(grouped.keys()):
                if dests_proc >= max_dests:
                    next_edge_priority = a_node.children[edge_i].priority if dest_j > 0 else a_node.children[edge_i + 1].priority
                    priority = (next_edge_priority, depth)
                    new_mask = yield Suspend(node_id, priority, depth)
                    if new_mask is not None:
                        remaining_mask = new_mask
                    if remaining_mask.is_empty():
                        return
                    dests_proc = 0

                if remaining_mask.is_empty():
                    break
                dest = edge.dests[dest_j]
                values_to_keep = grouped[dest_j]
                
                if len(values_to_keep) == len(peeked):
                    child_gss = source_after_apply
                else:
                    key_isolate = (id(source_after_apply), tuple(values_to_keep))
                    if isolate_cache is not None and key_isolate in isolate_cache:
                        child_gss = isolate_cache[key_isolate]
                    else:
                        child_gss = source_after_apply.isolate_many(values_to_keep)
                        if isolate_cache is not None:
                            isolate_cache[key_isolate] = child_gss
                
                if child_gss.is_empty(): continue
                
                d: NodeID = int(dest.dest_idx)
                new_mask = yield Enqueue(d, child_gss, depth + 1)
                if new_mask is not None:
                    remaining_mask = new_mask
                if remaining_mask.is_empty():
                    return
                dests_proc += 1

            if remaining_mask.is_empty():
                return

            edges_proc += 1
            if edges_proc >= max_edges:
                next_edge_priority = a_node.children[edge_i + 1].priority if (edge_i + 1) < len(a_node.children) else float('inf')
                priority = (next_edge_priority, depth)
                new_mask = yield Suspend(int(node_id), priority, depth)
                if new_mask is not None:
                    remaining_mask = new_mask
                if remaining_mask.is_empty():
                    return
                edges_proc = 0

    def get_mask(self) -> Union[RangeSetOut, Dict]:
        global_pop_cache: Dict[Tuple[int, int], Tuple[Any, Any, List[int], StateIDSet]] = {}
        apply_cache_by_bv: Dict[Tuple[int, int], GSS] = {}
        isolate_many_cache: Dict[Tuple[int, Tuple[int, ...]], GSS] = {}

        @dataclass
        class HeapItem:
            priority: Any
            item: Any
            def __lt__(self, other: 'HeapItem') -> bool:
                return self.priority < other.priority

        all_ones = self.all_internal_llm_tokens_bitset
        final_mask = RangeSet.empty()
        work_heap = []
        infinity = float('inf')

        @_acc_memoize
        def initialize_acc(acc: PyAcc) -> PyAcc:
            disallowed = RangeSet.empty()
            for tsid, terms in acc.terminals_union.items():
                if tsid in self.possible_matches_cache:
                    term_map = self.possible_matches_cache[tsid]
                    for term_id in terms.iter_indices():
                        if term_id in term_map: disallowed |= term_map[term_id]
            return PyAcc({}, all_ones.difference(disallowed))

        init_cache = {}
        for sid, gss in self.state.items():
            r = self.roots_map[int(sid)]
            gss_init = gss.apply(initialize_acc, init_cache)
            if not gss_init.is_empty():
                node = self.arena.get(r)
                priority = (node.children[0].priority, 0) if node and node.children else (infinity, 0)
                heapq.heappush(work_heap, HeapItem(priority, WorkItemNew(r, gss_init, 0)))

        remaining_mask = all_ones
        while work_heap:
            if remaining_mask.is_empty():
                break

            heap_item = heapq.heappop(work_heap)
            priority, work = heap_item.priority, heap_item.item

            gen = None
            is_new_gen = False

            if isinstance(work, WorkItemSuspended):
                gen, depth = work.generator, work.depth
            elif isinstance(work, WorkItemNew):
                is_new_gen = True
                node_id, gss_node, depth = work.node_id, work.gss, work.depth

                gss_acc = gss_node.reduce_acc()
                
                if self.is_end(node_id):
                    final_mask |= gss_acc.llm_mask
                    remaining_mask = all_ones.difference(final_mask)
                    if remaining_mask.is_empty():
                        break
                
                a_node = self.arena.get(node_id)
                if not a_node or not a_node.children:
                    continue

                if a_node.llm_bv_union.isdisjoint(remaining_mask):
                    continue

                gen = self._process_internal_node_gen(
                    node_id, gss_node, remaining_mask, gss_acc.llm_mask, depth,
                    global_pop_cache, apply_cache_by_bv, isolate_many_cache
                )
            else:
                raise ValueError(f'Unexpected work item: {work}')

            if not gen:
                continue

            try:
                if is_new_gen:
                    yielded = next(gen)
                else:
                    yielded = gen.send(remaining_mask)

                if isinstance(yielded, Enqueue):
                    new_node_id, new_gss, new_depth = yielded.node_id, yielded.gss, yielded.depth
                    new_node = self.arena.get(new_node_id)
                    child_priority = (new_node.children[0].priority, new_depth) if new_node and new_node.children else (infinity, new_depth)
                    
                    heapq.heappush(work_heap, HeapItem(child_priority, WorkItemNew(new_node_id, new_gss, new_depth)))
                    heapq.heappush(work_heap, HeapItem(priority, WorkItemSuspended(gen, depth)))
                elif isinstance(yielded, Suspend):
                    heapq.heappush(work_heap, HeapItem(yielded.priority, WorkItemSuspended(gen, yielded.depth)))
            except StopIteration:
                pass # Generator is done.

        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        
        return original_indices
