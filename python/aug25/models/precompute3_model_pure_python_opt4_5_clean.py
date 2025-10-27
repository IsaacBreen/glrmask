from __future__ import annotations

import collections
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set, Generator, Any
import json

import _sep1 as ffi
from tqdm import tqdm

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


def _acc_memoize(use_value_cache: bool = True):
    """Per-invocation memoization for PyAcc transformers."""
    def decorator(fn):
        id_memo = {}
        val_memo = {}
        def wrapper(acc):
            acc_id = id(acc)
            if acc_id in id_memo:
                return id_memo[acc_id]

            if use_value_cache:
                cached = val_memo.get(acc)
                if cached is not None:
                    id_memo[acc_id] = cached
                    return cached

            result = fn(acc)
            id_memo[acc_id] = result
            if use_value_cache and result is not None:
                val_memo[acc] = result
            return result
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

        for group_id in self.states[current_state].finalizers:
            if group_id in self.non_greedy_finalizers:
                matches.setdefault(group_id, 0)
            else:
                matches[group_id] = 0

        for i, byte in enumerate(text):
            state_data = self.states[current_state]
            next_state = state_data.transitions.get(byte)
            if next_state is None:
                return None, [(gid, width) for gid, width in matches.items() if width > 0]
            current_state = next_state
            for group_id in self.states[current_state].finalizers:
                if group_id in self.non_greedy_finalizers:
                    matches.setdefault(group_id, i + 1)
                else:
                    matches[group_id] = i + 1

        return current_state, [(gid, width) for gid, width in matches.items() if width > 0]

    def initial_state_id(self) -> int:
        return self.start_state


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
    min_distance_to_end: int = 999999

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
    llm_bv_descendant: LLMTokenSet = field(default_factory=RangeSet.empty)
    pop_to_state_union: Dict[int, StateIDSet] = field(default_factory=dict)
    pop_to_llm_union: Dict[int, LLMTokenSet] = field(default_factory=dict)
    fast_children: Dict[int, Dict[int, Dict[int, Tuple[LLMTokenSet, Optional[LLMTokenSet], LLMTokenSet]]]] = field(default_factory=dict)
    pop_state_to_edges: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)


def _optimize_intermediate_arena(intermediate_arena: Dict[NodeID, IntermediateArenaNode], max_depth: Dict[NodeID, int]):
    for node in tqdm(intermediate_arena.values(), desc="Optimizing intermediate arena"):
        if not node.children:
            continue
        node.children.sort(key=lambda e: (-max_depth.get(int(e.dests.dest_idx), 0), e.pop))


def _load_and_flatten_arena(loaded_arena: Dict[NodeID, LoadedArenaNode]) -> Dict[NodeID, IntermediateArenaNode]:
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
    edge_priority: int


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
    edge_priority: int


@dataclass
class WorkItemSuspended:
    generator: Generator
    depth: int
    next_edge_priority: int


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
    gm_max_edges: int = 256
    gm_max_dests: int = 4096
    node_distance_to_end: Dict[NodeID, int] = field(default_factory=dict)

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)

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
        dfa_states = [DFAState(
            transitions={int(k): v for k, v in s['transitions'].get('data', {}).items()},
            finalizers=set(s['finalizers']),
            possible_future_group_ids=set(s['possible_future_group_ids'])
        ) for s in dfa_data['states']]
        tokenizer = PyTokenizer(dfa_states, dfa_data['start_state'], set(dfa_data['non_greedy_finalizers']))

        # Parser Table
        parser_data = data['parser']
        py_table: Dict[int, Row] = {}
        for state_id_str, row_data in parser_data['stage_7_table']:
            state_id, py_row = int(state_id_str), Row()
            for term_id_str, action_data in row_data['shifts_and_reduces_full']:
                term_id, variant = int(term_id_str), action_data['variant']
                if variant == 'Shift':
                    py_row.actions[term_id] = action_data['state_id']
                elif variant == 'Reduce':
                    py_row.actions[term_id] = Reduce(action_data['nonterminal_id'], action_data['len'], tuple(sorted(action_data['production_ids'])))
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
            arena=arena,
            roots_map=roots_map,
            max_depth=max_depth,
            parser_table=parser_table,
            tokenizer=tokenizer,
            tokenizer_initial_state=tokenizer.initial_state_id(),
            possible_matches_cache=possible_matches_cache,
            id_to_token={v: bytes(k) for k, v in data['llm_token_map']},
            internal_to_original_map={int(k): RangeSetOut.from_indices(v) for k, v in dict(vocab['internal_to_original']).items()},
            all_internal_llm_tokens_bitset=all_internal_llm_tokens_bitset,
            ignore_terminal_id=parser_data.get('ignore_terminal_id'),
            state={tokenizer.initial_state_id(): initial_gss},
        )

        model._compute_descendant_llm_closure()
        model._shrink_edges_by_descendant()
        model._compute_edge_accelerators()
        model._compute_bucket_unions()
        model._build_fast_children()
        model._compute_static_priorities()
        model.optimize_traversal()
        model._build_pop_state_to_edges_index()

        return model

    def _compute_edge_accelerators(self) -> None:
        all_ones = self.all_internal_llm_tokens_bitset
        for node in self.arena.values():
            for edge in node.children:
                edge.llm_bv_not = all_ones.difference(edge.llm_bv)

    def _compute_bucket_unions(self) -> None:
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
        all_ones = self.all_internal_llm_tokens_bitset
        for nid, node in self.arena.items():
            node.llm_bv_descendant = all_ones if node.clean_end else RangeSet.empty()

        changed = True
        while changed:
            changed = False
            for nid, node in self.arena.items():
                if node.clean_end:
                    continue
                if not node.children:
                    continue
                new_accum = RangeSet.empty()
                for edge in node.children:
                    dest_union_closure = RangeSet.empty()
                    for d in edge.dests:
                        dest_union_closure |= self.arena[int(d.dest_idx)].llm_bv_descendant
                    if not dest_union_closure.is_empty():
                        new_accum |= edge.llm_bv.intersection(dest_union_closure)
                updated = node.llm_bv_descendant.union(new_accum)
                if updated != node.llm_bv_descendant:
                    node.llm_bv_descendant = updated
                    changed = True

    def _shrink_edges_by_descendant(self) -> None:
        for u, node in self.arena.items():
            if not node.children:
                node.llm_bv_union = RangeSet.empty()
                continue
            new_children = []
            llm_union = RangeSet.empty()
            for edge in node.children:
                child_union = RangeSet.empty()
                for d in edge.dests:
                    child_union |= self.arena[int(d.dest_idx)].llm_bv_descendant
                shrunk = edge.llm_bv.intersection(child_union)
                if shrunk.is_empty():
                    continue
                edge.llm_bv = shrunk
                new_children.append(edge)
                llm_union |= shrunk
            node.children = new_children
            node.llm_bv_union = llm_union

    def _build_fast_children(self) -> None:
        all_ones = self.all_internal_llm_tokens_bitset
        for node in self.arena.values():
            if not node.children:
                node.fast_children = {}
                continue
            mapping_by_pop: Dict[int, Dict[int, Dict[int, LLMTokenSet]]] = collections.defaultdict(lambda: collections.defaultdict(dict))

            for edge in node.children:
                for dest in edge.dests:
                    for sid in dest.state_bv.iter_indices():
                        state_map = mapping_by_pop[edge.pop][int(sid)]
                        prev = state_map.get(int(dest.dest_idx))
                        if prev is None:
                            state_map[int(dest.dest_idx)] = edge.llm_bv
                        else:
                            state_map[int(dest.dest_idx)] = prev.union(edge.llm_bv)

            final_children: Dict[int, Dict[int, Dict[int, Tuple[LLMTokenSet, Optional[LLMTokenSet], LLMTokenSet]]]] = {}
            for pop, smap in mapping_by_pop.items():
                per_state: Dict[int, Dict[int, Tuple[LLMTokenSet, Optional[LLMTokenSet], LLMTokenSet]]] = {}
                for sid, dmap in smap.items():
                    per_dest: Dict[int, Tuple[LLMTokenSet, Optional[LLMTokenSet], LLMTokenSet]] = {}
                    for dest_id, llm_bv in dmap.items():
                        child_desc = self.arena[int(dest_id)].llm_bv_descendant
                        llm_thru = llm_bv.intersection(child_desc)
                        if not llm_thru.is_empty():
                            per_dest[int(dest_id)] = (llm_bv, all_ones.difference(llm_bv), llm_thru)
                    per_state[int(sid)] = per_dest
                final_children[int(pop)] = per_state

            node.fast_children = final_children

    def _build_pop_state_to_edges_index(self) -> None:
        for node in self.arena.values():
            if not node.children:
                continue

            mapping_by_pop: Dict[int, Dict[int, List[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
            for i, edge in enumerate(node.children):
                if edge.state_to_dest:
                    for sid in edge.state_to_dest.keys():
                        mapping_by_pop[edge.pop][sid].append(i)

            node.pop_state_to_edges = {pop: dict(sids) for pop, sids in mapping_by_pop.items()}


    def _compute_static_priorities(self) -> None:
        """Compute minimum distance to end nodes for all nodes and edges using BFS."""
        distances: Dict[NodeID, int] = {}
        queue = collections.deque()

        # Initialize end nodes with distance 0
        for nid, node in self.arena.items():
            if node.clean_end:
                distances[nid] = 0
                queue.append(nid)

        # Build reverse adjacency
        reverse_adj: Dict[NodeID, Set[NodeID]] = collections.defaultdict(set)
        for u, node in self.arena.items():
            for edge in node.children:
                for dest in edge.dests:
                    reverse_adj[int(dest.dest_idx)].add(u)

        # BFS from end nodes backward
        while queue:
            v = queue.popleft()
            for u in reverse_adj[v]:
                if u not in distances:
                    distances[u] = distances[v] + 1
                    queue.append(u)

        self.node_distance_to_end = distances

        # Compute edge priorities (min distance among destinations)
        for node in self.arena.values():
            for edge in node.children:
                min_dist = 999999
                for dest in edge.dests:
                    dist = distances.get(int(dest.dest_idx), 999999)
                    min_dist = min(min_dist, dist)
                edge.min_distance_to_end = min_dist

    def optimize_traversal(self) -> None:
        for node in self.arena.values():
            for edge in node.children:
                edge.ensure_index()
            # Sort edges by priority (lower distance = higher priority)
            node.children.sort(key=lambda e: e.min_distance_to_end)

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        term_rs = RangeSet.from_indices([terminal_id])
        @_acc_memoize(use_value_cache=False)
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current = acc.terminals_union.get(state_id, RangeSet.empty())
            if current.contains(terminal_id):
                return acc
            new_map = acc.terminals_union.copy()
            new_map[state_id] = current.union(term_rs)
            return PyAcc(new_map, acc.llm_mask)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> NodeID:
        return self.roots_map[int(state_id)]

    def is_end(self, node: NodeID) -> bool:
        return self.arena[node].clean_end

    def iter_edges(self, node: NodeID, token: int):
        a_node = self.arena.get(node)
        if not a_node:
            return
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
            if end_state is not None:
                state_map[tsid] = end_state
            terminals_map[tsid] = RangeSet.from_indices([m[0] for m in matches])

        @_acc_memoize()
        def mutator(acc: PyAcc) -> Optional[PyAcc]:
            for tsid, matched in terminals_map.items():
                if acc.terminals_union.get(tsid, RangeSet.empty()).intersects(matched):
                    return None
            new_bvs = collections.defaultdict(RangeSet.empty)
            for old, new in state_map.items():
                if old in acc.terminals_union:
                    new_bvs[new] |= acc.terminals_union[old]
            return PyAcc(dict(new_bvs), acc.llm_mask)

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
                    if new_offset == len(token_bytes):
                        new_states[next_tsid].append(proc_gss)
                    else:
                        key = (new_offset, next_tsid)
                        if key in work:
                            work[key] = work[key].merge(proc_gss)
                        else:
                            work[key] = proc_gss
                            q.append(key)
            if end_state is not None:
                new_states[end_state].append(gss)

        merged = {sid: GSS.merge_many(gssl) for sid, gssl in new_states.items() if gssl}
        self.state = {sid: g for sid, g in merged.items() if not g.is_empty()}

    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        if self.ignore_terminal_id == terminal_id:
            return gss

        heads_by_state = collections.defaultdict(list)
        for state_id in gss.peek():
            heads_by_state[state_id].append(gss.isolate(state_id))

        shifted = []
        while heads_by_state:
            state_id, gss_list = heads_by_state.popitem()
            state_gss = GSS.merge_many(gss_list)
            row = self.parser_table.table.get(state_id)
            if not row:
                continue
            action = row.actions.get(terminal_id)
            if action is None:
                continue

            if isinstance(action, int):
                shifted.append(state_gss.push(action))
            elif isinstance(action, Reduce):
                popped = state_gss.popn(action.len)
                for from_id in popped.peek():
                    goto_id = self.parser_table.table[from_id].gotos[action.nonterminal_id]
                    heads_by_state[goto_id].append(popped.isolate(from_id).push(goto_id))
            elif isinstance(action, Split):
                if action.shift is not None:
                    shifted.append(state_gss.push(action.shift))
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
        global_pop_cache: Optional[Dict] = None,
        apply_cache: Optional[Dict] = None,
        isolate_cache: Optional[Dict] = None,
    ) -> Generator[Union[Enqueue, Suspend], Optional[LLMTokenSet], None]:
        a_node = self.arena.get(node_id)
        if not a_node:
            return

        peek0 = gss_node.peek()
        peek0_rs = RangeSetStates.from_indices(peek0) if peek0 else RangeSetStates.empty()

        if remaining_mask.is_empty():
            return

        active_remaining_mask = remaining_mask
        max_edges, max_dests = self.gm_max_edges, self.gm_max_dests
        edges_proc, dests_proc = 0, 0
        pop_cache: Dict[int, Tuple[Any, Any, List[int], StateIDSet]] = {}
        skip_pops: Set[int] = set()
        checked_pops: Set[int] = set()
        local_apply_cache: Dict[Tuple[int, int], GSS] = {} if apply_cache is None else apply_cache

        pops_in_node = sorted(a_node.fast_children.keys())
        group_index = 0

        for pop_val in pops_in_node:
            if pop_val in skip_pops:
                continue

            if pop_val in pop_cache:
                popped, popped_acc, peeked, peek_rs = pop_cache[pop_val]
            else:
                if pop_val == 0:
                    popped = gss_node
                    import types
                    popped_acc = types.SimpleNamespace(llm_mask=gss_mask)
                    peeked = peek0
                    peek_rs = peek0_rs
                    pop_cache[pop_val] = (popped, popped_acc, peeked, peek_rs)
                else:
                    popped = None
                    if global_pop_cache is not None:
                        key = (id(gss_node), pop_val)
                        cached = global_pop_cache.get(key)
                        if cached is not None:
                            popped, popped_acc, peeked, peek_rs = cached
                            pop_cache[pop_val] = cached
                    if popped is None:
                        popped = gss_node.popn(pop_val)
                    if popped.is_empty():
                        pop_cache[pop_val] = (popped, None, [], RangeSetStates.empty())
                        continue
                    popped_acc = popped.reduce_acc()
                    if not popped_acc or popped_acc.llm_mask.is_empty():
                        pop_cache[pop_val] = (GSS.empty(), None, [], RangeSetStates.empty())
                        continue
                    peeked = popped.peek()
                    peek_rs = RangeSetStates.from_indices(peeked)
                    pop_cache[pop_val] = (popped, popped_acc, peeked, peek_rs)
                    if global_pop_cache is not None:
                        global_pop_cache[(id(gss_node), pop_val)] = (popped, popped_acc, peeked, peek_rs)

            if pop_val not in checked_pops:
                checked_pops.add(pop_val)
                pop_states_union = a_node.pop_to_state_union.get(pop_val)
                if pop_states_union is not None and pop_states_union.isdisjoint(peek_rs):
                    skip_pops.add(pop_val)
                    continue
                pop_llm_union = a_node.pop_to_llm_union.get(pop_val)
                if pop_llm_union is not None and (
                    (popped_acc and popped_acc.llm_mask.isdisjoint(pop_llm_union)) or pop_llm_union.isdisjoint(active_remaining_mask)
                ):
                    skip_pops.add(pop_val)
                    continue

            groups: Dict[Tuple[int, int], Dict[str, Any]] = {}
            mapping_for_pop = a_node.fast_children.get(pop_val, {})
            for sid in peeked:
                per_state_map = mapping_for_pop.get(int(sid), {})
                if not per_state_map:
                    continue
                for dest_id, triple in per_state_map.items():
                    llm_bv, llm_bv_not, llm_bv_thru = triple
                    if popped_acc.llm_mask.isdisjoint(llm_bv_thru):
                        continue
                    key = (int(dest_id), id(llm_bv_thru))
                    entry = groups.get(key)
                    if entry is None:
                        groups[key] = {"llm_bv": llm_bv_thru, "states": [int(sid)]}
                    else:
                        entry["states"].append(int(sid))

            if not groups:
                continue

            # Sort by edge priority (lower distance = higher priority)
            dest_keys = sorted(groups.keys(), key=lambda k: (
                self.arena[int(k[0])].children[0].min_distance_to_end if self.arena.get(int(k[0])) and self.arena[int(k[0])].children else 999999
            ))

            for (dest_id, llm_bv_id) in dest_keys:
                entry = groups[(dest_id, llm_bv_id)]
                llm_bv = entry["llm_bv"]
                states_to_keep: List[int] = entry["states"]

                if llm_bv.isdisjoint(active_remaining_mask):
                    continue

                source_after_apply = popped
                key_apply = (id(popped), id(llm_bv))
                cached_apply = local_apply_cache.get(key_apply)
                if cached_apply is not None:
                    source_after_apply = cached_apply
                else:
                    @_acc_memoize(use_value_cache=False)
                    def intersect(acc: PyAcc):
                        new_mask = acc.llm_mask.intersection(llm_bv)
                        return None if new_mask.is_empty() else PyAcc(acc.terminals_union, new_mask)
                    tmp = popped.apply_and_prune(intersect)
                    if tmp.is_empty():
                        continue
                    source_after_apply = tmp
                    local_apply_cache[key_apply] = tmp

                if len(states_to_keep) == len(peeked):
                    child_gss = source_after_apply
                else:
                    key_isolate = (id(source_after_apply), tuple(states_to_keep))
                    if isolate_cache is not None and key_isolate in isolate_cache:
                        child_gss = isolate_cache[key_isolate]
                    else:
                        child_gss = source_after_apply.isolate_many(states_to_keep)
                        if isolate_cache is not None:
                            isolate_cache[key_isolate] = child_gss

                if child_gss.is_empty():
                    continue

                d: NodeID = int(dest_id)
                dest_node = self.arena.get(d)
                edge_priority = dest_node.children[0].min_distance_to_end if dest_node and dest_node.children else 999999

                new_mask = yield Enqueue(d, child_gss, depth + 1, edge_priority)
                if new_mask is not None:
                    active_remaining_mask = new_mask
                if active_remaining_mask.is_empty():
                    return

                dests_proc += 1
                if dests_proc >= max_dests:
                    next_priority = min((e.min_distance_to_end for e in a_node.children[group_index:]), default=999999)
                    priority = (next_priority, -self.max_depth.get(node_id, 0), group_index, 0)
                    new_mask = yield Suspend(node_id, priority, depth)
                    if new_mask is not None:
                        active_remaining_mask = new_mask
                    if active_remaining_mask.is_empty():
                        return
                    dests_proc = 0

                group_index += 1

            if active_remaining_mask.is_empty():
                return

            if edges_proc >= max_edges:
                next_priority = min((e.min_distance_to_end for e in a_node.children[group_index:]), default=999999)
                priority = (next_priority, -self.max_depth.get(node_id, 0), group_index, 0)
                new_mask = yield Suspend(int(node_id), priority, depth)
                if new_mask is not None:
                    active_remaining_mask = new_mask
                if active_remaining_mask.is_empty():
                    return
                edges_proc = 0
            edges_proc += 1

    def get_mask(self) -> Union[RangeSetOut, Dict]:
        all_ones = self.all_internal_llm_tokens_bitset
        final_mask = RangeSet.empty()
        work_heap = []

        @dataclass
        class HeapItem:
            priority: Any
            item: Any

            def __lt__(self, other: 'HeapItem') -> bool:
                if not isinstance(other, HeapItem):
                    return NotImplemented
                return self.priority < other.priority

        @_acc_memoize(use_value_cache=False)
        def initialize_acc(acc: PyAcc) -> PyAcc:
            disallowed = RangeSet.empty()
            for tsid, terms in acc.terminals_union.items():
                if tsid in self.possible_matches_cache:
                    term_map = self.possible_matches_cache[tsid]
                    for term_id in terms.iter_indices():
                        if term_id in term_map:
                            disallowed |= term_map[term_id]
            return PyAcc({}, all_ones.difference(disallowed))

        init_cache = {}
        for sid, gss in self.state.items():
            r = self.roots_map[int(sid)]
            gss_init = gss.apply(initialize_acc, init_cache)
            if not gss_init.is_empty():
                root_node = self.arena.get(r)
                edge_priority = root_node.children[0].min_distance_to_end if root_node and root_node.children else 999999
                priority = (edge_priority, -self.max_depth.get(r, 0), 0, 0)
                heapq.heappush(work_heap, HeapItem(priority, WorkItemNew(r, gss_init, 0, edge_priority)))

        remaining_mask = all_ones
        global_pop_cache: Dict[Tuple[int, int], Tuple[Any, Any, List[int], StateIDSet]] = {}
        apply_cache_by_bv: Dict[Tuple[int, int], GSS] = {}
        isolate_many_cache: Dict[Tuple[int, Tuple[int, ...]], GSS] = {}

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

                node_desc_mask = a_node.llm_bv_descendant
                work_llm_mask = node_desc_mask.intersection(gss_acc.llm_mask)

                if work_llm_mask.isdisjoint(remaining_mask):
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
                    new_node_id, new_gss, new_depth, new_edge_priority = yielded.node_id, yielded.gss, yielded.depth, yielded.edge_priority
                    child_priority = (new_edge_priority, -self.max_depth.get(new_node_id, 0), 0, 0)
                    heapq.heappush(work_heap, HeapItem(child_priority, WorkItemNew(new_node_id, new_gss, new_depth, new_edge_priority)))
                    heapq.heappush(work_heap, HeapItem(priority, WorkItemSuspended(gen, depth, new_edge_priority)))
                elif isinstance(yielded, Suspend):
                    susp_pri = yielded.priority
                    next_edge_priority = susp_pri[0] if isinstance(susp_pri, tuple) and len(susp_pri) > 0 else 999999
                    heapq.heappush(work_heap, HeapItem(susp_pri, WorkItemSuspended(gen, yielded.depth, next_edge_priority)))
            except StopIteration:
                pass

        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]

        return original_indices