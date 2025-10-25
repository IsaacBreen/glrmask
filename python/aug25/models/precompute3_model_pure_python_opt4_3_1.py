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

@dataclass(frozen=True)
class TraversalData:
    nodes: List[NodeID]
    pos_of_u: Dict[NodeID, int]
    comp_id: List[int]
    sccs: List[List[int]]  # Each inner list holds positions into `nodes`
    topo: List[int]        # Topological ordering of SCC IDs

def _compute_traversal_data(arena: Dict[NodeID, ArenaNode]) -> TraversalData:
    # Build a stable node list and a position map
    nodes: List[NodeID] = sorted(arena.keys())
    n = len(nodes)
    pos_of_u: Dict[NodeID, int] = {u: i for i, u in enumerate(nodes)}

    # Adjacency over positions
    adj: List[List[int]] = [[] for _ in range(n)]
    radj: List[List[int]] = [[] for _ in range(n)]
    for i, u in enumerate(nodes):
        a_node = arena.get(u)
        if not a_node:
            continue
        for edge in a_node.children:
            for dest in edge.dests:
                v = int(dest.dest_idx)
                j = pos_of_u.get(v)
                if j is not None:
                    adj[i].append(j)
                    radj[j].append(i)

    # Kosaraju (iterative) for SCCs
    visited = [False] * n
    order: List[int] = []
    for u in range(n):
        if not visited[u]:
            stack: List[Tuple[int, int]] = [(u, 0)]
            visited[u] = True
            while stack:
                node, next_i = stack[-1]
                if next_i < len(adj[node]):
                    v = adj[node][next_i]
                    stack[-1] = (node, next_i + 1)
                    if not visited[v]:
                        visited[v] = True
                        stack.append((v, 0))
                else:
                    order.append(node)
                    stack.pop()

    comp_id: List[Optional[int]] = [None] * n
    cid = 0
    for u in reversed(order):
        if comp_id[u] is None:
            stack = [u]
            comp_id[u] = cid
            while stack:
                x = stack.pop()
                for v in radj[x]:
                    if comp_id[v] is None:
                        comp_id[v] = cid
                        stack.append(v)
            cid += 1

    # Build SCC lists
    sccs: List[List[int]] = [[] for _ in range(cid)]
    for i in range(n):
        sccs[comp_id[i]].append(i)  # type: ignore[arg-type]

    # Condensation DAG and topo sort
    scc_adj: List[Set[int]] = [set() for _ in range(cid)]
    indeg: List[int] = [0] * cid
    for u in range(n):
        cu = comp_id[u]  # type: ignore[index]
        for v in adj[u]:
            cv = comp_id[v]  # type: ignore[index]
            if cu != cv and cv not in scc_adj[cu]:  # type: ignore[index]
                scc_adj[cu].add(cv)  # type: ignore[index]
                indeg[cv] += 1

    topo: List[int] = []
    from collections import deque
    q = deque([s for s in range(cid) if indeg[s] == 0])
    while q:
        s = q.popleft()
        topo.append(s)
        for t in scc_adj[s]:
            indeg[t] -= 1
            if indeg[t] == 0:
                q.append(t)

    # Mypy/type ignore above because we know comp_id is fully assigned.
    return TraversalData(
        nodes=nodes,
        pos_of_u=pos_of_u,
        comp_id=[int(c) for c in comp_id],  # type: ignore[arg-type]
        sccs=sccs,
        topo=topo,
    )

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

@dataclass
class Suspend:
    priority: Any

@dataclass
class WorkItemNew:
    node_id: NodeID
    gss: GSS

@dataclass
class WorkItemSuspended:
    generator: Generator
    llm_mask: LLMTokenSet


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
    traversal: TraversalData

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

        arena = _convert_arena(loaded_arena, max_depth)
        # Precompute SCC traversal data over the static arena graph.
        traversal = _compute_traversal_data(arena)
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
            traversal=traversal,
        )
        model._compute_edge_accelerators()
        model.optimize_traversal()
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

    def _process_internal_node_gen(self, node_id: NodeID, gss_node: GSS, max_edges: int = 1, max_dests: int = 1) -> Generator[Union[Enqueue, Suspend], None, None]:
        a_node = self.arena.get(node_id)
        if not a_node:
            return

        # Budget controls to slice work; now configurable by caller.
        edges_proc, dests_proc = 0, 0
        peek0_rs = None
        pop_cache = {}

        for edge_i, edge in enumerate(a_node.children):
            if edge.pop == 0:
                if peek0_rs is None: peek0_rs = RangeSetStates.from_indices(gss_node.peek())
                if edge.dest_states_union.isdisjoint(peek0_rs): continue

            if edge.pop in pop_cache: popped, popped_acc, peeked, peek_rs = pop_cache[edge.pop]
            else:
                popped = gss_node.popn(edge.pop)
                if popped.is_empty():
                    pop_cache[edge.pop] = (popped, None, [], RangeSetStates.empty())
                    continue
                popped_acc = popped.reduce_acc()
                if not popped_acc or popped_acc.llm_mask.is_empty():
                    pop_cache[edge.pop] = (GSS.empty(), None, [], RangeSetStates.empty())
                    continue
                # Avoid double-peek: call once and reuse both list and RangeSetStates
                peeked = popped.peek()
                peek_rs = RangeSetStates.from_indices(peeked)
                pop_cache[edge.pop] = (popped, popped_acc, peeked, peek_rs)

            if not popped_acc or edge.dest_states_union.isdisjoint(peek_rs): continue

            if not (edge.llm_bv_not and popped_acc.llm_mask.isdisjoint(edge.llm_bv_not)):
                if popped_acc.llm_mask.isdisjoint(edge.llm_bv): continue
                @_acc_memoize(use_value_cache=False)
                def intersect(acc: PyAcc):
                    new_mask = acc.llm_mask.intersection(edge.llm_bv)
                    return None if new_mask.is_empty() else PyAcc(acc.terminals_union, new_mask)
                popped = popped.apply_and_prune(intersect)
                if popped.is_empty(): continue
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

            # Iterate grouped dests in ascending order for locality
            for dest_j in sorted(grouped.keys()):
                if dests_proc >= max_dests:
                    priority = (-self.max_depth.get(node_id, 0), edge_i, dest_j)
                    yield Suspend(priority)
                    dests_proc = 0
                dest = edge.dests[dest_j]
                values_to_keep = grouped[dest_j]
                # If all heads survive, reuse popped directly
                if len(values_to_keep) == len(peeked):
                    child_gss = popped
                else:
                    child_gss = popped.isolate_many(values_to_keep)
                if child_gss.is_empty(): continue
                d: NodeID = int(dest.dest_idx)
                yield Enqueue(d, child_gss)
                dests_proc += 1

            if edges_proc >= max_edges:
                priority = (-self.max_depth.get(node_id, 0), edge_i + 1, 0)
                yield Suspend(priority)
                edges_proc = 0
            edges_proc += 1

    def get_mask(self) -> LLMTokenSet:
        # SCC-aware scheduler similar to Rust special_map_grouped: process SCCs in topological order,
        # run a local worklist until stabilization inside each SCC, and propagate across SCC boundaries
        # without looping back prematurely.
        all_ones = self.all_internal_llm_tokens_bitset
        final_mask = RangeSet.empty()
        t0 = time.perf_counter()

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

        # Seed per-node pending values from current parser states.
        init_cache = {}
        pending: Dict[NodeID, GSS] = {}
        for tsid, gss in self.state.items():
            root_id = self.roots_map[int(tsid)]
            gss_init = gss.apply(initialize_acc, init_cache)
            if not gss_init.is_empty():
                if root_id in pending:
                    pending[root_id] = pending[root_id].merge(gss_init)
                else:
                    pending[root_id] = gss_init
        t1 = time.perf_counter()

        tr = self.traversal
        remaining_mask = all_ones
        in_queue: Set[NodeID] = set()

        for s in tr.topo:
            scc_positions = tr.sccs[s]
            # Initialize local queue with nodes in this SCC having pending inputs.
            from collections import deque
            queue: collections.deque[NodeID] = deque()
            in_queue.clear()
            for pos in scc_positions:
                u = tr.nodes[pos]
                if u in pending:
                    queue.append(u)
                    in_queue.add(u)

            while queue:
                u = queue.popleft()
                in_queue.discard(u)
                gss_node = pending.pop(u, None)
                if gss_node is None or gss_node.is_empty():
                    continue

                gss_acc = gss_node.reduce_acc()
                if not gss_acc or gss_acc.llm_mask.is_empty():
                    continue

                # Record final mask if this node is an end.
                if self.is_end(u):
                    if not final_mask.issuperset(gss_acc.llm_mask):
                        final_mask |= gss_acc.llm_mask
                        remaining_mask = all_ones.difference(final_mask)

                a_node = self.arena.get(u)
                if not a_node or not a_node.children:
                    continue
                work_llm_mask = a_node.llm_bv_union.intersection(gss_acc.llm_mask)
                if work_llm_mask.isdisjoint(remaining_mask):
                    continue

                # Drive the internal generator with large budgets (effectively "no suspend").
                gen = self._process_internal_node_gen(u, gss_node, max_edges=(1 << 60), max_dests=(1 << 60))
                while True:
                    try:
                        yielded = next(gen)
                    except StopIteration:
                        break

                    if isinstance(yielded, Enqueue):
                        child_id, child_gss = yielded.node_id, yielded.gss
                        if child_gss.is_empty():
                            continue
                        if child_id in pending:
                            pending[child_id] = pending[child_id].merge(child_gss)
                        else:
                            pending[child_id] = child_gss

                        # If the child is in the same SCC, schedule immediately.
                        pos_d = tr.pos_of_u.get(child_id)
                        if pos_d is not None and tr.comp_id[pos_d] == s:
                            if child_id not in_queue:
                                queue.append(child_id)
                                in_queue.add(child_id)
                    elif isinstance(yielded, Suspend):
                        # With very high budgets we don't expect to suspend; ignore safely.
                        continue

        t2 = time.perf_counter()

        # Map internal mask to original indices
        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        t3 = time.perf_counter()
        print(f"Get mask times: init {(t1 - t0)*1000:.2f}ms, main {(t2 - t1)*1000:.2f}ms, map {(t3 - t2)*1000:.2f}ms")
        return original_indices
