from __future__ import annotations

import json
import heapq
import collections
import time
import random
from typing import Dict, List, Tuple, Optional, Union, Any, TypedDict, Set
from dataclasses import dataclass, field

from tqdm import tqdm

from ..common_interface import GraphProvider
from ..range_set.range_set_abc import RangeSet as RangeSetABC
from ..range_set.bitset_range_set import BitsetRangeSet
from ..range_set.py_range_set import PyRangeSet
import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
# from python.gss_tester.implementations.leveled_impl_cpp import Leveled_impl_cppGSS as GSS


NodeID = int

# Type aliases for different uses of PyRangeSet to improve clarity.
LLMTokenSet = RangeSetABC
StateIDSet = RangeSetABC
TerminalIdSet = RangeSetABC


# Add a dummy profiler for when not running under kernprof
try:
    # This will be injected by the kernprof script.
    profile
except NameError:
    # If not running under kernprof, create a dummy decorator.
    def profile(func): return func


# Optimization-time edge types for state propagation/unconditionalization
State = int
LLMToken = int

@dataclass(frozen=True)
class PopEdge:
    n: int

@dataclass(frozen=True)
class StateEdge:
    states: Set[State]

@dataclass(frozen=True)
class UnconditionalEdge:
    pass

# One of pop-only, state-masked, or unconditional. For storage we keep pop
# alongside Edge (i.e., children[token][dest] -> (pop, Edge)).
Edge = Union[PopEdge, StateEdge, UnconditionalEdge]

@dataclass
class NodeOpt:
    """
    Working optimization graph for a single-token analysis pass.
    children:
        token_id -> { dest_node_id -> Edge }
      Edge is exactly one of:
        - PopEdge(n)
        - StateEdge(states)
        - UnconditionalEdge()
    is_end:
        True if this node corresponds to an arena node with clean_end=True.
    """
    children: Dict[LLMToken, Dict[NodeID, List[Edge]]] = field(default_factory=dict)
    is_end: bool = False

@dataclass
class NodeOptGraph:
    nodes: Dict[NodeID, NodeOpt]
    roots_map: Dict[int, NodeID]


@dataclass(frozen=True)
class Reduce:
    nonterminal_id: int
    len: int
    production_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Split:
    shift: Optional[int]
    reduces: Dict[int, Dict[int, Tuple[int, ...]]]  # len -> nt_id -> pids


# Action can be a Shift (int), Reduce, or Split
Action = Union[int, Reduce, Split]


@dataclass
class Row:
    actions: Dict[int, Action] = field(default_factory=dict)  # terminal_id -> Action
    gotos: Dict[int, int] = field(default_factory=dict)  # nonterminal_id -> state_id


@dataclass
class ParserTable:
    start_state_id: int
    table: Dict[int, Row]


class ArenaValue(TypedDict, total=False):
    clean_end: bool


class ArenaNode(TypedDict, total=False):
    max_depth: int
    children: List[Tuple[Tuple[int, LLMTokenSet], List[Tuple[NodeID, StateIDSet]]]]
    llm_bv_union: LLMTokenSet
    value: ArenaValue


@dataclass(frozen=True, eq=False)
class PyAcc:
    terminals_union: Dict[int, TerminalIdSet]
    llm_mask: LLMTokenSet

    def __eq__(self, other):
        if not isinstance(other, PyAcc):
            return NotImplemented
        return self.llm_mask == other.llm_mask and self.terminals_union == other.terminals_union

    def __hash__(self):
        # frozenset of items for hashable dict
        return hash((len(self.terminals_union), self.llm_mask))

    def merge(self, other: "PyAcc") -> "PyAcc":
        # The dataclass is frozen, so we can't modify in-place.
        # But terminals_union is a dict, which is mutable.
        # We must be careful to create copies.
        d1 = self.terminals_union
        d2 = other.terminals_union
        new_terminals_union = d1.copy()
        for k, v in d2.items():
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
class Model(GraphProvider):
    """
    Precomputed trie model (third-generation), simplified and concise.
    """
    # Core data structures
    arena: Dict[NodeID, ArenaNode]
    roots_map: Dict[int, NodeID]
    max_depth: Dict[NodeID, int]

    # Parser-related fields
    parser_table: ParserTable
    glr_parser: ffi.GLRParser
    reverse_state_map: Dict[int, Set[int]]

    # Tokenizer-related fields
    tokenizer: ffi.Regex
    tokenizer_initial_state: int
    tokenizer_max_state: int
    possible_matches_cache: Dict[int, Dict[int, LLMTokenSet]]

    # Token/Terminal mapping fields
    id_to_token: Dict[int, bytes]
    internal_to_original_map: Dict[int, Set[int]]
    all_internal_llm_tokens_bitset: LLMTokenSet
    all_terminals_bitset: TerminalIdSet
    ignore_terminal_id: Optional[int]

    # State
    state: Dict[int, GSS]

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map_raw = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
 
        roots_map = {int(s): int(r) for s, r in roots_map_raw}
        max_depth: Dict[NodeID, int] = {}
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # Normalize arena children bitsets and cache max_depth
        for uid, node in arena.items():
            uid_int = int(uid)
            max_depth[uid_int] = int(node.get("max_depth", 0) or 0)

            children = node.get("children") or []
            if not children:
                node["children"] = []
                node["llm_bv_union"] = PyRangeSet.empty()
                continue

            new_children = []
            llm_bv_union: LLMTokenSet = PyRangeSet.empty()
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv_bitset = bs_from_json(dumps(llm_bv_json))
                # Convert to PyRangeSet for ffi-free operations in commit/get_mask
                llm_bv: LLMTokenSet = PyRangeSet.from_ranges(llm_bv_bitset.to_ranges())
                llm_bv_union = llm_bv_union.union(llm_bv)
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv_bitset = bs_from_json(dumps(state_bv_json))
                    state_bv: StateIDSet = PyRangeSet.from_ranges(state_bv_bitset.to_ranges())
                    new_dest_map.append((int(dest_idx), state_bv))
                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children
            node["llm_bv_union"] = llm_bv_union

        # Load tokenizer and parser table from the full constraint JSON
        constraint = ffi.GrammarConstraint.from_json_string(s)
        tokenizer = constraint.tokenizer()
        tokenizer_max_state = tokenizer.max_state()
        glr_parser = constraint.glr_parser()
        ignore_terminal_id = glr_parser.ignore_terminal_id
        tokenizer_initial_state = tokenizer.initial_state_id()

        parser_data = data['parser']
        table_data = parser_data['stage_7_table']
        start_state_id = parser_data['start_state_id']
        py_table: Dict[int, Row] = {}
        for state_id_str, row_data in table_data:
            state_id = int(state_id_str)
            py_row = Row()
            for term_id_str, action_data in row_data['shifts_and_reduces_full']:
                term_id = int(term_id_str)
                variant = action_data['variant']
                if variant == 'Shift':
                    py_row.actions[term_id] = action_data['state_id']
                elif variant == 'Reduce':
                    pids = tuple(sorted(action_data['production_ids']))
                    py_row.actions[term_id] = Reduce(action_data['nonterminal_id'], action_data['len'], pids)
                elif variant == 'Split':
                    shift = action_data['shift']
                    reduces: Dict[int, Dict[int, Tuple[int, ...]]] = {}
                    for len_str, nts_data in action_data['reduces']:
                        len_int = int(len_str)
                        nts: Dict[int, Tuple[int, ...]] = {}
                        for nt_id_str, pids in nts_data:
                            nt_id_int = int(nt_id_str)
                            nts[nt_id_int] = tuple(sorted(pids))
                        reduces[len_int] = nts
                    py_row.actions[term_id] = Split(shift, reduces)
            for nt_id_str, goto_data in row_data['gotos']:
                nt_id = int(nt_id_str)
                if goto_data['state_id'] is not None:
                    py_row.gotos[nt_id] = goto_data['state_id']
            py_table[state_id] = py_row
        parser_table = ParserTable(start_state_id, py_table)

        reverse_map: Dict[int, Set[int]] = collections.defaultdict(set)
        for from_state, row in parser_table.table.items():
            # Handle shifts
            for action in row.actions.values():
                if isinstance(action, int): # Shift
                    reverse_map[action].add(from_state)
                elif isinstance(action, Split):
                    if action.shift is not None:
                        reverse_map[action.shift].add(from_state)
            # Handle gotos
            for to_state in row.gotos.values():
                reverse_map[to_state].add(from_state)
        reverse_state_map = dict(reverse_map)

        all_terminals = set()
        for row in parser_table.table.values():
            all_terminals.update(row.actions.keys())
        if ignore_terminal_id is not None:
            all_terminals.add(ignore_terminal_id)
        all_terminals_bitset = PyRangeSet.from_indices(list(all_terminals))

        initial_acc = PyAcc(terminals_union={}, llm_mask=PyRangeSet.empty())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(parser_table.start_state_id)
        state = {tokenizer_initial_state: initial_gss}

        id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        # Convert possible_matches_cache to PyRangeSet
        pmc_ffi: Dict[int, Dict[int, ffi.Bitset]] = constraint.possible_matches()
        pmc_rs: Dict[int, Dict[int, LLMTokenSet]] = {}
        for tsid, inner in pmc_ffi.items():
            mapped: Dict[int, LLMTokenSet] = {}
            for term_id, bit in inner.items():
                mapped[int(term_id)] = PyRangeSet.from_ranges(bit.to_ranges())
            pmc_rs[int(tsid)] = mapped
        possible_matches_cache = pmc_rs
        internal_to_original_map_raw = constraint.internal_to_original_map()
        internal_to_original_map = {
            k: {v} for k, v in internal_to_original_map_raw.items()
        }
        # Convert universe LLM tokens bitset to PyRangeSet
        all_internal = constraint.all_internal_llm_tokens_bitset()
        all_internal_llm_tokens_bitset = PyRangeSet.from_ranges(all_internal.to_ranges())

        model = Model(
            arena=arena,
            roots_map=roots_map,
            max_depth=max_depth,
            parser_table=parser_table,
            glr_parser=glr_parser,
            reverse_state_map=reverse_state_map,
            tokenizer=tokenizer,
            tokenizer_initial_state=tokenizer_initial_state,
            tokenizer_max_state=tokenizer_max_state,
            possible_matches_cache=possible_matches_cache,
            id_to_token=id_to_token,
            internal_to_original_map=internal_to_original_map,
            all_internal_llm_tokens_bitset=all_internal_llm_tokens_bitset,
            all_terminals_bitset=all_terminals_bitset,
            ignore_terminal_id=ignore_terminal_id,
            state=state,
        )

        model._merge_equivalent_llm_tokens()
        model._reorder_llm_tokens_for_range_minimization()

        # model._unconditionalize_guaranteed_transitions()
        graph = model._to_nodeopt()
        model._from_nodeopt(graph)

        # model._convert_to_bitset_range_set()

        return model

    @profile
    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, TerminalIdSet]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            disallowed_terminals_map = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(state_id, BitsetRangeSet.empty())
                if not matched_bv.intersection(disallowed_for_state).is_empty():
                    return False
            return True
        return gss.prune(predicate)

    @profile
    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_map = acc.terminals_union
            new_bvs: Dict[int, TerminalIdSet] = collections.defaultdict(BitsetRangeSet.empty)
            for old_sid, new_sid in state_map.items():
                bv_source = old_map.get(old_sid, BitsetRangeSet.empty())
                new_bvs[new_sid] = new_bvs[new_sid].union(bv_source)

            return PyAcc(terminals_union=dict(new_bvs), llm_mask=acc.llm_mask)
        return gss.apply(apply_map)

    @profile
    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_map = acc.terminals_union.copy()
            curr_bv = current_map.get(state_id, BitsetRangeSet.empty())
            to_add = BitsetRangeSet.from_indices([terminal_id])
            new_bv = curr_bv.union(to_add)
            current_map[state_id] = new_bv
            return PyAcc(terminals_union=current_map, llm_mask=acc.llm_mask)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> NodeID:
        return self.roots_map[int(state_id)]

    def is_end(self, node: NodeID) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: NodeID, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        """
        children = self.arena.get(node, {}).get("children") or []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():
                        yield (int(pop), None, int(dest_idx))
                    else:
                        for start, end in state_bv.to_ranges():
                            for sid in range(start, end + 1):
                                yield (int(pop), sid, int(dest_idx))

    @profile
    def commit(self, token_id: int):
        t0 = time.perf_counter()
        token_bytes = self.id_to_token[token_id]

        # Build tokenizer maps
        terminals_map: Dict[int, TerminalIdSet] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            matched_terminals = [terminal_id for terminal_id, _ in matches]
            terminals_map[tokenizer_sid] = BitsetRangeSet.from_indices(matched_terminals)

        # Prune and map per-state GSS
        temp_states: Dict[int, GSS] = {}
        for tokenizer_sid, gss in self.state.items():
            pruned_gss = self._prune_disallowed_terminals(gss, terminals_map)
            if not pruned_gss.is_empty():
                mapped_gss = self._map_allowed_terminals_tokenizer_states(pruned_gss, state_map)
                temp_states[tokenizer_sid] = mapped_gss

        current_state_for_processing = temp_states

        new_states: Dict[int, List[GSS]] = collections.defaultdict(list)
        q = collections.deque()
        for tokenizer_sid, gss in current_state_for_processing.items():
            q.append((0, tokenizer_sid, gss))  # offset, tokenizer_state, gss

        visited_q_items: set = set()

        while q:
            offset, tokenizer_sid, gss = q.popleft()
            q_item_key = (offset, tokenizer_sid, id(gss))
            if q_item_key in visited_q_items:
                continue
            visited_q_items.add(q_item_key)

            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tokenizer_sid)

            for terminal_id, width in matches:
                processed_gss = gss if terminal_id == self.ignore_terminal_id else self._process_token(gss, terminal_id)

                # Immediate re-match disallow
                if end_state is not None:
                    accessible_terms = set(self.tokenizer.tokens_accessible_from_state(end_state))
                    if terminal_id in accessible_terms:
                        processed_gss = self._disallow_terminal_in_state(processed_gss, end_state, terminal_id)

                if not processed_gss.is_empty():
                    new_offset = offset + width
                    next_tokenizer_sid = self.tokenizer_initial_state
                    if new_offset == len(token_bytes):
                        new_states[next_tokenizer_sid].append(processed_gss)
                    else:
                        q.append((new_offset, next_tokenizer_sid, processed_gss))

            if end_state is not None:
                new_states[end_state].append(gss)

        merged_states = {
            sid: GSS.merge_many(gss_list)
            for sid, gss_list in new_states.items()
            if gss_list
        }
        merged_states = {sid: state for sid, state in merged_states.items() if not state.is_empty()}

        self.state = merged_states

        t1 = time.perf_counter()

    @profile
    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        heads_by_state: Dict[int, List[GSS]] = collections.defaultdict(list)
        for state_id in gss.peek():
            heads_by_state[state_id].append(gss.isolate(state_id))

        shifted_gsses: List[GSS] = []

        while heads_by_state:
            state_id, state_gsss = heads_by_state.popitem()
            state_gss = GSS.merge_many(state_gsss)
            row = self.parser_table.table.get(state_id)
            if not row:
                continue
            action = row.actions.get(terminal_id)
            if not action:
                continue

            def handle_shift(shift_to_state_id, gss_to_shift):
                shifted_gsses.append(gss_to_shift.push(shift_to_state_id))

            def handle_reduce(reduce_action: Reduce, gss_to_reduce: GSS):
                popped_gss = gss_to_reduce
                for _ in range(reduce_action.len):
                    popped_gss = popped_gss.pop()
                for from_state_id in popped_gss.peek():
                    goto_state_id = self.parser_table.table[from_state_id].gotos[reduce_action.nonterminal_id]
                    goto_gss = popped_gss.isolate(from_state_id).push(goto_state_id)
                    heads_by_state[goto_state_id].append(goto_gss)

            if isinstance(action, int):
                handle_shift(action, state_gss)
            elif isinstance(action, Reduce):
                handle_reduce(action, state_gss)
            elif isinstance(action, Split):
                if action.shift is not None:
                    handle_shift(action.shift, state_gss)
                for length, nts in action.reduces.items():
                    for nt_id, pids in nts.items():
                        handle_reduce(Reduce(nt_id, length, pids), state_gss)

        return GSS.merge_many(shifted_gsses)
    def get_mask(self) -> LLMTokenSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Changes for get_mask_only:
        - Initialize a per-accumulator LLM mask (PyAcc.llm_mask) BEFORE traversal by computing
          the forbidden terminals -> forbidden LLM tokens and taking the complement.
        - Consume terminals_union (set to HybridL2Bitset.all()) after initialization.
        - As we traverse edges, intersect llm_mask with the edge's LLM bitset using apply.
        - At end nodes, simply reduce acc over the GSS and union the llm_mask into the final.
        """
        state_map: Dict[int, GSS] = self.state

        all_ones: Optional[LLMTokenSet] = self.all_internal_llm_tokens_bitset
        final_mask: LLMTokenSet = BitsetRangeSet.empty()

        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask
        values: Dict[NodeID, GSS] = {}
        depth_heap: List[Tuple[int, NodeID]] = []  # Stores (-depth, node_id)
        enqueued_nodes: Set[NodeID] = set()

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, NodeID] = self.roots_map
        max_depth: Dict[NodeID, int] = self.max_depth
        arena: Dict[NodeID, ArenaNode] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, LLMTokenSet]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

        # Seed: Initialize llm_mask in each GSS, consume terminals_union, and enqueue roots.
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask: LLMTokenSet = BitsetRangeSet.empty()
            disallowed_map = acc.terminals_union

            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]
                for terminal_id in disallowed_terminals.to_indices():
                    if terminal_id in terminals_to_llm:
                        disallowed_llm_mask = disallowed_llm_mask.union(
                            terminals_to_llm[terminal_id]
                        )

            allowed_mask = (all_ones if all_ones is not None else BitsetRangeSet.empty()).difference(disallowed_llm_mask)
            return PyAcc(
                terminals_union={},  # consume
                llm_mask=allowed_mask,
            )

        for sid, gss in state_map.items():
            r: NodeID = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc)
            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            if r not in enqueued_nodes:
                enqueued_nodes.add(r)
                hp(depth_heap, (-d, r))

        def enqueue(d: int, n: NodeID) -> None:
            if n not in enqueued_nodes:
                enqueued_nodes.add(n)
                hp(depth_heap, (-d, n))

        # Main loop
        while depth_heap:
            neg_depth, node = hpop(depth_heap)
            gss_node: GSS = values.pop(node)

            # End-node handling: just union the allowed LLM tokens
            if is_end(node):
                reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                if reduced_acc:
                    final_mask = final_mask.union(reduced_acc.llm_mask)

            # Zombie traversal avoidance
            node_llm_bv_union: LLMTokenSet = arena.get(node, {}).get("llm_bv_union", BitsetRangeSet.empty())
            potential_new_tokens = node_llm_bv_union.difference(final_mask)
            if potential_new_tokens.is_empty():
                continue

            gss_mask_acc = gss_node.reduce_acc()
            if gss_mask_acc and gss_mask_acc.llm_mask.intersection(potential_new_tokens).is_empty():
                continue

            # Traverse edges and propagate masks
            edges = arena.get(node, {}).get("children") or []
            for (pop, llm_bv), dests in edges:
                llm_bv = llm_bv.difference(final_mask)
                if llm_bv.is_empty():
                    continue

                popped: GSS = gss_node.popn(pop)
                if popped.is_empty():
                    continue

                peeked = popped.peek()
                # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv
                acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}

                def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                    if acc in acc_memo:
                        return acc_memo[acc]
                    new_mask = acc.llm_mask.intersection(llm_bv)
                    if new_mask.is_empty():
                        result = None
                    else:
                        result = PyAcc(
                            terminals_union=acc.terminals_union,
                            llm_mask=new_mask
                        )
                    acc_memo[acc] = result
                    return result

                popped = popped.apply_and_prune(intersect_and_prune)
                if popped.is_empty():
                    continue

                reduced = popped.reduce_acc()
                if not reduced or reduced.is_empty():
                    continue

                for dest_idx, state_bv in dests:
                    # Empty state_bv in arena means unconditional (applies to all states).
                    if state_bv.is_empty():
                        values_to_keep = peeked
                        child_gss: GSS = popped
                        reduced_child = reduced
                    else:
                        values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]

                    if not values_to_keep:
                        continue

                    if state_bv.is_empty():
                        # Already assigned above.
                        pass
                    else:
                        child_gss = popped.isolate_many(values_to_keep)
                    if child_gss.is_empty():
                        continue

                    if not state_bv.is_empty():
                        reduced_child = child_gss.reduce_acc()
                    if not reduced_child or reduced_child.is_empty():
                        continue

                    d: NodeID = int(dest_idx)
                    if d in values:
                        values[d] = values[d].merge(child_gss)
                    else:
                        values[d] = child_gss
                    enqueue(max_depth[d], d)


        # Convert internal mask back to original IDs
        original_indices: List[int] = []
        for i in final_mask.to_indices():
            if i in self.internal_to_original_map:
                original_indices.extend(list(self.internal_to_original_map[i]))


        return BitsetRangeSet.from_indices(original_indices)

    # ===========================
    # Optimization/conversion API
    # ===========================
    def _nodeopt_pop_preimage(self, states: Set[int], pop: int) -> Set[int]:
        """
        Compute the preimage of 'states' by applying reverse_state_map 'pop' times.
        reverse_state_map maps to_state -> set(from_state) for one parser transition step.
        """
        if pop <= 0:
            return set(states)
        current = set(states)
        for _ in range(pop):
            if not current:
                return set()
            nxt: Set[int] = set()
            for s in current:
                preds = self.reverse_state_map.get(s, set())
                if preds:
                    nxt.update(preds)
            current = nxt
        return current

    def _count_total_ranges(self) -> int:
        count = 0
        for node in self.arena.values():
            for _, llm_bv in (c[0] for c in node.get("children", [])):
                count += len(llm_bv.to_ranges())
            union_bv = node.get("llm_bv_union", PyRangeSet.empty())
            count += len(union_bv.to_ranges())
        if self.possible_matches_cache:
            for inner in self.possible_matches_cache.values():
                for llm_bv in inner.values():
                    count += len(llm_bv.to_ranges())
        return count

    def _merge_equivalent_llm_tokens(self) -> None:
        """
        Merge internal LLM tokens that are indistinguishable across the entire model:
        two tokens are "equivalent" if they occur together in every PyRangeSet occurrence
        the model uses (arena edge llm_bv, node llm_bv_union, and possible_matches_cache).

        Implementation:
        - Build a family of all PyRangeSet occurrences that reference internal tokens.
        - For each internal token, build a signature: the ordered list of set indices in
          which it appears.
        - Group tokens by identical signatures; each group is merged into its smallest-id
          representative.
        - Apply a many-to-one remapping to Arena, possible_matches_cache, the internal
          token universe, and internal_to_original_map.
        """
        print("Merging equivalent internal LLM tokens...", end='', flush=True)

        # Universe of current internal tokens
        universe_list = list(self.all_internal_llm_tokens_bitset.to_indices())
        universe: Set[int] = set(int(t) for t in universe_list)
        if len(universe) <= 1:
            print(" done (not enough tokens to merge).")
            return

        # 1) Collect the family of RangeSets that define token co-occurrence
        family: List[LLMTokenSet] = []
        # Arena children sets and node unions
        for node in self.arena.values():
            children = node.get("children") or []
            for (pop, llm_bv), _dests in children:
                if not llm_bv.is_empty():
                    family.append(llm_bv)
            union_bv: LLMTokenSet = node.get("llm_bv_union") or PyRangeSet.empty()
            if not union_bv.is_empty():
                family.append(union_bv)
        # possible_matches_cache sets
        if self.possible_matches_cache:
            for _tsid, inner in self.possible_matches_cache.items():
                for _term_id, llm_bv in inner.items():
                    if not llm_bv.is_empty():
                        family.append(llm_bv)
        # Include the "universe" as one set so tokens that appear nowhere else
        # still get grouped meaningfully.
        if not self.all_internal_llm_tokens_bitset.is_empty():
            family.append(self.all_internal_llm_tokens_bitset)

        if not family:
            print(" done (no RangeSets to analyze).")
            return

        # 2) Build per-token signatures: which sets (by index) contain the token
        token_to_sets: Dict[int, List[int]] = {t: [] for t in universe}
        for si, rs in enumerate(family):
            for tok in rs.to_indices():
                t = int(tok)
                if t in token_to_sets:
                    token_to_sets[t].append(si)

        # 3) Group by signature
        signature_groups: Dict[Tuple[int, ...], List[int]] = collections.defaultdict(list)
        for t in sorted(universe):
            sig = tuple(token_to_sets.get(t, []))
            signature_groups[sig].append(t)

        # 4) Build many-to-one mapping old->representative
        old_to_new_map: Dict[int, int] = {}
        merges = 0
        for _sig, toks in signature_groups.items():
            if len(toks) <= 1:
                continue
            rep = int(min(toks))
            for tok in toks:
                old_to_new_map[int(tok)] = rep
            merges += (len(toks) - 1)

        if merges == 0:
            print(" done (no equivalent tokens found).")
            return

        # Fill identity for tokens not being merged (keeps remapper robust)
        for t in universe:
            if t not in old_to_new_map:
                old_to_new_map[t] = t

        # 5) Apply mapping
        before_cnt = len(universe)
        self._remap_llm_tokens_many_to_one(old_to_new_map)
        after_cnt = len(list(self.all_internal_llm_tokens_bitset.to_indices()))
        print(f" done. Tokens reduced from {before_cnt} to {after_cnt} ({merges} merged).", flush=True)

    def _unconditionalize_guaranteed_transitions(
        self,
        # etc
    ) -> None:
        print("Unconditionalizing guaranteed transitions...", end='', flush=True)
        ...
        print(" done.")

    def _convert_to_bitset_range_set(self) -> None:
        """
        Converts all PyRangeSet instances (which are py_range_set.PyRangeSet during
        optimization) to bitset_range_set.BitsetRangeSet for runtime efficiency.
        """
        print("Converting PyRangeSet implementations to BitsetRangeSet...", end='', flush=True)

        def convert(rs: RangeSetABC) -> BitsetRangeSet:
            if isinstance(rs, BitsetRangeSet):
                return rs
            return BitsetRangeSet.from_ranges(rs.to_ranges())

        # 1. Convert arena
        for node in self.arena.values():
            new_children = []
            for (pop, llm_bv), dests in node.get("children", []):
                new_dests = []
                for dest_idx, state_bv in dests:
                    new_dests.append((dest_idx, convert(state_bv)))
                new_children.append(((pop, convert(llm_bv)), new_dests))
            node["children"] = new_children
            node["llm_bv_union"] = convert(node.get("llm_bv_union", PyRangeSet.empty()))

        # 2. Convert possible_matches_cache
        if self.possible_matches_cache:
            for tsid, inner in self.possible_matches_cache.items():
                for term_id, llm_bv in inner.items():
                    inner[term_id] = convert(llm_bv)

        # 3. Convert top-level bitsets
        self.all_internal_llm_tokens_bitset = convert(self.all_internal_llm_tokens_bitset)
        self.all_terminals_bitset = convert(self.all_terminals_bitset)

        # 4. Convert GSS accumulators in self.state
        def convert_acc(acc: PyAcc) -> PyAcc:
            new_terminals_union = {
                state_id: convert(term_set)
                for state_id, term_set in acc.terminals_union.items()
            }
            return PyAcc(terminals_union=new_terminals_union, llm_mask=convert(acc.llm_mask))
        self.state = {tsid: gss.apply(convert_acc) for tsid, gss in self.state.items()}
        print(" done.")

    def _reorder_llm_tokens_for_range_minimization(self) -> None:
        """
        Permute internal LLM token IDs to reduce the number of ranges present in
        all PyRangeSet occurrences across the model (arena edges, unions, and possible_matches_cache).
        External token IDs remain unchanged; we only adjust internal indices used in RangeSets.

        Heuristic:
        - Build weighted "groups" of tokens that co-occur:
          * Each child edge's llm_bv contributes a high-weight group.
          * Each node's llm_bv_union contributes a small-weight group.
          * Each possible_matches_cache entry contributes a medium-weight group.
        - Greedily construct an ordering that keeps overlapping groups contiguous.
        - Apply a permutation remapping of internal tokens to this order.
        """
        print("Reordering LLM tokens for range minimization...", end='', flush=True)
        ranges_before = self._count_total_ranges()
        # Collect universe of current internal tokens
        all_tokens_list = list(self.all_internal_llm_tokens_bitset.to_indices())
        if len(all_tokens_list) <= 1:
            print(" done (not enough tokens to reorder).")
            return

        all_tokens: Set[int] = set(int(t) for t in all_tokens_list)

        # 1) Collect weighted co-occurrence groups.
        # Weights are chosen to prioritize edge groups over unions and pmc sets.
        # - Edge LLM sets: high weight (10)
        # - Node llm_bv_union: low weight (2)
        # - possible_matches_cache sets: medium weight (3)
        edge_weight = 10
        union_weight = 2
        pmc_weight = 3

        groups_counter: Dict[Tuple[int, ...], int] = collections.Counter()

        # Arena children groups
        for node in self.arena.values():
            children = node.get("children") or []
            if children:
                for (pop, llm_bv), _dests in children:
                    idxs = [int(x) for x in llm_bv.to_indices() if int(x) in all_tokens]
                    if len(idxs) > 1:
                        groups_counter[tuple(idxs)] += edge_weight
            # Node union (optional)
            union_bv: LLMTokenSet = node.get("llm_bv_union") or PyRangeSet.empty()
            if not union_bv.is_empty():
                idxs = [int(x) for x in union_bv.to_indices() if int(x) in all_tokens]
                if len(idxs) > 1:
                    groups_counter[tuple(idxs)] += union_weight

        # possible_matches_cache groups
        if self.possible_matches_cache:
            for _tsid, inner in self.possible_matches_cache.items():
                for _term_id, llm_bv in inner.items():
                    if llm_bv.is_empty():
                        continue
                    idxs = [int(x) for x in llm_bv.to_indices() if int(x) in all_tokens]
                    if len(idxs) > 1:
                        groups_counter[tuple(idxs)] += pmc_weight

        # Early exit if we don't have any groups to optimize against
        if not groups_counter:
            return

        # Deduplicate groups (tuples already sorted by to_indices), compute a stable id for tie-breaking.
        def stable_hash_of_ints(ints: Tuple[int, ...]) -> int:
            # 64-bit FNV-1a over the sorted ints (ints already sorted)
            h = 1469598103934665603
            for x in ints:
                v = x & 0xFFFFFFFFFFFFFFFF
                h ^= v
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            return h

        groups_tokens: List[Set[int]] = []
        groups_weight: List[int] = []
        groups_stable_id: List[int] = []

        for tup, w in groups_counter.items():
            # Filter out any token that somehow escaped the current universe
            s = [int(x) for x in tup if int(x) in all_tokens]
            if len(s) <= 1:
                continue
            st = set(s)
            groups_tokens.append(st)
            groups_weight.append(int(w))
            groups_stable_id.append(stable_hash_of_ints(tuple(sorted(st))))

        if not groups_tokens:
            return

        G = len(groups_tokens)

        # 2) Build inverted index: token -> groups, and token "importance"
        token_to_groups: Dict[int, List[int]] = collections.defaultdict(list)
        token_importance: Dict[int, int] = {t: 0 for t in all_tokens}

        for gi in range(G):
            g = groups_tokens[gi]
            w = groups_weight[gi]
            for t in g:
                token_to_groups[t].append(gi)
                token_importance[t] += w

        # 3) Greedy ordering to cluster co-occurring tokens
        unplaced: Set[int] = set(all_tokens)
        placed: Set[int] = set()
        order: List[int] = []

        # Track per-group placed/unplaced counts incrementally.
        placed_count = [0] * G
        unplaced_count = [len(groups_tokens[i]) for i in range(G)]

        def place_token(t: int):
            if t in placed:
                return
            placed.add(t)
            unplaced.discard(t)
            order.append(t)
            for gi in token_to_groups.get(t, []):
                placed_count[gi] += 1
                unplaced_count[gi] -= 1

        def pick_seed_token() -> Optional[int]:
            if not unplaced:
                return None
            # Highest importance, deterministic tie-break on token id
            best_t = None
            best_key = None
            for t in unplaced:
                key = (token_importance.get(t, 0), -int(t))
                if (best_key is None) or (key > best_key):
                    best_key = key
                    best_t = t
            return best_t

        def pick_best_group() -> Optional[int]:
            # Among groups with both placed and unplaced tokens, pick one maximizing:
            # score = weight * placed_count * unplaced_count
            best_gi = None
            best_score = None
            for gi in range(G):
                pc = placed_count[gi]
                uc = unplaced_count[gi]
                if pc > 0 and uc > 0:
                    score = groups_weight[gi] * pc * uc
                    key = (score, groups_weight[gi], -groups_stable_id[gi])
                    if (best_score is None) or (key > best_score):
                        best_score = key
                        best_gi = gi
            return best_gi

        # Seed with the best single token's heaviest group if available
        seed = pick_seed_token()
        if seed is None:
            return
        # Determine a heavy group containing the seed (if any)
        seed_groups = token_to_groups.get(seed, [])
        if seed_groups:
            best_gi = None
            best_key = None
            for gi in seed_groups:
                key = (groups_weight[gi], -groups_stable_id[gi])
                if (best_key is None) or (key > best_key):
                    best_key = key
                    best_gi = gi
            # Place all tokens in that group first (by descending importance)
            if best_gi is not None:
                candidates = [t for t in groups_tokens[best_gi] if t in unplaced]
                candidates.sort(key=lambda x: (token_importance.get(x, 0), -int(x)), reverse=True)
                for t in candidates:
                    place_token(t)
        else:
            # No groups include the seed; place it alone
            place_token(seed)

        # Grow ordering by picking best-overlap groups
        while unplaced:
            gi = pick_best_group()
            if gi is None:
                # Start a new cluster
                seed = pick_seed_token()
                if seed is None:
                    break
                seed_groups = token_to_groups.get(seed, [])
                if seed_groups:
                    best_gi = None
                    best_key = None
                    for sgi in seed_groups:
                        key = (groups_weight[sgi], -groups_stable_id[sgi])
                        if (best_key is None) or (key > best_key):
                            best_key = key
                            best_gi = sgi
                    if best_gi is not None:
                        candidates = [t for t in groups_tokens[best_gi] if t in unplaced]
                        candidates.sort(key=lambda x: (token_importance.get(x, 0), -int(x)), reverse=True)
                        for t in candidates:
                            place_token(t)
                        continue
                # If seed isn't in any group, just place it
                place_token(seed)
                continue

            # Place remaining tokens from the chosen group in importance order
            to_place = [t for t in groups_tokens[gi] if t in unplaced]
            if not to_place:
                # No-op (can happen due to incremental updates); pick again
                continue
            to_place.sort(key=lambda x: (token_importance.get(x, 0), -int(x)), reverse=True)
            for t in to_place:
                place_token(t)

        # Sanity check: every token must be placed exactly once
        if len(order) != len(all_tokens):
            # Fallback to identity if something went wrong
            return

        # 4) Build permutation mapping and apply it
        old_to_new: Dict[int, int] = {old: new for new, old in enumerate(order)}
        self._remap_llm_tokens_permutation(old_to_new)

        ranges_after = self._count_total_ranges()
        print(f" done. Ranges reduced from {ranges_before} to {ranges_after} ({ranges_before - ranges_after} fewer).", flush=True)

    # ==============================
    # Arena <-> NodeOpt conversions
    # ==============================
    def _to_nodeopt(self) -> NodeOptGraph:
        """
        Convert the current Arena into a NodeOptGraph, keeping the representation
        simple and faithful for get_mask.

        Encoding rules (per token -> dest -> List[Edge]):
        - If state_bv is unconditional (contains all parser states):
          * pop > 0 -> [PopEdge(pop)]
          * pop == 0 -> [UnconditionalEdge()]
        - If state_bv is masked (not unconditional):
          * pop > 0 -> [PopEdge(pop), StateEdge(states)]
          * pop == 0 -> [StateEdge(states)]

        We do not attempt to create new merges; if a duplicate token+dest pair
        somehow occurs, we keep the first instance we encountered.
        """
        nodes: Dict[NodeID, NodeOpt] = {}
        all_states = PyRangeSet.from_indices(int(s) for s in self.parser_table.table.keys())

        def is_unconditional_state_bv(bv: StateIDSet) -> bool:
            # If bv includes all parser states, treat as unconditional.
            return all_states.is_subset(bv)

        # Create NodeOpt nodes
        for nid, a_node in self.arena.items():
            nid_int = int(nid)
            nopt = NodeOpt()
            nopt.is_end = bool((a_node.get("value") or {}).get("clean_end", False))
            nodes[nid_int] = nopt

        # Populate edges
        for nid, a_node in self.arena.items():
            nid_int = int(nid)
            nopt = nodes[nid_int]

            for (pop, llm_bv), dests in a_node.get("children", []) or []:
                tokens = [int(t) for t in llm_bv.to_indices()]
                if not tokens:
                    continue
                pop_int = int(pop)
                for dest_id, state_bv in dests:
                    dest_int = int(dest_id)
                    unconditional = is_unconditional_state_bv(state_bv)
                    if not unconditional:
                        # Masked: encode states into StateEdge
                        states: Set[int] = set(int(s) for s in state_bv.to_indices())
                        seq: List[Edge] = []
                        if pop_int > 0:
                            seq.append(PopEdge(pop_int))
                        seq.append(StateEdge(states=states))
                    else:
                        # Unconditional
                        if pop_int > 0:
                            seq = [PopEdge(pop_int)]
                        else:
                            seq = [UnconditionalEdge()]

                    # Assign per token; keep first instance if duplicate (token, dest)
                    for tok in tokens:
                        if tok not in nopt.children:
                            nopt.children[tok] = {}
                        dest_map = nopt.children[tok]
                        if dest_int in dest_map:
                            # Keep the first encountered edge sequence.
                            continue
                        dest_map[dest_int] = list(seq)

        return NodeOptGraph(nodes=nodes, roots_map=dict(self.roots_map))

    def _from_nodeopt(self, graph: NodeOptGraph) -> None:
        """
        Rebuild Arena from a NodeOptGraph. We emit simple, correct edges that
        preserve get_mask behavior:
        - For each node, for each (token -> dest -> edge-seq), we derive a signature:
          (pop_total, dest, mask), where mask is None for unconditional or a tuple
          of allowed states. We then group tokens by signature and emit one arena
          child entry per signature: ((pop, llm_bv), [(dest, state_bv)]).
        - We don't attempt to merge multiple dests under the same ((pop, llm_bv), dests).
          This keeps reconstruction simple and preserves semantics.
        """
        new_arena: Dict[NodeID, ArenaNode] = {}
        all_states = PyRangeSet.from_indices(int(s) for s in self.parser_table.table.keys())

        def ranges_key(rs: RangeSetABC) -> Tuple[Tuple[int, int], ...]:
            try:
                return tuple((int(a), int(b)) for (a, b) in rs.to_ranges())
            except Exception:
                return ()

        for nid, nopt in graph.nodes.items():
            nid_int = int(nid)

            # Group tokens by (pop, dest, mask)
            # mask is None for unconditional; otherwise a sorted tuple of states.
            groups: Dict[Tuple[int, int, Optional[Tuple[int, ...]]], Set[int]] = collections.defaultdict(set)

            for tok, dest_map in (nopt.children or {}).items():
                tok_int = int(tok)
                for dest_id, seq in dest_map.items():
                    dest_int = int(dest_id)
                    pop_total = 0
                    mask_states: Optional[Set[int]] = None
                    unconditional = False

                    # Interpret the sequence
                    for e in seq or []:
                        if isinstance(e, PopEdge):
                            pop_total += int(e.n)
                        elif isinstance(e, StateEdge):
                            # Last one wins if multiple present (shouldn't happen in our encoding)
                            mask_states = set(int(s) for s in e.states)
                            unconditional = False
                        elif isinstance(e, UnconditionalEdge):
                            unconditional = True
                            mask_states = None

                    mask_key: Optional[Tuple[int, ...]]
                    if unconditional or mask_states is None:
                        mask_key = None
                    else:
                        mask_key = tuple(sorted(mask_states))

                    sig = (int(pop_total), dest_int, mask_key)
                    groups[sig].add(tok_int)

            # Build children list for this node
            children_list: List[Tuple[Tuple[int, LLMTokenSet], List[Tuple[int, StateIDSet]]]] = []
            # llm_union: LLMTokenSet = PyRangeSet.empty()
            llm_union: LLMTokenSet = PyRangeSet.from_ranges([(0, self.tokenizer_max_llm_token_id + 1)])

            for (pop_val, dest_val, mask_key), tok_set in groups.items():
                if not tok_set:
                    continue
                llm_bv = PyRangeSet.from_indices(sorted(tok_set))
                llm_union = llm_union.union(llm_bv)
                if mask_key is None:
                    state_bv = all_states
                else:
                    state_bv = PyRangeSet.from_indices(list(mask_key))
                children_list.append(((int(pop_val), llm_bv), [(int(dest_val), state_bv)]))

            # Deterministic sort for stability
            children_list.sort(
                key=lambda item: (
                    int(item[0][0]),
                    ranges_key(item[0][1]),
                    tuple(sorted(int(d[0]) for d in item[1])),
                )
            )

            node_value: ArenaValue = {}
            if nopt.is_end:
                node_value["clean_end"] = True

            new_arena[nid_int] = {
                "children": children_list,
                "llm_bv_union": llm_union,
                "value": node_value,
            }

        # Install reconstructed arena and metadata
        self.arena = new_arena
        self.roots_map = dict(graph.roots_map)
        self.max_depth = self._recompute_max_depth_from_arena()

    def _recompute_max_depth_from_arena(self) -> Dict[NodeID, int]:
        """
        Compute a simple upper bound on max depth (longest distance to sink/end),
        using N-iteration relaxation over the graph. End nodes start at depth 1.
        """
        arena = self.arena
        N = len(arena)
        if N == 0:
            return {}

        # Build adjacency: node -> set(dest nodes)
        adj: Dict[NodeID, Set[NodeID]] = {int(n): set() for n in arena.keys()}
        is_end_map: Dict[NodeID, bool] = {}
        for nid, node in arena.items():
            nid = int(nid)
            is_end_map[nid] = bool((node.get("value") or {}).get("clean_end", False))
            for (pop, llm_bv), dests in node.get("children") or []:
                for dest_id, _state_bv in dests:
                    adj[nid].add(int(dest_id))

        # Initialize depths: end nodes = 1, others = 0
        depth: Dict[NodeID, int] = {nid: (1 if is_end_map.get(nid, False) else 0) for nid in adj.keys()}

        # Relax up to N times
        for _ in range(N):
            updated = False
            for u in adj.keys():
                best = depth[u]
                for v in adj[u]:
                    cand = 1 + depth[v]
                    if cand > best:
                        best = cand
                if best > depth[u]:
                    depth[u] = best
                    updated = True
            if not updated:
                break

        return depth

    def _remap_llm_tokens_permutation(self, old_to_new_map: Dict[int, int]) -> None:
        """
        Apply a bijective mapping old_token_id -> new_token_id to all model RangeSets.
        Does not modify id_to_token (external mapping). Updates:
        - internal_to_original_map (permutation)
        - possible_matches_cache
        - all_internal_llm_tokens_bitset
        - arena (by regrouping tokens)
        """
        # Ensure mapping is bijection over the current universe
        domain = set(old_to_new_map.keys())
        image = set(old_to_new_map.values())
        if len(domain) != len(image):
            # Not a permutation; do nothing
            return

        def remap_llm_token_set(s: LLMTokenSet) -> LLMTokenSet:
            if s.is_empty():
                return s
            new_indices = [old_to_new_map[i] for i in s.to_indices() if i in old_to_new_map]
            if not new_indices:
                return PyRangeSet.empty()
            return PyRangeSet.from_indices(sorted(new_indices))

        # Remap possible_matches_cache
        if self.possible_matches_cache:
            new_pmc: Dict[int, Dict[int, LLMTokenSet]] = {}
            for tsid, inner in self.possible_matches_cache.items():
                new_inner: Dict[int, LLMTokenSet] = {}
                for term_id, llm_bv in inner.items():
                    new_inner[int(term_id)] = remap_llm_token_set(llm_bv)
                new_pmc[int(tsid)] = new_inner
            self.possible_matches_cache = new_pmc

        # Remap the universe
        self.all_internal_llm_tokens_bitset = remap_llm_token_set(self.all_internal_llm_tokens_bitset)

        # Remap arena directly by regrouping tokens
        for node in self.arena.values():
            children = node.get("children", [])
            if not children:
                continue

            # Group old tokens by their transition signature (pop, dests)
            groups: Dict[Tuple[int, Tuple[Tuple[int, Tuple[Tuple[int, int], ...]], ...]], Set[int]] = collections.defaultdict(set)
            for (pop, llm_bv), dests in children:
                # Sort dests for a canonical signature. Each dest is (dest_id, state_bv).
                # state_bv is a PyRangeSet, so we use its ranges for the signature.
                dests_sig = tuple(sorted(
                    (dest_id, tuple(state_bv.to_ranges())) for dest_id, state_bv in dests
                ))
                signature = (pop, dests_sig)
                groups[signature].update(llm_bv.to_indices())

            # Rebuild children with remapped tokens
            new_children = []
            llm_union = PyRangeSet.empty()
            for (pop, dests_sig), old_tokens in groups.items():
                new_tokens = {old_to_new_map[t] for t in old_tokens if t in old_to_new_map}
                if not new_tokens:
                    continue

                new_llm_bv = PyRangeSet.from_indices(sorted(list(new_tokens)))
                llm_union = llm_union.union(new_llm_bv)
                # Convert dests_sig back to list of (int, StateIDSet)
                dests_list = [
                    (dest_id, PyRangeSet.from_ranges(list(ranges)))
                    for dest_id, ranges in dests_sig
                ]
                new_children.append(((pop, new_llm_bv), dests_list))

            # Deterministic sort
            def _ranges_key(rs: LLMTokenSet) -> Tuple[Tuple[int, int], ...]:
                try:
                    return tuple((int(a), int(b)) for (a, b) in rs.to_ranges())
                except Exception:
                    return ()

            new_children.sort(
                key=lambda item: (
                    int(item[0][0]),
                    _ranges_key(item[0][1]),
                    tuple(sorted(int(d[0]) for d in item[1])),
                )
            )

            node["children"] = new_children
            node["llm_bv_union"] = llm_union

        # Remap internal_to_original_map (pure permutation)
        if self.internal_to_original_map:
            new_internal_to_original_map: Dict[int, Set[int]] = {}
            for old_tok, orig_set in self.internal_to_original_map.items():
                if old_tok in old_to_new_map:
                    new_internal_to_original_map[old_to_new_map[old_tok]] = set(orig_set)
            self.internal_to_original_map = new_internal_to_original_map


    def _remap_llm_tokens_many_to_one(self, old_to_new_map: Dict[int, int]) -> None:
        """
        Apply a many-to-one mapping old_token_id -> new_token_id to all model RangeSets.
        This merges internal tokens that are equivalent, and updates:
        - internal_to_original_map (merge original IDs into representatives)
        - possible_matches_cache
        - all_internal_llm_tokens_bitset
        - arena children and llm_bv_union (by regrouping tokens per identical transition signature)
        """
        if not old_to_new_map:
            return

        def remap_llm_token_set(s: LLMTokenSet) -> LLMTokenSet:
            if s.is_empty():
                return s
            mapped: Set[int] = set()
            for i in s.to_indices():
                ii = int(i)
                mapped.add(old_to_new_map.get(ii, ii))
            if not mapped:
                return PyRangeSet.empty()
            return PyRangeSet.from_indices(sorted(mapped))

        # Remap possible_matches_cache
        if self.possible_matches_cache:
            new_pmc: Dict[int, Dict[int, LLMTokenSet]] = {}
            for tsid, inner in self.possible_matches_cache.items():
                new_inner: Dict[int, LLMTokenSet] = {}
                for term_id, llm_bv in inner.items():
                    new_inner[int(term_id)] = remap_llm_token_set(llm_bv)
                new_pmc[int(tsid)] = new_inner
            self.possible_matches_cache = new_pmc

        # Remap arena by grouping tokens with identical (pop, dests) signatures
        for node in self.arena.values():
            children = node.get("children", [])
            if not children:
                # Still ensure union is remapped (it may be non-empty)
                union_bv = node.get("llm_bv_union") or PyRangeSet.empty()
                node["llm_bv_union"] = remap_llm_token_set(union_bv)
                continue

            # signature: (pop, ((dest_id, ((start,end),...)), ...))
            groups: Dict[
                Tuple[int, Tuple[Tuple[int, Tuple[Tuple[int, int], ...]], ...]],
                Set[int]
            ] = collections.defaultdict(set)

            for (pop, llm_bv), dests in children:
                # Stable signature based on dests state ranges
                dests_sig = tuple(sorted(
                    (int(dest_id), tuple((int(a), int(b)) for (a, b) in state_bv.to_ranges()))
                    for dest_id, state_bv in dests
                ))
                signature = (int(pop), dests_sig)

                # Map each token in llm_bv to its representative
                mapped_tokens: Set[int] = set()
                for t in llm_bv.to_indices():
                    ti = int(t)
                    mapped_tokens.add(old_to_new_map.get(ti, ti))

                if mapped_tokens:
                    groups[signature].update(mapped_tokens)

            # Rebuild children and union
            new_children: List[Tuple[Tuple[int, LLMTokenSet], List[Tuple[int, StateIDSet]]]] = []
            llm_union: LLMTokenSet = PyRangeSet.empty()

            for (pop, dests_sig), tokens_set in groups.items():
                if not tokens_set:
                    continue
                llm_bv_new = PyRangeSet.from_indices(sorted(tokens_set))
                llm_union = llm_union.union(llm_bv_new)

                dests_list: List[Tuple[int, StateIDSet]] = []
                for dest_id, ranges in dests_sig:
                    dests_list.append((int(dest_id), PyRangeSet.from_ranges(list(ranges))))
                new_children.append(((int(pop), llm_bv_new), dests_list))

            # Deterministic sort
            def _ranges_key(rs: LLMTokenSet) -> Tuple[Tuple[int, int], ...]:
                try:
                    return tuple((int(a), int(b)) for (a, b) in rs.to_ranges())
                except Exception:
                    return ()

            new_children.sort(
                key=lambda item: (
                    int(item[0][0]),
                    _ranges_key(item[0][1]),
                    tuple(sorted(int(d[0]) for d in item[1])),
                )
            )

            node["children"] = new_children
            node["llm_bv_union"] = llm_union

        # Remap the universe
        self.all_internal_llm_tokens_bitset = remap_llm_token_set(self.all_internal_llm_tokens_bitset)

        # Merge internal_to_original_map entries into representatives
        if self.internal_to_original_map:
            new_internal_to_original_map: Dict[int, Set[int]] = {}
            for old_tok, orig_set in self.internal_to_original_map.items():
                rep = old_to_new_map.get(int(old_tok), int(old_tok))
                if rep not in new_internal_to_original_map:
                    new_internal_to_original_map[rep] = set()
                new_internal_to_original_map[rep].update(orig_set)
            self.internal_to_original_map = new_internal_to_original_map
