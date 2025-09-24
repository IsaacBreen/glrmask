from __future__ import annotations

import json
import heapq
import collections
import time
import random
from typing import Dict, List, Tuple, Optional, Union, Any, TypedDict, Set
from dataclasses import dataclass, field

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
# from python.gss_tester.implementations.leveled_impl_cpp import Leveled_impl_cppGSS as GSS


NodeID = int

# Type aliases for different uses of RangeSet to improve clarity.
LLMTokenSet = RangeSet
StateIDSet = RangeSet
TerminalIdSet = RangeSet


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
    # children[token][dest] = (pop, Edge)
    children: Dict[LLMToken, Dict[int, Tuple[int, Edge]]]
    is_end: bool


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
    roots_map_raw: List[Tuple[int, NodeID]]
    arena: Dict[NodeID, ArenaNode]  # This is Dict[int, ArenaNode] after __post_init__

    roots_map: Dict[int, NodeID] = field(init=False)
    id_to_token: Dict[int, bytes] = field(init=False, default_factory=dict)
    max_depth: Dict[NodeID, int] = field(init=False, default_factory=dict)
    possible_matches_cache: Optional[Dict[int, Dict[int, LLMTokenSet]]] = field(init=False, default=None)
    tokenizer: Optional[ffi.Regex] = field(init=False, default=None)
    glr_parser: Optional[ffi.GLRParser] = field(init=False, default=None)
    ignore_terminal_id: Optional[int] = field(init=False, default=None)
    parser_table: Optional[ParserTable] = field(init=False, default=None)
    reverse_state_map: Dict[int, Set[int]] = field(init=False, default_factory=dict)
    state: Dict[int, GSS] = field(init=False, default_factory=dict)
    internal_to_original_map: Dict[int, int] = field(init=False, default_factory=dict)
    all_internal_llm_tokens_bitset: Optional[LLMTokenSet] = field(init=False, default=None)
    tokenizer_initial_state: Optional[int] = field(init=False, default=None)
    tokenizer_max_state: Optional[int] = field(init=False, default=None)
    all_terminals_bitset: Optional[TerminalIdSet] = field(init=False, default=None)

    def __post_init__(self):
        self.roots_map = {int(s): int(r) for s, r in self.roots_map_raw}

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # Normalize arena children bitsets and cache max_depth
        for uid, node in self.arena.items():
            uid_int = int(uid)
            self.max_depth[uid_int] = int(node.get("max_depth", 0) or 0)

            children = node.get("children") or []
            if not children:
                node["children"] = []
                node["llm_bv_union"] = LLMTokenSet.empty()
                continue

            new_children = []
            llm_bv_union: LLMTokenSet = LLMTokenSet.empty()
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv_bitset = bs_from_json(dumps(llm_bv_json))
                # Convert to RangeSet for ffi-free operations in commit/get_mask
                llm_bv: LLMTokenSet = LLMTokenSet.from_ranges(llm_bv_bitset.to_ranges())
                llm_bv_union = llm_bv_union.union(llm_bv)
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv_bitset = bs_from_json(dumps(state_bv_json))
                    state_bv: StateIDSet = StateIDSet.from_ranges(state_bv_bitset.to_ranges())
                    new_dest_map.append((int(dest_idx), state_bv))
                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children
            node["llm_bv_union"] = llm_bv_union

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena_values = arena_json.get("values", [])
        arena = {int(k): v for k, v in arena_values}
        model = Model(roots_map, arena)

        # Load tokenizer and parser table from the full constraint JSON
        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.tokenizer = constraint.tokenizer()
        model.tokenizer_max_state = model.tokenizer.max_state()
        model.glr_parser = constraint.glr_parser()
        model.ignore_terminal_id = model.glr_parser.ignore_terminal_id
        model.tokenizer_initial_state = model.tokenizer.initial_state_id()

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
        model.parser_table = ParserTable(start_state_id, py_table)

        reverse_map: Dict[int, Set[int]] = collections.defaultdict(set)
        for from_state, row in model.parser_table.table.items():
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
        model.reverse_state_map = dict(reverse_map)

        all_terminals = set()
        for row in model.parser_table.table.values():
            all_terminals.update(row.actions.keys())
        if model.ignore_terminal_id is not None:
            all_terminals.add(model.ignore_terminal_id)
        model.all_terminals_bitset = TerminalIdSet.from_indices(list(all_terminals))

        initial_acc = PyAcc(terminals_union={}, llm_mask=LLMTokenSet.empty())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        # Convert possible_matches_cache to RangeSet
        pmc_ffi: Dict[int, Dict[int, ffi.Bitset]] = constraint.possible_matches()
        pmc_rs: Dict[int, Dict[int, LLMTokenSet]] = {}
        for tsid, inner in pmc_ffi.items():
            mapped: Dict[int, LLMTokenSet] = {}
            for term_id, bit in inner.items():
                mapped[int(term_id)] = LLMTokenSet.from_ranges(bit.to_ranges())
            pmc_rs[int(tsid)] = mapped
        model.possible_matches_cache = pmc_rs
        model.internal_to_original_map = constraint.internal_to_original_map()
        # Convert universe LLM tokens bitset to RangeSet
        all_internal = constraint.all_internal_llm_tokens_bitset()
        model.all_internal_llm_tokens_bitset = LLMTokenSet.from_ranges(all_internal.to_ranges())

        model._unconditionalize_guaranteed_transitions()

        return model

    @profile
    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, TerminalIdSet]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            disallowed_terminals_map = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(state_id, TerminalIdSet.empty())
                if not matched_bv.intersection(disallowed_for_state).is_empty():
                    return False
            return True
        return gss.prune(predicate)

    @profile
    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_map = acc.terminals_union
            new_bvs: Dict[int, TerminalIdSet] = collections.defaultdict(TerminalIdSet.empty)
            for old_sid, new_sid in state_map.items():
                bv_source = old_map.get(old_sid, TerminalIdSet.empty())
                new_bvs[new_sid] = new_bvs[new_sid].union(bv_source)

            return PyAcc(terminals_union=dict(new_bvs), llm_mask=acc.llm_mask)
        return gss.apply(apply_map)

    @profile
    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_map = acc.terminals_union.copy()
            curr_bv = current_map.get(state_id, TerminalIdSet.empty())
            to_add = TerminalIdSet.from_indices([terminal_id])
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
            terminals_map[tokenizer_sid] = TerminalIdSet.from_indices(matched_terminals)

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
        # print(f"commit (ms): {round((t1 - t0) * 1000, 2)}")

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
        final_mask: LLMTokenSet = LLMTokenSet.empty()

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
            disallowed_llm_mask: LLMTokenSet = LLMTokenSet.empty()
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

            allowed_mask = (all_ones if all_ones is not None else LLMTokenSet.empty()).difference(disallowed_llm_mask)
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
            node_llm_bv_union: LLMTokenSet = arena.get(node, {}).get("llm_bv_union", LLMTokenSet.empty())
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
                    peeked = popped.peek()
                    values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]

                    if not values_to_keep:
                        continue

                    child_gss: GSS = popped.isolate_many(values_to_keep)
                    if child_gss.is_empty():
                        continue

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
                original_indices.append(self.internal_to_original_map[i])


        return LLMTokenSet.from_indices(original_indices)

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

    def _to_nodeopt_graph(self) -> Dict[NodeID, NodeOpt]:
        """
        Convert the arena children into a NodeOpt graph:
        - Explode llm_bv into individual tokens
        - Represent each edge as (pop, Edge), where Edge is either
          StateEdge(states) or UnconditionalEdge() (pop carried separately).
        """
        nodeopts: Dict[NodeID, NodeOpt] = {}
        for uid, node in self.arena.items():
            token_map: Dict[int, Dict[int, Tuple[int, Edge]]] = collections.defaultdict(dict)
            children = node.get("children") or []
            for (pop, llm_bv), dests in children:
                tokens = llm_bv.to_indices()
                for dest_idx, state_bv in dests:
                    if state_bv.is_empty():
                        edge_obj: Edge = UnconditionalEdge()
                    else:
                        edge_obj = StateEdge(set(state_bv.to_indices()))
                    for tok in tokens:
                        existing = token_map[tok].get(int(dest_idx))
                        if existing is None:
                            # store a copy of states set if StateEdge
                            if isinstance(edge_obj, StateEdge):
                                token_map[tok][int(dest_idx)] = (int(pop), StateEdge(set(edge_obj.states)))
                            else:
                                token_map[tok][int(dest_idx)] = (int(pop), UnconditionalEdge())
                        else:
                            # Merge if the exact same dest/token encountered again
                            existing_pop, existing_edge = existing
                            if existing_pop != int(pop):
                                # Extremely rare; conservatively collapse to unconditional with the existing pop.
                                token_map[tok][int(dest_idx)] = (existing_pop, UnconditionalEdge())
                            else:
                                if isinstance(existing_edge, UnconditionalEdge) or isinstance(edge_obj, UnconditionalEdge):
                                    token_map[tok][int(dest_idx)] = (existing_pop, UnconditionalEdge())
                                else:
                                    merged = set(existing_edge.states)
                                    merged.update(edge_obj.states)
                                    token_map[tok][int(dest_idx)] = (existing_pop, StateEdge(merged))

            nodeopts[int(uid)] = NodeOpt(children={t: dict(dm) for t, dm in token_map.items()}, is_end=self.is_end(int(uid)))
        return nodeopts

    def _compute_alive_states(self, nodeopts: Dict[NodeID, NodeOpt]) -> Dict[NodeID, Set[int]]:
        """
        Compute per-node 'alive' parser state sets at fixpoint.
        Seed with {start_state_id} at all root nodes.
        Propagation rules:
        - Across each edge (pop, Edge):
            S' = reverse_state_map^pop(S)
            If StateEdge(states): S'' = S' ∩ states
            If UnconditionalEdge: S'' = S'
        """
        start_state = self.parser_table.start_state_id
        alive: Dict[int, Set[int]] = collections.defaultdict(set)

        q = collections.deque()
        for _, root in self.roots_map.items():
            if start_state not in alive[root]:
                alive[root].add(start_state)
                q.append(root)

        while q:
            node_id = q.popleft()
            S = alive[node_id]
            if not S:
                continue
            children = nodeopts.get(node_id)
            if not children:
                continue
            for _tok, dest_map in children.children.items():
                for dest_id, (pop, edge) in dest_map.items():
                    pre = self._nodeopt_pop_preimage(S, pop)
                    if not pre:
                        continue
                    if isinstance(edge, StateEdge):
                        pre = pre.intersection(edge.states)
                        if not pre:
                            continue
                    # unconditional leaves 'pre' unchanged
                    dest_alive = alive[dest_id]
                    new_states = pre - dest_alive
                    if new_states:
                        dest_alive.update(new_states)
                        q.append(dest_id)
        return {k: set(v) for k, v in alive.items()}

    def _from_nodeopt_graph(self, nodeopts: Dict[NodeID, NodeOpt]) -> None:
        """
        Convert NodeOpt back into the arena node format:
        - Group tokens by identical (pop, dest -> state_bv) mapping into shared llm_bv bitsets.
        - Recompute llm_bv_union for each node.
        """
        for node_id, nodeopt in nodeopts.items():
            # 1. For each token, determine its transition map, partitioned by pop.
            #    token_transitions: {token -> {pop -> {dest -> states}}}
            token_transitions = collections.defaultdict(lambda: collections.defaultdict(dict))
            for tok, dests in nodeopt.children.items():
                for dest_id, (pop, edge) in dests.items():
                    allowed = tuple(sorted(edge.states)) if isinstance(edge, StateEdge) else None
                    token_transitions[tok][pop][dest_id] = allowed

            # 2. Invert the mapping to group tokens by identical (pop, dest_map).
            #    sig_to_tokens: {(pop, dest_map_sig) -> {token}}
            sig_to_tokens = collections.defaultdict(set)
            for tok, pop_maps in token_transitions.items():
                for pop, dest_map in pop_maps.items():
                    # The signature must be canonical and hashable.
                    sig = tuple(sorted(dest_map.items()))
                    sig_to_tokens[(pop, sig)].add(tok)

            # 3. Build the new `children` list for the arena.
            new_children: List[Tuple[Tuple[int, LLMTokenSet], List[Tuple[int, StateIDSet]]]] = []
            llm_union: LLMTokenSet = LLMTokenSet.empty()
            for (pop, dest_map_sig), tokens in sig_to_tokens.items():
                llm_bv = LLMTokenSet.from_indices(sorted(list(tokens)))
                llm_union = llm_union.union(llm_bv)

                dests_list: List[Tuple[int, StateIDSet]] = []
                for dest_id, allowed in dest_map_sig:  # sig is already a sorted tuple of items
                    state_bv = StateIDSet.empty()
                    if allowed is None:
                        state_bv = StateIDSet.empty()
                    else:
                        state_bv = StateIDSet.from_indices(list(allowed))
                    dests_list.append((int(dest_id), state_bv))

                new_children.append(((int(pop), llm_bv), dests_list))

            # 4. Update the arena node content in-place.
            node_ref = self.arena.get(int(node_id), {})
            node_ref["children"] = new_children
            node_ref["llm_bv_union"] = llm_union
            self.arena[int(node_id)] = node_ref

    def _unconditionalize_guaranteed_transitions(
        self,
        time_budget_sec: float = 10.0,
        stagnation_limit: int = 2000,
        rng_seed: Optional[int] = None
    ) -> None:
        """
        Stochastic optimization pass that attempts to convert state-filtered edges
        to unconditional edges when doing so is guaranteed not to introduce new
        accepting paths or merge with existing alive sets.

        Algorithm overview:
        1) Build NodeOpt graph and compute 'alive' parser states per node to fixpoint.
        2) Randomly pick state-filtered edges. Let S be alive at src; E be edge's allowed set.
           - X0 = S \ E
           - If X0 is empty, we can immediately unconditionalize (edge is as restrictive as src allows).
           - Else X1 = reverse_state_map^pop(X0). If X1 is empty, also safe to unconditionalize.
           - If X1 intersects alive at destination, reject (would merge with existing alive states).
           - Tentatively propagate X1 from destination; if it ever:
               a) reaches an end node (nodeopt.is_end), or
               b) intersects any alive set at any node,
              then reject.
           - Otherwise, accept and flip to UnconditionalEdge().
        3) Continue until time budget exhausted or no successes for 'stagnation_limit' trials.
        4) If any changes were made, convert back to arena.
        """
        t_start = time.perf_counter()
        deadline = t_start + float(time_budget_sec)
        rng = random.Random(rng_seed)

        nodeopts = self._to_nodeopt_graph()
        alive = self._compute_alive_states(nodeopts)

        # Collect candidate edges: (src, token, dest)
        candidates: List[Tuple[int, int, int]] = []
        for src_id, nodeopt in nodeopts.items():
            for tok, dest_map in nodeopt.children.items():
                for dest_id, (_pop, edge) in dest_map.items():
                    if isinstance(edge, StateEdge):
                        candidates.append((int(src_id), int(tok), int(dest_id)))

        if not candidates:
            print("No state-filtered edges found; skipping unconditionalization.")
            return

        def simulate_propagation_from(node_id: int, seed_states: Set[int]) -> bool:
            """
            Return True if unsafe:
              - intersects any alive set during propagation, or
              - reaches an end node with a non-empty set.
            Else return False (safe / ineffectual).
            """
            if not seed_states:
                return False
            # Immediate checks on starting node
            if nodeopts[node_id].is_end and seed_states:
                return True
            if seed_states & alive.get(node_id, set()):
                return True
            pending: collections.deque[Tuple[int, Set[int]]] = collections.deque()
            pending.append((node_id, set(seed_states)))
            added: Dict[int, Set[int]] = {node_id: set(seed_states)}

            while pending:
                nid, states_here = pending.popleft()
                if not states_here:
                    continue
                children = nodeopts[nid].children
                for _tok, dest_map in children.items():
                    for dest2, (pop2, edge2) in dest_map.items():
                        pre2 = self._nodeopt_pop_preimage(states_here, pop2)
                        if not pre2:
                            continue
                        if isinstance(edge2, StateEdge):
                            pre2 = pre2.intersection(edge2.states)
                            if not pre2:
                                continue
                        # Check end node or intersection with alive
                        if nodeopts[dest2].is_end and pre2:
                            return True
                        if pre2 & alive.get(dest2, set()):
                            return True
                        # Only propagate brand-new additions for this simulation
                        already = added.get(dest2)
                        if already is None:
                            added[dest2] = set(pre2)
                            pending.append((dest2, set(pre2)))
                        else:
                            diff = pre2 - already
                            if diff:
                                already.update(diff)
                                pending.append((dest2, diff))
            return False

        converted = 0
        stagnation = 0
        while candidates and time.perf_counter() < deadline and stagnation < stagnation_limit:
            src, tok, dst = rng.choice(candidates)
            edge_tuple = nodeopts[src].children.get(tok, {}).get(dst)
            if edge_tuple is None:
                # Removed or changed; drop from candidates
                candidates.remove((src, tok, dst))
                continue
            pop, edge = edge_tuple
            if isinstance(edge, UnconditionalEdge):
                # No longer a candidate
                candidates.remove((src, tok, dst))
                continue
            if not isinstance(edge, StateEdge):
                # Shouldn't happen, but skip to be safe
                candidates.remove((src, tok, dst))
                continue

            src_alive = alive.get(src, set())
            allowed = edge.states
            extra = src_alive - allowed
            safe = False
            if not extra:
                safe = True
            else:
                seed = self._nodeopt_pop_preimage(extra, pop)
                if not seed:
                    safe = True
                else:
                    # If new states at destination intersect existing alive, reject
                    if seed & alive.get(dst, set()):
                        safe = False
                    else:
                        unsafe = simulate_propagation_from(dst, seed)
                        safe = not unsafe

            if safe:
                # Convert to unconditional
                nodeopts[src].children[tok][dst] = (pop, UnconditionalEdge())
                candidates.remove((src, tok, dst))
                converted += 1
                stagnation = 0
            else:
                stagnation += 1

        if converted > 0:
            # Convert NodeOpt back to arena inplace
            self._from_nodeopt_graph(nodeopts)
        # If converted == 0, arena remains unchanged.

        print(f"Unconditionalized {converted} edges in {round(time.perf_counter() - t_start, 2)} sec")