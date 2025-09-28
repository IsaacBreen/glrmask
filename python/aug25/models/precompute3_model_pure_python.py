from __future__ import annotations

import json
import heapq
import collections
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field

# Progress bars (safe fallback if tqdm is unavailable)
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from ..common_interface import GraphProvider
from ..common_interface import RangeSet
# from ..range_set.py_range_set import PyRangeSet as RangeSet
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


@dataclass
class ArenaNode:
    children: List[Tuple[Tuple[int, LLMTokenSet], List[Tuple[NodeID, StateIDSet]]]] = field(default_factory=list)
    llm_bv_union: LLMTokenSet = field(default_factory=RangeSet.empty)
    clean_end: bool = False

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
        arena_dict = {int(k): v for k, v in arena_values}
 
        roots_map = {int(s): int(r) for s, r in roots_map_raw}
        max_depth: Dict[NodeID, int] = {}
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # Normalize arena children bitsets and cache max_depth
        for uid, node in arena_dict.items():
            uid_int = int(uid)
            max_depth[uid_int] = int(node.get("max_depth", 0) or 0)

            children = node.get("children") or []
            if not children:
                node["children"] = []
                node["llm_bv_union"] = RangeSet.empty()
                continue

            new_children = []
            llm_bv_union: LLMTokenSet = RangeSet.empty()
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv_bitset = bs_from_json(dumps(llm_bv_json))
                # Convert to RangeSet for ffi-free operations in commit/get_mask
                llm_bv: LLMTokenSet = RangeSet.from_ranges(llm_bv_bitset.to_ranges())
                llm_bv_union = llm_bv_union.union(llm_bv)
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv_bitset = bs_from_json(dumps(state_bv_json))
                    state_bv: StateIDSet = RangeSet.from_ranges(state_bv_bitset.to_ranges())
                    new_dest_map.append((int(dest_idx), state_bv))
                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children
            node["llm_bv_union"] = llm_bv_union

        arena: Dict[NodeID, ArenaNode] = {
            uid: ArenaNode(
                children=node_data.get("children", []),
                llm_bv_union=node_data.get("llm_bv_union", RangeSet.empty()),
                clean_end=node_data.get("value", {}).get("clean_end", False),
            )
            for uid, node_data in arena_dict.items()
        }
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
        all_terminals_bitset = RangeSet.from_indices(list(all_terminals))

        initial_acc = PyAcc(terminals_union={}, llm_mask=RangeSet.empty())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(parser_table.start_state_id)
        state = {tokenizer_initial_state: initial_gss}

        id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        # Convert possible_matches_cache to RangeSet
        pmc_ffi: Dict[int, Dict[int, ffi.Bitset]] = constraint.possible_matches()
        pmc_rs: Dict[int, Dict[int, LLMTokenSet]] = {}
        for tsid, inner in pmc_ffi.items():
            mapped: Dict[int, LLMTokenSet] = {}
            for term_id, bit in inner.items():
                mapped[int(term_id)] = RangeSet.from_ranges(bit.to_ranges())
            pmc_rs[int(tsid)] = mapped
        possible_matches_cache = pmc_rs

        vocab = data.get('precompute3_vocab') or data.get('precompute2_vocab') or data.get('precompute_vocab')
        if vocab:
            internal_to_original_map_raw = dict(vocab['internal_to_original'])
            internal_to_original_map = {
                int(k): set(v) for k, v in internal_to_original_map_raw.items()
            }
            internal_max = vocab['internal_max_llm_token']
            all_internal_llm_tokens_bitset = RangeSet.from_ranges([(0, internal_max)])
        else:
            internal_to_original_map_raw = constraint.internal_to_original_map()
            internal_to_original_map = {k: {v} for k, v in internal_to_original_map_raw.items()}
            all_internal = constraint.all_internal_llm_tokens_bitset()
            all_internal_llm_tokens_bitset = RangeSet.from_ranges(all_internal.to_ranges())

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

        # model._merge_equivalent_llm_tokens()
        # model._reorder_llm_tokens_for_range_minimization()
        # Run graph optimization after LLM token reorder, as requested
        # model._optimize_state_masks_and_edges()
        # # Additional compaction: merge equivalent subgraphs and coalesce edges
        # model._post_optimize_merge_subgraphs_and_edges()
        # model._merge_equivalent_llm_tokens()
        # model._reorder_llm_tokens_for_range_minimization()
        # model._recompute_max_depth_from_arena()
        return model

    @profile
    def _prune_and_map_gss_for_commit(self, gss: GSS, terminals_map: Dict[int, TerminalIdSet], state_map: Dict[int, int]) -> GSS:
        """
        Combines pruning and mapping of a GSS for the commit step in a single pass.
        """
        def mutator(acc: PyAcc) -> Optional[PyAcc]:
            # Pruning logic: check if any matched terminal is disallowed.
            disallowed_terminals_map = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(state_id, RangeSet.empty())
                if not matched_bv.intersection(disallowed_for_state).is_empty():
                    return None  # Prune this stack.

            # Mapping logic: update terminals_union for new tokenizer states.
            old_map = acc.terminals_union
            new_bvs: Dict[int, TerminalIdSet] = collections.defaultdict(RangeSet.empty)
            for old_sid, new_sid in state_map.items():
                bv_source = old_map.get(old_sid, RangeSet.empty())
                new_bvs[new_sid] = new_bvs[new_sid].union(bv_source)

            return PyAcc(terminals_union=dict(new_bvs), llm_mask=acc.llm_mask)

        return gss.apply_and_prune(mutator)

    @profile
    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_map = acc.terminals_union.copy()
            curr_bv = current_map.get(state_id, RangeSet.empty())
            to_add = RangeSet.from_indices([terminal_id])
            new_bv = curr_bv.union(to_add)
            current_map[state_id] = new_bv
            return PyAcc(terminals_union=current_map, llm_mask=acc.llm_mask)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> NodeID:
        return self.roots_map[int(state_id)]

    def is_end(self, node: NodeID) -> bool:
        a_node = self.arena.get(node)
        if not a_node:
            return False
        return a_node.clean_end

    def iter_edges(self, node: NodeID, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        """
        a_node = self.arena.get(node)
        children = a_node.children if a_node else []
        for (pop, llm_bv), dests in children:
            if llm_bv.contains(token):
                for dest_idx, state_bv in dests:
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
            terminals_map[tokenizer_sid] = RangeSet.from_indices(matched_terminals)

        t1 = time.perf_counter()

        # Prune and map per-state GSS
        temp_states: Dict[int, GSS] = {}
        for tokenizer_sid, gss in self.state.items():
            processed_gss = self._prune_and_map_gss_for_commit(gss, terminals_map, state_map)
            if not processed_gss.is_empty():
                temp_states[tokenizer_sid] = processed_gss

        t2 = time.perf_counter()

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

        t3 = time.perf_counter()

        merged_states = {
            sid: GSS.merge_many(gss_list)
            for sid, gss_list in new_states.items()
            if gss_list
        }
        merged_states = {sid: state for sid, state in merged_states.items() if not state.is_empty()}

        t4 = time.perf_counter()

        self.state = merged_states

        t5 = time.perf_counter()
        print(f"commit({token_id}, {token_bytes!r}):")
        print(f"  - build tokenizer maps: {t1 - t0:.6f}s")
        print(f"  - prune and map gss:    {t2 - t1:.6f}s")
        print(f"  - process queue:        {t3 - t2:.6f}s")
        print(f"  - merge states:         {t4 - t3:.6f}s")
        print(f"  - final assignment:     {t5 - t4:.6f}s")
        print(f"  - TOTAL:                {t5 - t0:.6f}s")

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
        state_map: Dict[int, GSS] = self.state.copy()

        all_ones: LLMTokenSet = self.all_internal_llm_tokens_bitset
        final_mask: LLMTokenSet = RangeSet.empty()

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
            disallowed_llm_mask: LLMTokenSet = RangeSet.empty()
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

            allowed_mask = all_ones.difference(disallowed_llm_mask)
            return PyAcc(
                terminals_union={},  # consume
                llm_mask=allowed_mask,
            )

        state_map = {sid: gss.apply(initialize_acc) for sid, gss in state_map.items()}

        for sid, gss in state_map.items():
            r: NodeID = roots_map[int(sid)]
            if r in values:
                values[r] = values[r].merge(gss)
            else:
                values[r] = gss

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
            enqueued_nodes.remove(node)

            # End-node handling: just union the allowed LLM tokens
            if is_end(node):
                reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                if reduced_acc:
                    final_mask = final_mask.union(reduced_acc.llm_mask)

            # Zombie traversal avoidance
            a_node = arena.get(node)
            node_llm_bv_union: LLMTokenSet = a_node.llm_bv_union if a_node else RangeSet.empty()
            potential_new_tokens = node_llm_bv_union.difference(final_mask)
            if potential_new_tokens.is_empty():
                continue

            gss_mask_acc = gss_node.reduce_acc()
            if gss_mask_acc and gss_mask_acc.llm_mask.intersection(potential_new_tokens).is_empty():
                continue

            # Traverse edges and propagate masks
            a_node = arena.get(node)
            edges = a_node.children if a_node else []
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
                    values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]

                    if not values_to_keep:
                        continue

                    child_gss = popped.isolate_many(values_to_keep)
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
                original_indices.extend(list(self.internal_to_original_map[i]))


        return RangeSet.from_indices(original_indices)

    # ===========================
    # Optimization/conversion API
    # ===========================
    def _count_total_ranges(self) -> int:
        count = 0
        for node in self.arena.values():
            for _, llm_bv in (c[0] for c in node.children):
                count += len(llm_bv.to_ranges())
            union_bv = node.llm_bv_union
            count += len(union_bv.to_ranges())
        if self.possible_matches_cache:
            for inner in self.possible_matches_cache.values():
                for llm_bv in inner.values():
                    count += len(llm_bv.to_ranges())
        return count

    def _merge_equivalent_llm_tokens(self) -> None:
        """
        Merge internal LLM tokens that are indistinguishable across the entire model:
        two tokens are "equivalent" if they occur together in every RangeSet occurrence
        the model uses (arena edge llm_bv, node llm_bv_union, and possible_matches_cache).

        Implementation:
        - Build a family of all RangeSet occurrences that reference internal tokens.
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
            children = node.children
            for (pop, llm_bv), _dests in children:
                if not llm_bv.is_empty():
                    family.append(llm_bv)
            union_bv: LLMTokenSet = node.llm_bv_union
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

    def _reorder_llm_tokens_for_range_minimization(self) -> None:
        """
        Permute internal LLM token IDs to reduce the number of ranges present in
        all RangeSet occurrences across the model (arena edges, unions, and possible_matches_cache).
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
            children = node.children
            if children:
                for (pop, llm_bv), _dests in children:
                    idxs = [int(x) for x in llm_bv.to_indices() if int(x) in all_tokens]
                    if len(idxs) > 1:
                        groups_counter[tuple(idxs)] += edge_weight
            # Node union (optional)
            union_bv: LLMTokenSet = node.llm_bv_union
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

    def _recompute_llm_bv_unions(self) -> None:
        """
        Iterate through the arena and recompute the llm_bv_union for each node
        based on the union of llm_bvs of its direct children edges.
        """
        for node in self.arena.values():
            union_bv: LLMTokenSet = RangeSet.empty()
            for (_pop, llm_bv), _dests in node.children:
                union_bv = union_bv.union(llm_bv)
            node.llm_bv_union = union_bv

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
            is_end_map[nid] = node.clean_end
            for (pop, llm_bv), dests in node.children:
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
                return RangeSet.empty()
            return RangeSet.from_indices(sorted(new_indices))

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
            children = node.children
            if not children:
                continue

            # Group old tokens by their transition signature (pop, dests)
            groups: Dict[Tuple[int, Tuple[Tuple[int, Tuple[Tuple[int, int], ...]], ...]], Set[int]] = collections.defaultdict(set)
            for (pop, llm_bv), dests in children:
                # Sort dests for a canonical signature. Each dest is (dest_id, state_bv).
                # state_bv is a RangeSet, so we use its ranges for the signature.
                dests_sig = tuple(sorted(
                    (dest_id, tuple(state_bv.to_ranges())) for dest_id, state_bv in dests
                ))
                signature = (pop, dests_sig)
                groups[signature].update(llm_bv.to_indices())

            # Rebuild children with remapped tokens
            new_children = []
            llm_union = RangeSet.empty()
            for (pop, dests_sig), old_tokens in groups.items():
                new_tokens = {old_to_new_map[t] for t in old_tokens if t in old_to_new_map}
                if not new_tokens:
                    continue

                new_llm_bv = RangeSet.from_indices(sorted(list(new_tokens)))
                llm_union = llm_union.union(new_llm_bv)
                # Convert dests_sig back to list of (int, StateIDSet)
                dests_list = [
                    (dest_id, RangeSet.from_ranges(list(ranges)))
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

            node.children = new_children
            node.llm_bv_union = llm_union

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
                return RangeSet.empty()
            return RangeSet.from_indices(sorted(mapped))

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
            children = node.children
            if not children:
                # Still ensure union is remapped (it may be non-empty)
                union_bv = node.llm_bv_union
                node.llm_bv_union = remap_llm_token_set(union_bv)
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
            llm_union: LLMTokenSet = RangeSet.empty()

            for (pop, dests_sig), tokens_set in groups.items():
                if not tokens_set:
                    continue
                llm_bv_new = RangeSet.from_indices(sorted(tokens_set))
                llm_union = llm_union.union(llm_bv_new)

                dests_list: List[Tuple[int, StateIDSet]] = []
                for dest_id, ranges in dests_sig:
                    dests_list.append((int(dest_id), RangeSet.from_ranges(list(ranges))))
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

            node.children = new_children
            node.llm_bv_union = llm_union

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

    # ===========================
    # Graph optimization (masks + edges)
    # ===========================
    def _count_edges(self) -> int:
        """
        Count total number of edge destinations across the arena.
        Each (dest_id, state_bv) pair inside a child group counts as one edge.
        """
        total = 0
        for node in self.arena.values():
            for (_pop, _llm_bv), dests in node.children:
                total += len(dests)
        return total

    def _build_parser_state_universe(self) -> Tuple[Set[int], StateIDSet]:
        """
        Returns:
          - universe_set: set of all parser LR states (actual)
          - universe_rs: contiguous [0..max_state] RangeSet for universal masks
        """
        if not self.parser_table or not self.parser_table.table:
            return set(), RangeSet.empty()
        all_states = set(int(s) for s in self.parser_table.table.keys())
        if not all_states:
            return set(), RangeSet.empty()
        max_sid = max(all_states)
        # Use contiguous 0..max_sid for maximal interval minimization
        universe_rs = RangeSet.from_ranges([(0, int(max_sid))])
        return all_states, universe_rs

    def _build_forward_state_map(self) -> Dict[int, Set[int]]:
        """
        Build forward one-pop map: for each state t, forward[t] = {y | t ∈ reverse_state_map[y]}.
        Then invR(X) = ⋃_{t in X} forward[t].
        """
        fwd: Dict[int, Set[int]] = collections.defaultdict(set)
        rev = self.reverse_state_map or {}
        for y, preds in rev.items():
            yy = int(y)
            for x in preds:
                fwd[int(x)].add(yy)
        # Ensure all known states are present with possibly-empty sets
        for s in self.parser_table.table.keys():
            _ = fwd[int(s)]
        return fwd

    def _compute_terminal_state_sets(self) -> Dict[NodeID, Set[int]]:
        """
        Compute T(n): the set of LR states that are terminal at node n (ignoring LLM tokens),
        i.e., starting at node n with top-of-stack in state s, there exists a path to an end node.

        Recurrence:
          T(n) = (n is end) ? U : ⋃ over edges e=(n --(p, S)-> v) of invR^p(S ∩ T(v))
        where invR(X) = { y | R({y}) ∩ X != ∅ } and R is the reverse_state_map mapping (apply after pop).
        """
        universe_set, _ = self._build_parser_state_universe()
        if not universe_set:
            return {}

        # Prepare forward map for invR
        forward = self._build_forward_state_map()

        def invR_pow(X: Set[int], p: int) -> Set[int]:
            if p <= 0:
                # No pop: preimage is the set itself
                return set(X)
            S: Set[int] = set(X)
            for _ in range(int(p)):
                nxt: Set[int] = set()
                for t in S:
                    nxt.update(forward.get(int(t), ()))
                S = nxt
                if not S:
                    break
            return S

        # Materialize edges per node: list of (pop, dest_id, state_set)
        edges: Dict[NodeID, List[Tuple[int, NodeID, Set[int]]]] = {}
        for nid, node in self.arena.items():
            lst: List[Tuple[int, NodeID, Set[int]]] = []
            for (pop, _llm_bv), dests in node.children:
                for dest_id, state_bv in dests:
                    state_set = set(int(x) for x in state_bv.to_indices())
                    lst.append((int(pop), int(dest_id), state_set))
            edges[int(nid)] = lst

        # Initialize T
        T: Dict[NodeID, Set[int]] = {int(n): set() for n in self.arena.keys()}
        for nid, node in self.arena.items():
            if node.clean_end:
                T[int(nid)] = set(universe_set)

        # Bottom-up order using max_depth (smallest depth first).
        order = sorted(self.arena.keys(), key=lambda n: int(self.max_depth.get(int(n), 0)))

        # Fixed-point iteration with a safe upper bound
        N = max(1, len(order))
        for _iter in range(N * 2):
            changed = False
            for nid in tqdm(order, desc="Compute terminal states (fixed-point)", leave=False):
                nid_int = int(nid)
                accum = set(T[nid_int])  # current terminal states at nid
                if self.arena[nid_int].clean_end:
                    # Already universal; no need to process edges
                    if accum != universe_set:
                        T[nid_int] = set(universe_set)
                        changed = True
                    continue
                for pop, dest_id, state_set in edges.get(nid_int, []):
                    Tv = T[int(dest_id)]
                    if not Tv:
                        continue
                    X = state_set & Tv
                    if not X:
                        continue
                    pre = invR_pow(X, pop)
                    if pre:
                        before = len(accum)
                        accum |= pre
                        if len(accum) != before:
                            changed = True
                if T[nid_int] != accum:
                    T[nid_int] = accum
            if not changed:
                break
        return T

    def _transform_state_masks_alt6(self, T: Dict[NodeID, Set[int]]) -> Dict[str, int]:
        """
        Goal 1 Alt 6: Make as many (0..num_states) masks as possible by safely adding
        non-terminal states to edge masks. For each edge to v with mask S:
            S' = (U \ T(v)) ∪ (S ∩ T(v))
        This preserves all existing terminal states while filling every non-terminal.
        """
        stats = {
            "dest_masks_total": 0,
            "dest_masks_already_universal": 0,
            "dest_masks_made_universal": 0,
            "dest_masks_changed": 0,
        }

        universe_set, universe_rs = self._build_parser_state_universe()
        univ_full_indices = set(range(0, max(universe_set) + 1)) if universe_set else set()

        def is_universal(rs: StateIDSet) -> bool:
            if rs.is_empty():
                return False
            return rs.issubset(universe_rs) and universe_rs.issubset(rs)

        for nid, node in tqdm(self.arena.items(), desc="Transform state masks (Alt 6)", leave=False):
            for gi, ((pop, llm_bv), dests) in enumerate(node.children):
                new_dests: List[Tuple[int, StateIDSet]] = []
                for (dest_id, state_bv) in dests:
                    stats["dest_masks_total"] += 1
                    already_univ = is_universal(state_bv)
                    if already_univ:
                        stats["dest_masks_already_universal"] += 1
                        new_dests.append((int(dest_id), state_bv))
                        continue
                    Tv = T.get(int(dest_id), set())
                    S_set = set(int(x) for x in state_bv.to_indices())
                    # S' = (U\T) ∪ (S∩T)
                    addable_non_term = set(univ_full_indices) - set(Tv)
                    keep_terminals = S_set & set(Tv)
                    S_new = addable_non_term | keep_terminals
                    # Build new RangeSet mask
                    new_bv = RangeSet.from_indices(sorted(S_new)) if S_new else RangeSet.empty()
                    if not new_bv.is_empty():
                        if is_universal(new_bv):
                            stats["dest_masks_made_universal"] += 1
                        if list(new_bv.to_ranges()) != list(state_bv.to_ranges()):
                            stats["dest_masks_changed"] += 1
                        new_dests.append((int(dest_id), new_bv))
                    else:
                        # Extremely unlikely (if Tv == U and S had no terminals)
                        new_dests.append((int(dest_id), state_bv))
                node.children[gi] = ((int(pop), llm_bv), new_dests)
        return stats

    def _build_incoming_map(self) -> Dict[NodeID, List[Tuple[int, int, LLMTokenSet, StateIDSet]]]:
        """
        Build a map dest_id -> list of (src_id, src_pop, src_llm_bv, src_state_bv_to_that_dest).
        """
        incoming: Dict[NodeID, List[Tuple[int, int, LLMTokenSet, StateIDSet]]] = collections.defaultdict(list)
        for src_id, src_node in self.arena.items():
            for (pop, llm_bv), dests in src_node.children:
                for dest_id, state_bv in dests:
                    incoming[int(dest_id)].append((int(src_id), int(pop), llm_bv, state_bv))
        return incoming

    def _add_or_merge_child_edge(
        self,
        src_node: ArenaNode,
        pop: int,
        llm_bv: LLMTokenSet,
        dest_id: int,
        state_bv: StateIDSet,
    ) -> None:
        """
        Add an edge (pop,llm)->dest_id with state_bv to src_node, merging into an existing group
        if (pop,llm) already exists. Merge state_bv for same dest by union.
        """
        def _ranges_key(rs: LLMTokenSet) -> Tuple[Tuple[int, int], ...]:
            try:
                return tuple((int(a), int(b)) for (a, b) in rs.to_ranges())
            except Exception:
                return ()

        # Try to find a matching (pop,llm_bv) group
        for idx, ((pop0, llm0), dests) in enumerate(src_node.children):
            if int(pop0) != int(pop):
                continue
            if _ranges_key(llm0) == _ranges_key(llm_bv):
                # Merge into this group
                merged = False
                new_dests: List[Tuple[int, StateIDSet]] = []
                for (d_id, d_bv) in dests:
                    if int(d_id) == int(dest_id):
                        # union
                        new_dests.append((int(d_id), d_bv.union(state_bv)))
                        merged = True
                    else:
                        new_dests.append((int(d_id), d_bv))
                if not merged:
                    new_dests.append((int(dest_id), state_bv))
                src_node.children[idx] = ((int(pop), llm0), new_dests)
                return

        # No existing group; create a new one
        src_node.children.append(((int(pop), llm_bv), [(int(dest_id), state_bv)]))
        # Deterministic sort
        def _ranges_key(rs: LLMTokenSet) -> Tuple[Tuple[int, int], ...]:
            try:
                return tuple((int(a), int(b)) for (a, b) in rs.to_ranges())
            except Exception:
                return ()
        src_node.children.sort(
            key=lambda item: (
                int(item[0][0]),
                _ranges_key(item[0][1]),
                tuple(sorted(int(d[0]) for d in item[1])),
            )
        )

    def _remove_dest_entry(self, src_node: ArenaNode, pop: int, llm_bv: LLMTokenSet, dest_id: int) -> bool:
        """
        Remove the dest entry for dest_id from the group (pop,llm_bv) in src_node.
        Returns True if something was removed.
        """
        def _ranges_key(rs: LLMTokenSet) -> Tuple[Tuple[int, int], ...]:
            try:
                return tuple((int(a), int(b)) for (a, b) in rs.to_ranges())
            except Exception:
                return ()
        removed = False
        new_children = []
        for (pop0, llm0), dests in src_node.children:
            if int(pop0) == int(pop) and _ranges_key(llm0) == _ranges_key(llm_bv):
                new_dests = [(d, bv) for (d, bv) in dests if int(d) != int(dest_id)]
                if len(new_dests) != len(dests):
                    removed = True
                if new_dests:
                    new_children.append(((int(pop0), llm0), new_dests))
                # If no dests remain, drop the group entirely
            else:
                new_children.append(((int(pop0), llm0), dests))
        if removed:
            src_node.children = new_children
        return removed

    def _gc_unreachable_nodes(self) -> int:
        """
        Remove nodes not reachable from any root in roots_map.
        Returns number of nodes removed.
        """
        roots = set(int(r) for r in self.roots_map.values())
        if not roots:
            return 0

        # BFS from roots
        visited: Set[int] = set()
        q = collections.deque()
        for r in roots:
            if r in self.arena:
                visited.add(int(r))
                q.append(int(r))
        while q:
            u = q.popleft()
            node = self.arena.get(int(u))
            if not node:
                continue
            for (_pop, _llm), dests in node.children:
                for (v, _sbv) in dests:
                    if int(v) not in visited and int(v) in self.arena:
                        visited.add(int(v))
                        q.append(int(v))
        # Remove unreachable
        to_remove = [nid for nid in self.arena.keys() if int(nid) not in visited]
        for nid in to_remove:
            self.arena.pop(int(nid), None)
            self.max_depth.pop(int(nid), None)
        return len(to_remove)

    def _edge_replacement_pass(self) -> Dict[str, int]:
        """
        Perform replacement/contraction to reduce edges (Goal 2 Alt 2).
        Strategy:
          - Replacement 1: If x (A->B mask) is universal, replace B->C with A->C using pop=n+m and state mask y,
            llm mask becomes intersection; apply only if indegree(B)==1 or outdegree(B)==1, and B not an end.
          - Replacement 2: If outdegree(B)==1 and m==0, replace A->B->C with A->C using pop=n and state mask x∩y,
            llm mask intersection; again require indegree(B)==1 or outdegree(B)==1.
          - Only contract nodes B that are not roots.
        """
        stats = {
            "contractions": 0,
            "replacement_1": 0,
            "replacement_2": 0,
            "edges_added": 0,
            "edges_removed": 0,
            "nodes_removed": 0,
        }
        _, universe_rs = self._build_parser_state_universe()

        def is_universal(rs: StateIDSet) -> bool:
            if rs.is_empty():
                return False
            return rs.issubset(universe_rs) and universe_rs.issubset(rs)

        roots = set(int(r) for r in self.roots_map.values())

        # Helper to count outdegree (number of dest entries)
        def outdegree(node_id: int) -> int:
            node = self.arena.get(int(node_id))
            if not node:
                return 0
            total = 0
            for (_p, _llm), dests in node.children:
                total += len(dests)
            return total

        # Try to contract nodes in a loop until no changes
        while True:
            incoming = self._build_incoming_map()
            node_ids = list(self.arena.keys())
            changed = False

            for B in tqdm(node_ids, desc="Edge replacement pass", leave=False):
                B = int(B)
                if B not in self.arena:
                    continue
                if self.arena[B].clean_end:
                    continue  # never contract end nodes
                if B in roots:
                    continue  # do not contract root nodes

                inc_list = incoming.get(B, [])
                indeg = len(inc_list)
                out_edges: List[Tuple[int, LLMTokenSet, int, StateIDSet]] = []
                # Gather B's outgoing edges as flat list
                for (m, llm_bc), dests in self.arena[B].children:
                    for (C, y) in dests:
                        out_edges.append((int(m), llm_bc, int(C), y))
                outdeg = len(out_edges)

                if indeg == 0 or outdeg == 0:
                    continue

                # We only attempt when it will reduce edges: indegree==1 or outdegree==1
                if not (indeg == 1 or outdeg == 1):
                    continue

                # Check for replacement 1 condition: all incoming masks are universal
                all_incoming_universal = True
                for (A, n, llm_ab, x_mask) in inc_list:
                    if not is_universal(x_mask):
                        all_incoming_universal = False
                        break

                # Determine if replacement 2 is applicable: outdegree==1 and m==0
                repl2_candidate = False
                only_out = None
                if outdeg == 1:
                    only_out = out_edges[0]
                    m0, _llm_bc0, _C0, _y0 = only_out
                    if int(m0) == 0:
                        repl2_candidate = True

                performed = False
                if all_incoming_universal:
                    # Replacement 1
                    # Replicate all outgoing edges from B to incoming sources
                    for (A, n, llm_ab, x_mask) in inc_list:
                        src_node = self.arena.get(int(A))
                        if not src_node:
                            continue
                        for (m, llm_bc, C, y) in out_edges:
                            llm_new = llm_ab.intersection(llm_bc)
                            if llm_new.is_empty():
                                continue
                            state_new = y  # x is universal; keep y
                            pop_new = int(n) + int(m)
                            self._add_or_merge_child_edge(src_node, pop_new, llm_new, int(C), state_new)
                            stats["edges_added"] += 1
                        # remove A->B
                        if self._remove_dest_entry(src_node, int(n), llm_ab, int(B)):
                            stats["edges_removed"] += 1
                    # Remove node B entirely
                    if B in self.arena:
                        del self.arena[B]
                        self.max_depth.pop(B, None)
                        stats["nodes_removed"] += 1
                    stats["contractions"] += 1
                    stats["replacement_1"] += 1
                    performed = True
                elif repl2_candidate:
                    # Replacement 2 (m == 0). Replicate to all incoming, intersect state masks.
                    m0, llm_bc0, C0, y0 = only_out
                    for (A, n, llm_ab, x_mask) in inc_list:
                        src_node = self.arena.get(int(A))
                        if not src_node:
                            continue
                        llm_new = llm_ab.intersection(llm_bc0)
                        if llm_new.is_empty():
                            # nothing to add for this incoming
                            pass
                        else:
                            new_state = x_mask.intersection(y0)
                            if not new_state.is_empty():
                                pop_new = int(n)  # since m0 == 0, n + 0 == n
                                self._add_or_merge_child_edge(src_node, pop_new, llm_new, int(C0), new_state)
                                stats["edges_added"] += 1
                        # remove A->B
                        if self._remove_dest_entry(src_node, int(n), llm_ab, int(B)):
                            stats["edges_removed"] += 1
                    # Remove node B entirely
                    if B in self.arena:
                        del self.arena[B]
                        self.max_depth.pop(B, None)
                        stats["nodes_removed"] += 1
                    stats["contractions"] += 1
                    stats["replacement_2"] += 1
                    performed = True

                if performed:
                    changed = True
                    # Restart the outer loop with fresh incoming map
                    break

            if not changed:
                break

        return stats

    def _optimize_state_masks_and_edges(self) -> None:
        """
        Orchestrates the optimization:
          - Compute terminal states per node (ignoring LLM masks)
          - Transform state masks (Goal 1 Alt 6)
          - Perform edge replacement pass (Goal 2 Alt 2) while it reduces edges
          - GC unreachable nodes
          - Recompute llm_bv_union and max_depth
          - Report concrete improvements
        """
        print("Optimizing graph state masks and edges...", end="", flush=True)
        t0 = time.perf_counter()

        before_ranges = self._count_total_ranges()
        before_edges = self._count_edges()
        before_nodes = len(self.arena)

        # Terminal sets
        T = self._compute_terminal_state_sets()

        # Transform state masks
        mask_stats = self._transform_state_masks_alt6(T)

        # Replacement pass
        repl_stats = self._edge_replacement_pass()

        # GC unreachable nodes (safety)
        gc_removed = self._gc_unreachable_nodes()

        # Recompute unions/depths
        self._recompute_llm_bv_unions()
        self.max_depth = self._recompute_max_depth_from_arena()

        after_ranges = self._count_total_ranges()
        after_edges = self._count_edges()
        after_nodes = len(self.arena)

        t1 = time.perf_counter()
        print(" done.")
        # Concrete report
        print("Optimization report:")
        print(f"- Time: {t1 - t0:.3f}s")
        print(f"- Ranges: {before_ranges} -> {after_ranges} ({before_ranges - after_ranges} fewer)")
        print(f"- Edges:  {before_edges} -> {after_edges} ({before_edges - after_edges} fewer)")
        print(f"- Nodes:  {before_nodes} -> {after_nodes} ({before_nodes - after_nodes} fewer, {gc_removed} GC removed)")
        print(f"- Masks total: {mask_stats['dest_masks_total']}, made universal: {mask_stats['dest_masks_made_universal']}, "
              f"already universal: {mask_stats['dest_masks_already_universal']}, changed: {mask_stats['dest_masks_changed']}")
        print(f"- Contractions: {repl_stats['contractions']} "
              f"(rep1: {repl_stats['replacement_1']}, rep2: {repl_stats['replacement_2']}), "
              f"edges +{repl_stats['edges_added']}/-{repl_stats['edges_removed']}, "
              f"nodes removed: {repl_stats['nodes_removed']}")

    # =====================================================
    # Post-optimization: subtree merging and edge coalescing
    # =====================================================
    def _rs_key(self, rs: RangeSet) -> Tuple[Tuple[int, int], ...]:
        """
        Canonical key for a RangeSet: tuple of (start, end) int pairs.
        Robust against different RangeSet implementations.
        """
        try:
            return tuple((int(a), int(b)) for (a, b) in rs.to_ranges())
        except Exception:
            return ()

    def _normalize_all_nodes(self) -> Dict[str, int]:
        """
        For each node:
          - Coalesce duplicate dest entries to the same dest_id by unioning masks.
          - Merge parallel edge groups that share the same (pop, dests) by unioning their LLM masks.
          - Recompute node.llm_bv_union.
        Returns simple stats.
        """
        stats = {
            "groups_before": 0,
            "groups_after": 0,
            "edges_before": self._count_edges(),
            "edges_after": 0,
        }
        for node in self.arena.values():
            children = node.children
            stats["groups_before"] += len(children)
            if not children:
                node.llm_bv_union = RangeSet.empty()
                continue

            # Group by (pop, dest signature)
            # dest signature = sorted((dest_id, rs_key(state_bv))) pairs
            grouped: Dict[
                Tuple[int, Tuple[Tuple[int, Tuple[Tuple[int, int], ...]], ...]],
                Dict[str, Any]
            ] = {}

            for (pop, llm_bv), dests in children:
                # 1) Union duplicate dest entries per dest_id for this group
                dest_union: Dict[int, StateIDSet] = {}
                for dest_id, state_bv in dests:
                    di = int(dest_id)
                    if di in dest_union:
                        dest_union[di] = dest_union[di].union(state_bv)
                    else:
                        dest_union[di] = state_bv

                # 2) Canonical dest signature
                # Keep actual dest list sorted by dest_id for reconstruction
                dest_items_sorted = sorted(dest_union.items(), key=lambda x: int(x[0]))
                dest_sig = tuple(
                    (int(did), self._rs_key(sbv)) for did, sbv in dest_items_sorted
                )
                key = (int(pop), dest_sig)

                # 3) Accumulate by unioning LLM masks across parallel groups
                entry = grouped.get(key)
                if entry is None:
                    grouped[key] = {
                        "llm": llm_bv,
                        "dests": [(int(did), sbv) for did, sbv in dest_items_sorted],
                    }
                else:
                    grouped[key]["llm"] = entry["llm"].union(llm_bv)

            # 4) Rebuild children deterministically
            new_children: List[Tuple[Tuple[int, LLMTokenSet], List[Tuple[int, StateIDSet]]]] = []
            llm_union = RangeSet.empty()
            for (pop, _dest_sig), data in grouped.items():
                llm = data["llm"]
                if llm.is_empty():
                    # If the unioned LLM mask is empty, drop the group
                    continue
                llm_union = llm_union.union(llm)
                # Dests are already deduplicated and sorted
                new_children.append(((int(pop), llm), data["dests"]))

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

            node.children = new_children
            node.llm_bv_union = llm_union
            stats["groups_after"] += len(new_children)

        stats["edges_after"] = self._count_edges()
        return stats

    def _compute_node_signatures(self, max_rounds: int = 8) -> Dict[int, Tuple]:
        """
        Compute WL-like iterative signatures to detect equivalent subgraphs.
        Signature of a node is a tuple:
          (is_end, sorted list of (pop, llm_key, state_key, child_color))
        Iteratively refines child_color; stable when colors stop changing, or
        after max_rounds.
        Returns node_id -> signature tuple (last iteration).
        """
        if not self.arena:
            return {}
        node_ids = sorted(int(n) for n in self.arena.keys())

        # Initial coarse signatures using local structure without child colors
        init_sigs: Dict[int, Tuple] = {}
        sig_intern: Dict[Tuple, int] = {}
        next_id = 0
        for nid in node_ids:
            node = self.arena[nid]
            group_summaries = []
            for (pop, llm_bv), dests in node.children:
                state_keys = [self._rs_key(sbv) for (_d, sbv) in dests]
                state_keys.sort()
                group_summaries.append((int(pop), self._rs_key(llm_bv), tuple(state_keys), int(len(dests))))
            group_summaries.sort()
            sig0 = (bool(node.clean_end), tuple(group_summaries))
            init_sigs[nid] = sig0
            if sig0 not in sig_intern:
                sig_intern[sig0] = next_id
                next_id += 1
        colors: Dict[int, int] = {nid: sig_intern[init_sigs[nid]] for nid in node_ids}
        sig_by_node = init_sigs

        # Refinement
        for _round in range(max(1, int(max_rounds))):
            new_sig_by_node: Dict[int, Tuple] = {}
            interner: Dict[Tuple, int] = {}
            next_c = 0
            for nid in node_ids:
                node = self.arena[nid]
                edges = []
                for (pop, llm_bv), dests in node.children:
                    llm_key = self._rs_key(llm_bv)
                    for dest_id, state_bv in dests:
                        edges.append((int(pop), llm_key, self._rs_key(state_bv), int(colors.get(int(dest_id), -1))))
                edges.sort()
                sig = (bool(node.clean_end), tuple(edges))
                new_sig_by_node[nid] = sig
                if sig not in interner:
                    interner[sig] = next_c
                    next_c += 1
            new_colors = {nid: interner[new_sig_by_node[nid]] for nid in node_ids}
            if new_colors == colors:
                sig_by_node = new_sig_by_node
                break
            colors = new_colors
            sig_by_node = new_sig_by_node

        return sig_by_node

    def _merge_equivalent_subgraphs_and_edges(self) -> Dict[str, int]:
        """
        Merge nodes that are structurally equivalent (same signature) by:
          - Choosing a representative per signature (smallest node id).
          - Rewiring all edges and roots_map to representatives.
          - Normalizing all nodes (merge parallel edges).
          - GC unreachable nodes.
          - Recompute llm_bv_union and max_depth.
        Returns stats about the compaction.
        """
        stats = {
            "nodes_before": len(self.arena),
            "edges_before": self._count_edges(),
            "ranges_before": self._count_total_ranges(),
            "nodes_merged": 0,
            "edges_after_rewire": 0,
            "gc_removed": 0,
        }

        # Normalize first to maximize merge opportunities
        self._normalize_all_nodes()

        sig_by_node = self._compute_node_signatures(max_rounds=8)
        if not sig_by_node:
            stats["edges_after_rewire"] = self._count_edges()
            return stats

        # Group by signature and choose representatives
        sig_groups: Dict[Tuple, List[int]] = collections.defaultdict(list)
        for nid, sig in sig_by_node.items():
            sig_groups[sig].append(int(nid))

        rep_of: Dict[int, int] = {}
        merged_count = 0
        for _sig, ids in sig_groups.items():
            ids_sorted = sorted(int(x) for x in ids)
            rep = ids_sorted[0]
            for x in ids_sorted:
                rep_of[int(x)] = int(rep)
            merged_count += max(0, len(ids_sorted) - 1)
        stats["nodes_merged"] = merged_count

        if merged_count == 0:
            stats["edges_after_rewire"] = self._count_edges()
            return stats

        # Rewire roots to representatives
        new_roots_map: Dict[int, int] = {}
        for k, v in self.roots_map.items():
            vv = int(v)
            new_roots_map[int(k)] = int(rep_of.get(vv, vv))
        self.roots_map = new_roots_map

        # Rewire edges to representatives and coalesce duplicate dests
        for node in self.arena.values():
            children = node.children
            if not children:
                continue
            new_children: List[Tuple[Tuple[int, LLMTokenSet], List[Tuple[int, StateIDSet]]]] = []
            for (pop, llm_bv), dests in children:
                dest_union: Dict[int, StateIDSet] = {}
                for dest_id, state_bv in dests:
                    rid = int(rep_of.get(int(dest_id), int(dest_id)))
                    if rid in dest_union:
                        dest_union[rid] = dest_union[rid].union(state_bv)
                    else:
                        dest_union[rid] = state_bv
                # Add group with remapped/unioned destinations
                merged_dests = sorted(((int(d), bv) for d, bv in dest_union.items()), key=lambda x: int(x[0]))
                new_children.append(((int(pop), llm_bv), merged_dests))
            node.children = new_children

        # Normalize again (merge parallel edges) and recompute unions/depths
        self._normalize_all_nodes()
        stats["edges_after_rewire"] = self._count_edges()

        # GC unreachable nodes
        stats["gc_removed"] = self._gc_unreachable_nodes()
        self._recompute_llm_bv_unions()
        self.max_depth = self._recompute_max_depth_from_arena()

        return stats

    def _post_optimize_merge_subgraphs_and_edges(self) -> None:
        """
        Orchestrates additional compaction after _optimize_state_masks_and_edges:
          - Normalize nodes (dedupe/merge parallel edges)
          - Merge equivalent subgraphs via signatures
          - Report concrete improvements
        """
        print("Post-optimizing graph (merge equivalent subgraphs and edges)...", end="", flush=True)
        t0 = time.perf_counter()
        before_nodes = len(self.arena)
        before_edges = self._count_edges()
        before_ranges = self._count_total_ranges()

        compaction_stats = self._merge_equivalent_subgraphs_and_edges()

        after_nodes = len(self.arena)
        after_edges = self._count_edges()
        after_ranges = self._count_total_ranges()
        t1 = time.perf_counter()
        print(" done.")
        print("Post-optimization report:")
        print(f"- Time: {t1 - t0:.3f}s")
        print(f"- Ranges: {before_ranges} -> {after_ranges} ({before_ranges - after_ranges} fewer)")
        print(f"- Edges:  {before_edges} -> {after_edges} ({before_edges - after_edges} fewer)")
        print(f"- Nodes:  {before_nodes} -> {after_nodes} ({before_nodes - after_nodes} fewer, {compaction_stats.get('gc_removed', 0)} GC removed)")
        print(f"- Nodes merged (equiv signatures): {compaction_stats.get('nodes_merged', 0)}")
        print(f"- Normalization: groups before/after not tracked across stages here")