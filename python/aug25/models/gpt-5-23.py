"""
Ultra-optimized get_mask implementation.

Key ideas:
- Drastically reduce GSS merges by accumulating contributions as lists and merging only
  when a node is dequeued (lazy merging per node).
- Group edges by pop value per node; compute GSS.popn(pop) once per pop group per node.
- For each (pop, llm_bv) edge group, pre-apply llm_bv to the popped GSS once using
  apply_and_prune (rather than per-destination); then only isolate_many for each dest.
- Avoid repeated peek computations and repeated pop calls.
- No Stats/metrics collection overhead.

The rest of the model (commit, tokenizer integration, parser table handling) follows
the precompute3 Python model to remain compatible with the surrounding system.
"""

import json
import heapq
import collections
from typing import Dict, List, Tuple, Optional, Union, Set, DefaultDict
from dataclasses import dataclass, field

try:
    # Local package layout
    from .common_interface import GraphProvider, RangeSet
    import _sep1 as ffi
    from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
except Exception:
    # Fallback: typical workspace layout
    from python.aug25.common_interface import GraphProvider, RangeSet
    import _sep1 as ffi
    from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS


# Add a dummy profiler for when not running under kernprof (kept as no-op to preserve decorators)
try:
    profile  # type: ignore
except NameError:
    def profile(func):  # type: ignore
        return func


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
    gotos: Dict[int, int] = field(default_factory=dict)       # nonterminal_id -> state_id


@dataclass
class ParserTable:
    start_state_id: int
    table: Dict[int, Row]


@dataclass(frozen=True, eq=False)
class PyAcc:
    terminals_union: Dict[int, RangeSet]
    llm_mask: RangeSet

    def __eq__(self, other):
        if not isinstance(other, PyAcc):
            return NotImplemented
        return self.llm_mask == other.llm_mask and self.terminals_union == other.terminals_union

    def __hash__(self):
        # Frozen; combine hash of terminals_union size and llm_mask
        return hash((len(self.terminals_union), self.llm_mask))

    def merge(self, other: "PyAcc") -> "PyAcc":
        # merge disallowed terminals per tokenizer state
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


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation), optimized get_mask.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.id_to_token: Dict[int, bytes] = {}
        self.max_depth: Dict[int, int] = {}
        self.possible_matches_cache: Optional[Dict[int, Dict[int, RangeSet]]] = None
        self.tokenizer: Optional[ffi.Regex] = None
        self.glr_parser: Optional[ffi.GLRParser] = None
        self.ignore_terminal_id: Optional[int] = None
        self.parser_table: Optional[ParserTable] = None
        self.state: Dict[int, GSS] = {}
        self.internal_to_original_map: Dict[int, int] = {}
        self.all_internal_llm_tokens_bitset: Optional[RangeSet] = None
        self.tokenizer_initial_state: Optional[int] = None
        self.tokenizer_max_state: Optional[int] = None
        self.all_terminals_bitset: Optional[RangeSet] = None

        # Per-node: pre-group edges by pop to minimize popn calls and repeated peek
        # edges_by_pop: node_id -> Dict[pop, List[Tuple[RangeSet(llm_bv), List[Tuple[dest_idx, RangeSet(state_bv)]]]]]
        self.edges_by_pop: Dict[int, Dict[int, List[Tuple[RangeSet, List[Tuple[int, RangeSet]]]]]] = {}

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # Normalize arena children bitsets and cache max_depth; build edges_by_pop
        for uid, node in self.arena.items():
            uid_int = int(uid)
            self.max_depth[uid_int] = int(node.get("max_depth", 0) or 0)

            children = node.get("children") or []
            if not children:
                node["children"] = []
                self.edges_by_pop[uid_int] = {}
                continue

            new_children = []
            pop_groups: DefaultDict[int, List[Tuple[RangeSet, List[Tuple[int, RangeSet]]]]] = collections.defaultdict(list)

            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv_bitset = bs_from_json(dumps(llm_bv_json))
                llm_bv = RangeSet.from_ranges(llm_bv_bitset.to_ranges())

                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv_bitset = bs_from_json(dumps(state_bv_json))
                    state_bv = RangeSet.from_ranges(state_bv_bitset.to_ranges())
                    new_dest_map.append((int(dest_idx), state_bv))
                new_children.append(((int(pop), llm_bv), new_dest_map))

                # Populate grouping for optimized traversal
                pop_groups[int(pop)].append((llm_bv, new_dest_map))

            node["children"] = new_children
            self.edges_by_pop[uid_int] = dict(pop_groups)

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

        all_terminals = set()
        for row in model.parser_table.table.values():
            all_terminals.update(row.actions.keys())
        if model.ignore_terminal_id is not None:
            all_terminals.add(model.ignore_terminal_id)
        model.all_terminals_bitset = RangeSet.from_indices(list(all_terminals))

        initial_acc = PyAcc(terminals_union={}, llm_mask=RangeSet.empty())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}

        # Convert possible_matches_cache to RangeSet
        pmc_ffi: Dict[int, Dict[int, ffi.Bitset]] = constraint.possible_matches()
        pmc_rs: Dict[int, Dict[int, RangeSet]] = {}
        for tsid, inner in pmc_ffi.items():
            mapped: Dict[int, RangeSet] = {}
            for term_id, bit in inner.items():
                mapped[int(term_id)] = RangeSet.from_ranges(bit.to_ranges())
            pmc_rs[int(tsid)] = mapped
        model.possible_matches_cache = pmc_rs

        model.internal_to_original_map = constraint.internal_to_original_map()

        # Universe of internal LLM tokens as RangeSet
        all_internal = constraint.all_internal_llm_tokens_bitset()
        model.all_internal_llm_tokens_bitset = RangeSet.from_ranges(all_internal.to_ranges())
        return model

    @profile
    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, RangeSet]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            disallowed_terminals_map = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(state_id, RangeSet.empty())
                if not matched_bv.intersection(disallowed_for_state).is_empty():
                    return False
            return True
        return gss.prune(predicate)

    @profile
    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_map = acc.terminals_union
            new_bvs: Dict[int, RangeSet] = collections.defaultdict(RangeSet.empty)
            for old_sid, new_sid in state_map.items():
                bv_source = old_map.get(old_sid, RangeSet.empty())
                new_bvs[new_sid] = new_bvs[new_sid].union(bv_source)

            return PyAcc(terminals_union=dict(new_bvs), llm_mask=acc.llm_mask)
        return gss.apply(apply_map)

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

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
        """
        Explode packed transitions into (pop, state_id or None, dest_idx).
        Unused in optimized path, but left for compatibility/testing.
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
        # Same as the pure Python version (commit is already fast enough)
        token_bytes = self.id_to_token[token_id]

        # Build tokenizer maps
        terminals_map: Dict[int, RangeSet] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            matched_terminals = [terminal_id for terminal_id, _ in matches]
            terminals_map[tokenizer_sid] = RangeSet.from_indices(matched_terminals)

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

    def get_mask(self) -> RangeSet:
        """
        Optimized traversal to compute final LLM token mask:
        - Initialize per-accumulator allowed LLM masks once (seed).
        - For each node:
          * Group edges by 'pop' and pop the GSS only once per group.
          * For each (pop, llm_bv) group, pre-apply llm_bv to the popped GSS once,
            then only filter by state_bv and isolate for each destination.
          * Accumulate child GSS contributions as lists; defer merges until the child
            node is actually processed. This slashes merge overhead.
        """
        state_map: Dict[int, GSS] = self.state
        if not state_map:
            return RangeSet.empty()

        all_ones: Optional[RangeSet] = self.all_internal_llm_tokens_bitset
        final_mask: RangeSet = RangeSet.empty()

        # values maps node_id -> list of GSS contributions (lazy-merge)
        values: Dict[int, List[GSS]] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        is_end = self.is_end
        pmc: Dict[int, Dict[int, RangeSet]] = self.possible_matches_cache or {}
        max_state: int = int(self.tokenizer_max_state or 0)

        # Seed: Initialize llm_mask in each GSS, consume terminals_union, enqueue roots.
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask = RangeSet.empty()
            disallowed_map = acc.terminals_union
            for tsid, disallowed_terminals in disallowed_map.items():
                # Skip out-of-range tokenizer states or missing cache
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]
                # Gather individual disallowed terminals; typically small per accumulator
                for terminal_id in disallowed_terminals.to_indices():
                    bit = terminals_to_llm.get(terminal_id)
                    if bit is not None:
                        disallowed_llm_mask = disallowed_llm_mask.union(bit)

            allowed_mask = (all_ones if all_ones is not None else RangeSet.empty()).difference(disallowed_llm_mask)
            return PyAcc(
                terminals_union={},  # consume after initialization
                llm_mask=allowed_mask,
            )

        apply_memo: Dict[PyAcc, PyAcc] = {}
        for sid, gss in state_map.items():
            r: int = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)
            if r in values:
                values[r].append(gss_initialized)
            else:
                values[r] = [gss_initialized]

            d: int = max_depth[r]
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {r}
                hp(depth_heap, d)
            else:
                bucket.add(r)

        def enqueue(d: int, n: int) -> None:
            bucket: Optional[Set[int]] = todo.get(d)
            if bucket is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                bucket.add(n)

        # Main loop over nodes by increasing max_depth
        while depth_heap:
            depth: int = hpop(depth_heap)
            while todo[depth]:
                node: int = todo[depth].pop()

                # Merge all contributions to this node only once here (lazy merge)
                gss_list = values.pop(node, [])
                if not gss_list:
                    continue
                gss_node: GSS = gss_list[0] if len(gss_list) == 1 else GSS.merge_many(gss_list)
                if gss_node.is_empty():
                    continue

                # End-node handling: just union the allowed LLM tokens
                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
                        final_mask = final_mask.union(reduced_acc.llm_mask)

                # Traverse edges grouped by 'pop'
                edges_by_pop = self.edges_by_pop.get(node)
                if not edges_by_pop:
                    continue

                # Accumulate child contributions from this node, then enqueue children
                child_contribs: DefaultDict[int, List[GSS]] = collections.defaultdict(list)

                for pop, groups in edges_by_pop.items():
                    # pop the GSS once per pop group
                    popped: GSS = gss_node.popn(pop)
                    if popped.is_empty():
                        continue

                    for llm_bv, dests in groups:
                        # Pre-apply llm_bv once for this (pop, llm_bv) group
                        acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}

                        def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                            if acc in acc_memo:
                                return acc_memo[acc]
                            new_mask = acc.llm_mask.intersection(llm_bv)
                            if new_mask.is_empty():
                                result = None
                            else:
                                result = PyAcc(terminals_union=acc.terminals_union, llm_mask=new_mask)
                            acc_memo[acc] = result
                            return result

                        popped_pruned: GSS = popped.apply_and_prune(intersect_and_prune)
                        if popped_pruned.is_empty():
                            continue

                        # Peek once after pruning; convert to a RangeSet for fast filtering by bitset intersection
                        peeked_states = popped_pruned.peek()
                        if not peeked_states:
                            continue
                        peeked_rs = RangeSet.from_indices(list(peeked_states))

                        # For each destination, filter peeked states by state_bv and isolate once
                        for dest_idx, state_bv in dests:
                            # Fast filter using RangeSet intersection
                            inter = state_bv.intersection(peeked_rs)
                            if inter.is_empty():
                                continue
                            keep_states = inter.to_indices()
                            if not keep_states:
                                continue

                            child_gss: GSS = popped_pruned.isolate_many(keep_states)
                            if child_gss.is_empty():
                                continue
                            child_contribs[int(dest_idx)].append(child_gss)

                # Commit child contributions
                for dest_idx, lst in child_contribs.items():
                    # Do not merge here; defer to when the child node is processed
                    if lst:
                        if dest_idx in values:
                            values[dest_idx].extend(lst)
                        else:
                            values[dest_idx] = lst
                        enqueue(self.max_depth[dest_idx], dest_idx)

            # bucket exhausted
            todo.pop(depth, None)

        # Convert internal mask back to original IDs
        # Note: if internal_to_original_map is identity, this remains a no-op mapping.
        original_indices: List[int] = []
        for i in final_mask.to_indices():
            mapped = self.internal_to_original_map.get(i)
            if mapped is not None:
                original_indices.append(mapped)

        return RangeSet.from_indices(original_indices)
