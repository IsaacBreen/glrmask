import json
import heapq
import collections
import time
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from gss_tester.leveled_impl import LeveledGSS as GSS


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


@dataclass(frozen=True)
class PyAcc:
    terminals_union: ffi.HybridL2Bitset

    def __hash__(self):
        return hash(self.terminals_union)

    def merge(self, other: "PyAcc") -> "PyAcc":
        return PyAcc(terminals_union=self.terminals_union.union(other.terminals_union))


def get_disallowed_terminals_py(gss: GSS) -> ffi.HybridL2Bitset:
    merged_acc = gss.reduce_acc()
    if merged_acc is None:
        return ffi.HybridL2Bitset.all()
    return merged_acc.terminals_union.complement()


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation), simplified and concise.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.id_to_token: Dict[int, bytes] = {}
        self.max_depth: Dict[int, int] = {}
        self.possible_matches_cache: Optional[Dict[int, Dict[int, ffi.Bitset]]] = None
        self.tokenizer: Optional[ffi.Regex] = None
        self.glr_parser: Optional[ffi.GLRParser] = None
        self.ignore_terminal_id: Optional[int] = None
        self.parser_table: Optional[ParserTable] = None
        self.state: Dict[int, GSS] = {}
        self.internal_to_original_map: Dict[int, int] = {}
        self.all_internal_llm_tokens_bitset: Optional[ffi.Bitset] = None
        self.tokenizer_initial_state: Optional[int] = None

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        # Normalize arena children bitsets and cache max_depth
        for uid, node in self.arena.items():
            uid_int = int(uid)
            self.max_depth[uid_int] = int(node.get("max_depth", 0) or 0)

            children = node.get("children") or []
            if not children:
                node["children"] = []
                continue

            new_children = []
            for edge_key, dest_map in children:
                pop, llm_bv_json = edge_key
                llm_bv = bs_from_json(dumps(llm_bv_json))
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv = bs_from_json(dumps(state_bv_json))
                    new_dest_map.append((int(dest_idx), state_bv))
                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children

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

        initial_acc = PyAcc(terminals_union=ffi.HybridL2Bitset.all())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        model.possible_matches_cache = constraint.possible_matches()
        model.internal_to_original_map = constraint.internal_to_original_map()
        model.all_internal_llm_tokens_bitset = constraint.all_internal_llm_tokens_bitset()
        return model

    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, ffi.Bitset]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            allowed_terminals_l2 = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                allowed_for_state = allowed_terminals_l2.get_l2_bitset(state_id)
                if not matched_bv.is_subset(allowed_for_state):
                    return False
            return True
        return gss.prune(predicate)

    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_l2 = acc.terminals_union
            new_bvs: Dict[int, ffi.Bitset] = collections.defaultdict(ffi.Bitset.zeros)
            for old_sid, new_sid in state_map.items():
                bv_source = old_l2.get_l2_bitset(old_sid)
                new_bvs[new_sid] = new_bvs[new_sid].union(bv_source)

            new_l2 = ffi.HybridL2Bitset.all()
            for new_sid, bv in new_bvs.items():
                new_l2.insert_l2_bitset(new_sid, bv)
            return PyAcc(terminals_union=new_l2)
        return gss.apply(apply_map)

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_l2 = acc.terminals_union
            new_l2 = current_l2.union(current_l2)  # clone
            curr_bv = current_l2.get_l2_bitset(state_id)
            if curr_bv.contains(terminal_id):
                to_remove = ffi.Bitset.from_indices([terminal_id])
                new_bv = curr_bv.difference(to_remove)
                new_l2.insert_l2_bitset(state_id, new_bv)
            return PyAcc(terminals_union=new_l2)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
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

    def commit(self, token_id: int):
        t0 = time.perf_counter()
        token_bytes = self.id_to_token[token_id]

        # Build tokenizer maps
        terminals_map: Dict[int, ffi.Bitset] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            terminals = ffi.Bitset.zeros()
            for terminal_id, _ in matches:
                terminals.insert(terminal_id)
            terminals_map[tokenizer_sid] = terminals

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
            # Deduplicate by semantic GSS equality instead of object identity to avoid exponential blow-up.
            q_item_key = (offset, tokenizer_sid, gss)
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
            sid: GSS.merge(gss_list)
            for sid, gss_list in new_states.items()
            if gss_list
        }
        merged_states = {sid: state for sid, state in merged_states.items() if not state.is_empty()}

        self.state = merged_states

        t1 = time.perf_counter()
        print(f"commit (ms): {round((t1 - t0) * 1000, 2)}")

    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        heads_by_state: Dict[int, List[GSS]] = collections.defaultdict(list)
        for state_id in gss.peek():
            heads_by_state[state_id].append(gss.isolate(state_id))

        shifted_gsses: List[GSS] = []

        while heads_by_state:
            state_id, state_gsss = heads_by_state.popitem()
            state_gss = GSS.merge(state_gsss)
            row = self.parser_table.table.get(state_id)
            if not row:
                continue
            action = row.actions.get(terminal_id)
            if not action:
                continue

            def handle_shift(shift_to_state_id, gss_to_shift):
                shifted_gsses.append(gss_to_shift.push(shift_to_state_id))

            def handle_reduce_grouped(length: int, nt_id: int, gss_to_reduce: GSS):
                # Cache popn by reduction length
                popped_gss = gss_to_reduce.popn(length)
                from_states: Set[int] = set(popped_gss.peek())
                if not from_states:
                    return
                # Group by goto state to reduce isolate() calls
                by_goto: Dict[int, Set[int]] = collections.defaultdict(set)
                for from_state_id in from_states:
                    goto_row = self.parser_table.table.get(from_state_id)
                    if not goto_row:
                        continue
                    goto_state_id = goto_row.gotos.get(nt_id)
                    if goto_state_id is None:
                        continue
                    by_goto[goto_state_id].add(from_state_id)
                for goto_state_id, from_group in by_goto.items():
                    iso_group = popped_gss.isolate_values(from_group)
                    if iso_group.is_empty():
                        continue
                    goto_gss = iso_group.push(goto_state_id)
                    heads_by_state[goto_state_id].append(goto_gss)

            if isinstance(action, int):
                handle_shift(action, state_gss)
            elif isinstance(action, Reduce):
                handle_reduce_grouped(action.len, action.nonterminal_id, state_gss)
            elif isinstance(action, Split):
                if action.shift is not None:
                    handle_shift(action.shift, state_gss)
                # For Split, reuse the same popped and from-states per length across all nonterminals
                for length, nts in action.reduces.items():
                    if not nts:
                        continue
                    for nt_id in nts.keys():
                        handle_reduce_grouped(length, nt_id, state_gss)

        return GSS(node_factory=gss._factory) if not shifted_gsses else GSS.merge(shifted_gsses)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.
        """
        state_map = self.state
        all_ones_mask = self.all_internal_llm_tokens_bitset
        final_mask = ffi.Bitset.zeros()

        values: Dict[int, Tuple[GSS, ffi.Bitset]] = {}
        stopped: set[int] = set()
        todo: Dict[int, set[int]] = {}
        depth_heap: List[int] = []

        heappush = heapq.heappush
        heappop = heapq.heappop
        roots_map = self.roots_map
        max_depth = self.max_depth
        arena = self.arena
        is_end = self.is_end

        # Seed
        for sid, gss in state_map.items():
            new_mask = all_ones_mask
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            existing = values.get(root_idx)
            if existing is not None:
                existing_gss, existing_mask = existing
                merged_gss = GSS.merge([existing_gss, gss])
                values[root_idx] = (merged_gss, existing_mask.union(new_mask))
            else:
                values[root_idx] = (gss, new_mask)

            depth = max_depth[root_idx]
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {root_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(root_idx)

        def enqueue(depth: int, node_idx: int) -> None:
            bucket = todo.get(depth)
            if bucket is None:
                todo[depth] = {node_idx}
                heappush(depth_heap, depth)
            else:
                bucket.add(node_idx)

        # Main loop
        while True:
            node_indices: Optional[set[int]] = None
            current_depth = -1
            while depth_heap:
                current_depth = heappop(depth_heap)
                node_indices = todo.pop(current_depth, None)
                if node_indices:
                    break
            if not node_indices:
                break

            for node_idx in node_indices:
                if node_idx in stopped:
                    continue

                item = values.pop(node_idx, None)
                if item is None:
                    continue
                gss_node, llm_mask = item

                # End-node handling
                if is_end(node_idx):
                    forbidden_llm_tokens = ffi.Bitset.zeros()
                    disallowed_terminals_l2 = get_disallowed_terminals_py(gss_node)
                    possible_matches = self.possible_matches_cache

                    for (start, end), disallowed_bv in disallowed_terminals_l2.range_values():
                        if disallowed_bv.is_empty():
                            continue
                        end = min(end, self.tokenizer.max_state())
                        for tsid in range(start, end + 1):
                            possible_matches_for_state = possible_matches.get(tsid)
                            if not possible_matches_for_state:
                                continue
                            for terminal_id_str, llm_tokens_for_terminal in possible_matches_for_state.items():
                                terminal_id = int(terminal_id_str)
                                if disallowed_bv.contains(terminal_id):
                                    forbidden_llm_tokens = forbidden_llm_tokens.union(llm_tokens_for_terminal)

                    gss_active_tokens = all_ones_mask
                    glr_active_tokens = llm_mask.intersection(gss_active_tokens)
                    final_allowed_tokens = glr_active_tokens.difference(forbidden_llm_tokens)
                    if not final_allowed_tokens.is_empty():
                        final_mask = final_mask.union(final_allowed_tokens)

                if llm_mask.is_empty():
                    stopped.add(node_idx)
                    continue

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                # Cache popn per 'pop' value for this node to avoid recomputation
                _popped_cache: Dict[int, GSS] = {}
                for (pop, llm_bv), dests in children:
                    popped = _popped_cache.get(pop)
                    if popped is None:
                        popped = gss_node.popn(pop)
                        _popped_cache[pop] = popped
                    llm_empty = llm_bv.is_empty()

                    for dest_idx, state_bv in dests:
                        if state_bv.is_empty():
                            continue
                        # Filter in a single pass using allowed states
                        allowed_states: Set[int] = set(state_bv.to_indices())
                        child_gss_node = popped.isolate_values(allowed_states)
                        if child_gss_node.is_empty():
                            continue
                        child_llm_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)
                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            existing_gss, existing_mask = existing
                            merged_gss = GSS.merge([existing_gss, child_gss_node])
                            combined_mask = existing_mask.union(child_llm_mask)
                            values[d] = (merged_gss, combined_mask)
                        else:
                            values[d] = (child_gss_node, child_llm_mask)

                        enqueue(max_depth[d], d)

        # Convert internal mask back to original IDs
        original_mask = ffi.Bitset.zeros()
        for internal_id in final_mask.to_indices():
            if internal_id in self.internal_to_original_map:
                original_mask.insert(self.internal_to_original_map[internal_id])
        return RangeSet.from_ranges(original_mask.to_ranges())
