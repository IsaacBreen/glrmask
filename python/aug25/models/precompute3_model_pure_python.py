import json
import heapq
import collections
import time
from typing import Dict, List, Tuple, Optional, Union, Iterable
from dataclasses import dataclass, field

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
import portion as P
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS


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


@dataclass(frozen=True)
class PyAcc:
    terminals_union: Tuple[Tuple[int, P.Interval], ...]
    llm_mask: P.Interval

    def merge(self, other: "PyAcc") -> "PyAcc":
        d1 = dict(self.terminals_union)
        d2 = dict(other.terminals_union)
        new_terminals_union: Dict[int, P.Interval] = d1.copy()
        for k, v in d2.items():
            if k in new_terminals_union:
                new_terminals_union[k] = new_terminals_union[k] | v
            else:
                new_terminals_union[k] = v

        return PyAcc(
            terminals_union=tuple(sorted(new_terminals_union.items())),
            llm_mask=self.llm_mask | other.llm_mask,
        )


def bitset_to_interval(bv: ffi.Bitset) -> P.Interval:
    """
    Convert an ffi.Bitset into a portion interval (as a union of closed integer intervals).
    """
    iv = P.empty()
    for start, end in bv.to_ranges():
        iv = iv | P.closed(int(start), int(end))
    return iv


def ints_to_interval(indices: Iterable[int]) -> P.Interval:
    iv = P.empty()
    for i in indices:
        iv = iv | P.singleton(int(i))
    return iv


def interval_to_int_ranges(iv: P.Interval) -> List[Tuple[int, int]]:
    """
    Convert a portion interval (union of intervals) into a list of closed integer ranges (start, end).
    Open endpoints are adjusted to integers (e.g., (1,3) -> [2,2]).
    """
    ranges: List[Tuple[int, int]] = []
    for atom in iv:
        # Determine integer inclusive lower/upper
        lower = atom.lower
        upper = atom.upper
        if atom.left is P.OPEN:
            lower = lower + 1
        if atom.right is P.OPEN:
            upper = upper - 1
        if lower <= upper:
            ranges.append((int(lower), int(upper)))
    return ranges


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
        self.all_internal_llm_tokens_bitset: Optional[P.Interval] = None
        self.tokenizer_initial_state: Optional[int] = None
        self.all_terminals_bitset: Optional[P.Interval] = None

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
                # Convert to portion interval
                llm_bv = bitset_to_interval(bs_from_json(dumps(llm_bv_json)))
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    # Convert to portion interval
                    state_bv = bitset_to_interval(bs_from_json(dumps(state_bv_json)))
                    new_dest_map.append((int(dest_idx), state_bv))
                # Store llm_bv and state_bv as portion intervals
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

        all_terminals = set()
        for row in model.parser_table.table.values():
            all_terminals.update(row.actions.keys())
        if model.ignore_terminal_id is not None:
            all_terminals.add(model.ignore_terminal_id)
        model.all_terminals_bitset = ints_to_interval(list(all_terminals))

        initial_acc = PyAcc(terminals_union=tuple(), llm_mask=P.empty())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        # Convert possible_matches_cache to use portion intervals
        pmc_raw: Dict[int, Dict[int, ffi.Bitset]] = constraint.possible_matches()
        pmc_conv: Dict[int, Dict[int, P.Interval]] = {}
        for tsid, inner in pmc_raw.items():
            tsid_int = int(tsid)
            pm_conv: Dict[int, P.Interval] = {}
            for term_id, bitset in inner.items():
                pm_conv[int(term_id)] = bitset_to_interval(bitset)
            pmc_conv[tsid_int] = pm_conv
        model.possible_matches_cache = pmc_conv
        model.internal_to_original_map = constraint.internal_to_original_map()
        model.all_internal_llm_tokens_bitset = bitset_to_interval(constraint.all_internal_llm_tokens_bitset())
        return model

    @profile
    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, P.Interval]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            disallowed_terminals_map = dict(acc.terminals_union)
            for state_id, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(state_id, P.empty())
                if not (matched_bv & disallowed_for_state).empty:
                    return False
            return True
        return gss.prune(predicate)

    @profile
    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_map = dict(acc.terminals_union)
            new_bvs: Dict[int, P.Interval] = {}
            for old_sid, new_sid in state_map.items():
                bv_source = old_map.get(old_sid, P.empty())
                if new_sid in new_bvs:
                    new_bvs[new_sid] = new_bvs[new_sid] | bv_source
                else:
                    new_bvs[new_sid] = bv_source

            new_map_tuple = tuple(sorted(new_bvs.items()))
            return PyAcc(terminals_union=new_map_tuple, llm_mask=acc.llm_mask)
        return gss.apply(apply_map)

    @profile
    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_map = dict(acc.terminals_union)
            curr_iv = current_map.get(state_id, P.empty())
            current_map[state_id] = curr_iv | P.singleton(int(terminal_id))
            new_map_tuple = tuple(sorted(current_map.items()))
            return PyAcc(terminals_union=new_map_tuple, llm_mask=acc.llm_mask)
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
        for (pop, llm_iv), dests in children:
            if token in llm_iv:
                for dest_idx, state_iv in dests:
                    if state_iv.empty:
                        yield (int(pop), None, int(dest_idx))
                    else:
                        # Iterate over integer states covered by the interval
                        for atom in state_iv:
                            lower = atom.lower
                            upper = atom.upper
                            if atom.left is P.OPEN:
                                lower = lower + 1
                            if atom.right is P.OPEN:
                                upper = upper - 1
                            if lower <= upper:
                                for sid in range(int(lower), int(upper) + 1):
                                    yield (int(pop), sid, int(dest_idx))

    @profile
    def commit(self, token_id: int):
        t0 = time.perf_counter()
        token_bytes = self.id_to_token[token_id]

        # Build tokenizer maps
        terminals_map: Dict[int, P.Interval] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            terminals = P.empty()
            for terminal_id, _ in matches:
                terminals = terminals | P.singleton(int(terminal_id))
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
        print(f"commit (ms): {round((t1 - t0) * 1000, 2)}")

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

    @profile
    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Changes:
        - Initialize LLM mask per-accumulator (PyAcc.llm_mask) BEFORE traversal by computing
          the forbidden terminals -> forbidden LLM tokens and taking the complement.
        - Consume terminals_union (set to HybridL2Bitset.all()) after initialization.
        - As we traverse edges, intersect llm_mask with the edge's LLM bitset using apply.
        - At end nodes, simply reduce acc over the GSS and union the llm_mask into the final.
        """
        state_map = self.state
        all_ones_mask: P.Interval = self.all_internal_llm_tokens_bitset or P.empty()
        final_mask: P.Interval = P.empty()

        values: Dict[int, GSS] = {}
        stopped: set[int] = set()
        todo: Dict[int, set[int]] = {}
        depth_heap: List[int] = []

        heappush = heapq.heappush
        heappop = heapq.heappop
        roots_map = self.roots_map
        max_depth = self.max_depth
        arena = self.arena
        is_end = self.is_end

        pmc: Dict[int, Dict[int, P.Interval]] = self.possible_matches_cache or {}
        max_state = self.tokenizer.max_state()

        # Seed: Initialize llm_mask in each GSS, consume terminals union, and enqueue roots.
        for sid, gss in state_map.items():
            # Set initial llm_mask on each accumulator and consume terminals_union
            def initialize_acc(acc: PyAcc) -> PyAcc:
                # Compute allowed LLM tokens from disallowed terminals
                disallowed_llm_mask: P.Interval = P.empty()
                disallowed_map = dict(acc.terminals_union)
                if disallowed_map:
                    for tsid, disallowed_terminals_iv in disallowed_map.items():
                        if tsid > max_state or tsid not in pmc:
                            continue
                        terminals_to_llm: Dict[int, P.Interval] = pmc[tsid]
                        # Iterate integer terminals contained in the interval
                        for atom in disallowed_terminals_iv:
                            low, up = atom.lower, atom.upper
                            if atom.left is P.OPEN: low += 1
                            if atom.right is P.OPEN: up -= 1
                            for terminal_id in range(int(low), int(up) + 1):
                                if terminal_id in terminals_to_llm:
                                    disallowed_llm_mask = disallowed_llm_mask | terminals_to_llm[terminal_id]

                allowed_mask = all_ones_mask - disallowed_llm_mask

                return PyAcc(
                    terminals_union=tuple(),  # consume
                    llm_mask=allowed_mask,
                )

            gss_initialized = gss.apply(initialize_acc)

            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)

            existing = values.get(root_idx)
            if existing is not None:
                merged_gss = existing.merge(gss_initialized)
                values[root_idx] = merged_gss
            else:
                values[root_idx] = gss_initialized

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

                gss_node = values.pop(node_idx, None)
                if gss_node is None:
                    continue

                # Compute current allowed LLM tokens at this node by reducing acc
                reduced_acc = gss_node.reduce_acc()
                if reduced_acc is None or reduced_acc.llm_mask.empty:
                    # No possible tokens here -> stop this branch
                    stopped.add(node_idx)
                    continue

                # End-node handling: just union the allowed LLM tokens
                if is_end(node_idx):
                    final_mask = final_mask.union(reduced_acc.llm_mask)

                # Transitions grouped by (pop, llm_bv)
                node_data = arena.get(node_idx, {})
                children = node_data.get("children") or []
                for (pop, llm_iv), dests in children:
                    popped = gss_node.popn(pop)
                    llm_empty = llm_iv.empty

                    for dest_idx, state_bv in dests:
                        matched: List[GSS] = []
                        if not state_bv.empty:
                            for sid_val in popped.peek():
                                if sid_val in state_bv:
                                    matched.append(popped.isolate(sid_val))
                        if not matched:
                            continue

                        child_gss_node = GSS.merge_many(matched)

                        # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv
                        if not llm_empty:
                            def intersect_edge(acc: PyAcc) -> PyAcc:
                                return PyAcc(
                                    terminals_union=acc.terminals_union,
                                    llm_mask=acc.llm_mask & llm_iv
                                )
                            child_gss_node = child_gss_node.apply(intersect_edge)

                        d = int(dest_idx)
                        existing_child = values.get(d)
                        if existing_child is not None:
                            merged_gss = existing_child.merge(child_gss_node)
                            values[d] = merged_gss
                        else:
                            values[d] = child_gss_node

                        enqueue(max_depth[d], d)

        # Convert internal mask back to original IDs and then to RangeSet
        original_iv: P.Interval = P.empty()
        for atom in final_mask:
            low, up = atom.lower, atom.upper
            if atom.left is P.OPEN: low += 1
            if atom.right is P.OPEN: up -= 1
            if low <= up:
                for internal_id in range(int(low), int(up) + 1):
                    if internal_id in self.internal_to_original_map:
                        original_iv = original_iv | P.singleton(self.internal_to_original_map[internal_id])
        return RangeSet.from_ranges(interval_to_int_ranges(original_iv))

