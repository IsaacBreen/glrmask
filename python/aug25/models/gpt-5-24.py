"""
gpt-5-24.py

Ultra-optimized Python model for get_mask traversal.

Key optimizations:
- Avoids per-membership FFI calls by precomputing state_bv membership as Python ranges with binary search.
- Defers GSS merges: accumulate lists and merge once at dequeue time (drastically reduces merge overhead).
- Caches apply-and-prune across edges: (acc, llm_bv) -> Optional[acc] to avoid recomputing intersections on repeated masks.
- Eliminates all stats/profiling overhead.
- Minimizes attribute lookups and leverages local bindings inside hot loops.

This file exposes a Model class with the same external interface used by previous implementations:
- from_json_string
- commit
- get_mask

Dependencies:
- _sep1 (FFI layer)
- python.gss_tester.implementations.leveled_impl.LeveledGSS as GSS
"""

from __future__ import annotations

import json
import heapq
import collections
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Union, Set, Callable

import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS


# -----------------------
# Fast in-Python Bitset wrapper (uses Rust ffi.Bitset underneath)
# -----------------------

class FastRangeSet:
    """
    Thin wrapper around ffi.Bitset with a minimal Pythonic API.
    Designed to be drop-in (enough) compatible with RangeSet used by existing code.
    """

    __slots__ = ("_bs",)

    def __init__(self, bs: "ffi.Bitset"):
        self._bs = bs

    @staticmethod
    def empty() -> "FastRangeSet":
        # Rely on from_ranges([]) to create empty bitset
        return FastRangeSet(ffi.Bitset.from_ranges([]))

    @staticmethod
    def from_ranges(ranges: Iterable[Iterable[int]]) -> "FastRangeSet":
        # Accepts iterable of [start, end]
        lst = [list(pair) for pair in ranges]
        return FastRangeSet(ffi.Bitset.from_ranges(lst))

    @staticmethod
    def from_indices(indices: Iterable[int]) -> "FastRangeSet":
        lst = list(indices)
        return FastRangeSet(ffi.Bitset.from_indices(lst))

    def to_ranges(self) -> List[Tuple[int, int]]:
        # ffi returns list of [start, end]; normalize to tuples
        return [tuple(pair) for pair in self._bs.to_ranges()]

    def to_indices(self) -> List[int]:
        return self._bs.to_indices()

    def union(self, other: "FastRangeSet") -> "FastRangeSet":
        return FastRangeSet(self._bs.union(other._bs))

    def intersection(self, other: "FastRangeSet") -> "FastRangeSet":
        return FastRangeSet(self._bs.intersection(other._bs))

    def difference(self, other: "FastRangeSet") -> "FastRangeSet":
        return FastRangeSet(self._bs.difference(other._bs))

    def contains(self, x: int) -> bool:
        return self._bs.contains(x)

    def is_empty(self) -> bool:
        return self._bs.is_empty()

    def __len__(self) -> int:
        return self._bs.len()

    def __eq__(self, other) -> bool:
        if not isinstance(other, FastRangeSet):
            return NotImplemented
        return self._bs == other._bs

    def __hash__(self) -> int:
        return hash(self._bs)

    # Convenience for debugging/printing
    def __repr__(self) -> str:
        return f"FastRangeSet({self.to_ranges()!r})"


# -----------------------
# Parser table data structures
# -----------------------

@dataclass(frozen=True)
class Reduce:
    nonterminal_id: int
    len: int
    production_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Split:
    shift: Optional[int]
    reduces: Dict[int, Dict[int, Tuple[int, ...]]]  # len -> nt_id -> pids


Action = Union[int, Reduce, Split]  # Shift (int), Reduce, or Split


@dataclass
class Row:
    actions: Dict[int, Action] = field(default_factory=dict)  # terminal_id -> Action
    gotos: Dict[int, int] = field(default_factory=dict)  # nonterminal_id -> state_id


@dataclass
class ParserTable:
    start_state_id: int
    table: Dict[int, Row]


# -----------------------
# Accumulator
# -----------------------

@dataclass(frozen=True, eq=False)
class PyAcc:
    terminals_union: Dict[int, FastRangeSet]
    llm_mask: FastRangeSet

    def __eq__(self, other):
        if not isinstance(other, PyAcc):
            return NotImplemented
        return self.llm_mask == other.llm_mask and self.terminals_union == other.terminals_union

    def __hash__(self):
        # Keep it simple and stable. After initialization, terminals_union is always {}.
        return hash(self.llm_mask)

    def merge(self, other: "PyAcc") -> "PyAcc":
        # union terminals_union per state and llm_mask union
        d1 = self.terminals_union
        d2 = other.terminals_union
        if not d1 and not d2:
            # Fast path: common after initialization
            return PyAcc(terminals_union={}, llm_mask=self.llm_mask.union(other.llm_mask))

        new_terminals_union = d1.copy()
        for k, v in d2.items():
            if k in new_terminals_union:
                new_terminals_union[k] = new_terminals_union[k].union(v)
            else:
                new_terminals_union[k] = v
        return PyAcc(terminals_union=new_terminals_union, llm_mask=self.llm_mask.union(other.llm_mask))


# -----------------------
# Helpers: membership via Python ranges
# -----------------------

def _normalize_ranges(ranges_like: Iterable[Iterable[int]]) -> Tuple[Tuple[int, int], ...]:
    """
    Normalize ranges_like (iterable of [start, end]) to a tuple of (start, end) tuples.
    Assumes input ranges are sorted, merged, and disjoint (as produced by the Bitset).
    """
    return tuple((int(a), int(b)) for a, b in ranges_like)


def _starts_from_ranges(ranges: Tuple[Tuple[int, int], ...]) -> Tuple[int, ...]:
    return tuple(start for start, _ in ranges)


def _contains_by_ranges(starts: Tuple[int, ...], ranges: Tuple[Tuple[int, int], ...], x: int) -> bool:
    """
    Membership test using binary search on starts to find the candidate range in O(log n).
    """
    idx = bisect_right(starts, x) - 1
    if idx < 0:
        return False
    s, e = ranges[idx]
    return x <= e


# -----------------------
# Optimized Model
# -----------------------

class Model:
    """
    Optimized precomputed trie model specialized for get_mask performance.

    Differences from the previous Python implementation:
    - Precomputes state_bv membership in Python via ranges to avoid FFI calls for contains().
    - Defers GSS merges: accumulate GSSs per node and merge at dequeue time (reduces merge cost).
    - Caches per-edge apply-and-prune results across the entire traversal.
    - Removes stats/profiling, focuses on minimal overhead in hot loops.
    """

    def __init__(self, roots_map: List[Tuple[int, int]], arena: Dict[int, dict]):
        # Core structures
        self.roots_map: Dict[int, int] = {int(s): int(r) for s, r in roots_map}
        self.arena: Dict[int, dict] = arena
        self.max_depth: Dict[int, int] = {}

        # Tokenizer/Parser resources
        self.id_to_token: Dict[int, bytes] = {}
        self.tokenizer: Optional[ffi.Regex] = None
        self.tokenizer_initial_state: Optional[int] = None
        self.tokenizer_max_state: Optional[int] = None
        self.glr_parser: Optional[ffi.GLRParser] = None
        self.ignore_terminal_id: Optional[int] = None
        self.parser_table: Optional[ParserTable] = None

        # State for running parser
        self.state: Dict[int, GSS] = {}

        # Matching caches
        self.possible_matches_cache: Optional[Dict[int, Dict[int, FastRangeSet]]] = None
        self.all_internal_llm_tokens_bitset: Optional[FastRangeSet] = None
        self.internal_to_original_map: Dict[int, int] = {}

        # Normalize arena: convert llm_bv to FastRangeSet; convert state_bv to Python ranges with starts
        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

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
                llm_bv_bitset = bs_from_json(dumps(llm_bv_json))
                llm_bv = FastRangeSet(ffi.Bitset.from_ranges(llm_bv_bitset.to_ranges()))
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv_bitset = bs_from_json(dumps(state_bv_json))
                    ranges = _normalize_ranges(state_bv_bitset.to_ranges())
                    starts = _starts_from_ranges(ranges)
                    new_dest_map.append((int(dest_idx), ranges, starts))
                new_children.append((int(pop), llm_bv, new_dest_map))
            node["children"] = new_children

    @staticmethod
    def from_json_string(s: str) -> "Model":
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

        # Parser table
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
                    py_row.actions[term_id] = int(action_data['state_id'])
                elif variant == 'Reduce':
                    pids = tuple(sorted(action_data['production_ids']))
                    py_row.actions[term_id] = Reduce(int(action_data['nonterminal_id']), int(action_data['len']), pids)
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
                    py_row.actions[term_id] = Split(shift if shift is None else int(shift), reduces)
            for nt_id_str, goto_data in row_data['gotos']:
                nt_id = int(nt_id_str)
                if goto_data['state_id'] is not None:
                    py_row.gotos[nt_id] = int(goto_data['state_id'])
            py_table[state_id] = py_row
        model.parser_table = ParserTable(int(start_state_id), py_table)

        # Initialize initial state and accumulators
        initial_acc = PyAcc(terminals_union={}, llm_mask=FastRangeSet.empty())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        # Token maps and caches
        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}

        # Convert possible_matches_cache to FastRangeSet
        pmc_ffi: Dict[int, Dict[int, ffi.Bitset]] = constraint.possible_matches()
        pmc_rs: Dict[int, Dict[int, FastRangeSet]] = {}
        for tsid, inner in pmc_ffi.items():
            mapped: Dict[int, FastRangeSet] = {}
            for term_id, bit in inner.items():
                mapped[int(term_id)] = FastRangeSet(ffi.Bitset.from_ranges(bit.to_ranges()))
            pmc_rs[int(tsid)] = mapped
        model.possible_matches_cache = pmc_rs

        model.internal_to_original_map = constraint.internal_to_original_map()
        all_internal = constraint.all_internal_llm_tokens_bitset()
        model.all_internal_llm_tokens_bitset = FastRangeSet(ffi.Bitset.from_ranges(all_internal.to_ranges()))
        return model

    # -----------------------
    # Minimal helpers needed for commit()
    # -----------------------

    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, FastRangeSet]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            disallowed_terminals_map = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(state_id, FastRangeSet.empty())
                if not matched_bv.intersection(disallowed_for_state).is_empty():
                    return False
            return True
        return gss.prune(predicate)

    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_map = acc.terminals_union
            new_bvs: Dict[int, FastRangeSet] = collections.defaultdict(FastRangeSet.empty)
            for old_sid, new_sid in state_map.items():
                bv_source = old_map.get(old_sid, FastRangeSet.empty())
                new_bvs[new_sid] = new_bvs[new_sid].union(bv_source)
            return PyAcc(terminals_union=dict(new_bvs), llm_mask=acc.llm_mask)
        return gss.apply(apply_map)

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_map = acc.terminals_union.copy()
            curr_bv = current_map.get(state_id, FastRangeSet.empty())
            to_add = FastRangeSet.from_indices([terminal_id])
            new_bv = curr_bv.union(to_add)
            current_map[state_id] = new_bv
            return PyAcc(terminals_union=current_map, llm_mask=acc.llm_mask)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        # Same as previous version, left unchanged for correctness.
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

    # -----------------------
    # Public API: commit
    # -----------------------

    def commit(self, token_id: int):
        """
        Updates self.state by committing the given token with the tokenizer and parser.
        This is essentially the same as the precompute3_model_pure_python version,
        with FastRangeSet substituted in.
        """
        token_bytes = self.id_to_token[token_id]

        # Build tokenizer maps
        terminals_map: Dict[int, FastRangeSet] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            matched_terminals = [terminal_id for terminal_id, _ in matches]
            terminals_map[tokenizer_sid] = FastRangeSet.from_indices(matched_terminals)

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

    # -----------------------
    # Public API: get_mask (optimized)
    # -----------------------

    def get_mask(self) -> FastRangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Optimizations in this implementation:
        - Avoids per-membership FFI calls by precomputing state_bv as Python ranges and using binary search.
        - Defers GSS merges: accumulates children per node and merges once on dequeue.
        - Caches apply-and-prune results across edges to avoid duplicate intersections.
        - No stats/profiling overhead.
        """
        state_map: Dict[int, GSS] = self.state
        if not state_map:
            return FastRangeSet.empty()

        all_ones: Optional[FastRangeSet] = self.all_internal_llm_tokens_bitset
        final_mask: FastRangeSet = FastRangeSet.empty()

        # values_pending holds lists of GSS to be merged when processing the node.
        values_pending: Dict[int, List[GSS]] = {}
        todo: Dict[int, Set[int]] = {}
        depth_heap: List[int] = []

        hp, hpop = heapq.heappush, heapq.heappop
        roots_map: Dict[int, int] = self.roots_map
        max_depth: Dict[int, int] = self.max_depth
        arena: Dict[int, dict] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, FastRangeSet]] = self.possible_matches_cache or {}
        max_state: int = int(self.tokenizer_max_state) if self.tokenizer_max_state is not None else 0

        # Seed: initialize per-acc llm_mask (and consume terminals_union), then enqueue root nodes
        def initialize_acc(acc: PyAcc) -> PyAcc:
            disallowed_llm_mask = FastRangeSet.empty()
            disallowed_map = acc.terminals_union

            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]
                for terminal_id in disallowed_terminals.to_indices():
                    bv = terminals_to_llm.get(terminal_id)
                    if bv is not None:
                        disallowed_llm_mask = disallowed_llm_mask.union(bv)

            allowed_mask = (all_ones if all_ones is not None else FastRangeSet.empty()).difference(disallowed_llm_mask)
            return PyAcc(terminals_union={}, llm_mask=allowed_mask)

        apply_memo: Dict[PyAcc, PyAcc] = {}

        for sid, gss in state_map.items():
            r: int = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)
            lst = values_pending.get(r)
            if lst is None:
                values_pending[r] = [gss_initialized]
                d = max_depth[r]
                bucket = todo.get(d)
                if bucket is None:
                    todo[d] = {r}
                    hp(depth_heap, d)
                else:
                    bucket.add(r)
            else:
                lst.append(gss_initialized)

        # Utility: enqueue node by depth
        def enqueue(d: int, n: int) -> None:
            bucket = todo.get(d)
            if bucket is None:
                todo[d] = {n}
                hp(depth_heap, d)
            else:
                bucket.add(n)

        # Global cache for apply-and-prune across edges:
        # Key: (PyAcc, FastRangeSet llm_bv) -> Optional[PyAcc]
        # Note: After initialization, acc.terminals_union is {}, so acc hash is effectively llm_mask hash.
        acc_intersect_cache: Dict[Tuple[PyAcc, FastRangeSet], Optional[PyAcc]] = {}

        # Main traversal
        while depth_heap:
            depth: int = hpop(depth_heap)
            bucket_nodes = todo.get(depth)
            if not bucket_nodes:
                # Shouldn't happen; safety
                continue

            while bucket_nodes:
                node: int = bucket_nodes.pop()
                gss_list: Optional[List[GSS]] = values_pending.pop(node, None)
                if not gss_list:
                    continue

                # Merge once per node
                gss_node: GSS = gss_list[0] if len(gss_list) == 1 else GSS.merge_many(gss_list)

                # End-node: reduce acc and union llm_mask to final
                if is_end(node):
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    if reduced_acc:
                        final_mask = final_mask.union(reduced_acc.llm_mask)

                # Traverse edges from this trie node
                edges = arena.get(node, {}).get("children") or []
                if not edges:
                    continue

                # Hot loop: bind methods locally to avoid attribute lookup overhead
                _popn: Callable[[int], GSS] = gss_node.popn
                _enqueue = enqueue
                _max_depth = max_depth
                for pop, llm_bv, dests in edges:
                    popped: GSS = _popn(pop)
                    if popped.is_empty():
                        continue

                    peeked_states = tuple(popped.peek())
                    if not peeked_states:
                        continue

                    # Group sids by destination index using fast range membership
                    # dests: List[(dest_idx, ranges, starts)]
                    # Create mapping d -> list[sid]
                    groups: Dict[int, List[int]] = {}

                    # Localize to reduce lookups
                    _contains_local = _contains_by_ranges
                    for sid in peeked_states:
                        added_any = False
                        for dest_idx, ranges, starts in dests:
                            if _contains_local(starts, ranges, sid):
                                lst = groups.get(dest_idx)
                                if lst is None:
                                    groups[dest_idx] = [sid]
                                else:
                                    lst.append(sid)
                                added_any = True
                        # If this sid didn't match any dest, we drop it; that's fine.

                    if not groups:
                        continue

                    # Build intersect-and-prune function, cached across edges via acc_intersect_cache
                    cache = acc_intersect_cache
                    llm_bv_local = llm_bv

                    def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                        key = (acc, llm_bv_local)
                        cached = cache.get(key)
                        if cached is not None or key in cache:
                            return cached
                        new_mask = acc.llm_mask.intersection(llm_bv_local)
                        if new_mask.is_empty():
                            cache[key] = None
                            return None
                        result = PyAcc(terminals_union=acc.terminals_union, llm_mask=new_mask)
                        cache[key] = result
                        return result

                    # For each destination, isolate subset and apply-and-prune
                    for dest_idx, sid_list in groups.items():
                        child_gss: GSS = popped.isolate_many(sid_list) if len(sid_list) > 1 else popped.isolate(sid_list[0])
                        if child_gss.is_empty():
                            continue

                        child_gss = child_gss.apply_and_prune(intersect_and_prune)
                        if child_gss.is_empty():
                            continue

                        # Defer merge: append to list for dest node
                        lst = values_pending.get(dest_idx)
                        if lst is None:
                            values_pending[dest_idx] = [child_gss]
                            _enqueue(_max_depth[dest_idx], dest_idx)
                        else:
                            lst.append(child_gss)

            # Done with this depth
            todo.pop(depth, None)

        # Convert internal mask to original IDs
        original_indices: List[int] = []
        for i in final_mask.to_indices():
            j = self.internal_to_original_map.get(i)
            if j is not None:
                original_indices.append(j)
        return FastRangeSet.from_indices(original_indices)
