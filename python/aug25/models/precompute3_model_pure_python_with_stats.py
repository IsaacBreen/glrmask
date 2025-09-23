import json
import heapq
import collections
import time
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field

from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from ..stats import Stats

# --- Monkey-patch RangeSet to collect stats on union/intersection ---
# This is to fulfill the request of tracking ffi.Bitset.union and intersection calls.
# Since the code was refactored to use a pure Python RangeSet, we track its methods instead.
_original_rangeset_union = RangeSet.union
_original_rangeset_intersection = RangeSet.intersection

def _patched_union(self, other: "RangeSet") -> "RangeSet":
    """Patched version of RangeSet.union that increments a stats counter."""
    Stats.get().inc('bitset.union.calls')
    return _original_rangeset_union(self, other)

def _patched_intersection(self, other: "RangeSet") -> "RangeSet":
    """Patched version of RangeSet.intersection that increments a stats counter."""
    Stats.get().inc('bitset.intersection.calls')
    return _original_rangeset_intersection(self, other)

# Apply the patches
RangeSet.union = _patched_union
RangeSet.intersection = _patched_intersection
# --- End of monkey-patch ---


# Add a dummy profiler for when not running under kernprof
try:
    # This will be injected by the kernprof script.
    profile
except NameError:
    # If not running under kernprof, create a dummy decorator.
    def profile(func):
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
    gotos: Dict[int, int] = field(default_factory=dict)  # nonterminal_id -> state_id


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


class Model(GraphProvider):
    """
    Precomputed trie model (third-generation), simplified and concise.
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
                llm_bv_bitset = bs_from_json(dumps(llm_bv_json))
                # Convert to RangeSet for ffi-free operations in commit/get_mask
                llm_bv = RangeSet.from_ranges(llm_bv_bitset.to_ranges())
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv_bitset = bs_from_json(dumps(state_bv_json))
                    state_bv = RangeSet.from_ranges(state_bv_bitset.to_ranges())
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
        # Convert universe LLM tokens bitset to RangeSet
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
    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.
        """
        # NOTE on GSS methods: We assume the GSS object has methods like `get_all_accs()`
        # to retrieve all unique accumulators and that `len(gss)` returns the number of stacks.
        # These are reasonable assumptions for instrumenting this GSS implementation.

        stats = Stats.get()
        with stats:
            stats.start('get_mask.total')
            state_map: Dict[int, GSS] = self.state
            stats.inc('get_mask.initial_tokenizer_states', len(state_map))

            all_ones: Optional[RangeSet] = self.all_internal_llm_tokens_bitset
            final_mask: RangeSet = RangeSet.empty()

            # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask
            values: Dict[int, GSS] = {}
            depth_heap: List[Tuple[int, int]] = []  # Stores (-depth, node_id)
            enqueued_nodes: Set[int] = set()

            hp, hpop = heapq.heappush, heapq.heappop
            roots_map: Dict[int, int] = self.roots_map
            max_depth: Dict[int, int] = self.max_depth
            arena: Dict[int, dict] = self.arena
            is_end = self.is_end
            pmc: Dict[int, Dict[int, RangeSet]] = self.possible_matches_cache or {}
            max_state: int = self.tokenizer_max_state

            def enqueue(d: int, n: int) -> None:
                stats.inc('get_mask.traversal.enqueues')
                if n not in enqueued_nodes:
                    enqueued_nodes.add(n)
                    hp(depth_heap, (-d, n))

            # --- Initial GSS Stats ---
            stats.start('get_mask.initial_stats')
            all_initial_accs = set()
            for gss in state_map.values():
                # We assume gss.get_all_accs() exists for stats gathering.
                accs = getattr(gss, 'get_all_accs', lambda: [])()
                all_initial_accs.update(accs)
                stats.inc('get_mask.initial.gss_heads.sum', len(gss.peek()))
            stats.inc('get_mask.initial.unique_accs', len(all_initial_accs))
            for acc in all_initial_accs:
                stats.inc('get_mask.initial.terminals_union_size.sum', len(acc.terminals_union))
            stats.stop('get_mask.initial_stats')

            stats.start('get_mask.seeding')
            # Seed: Initialize llm_mask in each GSS, consume terminals_union, and enqueue roots.
            def initialize_acc(acc: PyAcc) -> PyAcc:
                stats.inc('get_mask.initialize_acc.calls')
                stats.start('get_mask.initialize_acc.total')
                # Compute allowed LLM tokens from disallowed terminals for this accumulator
                disallowed_llm_mask = RangeSet.empty()
                disallowed_map = acc.terminals_union
                stats.inc('get_mask.initialize_acc.disallowed_map_size.sum', len(disallowed_map))

                for tsid, disallowed_terminals in disallowed_map.items():
                    stats.inc('get_mask.initialize_acc.disallowed_terminals_loops')
                    if tsid > max_state or tsid not in pmc:
                        continue
                    terminals_to_llm = pmc[tsid]

                    stats.start('get_mask.initialize_acc.to_indices')
                    indices = disallowed_terminals.to_indices()
                    stats.stop('get_mask.initialize_acc.to_indices')

                    stats.inc('get_mask.initialize_acc.disallowed_terminals_count.sum', len(indices))
                    for terminal_id in indices:
                        stats.inc('get_mask.initialize_acc.disallowed_terminals_inner_loops')
                        if terminal_id in terminals_to_llm:
                            stats.start('get_mask.initialize_acc.union')
                            disallowed_llm_mask = disallowed_llm_mask.union(
                                terminals_to_llm[terminal_id]
                            )
                            stats.stop('get_mask.initialize_acc.union')

                stats.start('get_mask.initialize_acc.difference')
                allowed_mask = (all_ones if all_ones is not None else RangeSet.empty()).difference(disallowed_llm_mask)
                stats.stop('get_mask.initialize_acc.difference')

                stats.stop('get_mask.initialize_acc.total')
                return PyAcc(
                    terminals_union={},  # consume
                    llm_mask=allowed_mask,
                )

            apply_memo: Dict[PyAcc, PyAcc] = {}
            for sid, gss in state_map.items():
                stats.inc('get_mask.seeding.gss_loops')
                r: int = roots_map[int(sid)]

                stats.start('get_mask.seeding.gss.apply')
                gss_initialized: GSS = gss.apply(initialize_acc, apply_memo)
                stats.stop('get_mask.seeding.gss.apply')

                if r in values:
                    stats.start('get_mask.seeding.gss.merge')
                    values[r] = values[r].merge(gss_initialized)
                    stats.stop('get_mask.seeding.gss.merge')
                else:
                    values[r] = gss_initialized

                d: int = max_depth[r]
                enqueue(d, r)
            stats.inc('get_mask.seeding.apply_memo.size', len(apply_memo))
            stats.stop('get_mask.seeding')

            # Main loop
            stats.start('get_mask.main_loop')
            max_depth_reached = 0
            visited_nodes = set()
            while depth_heap:
                neg_depth, node = hpop(depth_heap)
                depth = -neg_depth
                max_depth_reached = max(max_depth_reached, depth)
                stats.inc('get_mask.traversal.depth_heap.pops')
                stats.inc('get_mask.traversal.nodes_processed')
                visited_nodes.add(node)
                gss_node: GSS = values.pop(node)
                stats.inc('get_mask.gss.at_node.accs.sum', len(getattr(gss_node, 'get_all_accs', lambda: [])()))

                # End-node handling: just union the allowed LLM tokens
                if is_end(node):
                    stats.inc('get_mask.traversal.end_nodes')
                    stats.start('get_mask.main_loop.end_node.reduce_acc')
                    reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                    stats.stop('get_mask.main_loop.end_node.reduce_acc')
                    if reduced_acc:
                        stats.start('get_mask.main_loop.end_node.final_mask_union')
                        final_mask = final_mask.union(reduced_acc.llm_mask)
                        stats.stop('get_mask.main_loop.end_node.final_mask_union')

                # Traverse edges and propagate masks
                edges = arena.get(node, {}).get("children") or []
                stats.inc('get_mask.traversal.edge_blocks.sum', len(edges))
                if len(edges) > 10:
                    print("len(edges)", len(edges))
                stats.inc('get_mask.traversal.dests_blocks.sum', sum(len(dests) for _, dests in edges))
                if sum(len(dests) for _, dests in edges) > 10:
                    print("sum(len(dests) for _, dests in edges)", sum(len(dests) for _, dests in edges))
                for (pop, llm_bv), dests in edges:
                    llm_bv = llm_bv.difference(final_mask)

                    if llm_bv.is_empty():
                        stats.inc('get_mask.traversal.edge.llm_bv_empty')
                        continue

                    stats.inc('get_mask.traversal.edges_traversed')
                    stats.inc(f'get_mask.traversal.edge_pop_val.{pop}')
                    stats.inc('get_mask.data.llm_bv_on_edge.len.sum', len(llm_bv))

                    stats.start('get_mask.main_loop.edge.popn')
                    popped: GSS = gss_node.popn(pop)
                    stats.stop('get_mask.main_loop.edge.popn')
                    if popped.is_empty():
                        stats.inc('get_mask.traversal.edge.popped_empty')
                        continue

                    # Apply edge LLM mask by intersecting per-acc llm_mask with llm_bv
                    acc_memo: Dict[PyAcc, Optional[PyAcc]] = {}

                    def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                        stats.inc('get_mask.intersect_and_prune.calls')
                        if acc in acc_memo:
                            stats.inc('get_mask.intersect_and_prune.memo_hits')
                            return acc_memo[acc]

                        stats.start('get_mask.intersect_and_prune.intersection')
                        new_mask = acc.llm_mask.intersection(llm_bv)
                        stats.stop('get_mask.intersect_and_prune.intersection')
                        stats.inc('get_mask.data.llm_mask_after_intersect.len.sum', len(new_mask))

                        if new_mask.is_empty():
                            stats.inc('get_mask.intersect_and_prune.pruned_accs')
                            result = None
                        else:
                            result = PyAcc(
                                terminals_union=acc.terminals_union,
                                llm_mask=new_mask
                            )
                        acc_memo[acc] = result
                        return result

                    stats.start('get_mask.main_loop.edge.apply_and_prune')
                    popped = popped.apply_and_prune(intersect_and_prune)
                    stats.stop('get_mask.main_loop.edge.apply_and_prune')

                    if popped.is_empty():
                        stats.inc('get_mask.traversal.edge.popped_pruned_empty')
                        continue

                    if popped.reduce_acc().is_empty():
                        stats.inc('get_mask.traversal.edge.popped_reduced_empty')
                        continue

                    for dest_idx, state_bv in dests:
                        stats.inc('get_mask.traversal.dests_traversed')
                        stats.inc('get_mask.data.state_bv_on_edge.len.sum', len(state_bv))

                        stats.start('get_mask.main_loop.edge.peek_and_filter')
                        peeked = popped.peek()
                        values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]
                        stats.stop('get_mask.main_loop.edge.peek_and_filter')

                        if not values_to_keep:
                            continue

                        stats.start('get_mask.main_loop.edge.isolate_many')
                        child_gss: GSS = popped.isolate_many(values_to_keep)
                        stats.stop('get_mask.main_loop.edge.isolate_many')
                        if child_gss.is_empty():
                            continue

                        stats.inc('get_mask.intersect_and_prune.memo_size.sum', len(acc_memo))
                        if child_gss.is_empty():
                            stats.inc('get_mask.traversal.edge.child_gss_pruned_empty')
                            continue

                        if child_gss.reduce_acc().is_empty():
                            continue

                        d: int = int(dest_idx)
                        if d in values:
                            stats.inc('get_mask.traversal.edge.gss_merges')
                            stats.start('get_mask.main_loop.edge.gss_merge')
                            values[d] = values[d].merge(child_gss)
                            stats.stop('get_mask.main_loop.edge.gss_merge')
                        else:
                            values[d] = child_gss
                        enqueue(max_depth[d], d)
            stats.stop('get_mask.main_loop')
            stats.inc('get_mask.traversal.max_depth_reached', max_depth_reached)
            stats.inc('get_mask.traversal.nodes_visited.unique', len(visited_nodes))

            stats.start('get_mask.final_conversion')
            # Convert internal mask back to original IDs
            original_indices: List[int] = []

            stats.start('get_mask.final_conversion.to_indices')
            final_indices = final_mask.to_indices()
            stats.stop('get_mask.final_conversion.to_indices')

            stats.inc('get_mask.final_mask.internal_indices', len(final_indices))
            for i in final_indices:
                if i in self.internal_to_original_map:
                    original_indices.append(self.internal_to_original_map[i])
            stats.inc('get_mask.final_mask.original_indices', len(original_indices))

            stats.start('get_mask.final_conversion.from_indices')
            result = RangeSet.from_indices(original_indices)
            stats.stop('get_mask.final_conversion.from_indices')

            stats.stop('get_mask.final_conversion')

            stats.stop('get_mask.total')
            # stats.report()
            return result