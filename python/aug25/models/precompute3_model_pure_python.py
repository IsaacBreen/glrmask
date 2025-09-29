from __future__ import annotations

import json
import heapq
import collections
import textwrap
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
from ..stats import Stats
# from ..common_interface import RangeSet
# from ..range_set.set_range_set import SetRangeSet as RangeSet
# from ..range_set.py_range_set import PyRangeSet as RangeSet
# from ..range_set.bitset_range_set import BitsetRangeSet as RangeSet
from ..range_set.ffi_range_set import FFIRangeSet as RangeSet
# from ..range_set.roaring_range_set import RoaringRangeSet as RangeSet

# from ..common_interface import RangeSetOut
from ..range_set.set_range_set import SetRangeSet as RangeSetOut
# from ..range_set.py_range_set import PyRangeSet as RangeSetOut
# from ..range_set.bitset_range_set import BitsetRangeSet as RangeSetOut
# from ..range_set.ffi_range_set import FFIRangeSet as RangeSetOut
# from ..range_set.roaring_range_set import RoaringRangeSet as RangeSetOut

# from ..common_interface import RangeSetStates
from ..range_set.set_range_set import SetRangeSet as RangeSetStates
# from ..range_set.py_range_set import PyRangeSet as RangeSetStates
# from ..range_set.bitset_range_set import BitsetRangeSet as RangeSetStates
# from ..range_set.ffi_range_set import FFIRangeSet as RangeSetStates
# from ..range_set.roaring_range_set import RoaringRangeSet as RangeSetStates

import _sep1 as ffi
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
# from python.gss_tester.implementations.leveled_per_acc_impl import LeveledPerAccGSS as GSS
# from python.gss_tester.implementations.leveled_impl_cpp import Leveled_impl_cppGSS as GSS


# --- Monkey-patch RangeSet to collect stats on union/intersection ---
# This is to fulfill the request of tracking ffi.Bitset.union and intersection calls.
# Since the code was refactored to use a pure Python RangeSet, we track its methods instead.
_original_rangeset_union = RangeSet.union
_original_rangeset_intersection = RangeSet.intersection

def _patched_union(self, other: "RangeSet") -> "RangeSet":
    """Patched version of RangeSet.union that increments a stats counter."""
    stats = Stats.get()
    stats.inc('bitset.union.calls')
    stats.start('bitset.union.time')
    result = _original_rangeset_union(self, other)
    stats.stop('bitset.union.time')
    return result

def _patched_intersection(self, other: "RangeSet") -> "RangeSet":
    """Patched version of RangeSet.intersection that increments a stats counter."""
    stats = Stats.get()
    stats.inc('bitset.intersection.calls')
    stats.start('bitset.intersection.time')
    result = _original_rangeset_intersection(self, other)
    stats.stop('bitset.intersection.time')
    return result

# Apply the patches
RangeSet.union = _patched_union
RangeSet.intersection = _patched_intersection
# --- End of monkey-patch ---


NodeID = int

# Type aliases for different uses of RangeSet to improve clarity.
LLMTokenSet = RangeSet
StateIDSet = RangeSetStates
TerminalIdSet = RangeSet


# Add a dummy profiler for when not running under kernprof
try:
    # This will be injected by the kernprof script.
    profile
except NameError:
    # If not running under kernprof, create a dummy decorator.
    def profile(func): return func

# --- Accumulator memoization decorator ---
def _acc_memoize(stats_prefix: Optional[str] = None, use_value_cache: bool = True):
    """
    Per-invocation memoization for PyAcc transformers.
    - Caches by id(acc) (including None results).
    - Caches by value (acc) for non-None results, if use_value_cache is True.
    If stats_prefix is provided, increments '{prefix}.memo_hits' on cache hits.
    Exposes _acc_memo_size() to inspect id-cache size for stats.
    """
    def decorator(fn):
        id_memo = {}
        val_memo = {}
        def wrapper(acc):
            # Identity-based fast path
            if id(acc) in id_memo:
                if stats_prefix:
                    Stats.get().inc(f'{stats_prefix}.memo_hits')
                return id_memo[id(acc)]

            if use_value_cache:
                # Structural equality-based cache (only non-None results are useful here)
                cached = val_memo.get(acc)
                if cached is not None:
                    id_memo[id(acc)] = cached
                    if stats_prefix:
                        Stats.get().inc(f'{stats_prefix}.memo_hits')
                    return cached
            result = fn(acc)
            id_memo[id(acc)] = result
            if use_value_cache:
                val_memo[acc] = result
            return result
        wrapper._acc_memo_size = lambda: len(id_memo)
        return wrapper
    return decorator


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
    stats = Stats.get()
    stats.add_group('get_mask')
    stats.add_group('commit')

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
    internal_to_original_map: Dict[int, RangeSetOut]
    all_internal_llm_tokens_bitset: LLMTokenSet
    all_terminals_bitset: TerminalIdSet
    ignore_terminal_id: Optional[int]

    # State
    state: Dict[int, GSS]

    # --- Precomputed traversal optimization ---
    # Minimum steps from node to any end (clean_end=True)
    node_min_dist: Dict[NodeID, int] = field(default_factory=dict)

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        Stats.get().reset()
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
                llm_bv_union |= llm_bv
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    state_bv_bitset = bs_from_json(dumps(state_bv_json))
                    state_bv: StateIDSet = RangeSetStates.from_ranges(state_bv_bitset.to_ranges())
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

        vocab = data['precompute3_vocab']
        internal_to_original_map_raw = dict(vocab['internal_to_original'])
        internal_to_original_map = {
            int(k): RangeSetOut.from_indices(v) for k, v in internal_to_original_map_raw.items()
        }
        internal_max = vocab['internal_max_llm_token']
        all_internal_llm_tokens_bitset = RangeSet.from_ranges([(0, internal_max)])

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

        # Pre-optimization passes to make get_mask fast
        # 1) reorder edges/dests to approach end nodes ASAP and compute node_min_dist
        model._optimize_graph_for_end_bfs()

        return model
    @profile
    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        terminal_to_add_rs = RangeSet.from_indices([terminal_id])

        @_acc_memoize(use_value_cache=False)
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_set = acc.terminals_union.get(state_id, RangeSet.empty())
            if current_set.contains(terminal_id):
                return acc

            new_map = acc.terminals_union.copy()
            new_map[state_id] = current_set.union(terminal_to_add_rs)
            return PyAcc(terminals_union=new_map, llm_mask=acc.llm_mask)

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
        stats = Stats.get()
        stats.start('commit')
        token_bytes = self.id_to_token[token_id]

        # Build tokenizer maps
        stats.start('commit.build_tokenizer_maps')
        stats.inc('commit.tokenizer_states_in', len(self.state))
        terminals_map: Dict[int, TerminalIdSet] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            matched_terminals = [terminal_id for terminal_id, _ in matches]
            terminals_map[tokenizer_sid] = RangeSet.from_indices(matched_terminals)

        stats.stop('commit.build_tokenizer_maps')
        # Prune and map per-state GSS in a single pass
        temp_states: Dict[int, GSS] = {}
        stats.start('commit.prune_and_map_gss')
        @_acc_memoize()
        def mutator(acc: PyAcc) -> Optional[PyAcc]:
            # Prune condition
            disallowed_terminals_map = acc.terminals_union
            for tsid, matched_bv in terminals_map.items():
                disallowed_for_state = disallowed_terminals_map.get(tsid)
                if disallowed_for_state and not matched_bv.isdisjoint(disallowed_for_state):
                    return None
            # Map
            old_map = acc.terminals_union
            new_bvs: Dict[int, TerminalIdSet] = {}
            for old_sid, new_sid in state_map.items():
                bv_source = old_map.get(old_sid)
                if bv_source and not bv_source.is_empty():
                    if new_sid in new_bvs:
                        new_bvs[new_sid] |= bv_source
                    else:
                        new_bvs[new_sid] = bv_source
            return PyAcc(terminals_union=new_bvs, llm_mask=acc.llm_mask)
        cache = {}
        current_state_for_processing = {tsid: gss.apply_and_prune(mutator, cache) for tsid, gss in self.state.items()}
        current_state_for_processing = {tsid: gss for tsid, gss in current_state_for_processing.items() if not gss.is_empty()}
        stats.stop('commit.prune_and_map_gss')

        new_states: Dict[int, List[GSS]] = collections.defaultdict(list)
        stats.start('commit.main_loop')
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
        stats.stop('commit.main_loop')

        stats.start('commit.merge_states')
        merged_states = {
            sid: GSS.merge_many(gss_list)
            for sid, gss_list in new_states.items()
            if gss_list
        }
        merged_states = {sid: state for sid, state in merged_states.items() if not state.is_empty()}
        stats.stop('commit.merge_states')

        stats.start('commit.fuse')
        # memo = {}
        # merged_states = {tsid: gss.fuse("to_interface", memo) for tsid, gss in merged_states.items()}
        # memo = {}
        # merged_states = {tsid: gss.fuse(1, memo) for tsid, gss in merged_states.items()}
        stats.stop('commit.fuse')
        # print(GSS.merge_many(merged_states.values()).stats())
        # print(GSS.merge_many(merged_states.values()).to_graph_string(upper_only=True))


        stats.inc('commit.tokenizer_states_out', len(merged_states))
        self.state = merged_states
        stats.stop('commit')

    @profile
    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        stats = Stats.get()
        p = 'commit.main_loop._process_token'
        stats.start(f'{p}.total')
        stats.inc(f'{p}.calls')
        stats.inc(f'{p}.initial_heads', len(gss.peek()))

        heads_by_state: Dict[int, List[GSS]] = collections.defaultdict(list)
        for state_id in gss.peek():
            heads_by_state[state_id].append(gss.isolate(state_id))

        shifted_gsses: List[GSS] = []
        reduces_handled = 0

        while heads_by_state:
            stats.inc(f'{p}.loop_iterations')
            state_id, state_gsss = heads_by_state.popitem()
            stats.start(f'{p}.merge_many.heads')
            state_gss = GSS.merge_many(state_gsss)
            stats.stop(f'{p}.merge_many.heads')
            row = self.parser_table.table.get(state_id)
            if not row:
                continue
            action = row.actions.get(terminal_id)
            if not action:
                continue

            def handle_shift(shift_to_state_id, gss_to_shift):
                stats.inc(f'{p}.shifts')
                shifted_gsses.append(gss_to_shift.push(shift_to_state_id))

            def handle_reduce(reduce_action: Reduce, gss_to_reduce: GSS):
                stats.inc(f'{p}.reduces')
                stats.start(f'{p}.reduce.pop')
                popped_gss = gss_to_reduce
                for _ in range(reduce_action.len):
                    popped_gss = popped_gss.pop()
                stats.stop(f'{p}.reduce.pop')
                for from_state_id in popped_gss.peek():
                    goto_state_id = self.parser_table.table[from_state_id].gotos[reduce_action.nonterminal_id]
                    goto_gss = popped_gss.isolate(from_state_id).push(goto_state_id)
                    heads_by_state[goto_state_id].append(goto_gss)
                    stats.inc(f'{p}.reduce.new_heads')

            if isinstance(action, int):
                handle_shift(action, state_gss)
            elif isinstance(action, Reduce):
                handle_reduce(action, state_gss)
            elif isinstance(action, Split):
                stats.inc(f'{p}.splits')
                if action.shift is not None:
                    handle_shift(action.shift, state_gss)
                for length, nts in action.reduces.items():
                    for nt_id, pids in nts.items():
                        handle_reduce(Reduce(nt_id, length, pids), state_gss)

        stats.start(f'{p}.merge_many.final')
        result = GSS.merge_many(shifted_gsses)
        stats.stop(f'{p}.merge_many.final')
        stats.inc(f'{p}.final_heads', len(result.peek()))
        stats.stop(f'{p}.total')
        return result

    def _is_zombie_path(self, gss: GSS, path_token_union: LLMTokenSet, final_mask: LLMTokenSet, stat_context: str) -> bool:
        """
        Checks if a given traversal path is a "zombie" path, meaning it cannot
        contribute any new tokens to the final_mask.
        """
        stats = Stats.get()
        p = f'get_mask.zombie_check.{stat_context}'
        stats.start(p)

        # We only care about tokens that are not yet in the final mask.
        potential_new_tokens = path_token_union.difference(final_mask)
        if potential_new_tokens.is_empty():
            stats.inc(f'{p}.skipped_no_potential')
            stats.stop(p)
            return True

        gss_mask_acc = gss.reduce_acc()
        if gss_mask_acc and gss_mask_acc.llm_mask.isdisjoint(potential_new_tokens):
            stats.inc(f'{p}.skipped_no_overlap_disjoint')
            stats.stop(p)
            return True

        stats.stop(p)
        return False

    def _optimize_graph_for_end_bfs(self):
        """
        Compute min distance to an end node for each arena node and reorder:
        - inner dests: ascending by min-dist
        - outer edges: ascending by best dest min-dist
        This biases traversal to reach an end node as soon as possible.
        """
        stats = Stats.get()
        stats.start('optimize_graph')

        arena = self.arena
        if not arena:
            stats.stop('optimize_graph')
            return

        # Build reverse adjacency (dest -> sources)
        rev: Dict[NodeID, Set[NodeID]] = collections.defaultdict(set)
        for src, node in arena.items():
            for _edge_key, dests in (node.children or []):
                for dest_idx, _state_bv in dests:
                    rev[int(dest_idx)].add(int(src))

        # Multi-source BFS from all end nodes
        INF = 10**9
        dist: Dict[NodeID, int] = { }
        dq = collections.deque()
        for nid, node in arena.items():
            if node.clean_end:
                dist[int(nid)] = 0
                dq.append(int(nid))

        while dq:
            dnode = dq.popleft()
            base = dist[dnode]
            for parent in rev.get(dnode, ()):
                if dist.get(parent, INF) > base + 1:
                    dist[parent] = base + 1
                    dq.append(parent)

        # Fallback to max depth if a node is not connected to any end
        for nid in arena.keys():
            if nid not in dist:
                dist[nid] = self.max_depth.get(nid, INF)

        # Reorder dests and edges using this min-dist to end
        for node in arena.values():
            for _edge_key, dests in (node.children or []):
                dests.sort(key=lambda item: dist.get(int(item[0]), INF))
            node.children.sort(
                key=lambda edge: (dist.get(int(edge[1][0][0]), INF) if edge[1] else INF)
            )

        self.node_min_dist = dist
        stats.stop('optimize_graph')
    @profile
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
        stats = Stats.get()
        stats.start('get_mask')
        state_map: Dict[int, GSS] = self.state
        stats.inc('get_mask.initial_tokenizer_states', len(state_map))

        all_ones: LLMTokenSet = self.all_internal_llm_tokens_bitset
        final_mask: LLMTokenSet = RangeSet.empty()

        # We store (GSS, next_edge_idx, next_dest_idx) per node.
        values: Dict[NodeID, Tuple[GSS, int, int]] = {}
        # Min-heap by priority (closer to end first). Priority is node_min_dist.
        depth_heap: List[Tuple[int, NodeID]] = []  # Stores (priority, node_id)

        roots_map: Dict[int, NodeID] = self.roots_map
        max_depth: Dict[NodeID, int] = self.max_depth
        arena: Dict[NodeID, ArenaNode] = self.arena
        is_end = self.is_end
        pmc: Dict[int, Dict[int, LLMTokenSet]] = self.possible_matches_cache or {}
        max_state: int = self.tokenizer_max_state

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
        @_acc_memoize(use_value_cache=False)
        def initialize_acc(acc: PyAcc) -> PyAcc:
            if False:  # placeholder to keep minimal diff context; decorator handles memoization
                if cached_acc is not None:
                    return cached_acc

            p = 'get_mask.seeding.initialize_acc'
            stats.inc(f'{p}.calls')
            stats.start(f'{p}.total')
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask: LLMTokenSet = RangeSet.empty()
            disallowed_map = acc.terminals_union
            stats.inc(f'{p}.disallowed_map_size.sum', len(disallowed_map))

            for tsid, disallowed_terminals in disallowed_map.items():
                stats.inc(f'{p}.disallowed_terminals_loops')
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]

                stats.start(f'{p}.iter_indices')
                indices = disallowed_terminals.iter_indices()
                stats.stop(f'{p}.iter_indices')

                for terminal_id in indices:
                    stats.inc(f'{p}.disallowed_terminals_inner_loops')
                    if terminal_id in terminals_to_llm:
                        stats.start(f'{p}.union')
                        disallowed_llm_mask |= terminals_to_llm[terminal_id]
                        stats.stop(f'{p}.union')

            stats.start(f'{p}.difference')
            allowed_mask = all_ones.difference(disallowed_llm_mask)
            stats.stop(f'{p}.difference')

            stats.stop(f'{p}.total')
            result = PyAcc(
                terminals_union={},  # consume
                llm_mask=allowed_mask,
            )
            return result

        initial_gss_map: Dict[NodeID, List[GSS]] = collections.defaultdict(list)
        cache = {}
        for sid, gss in state_map.items():
            stats.inc('get_mask.seeding.gss_loops')
            r: NodeID = roots_map[int(sid)]

            stats.start('get_mask.seeding.gss.apply')
            gss_initialized = gss.apply(initialize_acc, cache)
            stats.stop('get_mask.seeding.gss.apply')
            initial_gss_map[r].append(gss_initialized)

        for r, gss_list in initial_gss_map.items():
            merged_gss = GSS.merge_many(gss_list)
            if not merged_gss.is_empty():
                values[r] = (merged_gss, 0, 0)
                # Prefer nodes closer to end first
                prio = self.node_min_dist.get(r, max_depth.get(r, 0))
                heapq.heappush(depth_heap, (prio, r))
        stats.stop('get_mask.seeding')

        # Main loop
        stats.start('get_mask.main_loop')
        visited_nodes = set()
        while depth_heap:
            prio, node = heapq.heappop(depth_heap)

            if node not in values:
                stats.inc('get_mask.traversal.stale_pops')
                continue

            gss_node, edge_idx, dest_idx = values[node]
            stats.inc('get_mask.traversal.depth_heap.pops')

            if edge_idx == 0 and dest_idx == 0:  # First time processing this node
                stats.inc('get_mask.traversal.nodes_processed')
                visited_nodes.add(node)
                stats.inc('get_mask.gss.at_node.accs.sum', len(getattr(gss_node, 'get_all_accs', lambda: [])()))

                if is_end(node):
                    stats.inc('get_mask.traversal.end_nodes')
                    reduced_acc = gss_node.reduce_acc()
                    if reduced_acc:
                        # Only union if there are new tokens
                        new_tokens = reduced_acc.llm_mask.difference(final_mask)
                        if not new_tokens.is_empty():
                            final_mask |= new_tokens

                a_node = arena.get(node)
                node_llm_bv_union = a_node.llm_bv_union if a_node else RangeSet.empty()
                if self._is_zombie_path(gss_node, node_llm_bv_union, final_mask, 'node'):
                    del values[node]
                    continue

            a_node = arena.get(node)
            edges = a_node.children if a_node else []
            if edge_idx >= len(edges):
                del values[node]
                continue
            # Compute reduced acc once to cheaply test edges for novelty
            reduced_acc = gss_node.reduce_acc()
            if not reduced_acc or reduced_acc.is_empty():
                del values[node]
                continue
            # Potentially new tokens available at this node:
            potential_new = reduced_acc.llm_mask.difference(final_mask)

            # Advance to the next edge that can contribute new tokens
            while edge_idx < len(edges):
                (pop, llm_bv), dests = edges[edge_idx]
                # Skip edges that can't add anything new for this node
                if potential_new.isdisjoint(llm_bv):
                    stats.inc('get_mask.edge.skipped_by_potential')
                    edge_idx += 1
                    dest_idx = 0
                    continue
                break

            if edge_idx >= len(edges):
                del values[node]
                continue

            # Process ONE destination from this edge (best-first), then reschedule this node
            (pop, llm_bv), dests = edges[edge_idx]
            stats.inc('get_mask.traversal.edges_traversed')
            stats.inc(f'get_mask.traversal.edge_pop_val.{pop}')

            popped = gss_node.popn(pop)
            if not popped.is_empty():
                @_acc_memoize(stats_prefix='get_mask.main_loop.edge.intersect_and_prune', use_value_cache=False)
                def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                    new_mask = acc.llm_mask.intersection(llm_bv)
                    if new_mask.is_empty():
                        return None
                    return PyAcc(terminals_union=acc.terminals_union, llm_mask=new_mask)

                popped = popped.apply_and_prune(intersect_and_prune)

                if not popped.is_empty():
                    peeked = popped.peek()
                    advanced = False
                    # Dests are pre-sorted by min distance to end; try one, then reschedule
                    for didx in range(dest_idx, len(dests)):
                        d_node, state_bv = dests[didx]
                        values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]
                        if not values_to_keep:
                            continue
                        child_gss = popped.isolate_many(values_to_keep)
                        if child_gss.is_empty():
                            continue
                        child_acc = child_gss.reduce_acc()
                        if not child_acc or child_acc.is_empty():
                            continue

                        d: NodeID = int(d_node)
                        if d in values:
                            stats.inc('get_mask.traversal.edge.gss_merges')
                            old_gss, old_ei, old_di = values[d]
                            values[d] = (old_gss.merge(child_gss), old_ei, old_di)
                        else:
                            values[d] = (child_gss, 0, 0)
                            heapq.heappush(depth_heap, (self.node_min_dist.get(d, max_depth.get(d, 0)), d))

                        # Reschedule this node for the next dest within the same edge (one dest at a time)
                        values[node] = (gss_node, edge_idx, didx + 1)
                        # Compute priority for the next dest of this edge; else next edge
                        next_prio = None
                        if didx + 1 < len(dests):
                            nd = dests[didx + 1][0]
                            next_prio = self.node_min_dist.get(int(nd), max_depth.get(int(nd), 0))
                        else:
                            # No more dests in this edge, schedule the next edge if any
                            if edge_idx + 1 < len(edges):
                                next_edge = edges[edge_idx + 1]
                                if next_edge[1]:
                                    nd2 = next_edge[1][0][0]
                                    next_prio = self.node_min_dist.get(int(nd2), max_depth.get(int(nd2), 0))
                                else:
                                    next_prio = self.node_min_dist.get(node, max_depth.get(node, 0))
                            else:
                                # No more edges: remove from values and continue
                                del values[node]
                                advanced = True
                                break
                        if next_prio is not None:
                            heapq.heappush(depth_heap, (next_prio, node))
                            advanced = True
                            break

                    if not advanced:
                        # We tried all dests in this edge; move to the next edge
                        edge_idx += 1
                        dest_idx = 0
                        if edge_idx < len(edges):
                            # Schedule node for next edge
                            ne = edges[edge_idx]
                            if ne[1]:  # has dests
                                nd3 = ne[1][0][0]
                                pr = self.node_min_dist.get(int(nd3), max_depth.get(int(nd3), 0))
                                values[node] = (gss_node, edge_idx, dest_idx)
                                heapq.heappush(depth_heap, (pr, node))
                            else:
                                # Edge without destinations, keep advancing
                                values[node] = (gss_node, edge_idx, dest_idx)
                                heapq.heappush(depth_heap, (self.node_min_dist.get(node, max_depth.get(node, 0)), node))
                        else:
                            # No more edges; done with this node
                            if node in values:
                                del values[node]
        stats.stop('get_mask.main_loop')
        stats.inc('get_mask.traversal.nodes_visited.unique', len(visited_nodes))

        stats.start('get_mask.final_conversion')
        # Convert internal mask back to original IDs
        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        stats.stop('get_mask.final_conversion')

        stats.stop('get_mask')
        return original_indices

    def finalize(self):
        """Called at the end of a benchmark run to perform any final actions, like printing stats."""
        print("\n--- Final Stats Report from Model ---")
        Stats.get().report()
