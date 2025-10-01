from __future__ import annotations

import json
import heapq
import collections
import os
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

        # Reorder edges/dests to prioritize reaching end nodes quickly
        # model.optimize_traversal()

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
        return self.arena[node].clean_end

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

    def _is_zombie_path(self, gss: GSS, path_token_union: LLMTokenSet, final_mask: LLMTokenSet) -> bool:
        """
        Checks if a given traversal path is a "zombie" path, meaning it cannot
        contribute any new tokens to the final_mask.
        """
        # We only care about tokens that are not yet in the final mask.
        potential_new_tokens = path_token_union.difference(final_mask)
        if potential_new_tokens.is_empty():
            return True

        gss_mask_acc = gss.reduce_acc()
        if gss_mask_acc and gss_mask_acc.llm_mask.isdisjoint(potential_new_tokens):
            return True

        return False

    def optimize_traversal(self) -> None:
        """
        Reorder edges and their destination lists to favor reaching end nodes ASAP.
        Heuristic:
          - Inner dests: sort by max_depth[dest] descending
          - Outer edges: sort by best dest depth (after inner sort) descending,
            then by pop asc as a tie-breaker.
        """
        stats = Stats.get()
        stats.start('optimize_traversal')
        md = self.max_depth
        for node in self.arena.values():
            if not node.children:
                continue
            # Sort inner dests first (so the first dest has the highest depth)
            for _edge_key, dests in node.children:
                dests.sort(key=lambda item: md.get(int(item[0]), 0), reverse=True)
            # Sort edges by best dest depth desc, then pop asc
            def _edge_key(edge):
                (pop, _llm_bv), dests = edge
                best = md.get(int(dests[0][0]), 0) if dests else -1
                return (-best, pop)
            node.children.sort(key=_edge_key)
    def get_mask(self) -> LLMTokenSet:
        m1 = self.get_mask1()
        m2 = self.get_mask2()
        assert m1 == m2, f"Mask mismatch: {m1} vs {m2}"
        return m1
    def get_mask1(self) -> LLMTokenSet:
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

        all_ones: LLMTokenSet = self.all_internal_llm_tokens_bitset
        final_mask: LLMTokenSet = RangeSet.empty()

        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask.
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
        @_acc_memoize(use_value_cache=False)
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask: LLMTokenSet = RangeSet.empty()
            disallowed_map = acc.terminals_union

            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]

                for terminal_id in disallowed_terminals.iter_indices():
                    if terminal_id in terminals_to_llm:
                        disallowed_llm_mask |= terminals_to_llm[terminal_id]

            allowed_mask = all_ones.difference(disallowed_llm_mask)

            return PyAcc(
                terminals_union={},  # consume
                llm_mask=allowed_mask,
            )

        cache = {}
        for sid, gss in state_map.items():
            r: NodeID = roots_map[int(sid)]

            gss_initialized: GSS = gss.apply(initialize_acc, cache)

            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            if r not in enqueued_nodes:
                enqueued_nodes.add(r)
                hp(depth_heap, (-d, r))

        # Main loop
        while depth_heap:
            neg_depth, node = hpop(depth_heap)
            gss_node: GSS = values.pop(node)
            enqueued_nodes.remove(node)

            # End-node handling: just union the allowed LLM tokens
            if is_end(node):
                reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                if reduced_acc:
                    final_mask |= reduced_acc.llm_mask

            # Zombie traversal avoidance: if this node's edges can't possibly add
            # to the final mask given the current GSS mask, skip it.
            a_node = arena.get(node)
            node_llm_bv_union: LLMTokenSet = a_node.llm_bv_union if a_node else RangeSet.empty()
            if self._is_zombie_path(gss_node, node_llm_bv_union, final_mask):
                continue

            # Traverse edges and propagate masks
            edges = a_node.children if a_node else []
            for (pop, llm_bv), dests in edges:
                popped_gss: GSS = gss_node.popn(pop)
                if popped_gss.is_empty():
                    continue

                @_acc_memoize(use_value_cache=False)
                def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                    # NOTE: get_mask1 uses the edge's llm_bv directly.
                    new_mask = acc.llm_mask.intersection(llm_bv)
                    if new_mask.is_empty():
                        return None
                    return PyAcc(terminals_union=acc.terminals_union, llm_mask=new_mask)

                pruned_gss = popped_gss.apply_and_prune(intersect_and_prune)
                if pruned_gss.is_empty():
                    continue

                if pruned_gss.reduce_acc().is_empty():
                    continue

                peeked = pruned_gss.peek()
                for dest_idx, state_bv in dests:
                    values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]
                    if not values_to_keep:
                        continue

                    child_gss = pruned_gss.isolate_many(values_to_keep)
                    if child_gss.is_empty():
                        continue

                    if child_gss.reduce_acc().is_empty():
                        continue

                    d = int(dest_idx)
                    if d in values:
                        values[d] = values[d].merge(child_gss)
                    else:
                        values[d] = child_gss
                    if d not in enqueued_nodes:
                        enqueued_nodes.add(d)
                        hp(depth_heap, (-max_depth[d], d))

        # Convert internal mask back to original IDs
        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        return original_indices
    def get_mask2(self) -> LLMTokenSet:
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
        @_acc_memoize(use_value_cache=False)
        def initialize_acc(acc: PyAcc) -> PyAcc:
            # Compute allowed LLM tokens from disallowed terminals for this accumulator
            disallowed_llm_mask = RangeSet.empty()
            disallowed_map = acc.terminals_union

            for tsid, disallowed_terminals in disallowed_map.items():
                if tsid > max_state or tsid not in pmc:
                    continue
                terminals_to_llm = pmc[tsid]
                for terminal_id in disallowed_terminals.iter_indices():
                    if terminal_id in terminals_to_llm:
                        disallowed_llm_mask |= terminals_to_llm[terminal_id]

            allowed_mask = all_ones.difference(disallowed_llm_mask)

            return PyAcc(
                terminals_union={},  # consume
                llm_mask=allowed_mask,
            )

        cache = {}
        for sid, gss in state_map.items():
            r: NodeID = roots_map[int(sid)]
            gss_initialized: GSS = gss.apply(initialize_acc, cache)
            if r in values:
                values[r] = values[r].merge(gss_initialized)
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            if r not in enqueued_nodes:
                enqueued_nodes.add(r)
                hp(depth_heap, (-d, r))

        # Main loop
        while depth_heap:
            neg_depth, node = hpop(depth_heap)
            gss_node: GSS = values.pop(node)
            enqueued_nodes.remove(node)

            # End-node handling: just union the allowed LLM tokens
            if is_end(node):
                reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                if reduced_acc:
                    final_mask |= reduced_acc.llm_mask

            # Zombie traversal avoidance
            a_node = arena.get(node)
            node_llm_bv_union: LLMTokenSet = a_node.llm_bv_union if a_node else RangeSet.empty()
            if self._is_zombie_path(gss_node, node_llm_bv_union, final_mask):
                continue

            # Traverse edges and propagate masks
            edges = a_node.children if a_node else []
            for (pop, llm_bv_from_edge), dests in edges:
                # NOTE: get_mask2 prunes the edge's bitset with the final_mask.
                llm_bv = llm_bv_from_edge.difference(final_mask)
                if llm_bv.is_empty():
                    continue

                popped_gss: GSS = gss_node.popn(pop)
                if popped_gss.is_empty():
                    continue

                @_acc_memoize(use_value_cache=False)
                def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                    new_mask = acc.llm_mask.intersection(llm_bv)
                    if new_mask.is_empty():
                        return None
                    return PyAcc(terminals_union=acc.terminals_union, llm_mask=new_mask)

                pruned_gss = popped_gss.apply_and_prune(intersect_and_prune)
                if pruned_gss.is_empty():
                    continue

                if pruned_gss.reduce_acc().is_empty():
                    continue

                peeked = pruned_gss.peek()
                for dest_idx, state_bv in dests:
                    values_to_keep = [sid for sid in peeked if state_bv.contains(sid)]
                    if not values_to_keep:
                        continue

                    child_gss = pruned_gss.isolate_many(values_to_keep)
                    if child_gss.is_empty():
                        continue

                    if child_gss.reduce_acc().is_empty():
                        continue

                    d = int(dest_idx)
                    if d in values:
                        values[d] = values[d].merge(child_gss)
                    else:
                        values[d] = child_gss
                    if d not in enqueued_nodes:
                        enqueued_nodes.add(d)
                        hp(depth_heap, (-max_depth[d], d))

        # Convert internal mask back to original IDs
        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        return original_indices

    def finalize(self):
        """Called at the end of a benchmark run to perform any final actions, like printing stats."""
        print("\n--- Final Stats Report from Model ---")
        Stats.get().report()
