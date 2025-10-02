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

@dataclass
class DFAState:
    transitions: Dict[int, int] = field(default_factory=dict)
    finalizers: Set[int] = field(default_factory=set)
    possible_future_group_ids: Set[int] = field(default_factory=set)
    group_id_to_u8set: Dict[int, TerminalIdSet] = field(default_factory=dict)

@dataclass
class PyTokenizer:
    states: List[DFAState]
    start_state: int
    non_greedy_finalizers: Set[int]

    def __repr__(self) -> str:
        """Compact string representation of the tokenizer DFA."""

        def _format_transitions(transitions: Dict[int, int]) -> str:
            if not transitions:
                return ""

            # Group transitions by destination state
            grouped_by_dest = collections.defaultdict(list)
            for byte, dest in transitions.items():
                grouped_by_dest[dest].append(byte)

            parts = []
            for dest, bytes_list in grouped_by_dest.items():
                bytes_list.sort()
                ranges = []
                if not bytes_list:
                    continue

                start = bytes_list[0]
                end = bytes_list[0]

                for i in range(1, len(bytes_list)):
                    if bytes_list[i] == end + 1:
                        end = bytes_list[i]
                    else:
                        if start == end:
                            ranges.append(str(start))
                        else:
                            ranges.append(f"{start}-{end}")
                        start = bytes_list[i]
                        end = bytes_list[i]

                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")

                parts.append(f"[{', '.join(ranges)}] -> {dest}")

            return "Transitions: " + "; ".join(parts)

        lines = [
            f"PyTokenizer(states={len(self.states)}, start={self.start_state}, non_greedy={self.non_greedy_finalizers})"
        ]

        for i, state in enumerate(self.states):
            state_lines = [f"State {i}:"]

            trans_str = _format_transitions(state.transitions)
            if trans_str:
                state_lines.append(trans_str)

            if state.finalizers:
                state_lines.append(f"Finalizers: {sorted(list(state.finalizers))}")

            if state.possible_future_group_ids:
                state_lines.append(f"Future Groups: {sorted(list(state.possible_future_group_ids))}")

            if len(state_lines) > 1:
                lines.append(textwrap.indent("\n".join(state_lines), "  "))
            else:
                lines.append(f"  State {i}: (No transitions, finalizers, or future groups)")

        return "\n".join(lines)

    def execute_from_state(self, text: bytes, state_id: int) -> Tuple[Optional[int], List[Tuple[int, int]]]:
        current_state = state_id
        matches = {}
        done = False

        # Check for initial matches (epsilon)
        initial_state_data = self.states[current_state]
        for group_id in initial_state_data.finalizers:
            if group_id in self.non_greedy_finalizers:
                matches.setdefault(group_id, 0)
            else:
                matches[group_id] = 0

        for i, byte in enumerate(text):
            state_data = self.states[current_state]
            next_state = state_data.transitions.get(byte)

            if next_state is None:
                done = True
                break

            current_state = next_state

            # Update matches
            next_state_data = self.states[current_state]
            for group_id in next_state_data.finalizers:
                if group_id in self.non_greedy_finalizers:
                    matches.setdefault(group_id, i + 1)
                else:
                    matches[group_id] = i + 1

        end_state = None if done else current_state

        result_matches = [(gid, width) for gid, width in matches.items() if width > 0]

        return end_state, result_matches

    def tokens_accessible_from_state(self, state_id: int) -> List[int]:
        return list(self.states[state_id].possible_future_group_ids)

    def initial_state_id(self) -> int:
        return self.start_state

    def max_state(self) -> int:
        return len(self.states)

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
    tokenizer: PyTokenizer
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
        # Load tokenizer DFA from JSON
        dfa_data = data['tokenizer']['dfa']
        dfa_states = []
        for state_data in dfa_data['states']:
            transitions_json = state_data['transitions']
            # The 'data' field of the TrieMap JSON contains string keys for byte values.
            transitions = {int(k): v for k, v in transitions_json.get('data', {}).items()}

            group_id_to_u8set_json = state_data.get('group_id_to_u8set', [])
            group_id_to_u8set = {}
            for group_id, u8set_ranges in group_id_to_u8set_json:
                u8set_ranges2 = []
                for r in u8set_ranges:
                    if isinstance(r, list):
                        u8set_ranges2.append(r)
                    elif isinstance(r, int):
                        u8set_ranges2.append([r, r])
                    else:
                        raise ValueError(f"Invalid range value: {r}")
                group_id_to_u8set[group_id] = TerminalIdSet.from_ranges(u8set_ranges2)

            dfa_states.append(DFAState(
                transitions=transitions,
                finalizers=set(state_data['finalizers']),
                possible_future_group_ids=set(state_data['possible_future_group_ids']),
                group_id_to_u8set=group_id_to_u8set
            ))

        tokenizer = PyTokenizer(
            states=dfa_states,
            start_state=dfa_data['start_state'],
            non_greedy_finalizers=set(dfa_data['non_greedy_finalizers'])
        )
        tokenizer_max_state = tokenizer.max_state()
        tokenizer_initial_state = tokenizer.initial_state_id()

        # Load other things from FFI for now, as they are not part of this refactoring
        constraint = ffi.GrammarConstraint.from_json_string(s)
        glr_parser = constraint.glr_parser()
        ignore_terminal_id = glr_parser.ignore_terminal_id

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

        initial_acc = PyAcc(terminals_union={}, llm_mask=all_internal_llm_tokens_bitset)
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(parser_table.start_state_id)
        state = {tokenizer_initial_state: initial_gss}

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
        model.optimize_traversal()

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

        work_map: Dict[Tuple[int, int], GSS] = {}
        q = collections.deque()

        for tokenizer_sid, gss in current_state_for_processing.items():
            key = (0, tokenizer_sid)
            work_map[key] = gss
        q.extend(work_map.keys())

        while q:
            offset, tokenizer_sid = q.popleft()
            gss = work_map.pop((offset, tokenizer_sid))

            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tokenizer_sid)

            for terminal_id, width in matches:
                processed_gss = self._process_token(gss, terminal_id)
                # Immediate re-match disallow
                if end_state is not None:
                    accessible_terms = set(self.tokenizer.tokens_accessible_from_state(end_state))
                    if terminal_id in accessible_terms:
                        processed_gss = self._disallow_terminal_in_state(processed_gss, end_state, terminal_id)

                if not processed_gss.is_empty():
                    new_offset = offset + width
                    next_tokenizer_sid = self.tokenizer.initial_state_id()
                    if new_offset == len(token_bytes):
                        new_states[next_tokenizer_sid].append(processed_gss)
                    else:
                        key = (new_offset, next_tokenizer_sid)
                        existing_gss = work_map.get(key)
                        if existing_gss is None:
                            work_map[key] = processed_gss
                            q.append(key)
                        else:
                            work_map[key] = existing_gss.merge(processed_gss)

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

        if self.ignore_terminal_id is not None:
            if self.ignore_terminal_id == terminal_id:
                stats.inc(f'{p}.ignored_terminal')
                return gss

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
            if action is None:
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
            else:
                raise TypeError(f"Unknown action type: {type(action)}")

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

    def optimize_traversal(self) -> None:
        """
        Reorder edges and their destination lists to favor reaching end nodes ASAP.
        Heuristic:
          - Inner dests: sort by max_depth[dest] descending
          - Outer edges: sort by best dest depth (after inner sort) descending,
            then by pop asc as a tie-breaker, then by fewest dests (smaller fan-out) asc.
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
            # Sort edges by best dest depth desc, then pop asc, then fewest dests asc
            def _edge_key(edge):
                (pop, _llm_bv), dests = edge
                best = md.get(int(dests[0][0]), 0) if dests else -1
                return (-best, pop, len(dests))
            node.children.sort(key=_edge_key)
    def get_mask(self) -> LLMTokenSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.

        Changes for get_mask_only:
        - Initialize a per-accumulator LLM mask (PyAcc.llm_mask) BEFORE traversal by computing
          the forbidden terminals -> forbidden LLM tokens and taking the complement.
        - Consume terminals_union (set to HybridL2Bitset.all()) after initialization.
        - As we traverse edges, intersect llm_mask with the edge's LLM bitset using apply.
        - At end nodes, simply reduce acc over the GSS and union the llm_mask into the final.

        Performance improvements:
        - Subtract currently known final_mask from per-edge accumulator masks (aggressive zombie avoidance).
        - Resume within an edge (dest index) and spawn at most one child per pop to reach deep nodes sooner.
        - Replace Python loops in peek/filter with set-based intersections (C-accelerated).
        - Cache popn+masking per (node, pop) within a single pop cycle.
        - Early edge skip via union of dest state-sets and peeked states.

        Additional optimizations to smooth spikes:
        - Use isdisjoint checks instead of building temporary intersections when only emptiness is needed.
        - Edge-level precheck: skip an edge if its llm_bv has no overlap with the node's remaining mask
          (i.e., gss.reduce_acc().llm_mask minus final_mask), avoiding pop/apply work entirely.
        - Early union-vs-peek skip uses isdisjoint to avoid intermediate set allocations.
        - Compute intersection for a dest only for the first non-disjoint dest we spawn for that edge.
        """
        stats = Stats.get()
        stats.start('get_mask')
        state_map: Dict[int, GSS] = self.state
        stats.inc('get_mask.initial_tokenizer_states', len(state_map))

        all_ones: LLMTokenSet = self.all_internal_llm_tokens_bitset
        final_mask: LLMTokenSet = RangeSet.empty()
        final_mask_version: int = 0  # used to invalidate per-edge memo when final_mask updates

        # Helper to avoid temporary allocations when only emptiness is needed.
        def _rs_isdisjoint(a, b) -> bool:
            # Some RangeSet implementations provide isdisjoint; fall back if missing.
            if hasattr(a, "isdisjoint"):
                return a.isdisjoint(b)
            return a.intersection(b).is_empty()

        # We carry only GSS per node; the per-path LLM mask lives inside PyAcc.llm_mask.
        values: Dict[NodeID, GSS] = {}
        depth_heap: List[Tuple[int, NodeID]] = []  # Stores (-depth, node_id)
        enqueued_nodes: Set[NodeID] = set()
        # node -> (edge_idx, dest_idx, gss_id_tag)
        edge_cursor: Dict[NodeID, Tuple[int, int, int]] = {}

        hp, hpop = heapq.heappush, heapq.heappop
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
                pass

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

        cache = {}
        for sid, gss in state_map.items():
            stats.inc('get_mask.seeding.gss_loops')
            r: NodeID = roots_map[int(sid)]

            stats.start('get_mask.seeding.gss.apply')
            gss_initialized: GSS = gss.apply(initialize_acc, cache)
            stats.stop('get_mask.seeding.gss.apply')

            if r in values:
                stats.start('get_mask.seeding.gss.merge')
                values[r] = values[r].merge(gss_initialized)
                stats.stop('get_mask.seeding.gss.merge')
            else:
                values[r] = gss_initialized

            d: int = max_depth[r]
            if r not in enqueued_nodes:
                enqueued_nodes.add(r)
                hp(depth_heap, (-d, r))

        def enqueue(d: int, n: NodeID) -> None:
            stats.inc('get_mask.traversal.enqueues')
            if n not in enqueued_nodes:
                enqueued_nodes.add(n)
                hp(depth_heap, (-d, n))

        def dequeue() -> Tuple[int, int]:
            neg_d, n = hpop(depth_heap)
            return -neg_d, n
        stats.stop('get_mask.seeding')

        # Local caches
        # - union-of-dests state sets for edges (node_id, edge_idx) -> RangeSetStates
        edge_states_union_cache: Dict[Tuple[int, int], StateIDSet] = {}
        # - whether an edge's llm_bv has any overlap with the node's remaining llm mask
        edge_remaining_nonempty_cache: Dict[Tuple[int, int, int, int], bool] = {}
        # key: (node_id, edge_idx, final_mask_version, gss_id)

        # Main loop
        stats.start('get_mask.main_loop')
        max_depth_reached = 0
        visited_nodes = set()
        while depth_heap:
            depth, node = dequeue()
            max_depth_reached = max(max_depth_reached, depth)
            stats.inc('get_mask.traversal.depth_heap.pops')
            stats.inc('get_mask.traversal.nodes_processed')
            visited_nodes.add(node)
            gss_node: GSS = values.pop(node)
            gss_id = id(gss_node)
            enqueued_nodes.remove(node)
            stats.inc('get_mask.gss.at_node.accs.sum', len(getattr(gss_node, 'get_all_accs', lambda: [])()))

            # End-node handling: just union the allowed LLM tokens
            if is_end(node):
                stats.inc('get_mask.traversal.end_nodes')
                stats.start('get_mask.main_loop.end_node.reduce_acc')
                reduced_acc: Optional[PyAcc] = gss_node.reduce_acc()
                stats.stop('get_mask.main_loop.end_node.reduce_acc')
                if reduced_acc:
                    stats.start('get_mask.main_loop.end_node.final_mask_union')
                    # Union and mark version bump to invalidate per-edge memo using final_mask
                    final_mask = final_mask.union(reduced_acc.llm_mask)
                    final_mask_version += 1
                    stats.stop('get_mask.main_loop.end_node.final_mask_union')

            # Zombie traversal avoidance: if this node's edges can't possibly add
            # to the final mask given the current GSS mask, skip it.
            a_node = arena.get(node)
            node_llm_bv_union: LLMTokenSet = a_node.llm_bv_union if a_node else RangeSet.empty()
            if self._is_zombie_path(gss_node, node_llm_bv_union, final_mask, 'node'):
                continue

            # Compute the node's remaining mask once (gss.reduce_acc().llm_mask - final_mask).
            # This is used to skip edges whose llm_bv has no overlap with what the current GSS can still contribute.
            node_remaining_llm_mask: Optional[LLMTokenSet] = None
            gss_mask_acc = gss_node.reduce_acc()
            if gss_mask_acc:
                node_remaining_llm_mask = gss_mask_acc.llm_mask.difference(final_mask)

            # Traverse edges and propagate masks
            a_node = arena.get(node)
            edges = a_node.children if a_node else []
            stats.inc('get_mask.traversal.edge_blocks.sum', len(edges))
            stats.inc('get_mask.traversal.dests_blocks.sum', sum(len(dests) for _, dests in edges))

            # Resume where we left off: both edge index and dest index
            saved_cursor = edge_cursor.get(node)
            if saved_cursor and saved_cursor[2] == gss_id:
                start_edge_idx = saved_cursor[0]
                start_dest_idx = saved_cursor[1]
            else:
                start_edge_idx = 0
                start_dest_idx = 0

            spawned_any = False
            # Cache popn results for this node pop cycle (by pop amount)
            pop_cache: Dict[int, Tuple[Optional[GSS], Optional[List[int]], Optional[StateIDSet]]] = {}
            # Each entry: pop -> (popped_gss, peeked_list, peeked_as_rangeset)

            for edge_i in range(start_edge_idx, len(edges)):
                (pop, llm_bv), dests = edges[edge_i]

                stats.inc('get_mask.traversal.edges_traversed')

                # Edge-level precheck vs node's remaining mask (stronger than final_mask-only).
                # Skip if there is no possible contribution from this edge given the current GSS mask.
                if node_remaining_llm_mask is not None:
                    cache_key = (node, edge_i, final_mask_version, gss_id)
                    cached_nonempty = edge_remaining_nonempty_cache.get(cache_key, None)
                    if cached_nonempty is None:
                        # True means we should keep (non-disjoint), False means we can skip
                        keep_edge = not _rs_isdisjoint(llm_bv, node_remaining_llm_mask)
                        edge_remaining_nonempty_cache[cache_key] = keep_edge
                    else:
                        keep_edge = cached_nonempty
                    if not keep_edge:
                        stats.inc('get_mask.main_loop.edge.skipped_by_node_remaining')
                        start_dest_idx = 0
                        continue
                else:
                    # Fall back to final_mask-only precheck (rare, conservative)
                    if not final_mask.is_empty():
                        not_final = all_ones.difference(final_mask)
                        if _rs_isdisjoint(llm_bv, not_final):
                            stats.inc('get_mask.main_loop.edge.skipped_by_final_only')
                            start_dest_idx = 0
                            continue

                # popn + intersect-and-prune (with final_mask subtraction) per pop value (cache within this node pop)
                if pop not in pop_cache or pop_cache[pop][0] is None:
                    stats.inc(f'get_mask.traversal.edge_pop_val.{pop}')
                    stats.start('get_mask.main_loop.edge.popn')
                    popped: GSS = gss_node.popn(pop)
                    stats.stop('get_mask.main_loop.edge.popn')

                    if popped.is_empty():
                        stats.inc('get_mask.traversal.edge.popped_empty')
                        pop_cache[pop] = (None, None, None)
                        # This edge cannot produce children; resume at next edge
                        start_dest_idx = 0
                        continue

                    # Per-edge memo with final_mask_version in key to avoid stale cache
                    id_memo: Dict[Tuple[int, int], Optional[PyAcc]] = {}
                    def intersect_and_prune(acc: PyAcc) -> Optional[PyAcc]:
                        key = (id(acc), final_mask_version)
                        cached = id_memo.get(key, None) if key in id_memo else None
                        if key in id_memo:
                            return cached
                        p = 'get_mask.main_loop.edge.intersect_and_prune'
                        stats.inc(f'{p}.calls')
                        # Intersect with llm_bv
                        stats.start(f'{p}.intersection')
                        new_mask = acc.llm_mask.intersection(llm_bv)
                        stats.stop(f'{p}.intersection')
                        if new_mask.is_empty():
                            stats.inc(f'{p}.pruned_accs')
                            id_memo[key] = None
                            return None
                        # Subtract final_mask to avoid re-propagating already-known tokens
                        if not final_mask.is_empty():
                            # We intentionally do not record timing for difference to keep existing stats stable
                            reduced_mask = new_mask.difference(final_mask)
                            if reduced_mask.is_empty():
                                stats.inc(f'{p}.pruned_accs')
                                id_memo[key] = None
                                return None
                            new_mask = reduced_mask
                        result = PyAcc(terminals_union=acc.terminals_union, llm_mask=new_mask)
                        id_memo[key] = result
                        return result

                    stats.start('get_mask.main_loop.edge.apply_and_prune')
                    popped = popped.apply_and_prune(intersect_and_prune)
                    stats.stop('get_mask.main_loop.edge.apply_and_prune')

                    if popped.is_empty():
                        stats.inc('get_mask.traversal.edge.popped_pruned_empty')
                        pop_cache[pop] = (None, None, None)
                        start_dest_idx = 0
                        continue

                    # Prepare peek info, both as list and RangeSetStates for fast set ops
                    peeked_list = popped.peek()
                    peeked_rs = RangeSetStates.from_indices(peeked_list) if peeked_list else RangeSetStates.from_indices([])
                    pop_cache[pop] = (popped, peeked_list, peeked_rs)
                else:
                    popped, peeked_list, peeked_rs = pop_cache[pop]
                    if popped is None:
                        # Cached as impossible; resume at next edge
                        start_dest_idx = 0
                        continue

                # Early skip: if no peeked state belongs to any dest of this edge, skip edge entirely
                # Cache union of dest state-sets for this edge
                edge_key = (node, edge_i)
                union_states = edge_states_union_cache.get(edge_key)
                if union_states is None:
                    # Build lazily
                    tmp_union = None
                    for _dest_idx, state_bv in dests:
                        tmp_union = state_bv if tmp_union is None else tmp_union.union(state_bv)
                    union_states = tmp_union if tmp_union is not None else RangeSetStates.from_indices([])
                    edge_states_union_cache[edge_key] = union_states

                # If no overlap between current heads and this edge's dest states, skip
                # Use isdisjoint to avoid allocating intersection sets
                fast_inter_empty = _rs_isdisjoint(union_states, peeked_rs)
                if fast_inter_empty:
                    start_dest_idx = 0
                    continue

                # Traverse dests; spawn at most a single child per pop cycle to push end discovery earlier
                child_spawned = False
                dests_len = len(dests)
                dest_range = range(start_dest_idx if edge_i == start_edge_idx else 0, dests_len)
                for dest_j in dest_range:
                    dest_idx, state_bv = dests[dest_j]
                    stats.inc('get_mask.traversal.dests_traversed')

                    # Check disjointness first (no allocation), compute intersection only when needed
                    stats.start('get_mask.main_loop.edge.peek_and_filter')
                    disjoint = _rs_isdisjoint(state_bv, peeked_rs)
                    if disjoint:
                        stats.stop('get_mask.main_loop.edge.peek_and_filter')
                        continue

                    # Build child GSS (fast-path if single sid)
                    keep_rs = state_bv.intersection(peeked_rs)
                    indices = list(keep_rs.iter_indices())
                    stats.stop('get_mask.main_loop.edge.peek_and_filter')

                    stats.start('get_mask.main_loop.edge.isolate_many')
                    if len(indices) == 1:
                        child_gss = popped.isolate(indices[0])
                    else:
                        child_gss = popped.isolate_many(indices)
                    stats.stop('get_mask.main_loop.edge.isolate_many')

                    if child_gss.is_empty():
                        stats.inc('get_mask.traversal.edge.child_gss_pruned_empty')
                        # continue searching other dests
                        continue

                    d: NodeID = int(dest_idx)
                    if d in values:
                        stats.inc('get_mask.traversal.edge.gss_merges')
                        stats.start('get_mask.main_loop.edge.gss_merge')
                        values[d] = values[d].merge(child_gss)
                        stats.stop('get_mask.main_loop.edge.gss_merge')
                    else:
                        values[d] = child_gss
                    enqueue(max_depth[d], d)
                    child_spawned = True

                    # Save progress and re-enqueue this node to process the next dest (or next edge) later
                    merged_for_requeue = values[node].merge(gss_node) if node in values else gss_node
                    values[node] = merged_for_requeue
                    next_edge_i = edge_i
                    next_dest_j = dest_j + 1
                    if next_dest_j >= dests_len:
                        next_edge_i = edge_i + 1
                        next_dest_j = 0
                    edge_cursor[node] = (next_edge_i, next_dest_j, id(merged_for_requeue))
                    enqueue(max_depth[node], node)
                    spawned_any = True
                    break  # spawn only one child per pop

                # If we have spawned one child for this edge, stop processing further edges now.
                if child_spawned:
                    break

                # Reset dest start index if we move to next edge
                start_dest_idx = 0

            # If we didn't spawn any child across remaining edges, this node is done
            if not spawned_any:
                edge_cursor.pop(node, None)
        stats.stop('get_mask.main_loop')
        stats.inc('get_mask.traversal.max_depth_reached', max_depth_reached)
        stats.inc('get_mask.traversal.nodes_visited.unique', len(visited_nodes))

        stats.start('get_mask.final_conversion')
        # Convert internal mask back to original IDs
        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        stats.stop('get_mask.final_conversion')

        stats.stop('get_mask')
        stats.report()
        stats.reset()
        return original_indices

    def finalize(self):
        """Called at the end of a benchmark run to perform any final actions, like printing stats."""
        print("\n--- Final Stats Report from Model ---")
        Stats.get().report()
