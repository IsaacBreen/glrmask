from __future__ import annotations

import collections
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set, Generator, Any
import json
import types

import _sep1 as ffi
from tqdm import tqdm

from ..stats import Stats
from python.gss_tester.implementations.leveled_impl import LeveledGSS as GSS
from ..common_interface import GraphProvider
from ..range_set import FFIRangeSet as RangeSet
from ..range_set import SetRangeSet as RangeSetOut
from ..range_set import SetRangeSet as RangeSetStates

# Type Aliases
NodeID = int
LLMTokenSet = RangeSet
StateIDSet = RangeSetStates
TerminalIdSet = RangeSet

# --- Monkey-patch RangeSet to collect stats on union/intersection ---
_original_rangeset_union = RangeSet.union
_original_rangeset_intersection = RangeSet.intersection
_original_rangeset_isdisjoint = RangeSet.isdisjoint
_original_rangeset_len = RangeSet.__len__

def _patched_union(self, other: "RangeSet") -> "RangeSet":
    stats = Stats.get()
    stats.inc('bitset.union.calls')
    stats.start('bitset.union.time')
    result = _original_rangeset_union(self, other)
    stats.stop('bitset.union.time')
    return result

def _patched_intersection(self, other: "RangeSet") -> "RangeSet":
    stats = Stats.get()
    stats.inc('bitset.intersection.calls')
    stats.start('bitset.intersection.time')
    result = _original_rangeset_intersection(self, other)
    stats.stop('bitset.intersection.time')
    return result

def _patched_isdisjoint(self, other: "RangeSet") -> bool:
    stats = Stats.get()
    stats.inc('bitset.isdisjoint.calls')
    stats.start('bitset.isdisjoint.time')
    result = _original_rangeset_isdisjoint(self, other)
    stats.stop('bitset.isdisjoint.time')
    return result

def _patched_len(self) -> int:
    stats = Stats.get()
    stats.inc('bitset.len.calls')
    stats.start('bitset.len.time')
    result = _original_rangeset_len(self)
    stats.stop('bitset.len.time')
    return result

# --- Monkey-patch RangeSetStates ---
_original_rangesetstates_isdisjoint = RangeSetStates.isdisjoint

def _patched_states_isdisjoint(self, other: "RangeSetStates") -> bool:
    stats = Stats.get()
    stats.inc('bitset.states.isdisjoint.calls')
    stats.start('bitset.states.isdisjoint.time')
    result = _original_rangesetstates_isdisjoint(self, other)
    stats.stop('bitset.states.isdisjoint.time')
    return result

# Apply the patches
RangeSet.union = _patched_union
RangeSet.intersection = _patched_intersection
RangeSet.isdisjoint = _patched_isdisjoint
RangeSet.__len__ = _patched_len
RangeSetStates.isdisjoint = _patched_states_isdisjoint
# --- End of monkey-patch ---

def _acc_memoize(stats_prefix: Optional[str] = None, use_value_cache: bool = True):
    def decorator(fn):
        id_memo = {}
        val_memo = {}
        def wrapper(acc):
            acc_id = id(acc)
            if acc_id in id_memo:
                if stats_prefix:
                    Stats.get().inc(f'{stats_prefix}.memo_hits')
                return id_memo[acc_id]

            if use_value_cache:
                cached = val_memo.get(acc)
                if cached is not None:
                    id_memo[acc_id] = cached
                    if stats_prefix:
                        Stats.get().inc(f'{stats_prefix}.memo_hits')
                    return cached

            result = fn(acc)
            id_memo[acc_id] = result
            if use_value_cache and result is not None:
                val_memo[acc] = result
            return result
        wrapper._acc_memo_size = lambda: len(id_memo)
        return wrapper
    return decorator


@dataclass(frozen=True)
class DFAState:
    transitions: Dict[int, int]
    finalizers: Set[int]
    possible_future_group_ids: Set[int]


@dataclass(frozen=True)
class PyTokenizer:
    states: List[DFAState]
    start_state: int
    non_greedy_finalizers: Set[int]

    def execute_from_state(self, text: bytes, state_id: int) -> Tuple[Optional[int], List[Tuple[int, int]]]:
        current_state = state_id
        matches = {}

        for group_id in self.states[current_state].finalizers:
            if group_id in self.non_greedy_finalizers:
                matches.setdefault(group_id, 0)
            else:
                matches[group_id] = 0

        for i, byte in enumerate(text):
            state_data = self.states[current_state]
            next_state = state_data.transitions.get(byte)
            if next_state is None:
                return None, [(gid, width) for gid, width in matches.items() if width > 0]
            current_state = next_state
            for group_id in self.states[current_state].finalizers:
                if group_id in self.non_greedy_finalizers:
                    matches.setdefault(group_id, i + 1)
                else:
                    matches[group_id] = i + 1

        return current_state, [(gid, width) for gid, width in matches.items() if width > 0]

    def initial_state_id(self) -> int:
        return self.start_state


@dataclass(frozen=True)
class Reduce:
    nonterminal_id: int
    len: int
    production_ids: Tuple[int, ...]


@dataclass(frozen=True)
class Split:
    shift: Optional[int]
    reduces: Dict[int, Dict[int, Tuple[int, ...]]]


Action = Union[int, Reduce, Split]


@dataclass
class Row:
    actions: Dict[int, Action] = field(default_factory=dict)
    gotos: Dict[int, int] = field(default_factory=dict)


@dataclass
class ParserTable:
    start_state_id: int
    table: Dict[int, Row]


@dataclass(frozen=True, eq=False)
class PyAcc:
    terminals_union: Dict[int, TerminalIdSet]
    llm_mask: LLMTokenSet

    def __eq__(self, other):
        if not isinstance(other, PyAcc):
            return NotImplemented
        return self.llm_mask == other.llm_mask and self.terminals_union == other.terminals_union

    def __hash__(self):
        return hash((len(self.terminals_union), self.llm_mask))

    def merge(self, other: "PyAcc") -> "PyAcc":
        new_terminals_union = self.terminals_union.copy()
        for k, v in other.terminals_union.items():
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
class DWAState:
    # Maps label (int) -> (target_state_id, weight)
    transitions: Dict[int, Tuple[int, LLMTokenSet]]
    state_weight: Optional[LLMTokenSet]
    final_weight: Optional[LLMTokenSet]


@dataclass
class DWA:
    states: List[DWAState]
    start_state: int


DEFAULT_TRANSITION_SYMBOL = 32767


@dataclass
class Model(GraphProvider):
    stats = Stats.get()
    stats.add_group('get_mask')
    stats.add_group('commit')

    dwa: DWA
    parser_table: ParserTable
    tokenizer: PyTokenizer
    tokenizer_initial_state: int
    possible_matches_cache: Dict[int, Dict[int, LLMTokenSet]]
    id_to_token: Dict[int, bytes]
    internal_to_original_map: Dict[int, RangeSetOut]
    all_internal_llm_tokens_bitset: LLMTokenSet
    ignore_terminal_id: Optional[int]
    original_to_dummy_map: Dict[int, int]
    state: Dict[int, GSS]
    suppress_stats_report: bool = False
    last_get_mask_cost: int = 0
    last_get_mask_metrics: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        Stats.get().reset()
        data = json.loads(s)
        dumps, bs_from_json = json.dumps, ffi.Bitset.from_json_string

        vocab = data['precompute_vocab']
        all_internal_llm_tokens_bitset = RangeSet.from_ranges([(0, vocab['internal_max_llm_token'])])

        def _parse_weight(w_json: Any) -> LLMTokenSet:
            if w_json is None:
                return all_internal_llm_tokens_bitset
            # Handle SimpleBitset serialization (assuming it might be wrapped or direct RangeSetBlaze)
            # If it's a dict with "rsb", use that. If it's a list, assume ranges or integers.
            # For now, assuming it matches what RangeSet.from_json_string expects or is a list of ranges.
            # If it's from Rust's RangeSetBlaze serde, it might be a list of integers.
            # But here we assume it's compatible with our RangeSet loading or we adapt.
            # Given we don't have the exact JSON format of SimpleBitset here, we try standard approaches.
            try:
                if isinstance(w_json, dict) and 'inner' in w_json:
                    # HybridBitset style
                    return RangeSet.from_ranges(bs_from_json(dumps(w_json)).to_ranges())
                if isinstance(w_json, dict) and 'rsb' in w_json:
                    # SimpleBitset style?
                    return RangeSet.from_ranges(bs_from_json(dumps(w_json['rsb'])).to_ranges())
                # Fallback: try parsing as HybridBitset directly
                return RangeSet.from_ranges(bs_from_json(dumps(w_json)).to_ranges())
            except Exception:
                # If it's a list of integers (RangeSetBlaze default serde), we might need to handle it.
                # But let's assume the provided JSON uses the compatible format.
                return all_internal_llm_tokens_bitset

        # Load DWA
        dwa_json = data['precomputed4']
        states_data = dwa_json['states']
        dwa_states = []
        for s in states_data:
            trans_map = dict(s['transitions'])  # label -> target
            weights_map = dict(s['trans_weights'])  # label -> weight

            merged_trans = {}
            for label_str, target in trans_map.items():
                label = int(label_str)
                w_json = weights_map.get(label_str)
                weight = _parse_weight(w_json) if w_json is not None else all_internal_llm_tokens_bitset
                merged_trans[label] = (target, weight)

            st_weight = _parse_weight(s.get('state_weight')) if s.get('state_weight') is not None else None
            fin_weight = _parse_weight(s.get('final_weight')) if s.get('final_weight') is not None else None

            dwa_states.append(DWAState(merged_trans, st_weight, fin_weight))

        start_state = dwa_json['start_state']
        dwa = DWA(dwa_states, start_state)

        # Tokenizer
        dfa_data = data['tokenizer']['dfa']
        dfa_states = [DFAState(
            transitions={int(k): v for k, v in s['transitions'].get('data', {}).items()},
            finalizers=set(s['finalizers']),
            possible_future_group_ids=set(s['possible_future_group_ids'])
        ) for s in dfa_data['states']]
        tokenizer = PyTokenizer(dfa_states, dfa_data['start_state'], set(dfa_data['non_greedy_finalizers']))

        # Parser Table
        parser_data = data['parser']
        py_table: Dict[int, Row] = {}
        for state_id_str, row_data in parser_data['stage_7_table']:
            state_id, py_row = int(state_id_str), Row()
            for term_id_str, action_data in row_data['shifts_and_reduces_full']:
                term_id, variant = int(term_id_str), action_data['variant']
                if variant == 'Shift':
                    py_row.actions[term_id] = action_data['state_id']
                elif variant == 'Reduce':
                    py_row.actions[term_id] = Reduce(action_data['nonterminal_id'], action_data['len'], tuple(sorted(action_data['production_ids'])))
                elif variant == 'Split':
                    reduces = {int(l): {int(n): tuple(sorted(p)) for n, p in nd} for l, nd in action_data['reduces']}
                    py_row.actions[term_id] = Split(action_data['shift'], reduces)
            py_row.gotos = {int(nt): goto['state_id'] for nt, goto in row_data['gotos'] if goto['state_id'] is not None}
            py_table[state_id] = py_row
        parser_table = ParserTable(parser_data['start_state_id'], py_table)

        # Misc data
        pmc_json = data['possible_matches_precompute1']
        possible_matches_cache = {}
        for tsid_json, term_map_json in pmc_json:
            tsid = int(tsid_json)
            term_map = {}
            for term_id_json, bv_json in term_map_json:
                term_id = int(term_id_json)
                bv = RangeSet.from_ranges(bs_from_json(dumps(bv_json)).to_ranges())
                term_map[term_id] = bv
            possible_matches_cache[tsid] = term_map

        original_to_dummy_map_json = data.get('original_to_dummy_map', [])
        original_to_dummy_map = {int(k): int(v) for k, v in original_to_dummy_map_json}

        # Initial state
        initial_acc = PyAcc({}, all_internal_llm_tokens_bitset)
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(parser_table.start_state_id)

        print("Precompute4 Model loaded.")
        print(f"  DWA states: {len(dwa.states)}")
        for i, st in enumerate(dwa.states):
            print(f"  DWA state {i}:")
            print(f"    state weight: {st.state_weight}")
            print(f"    final weight: {st.final_weight}")
            print(f"    transitions:")
            for label, (target, weight) in st.transitions.items():
                print(f"      {label} -> {target} ({weight})")
        print(f"  Tokenizer initial state: {tokenizer.initial_state_id()}")
        print(f"  Parser table start state: {parser_table.start_state_id}")
        print(f"  Original to dummy map: {original_to_dummy_map}")
        print(f"  Possible matches cache: {len(possible_matches_cache)}")

        model = Model(
            dwa=dwa,
            parser_table=parser_table,
            tokenizer=tokenizer,
            tokenizer_initial_state=tokenizer.initial_state_id(),
            possible_matches_cache=possible_matches_cache,
            id_to_token={v: bytes(k) for k, v in data['llm_token_map']},
            internal_to_original_map={int(k): RangeSetOut.from_indices(v) for k, v in dict(vocab['internal_to_original']).items()},
            all_internal_llm_tokens_bitset=all_internal_llm_tokens_bitset,
            ignore_terminal_id=parser_data.get('ignore_terminal_id'),
            original_to_dummy_map=original_to_dummy_map,
            state={tokenizer.initial_state_id(): initial_gss},
        )

        return model

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        term_rs = RangeSet.from_indices([terminal_id])
        @_acc_memoize(use_value_cache=False)
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current = acc.terminals_union.get(state_id, RangeSet.empty())
            if current.contains(terminal_id):
                return acc
            new_map = acc.terminals_union.copy()
            new_map[state_id] = current.union(term_rs)
            return PyAcc(new_map, acc.llm_mask)
        return gss.apply(apply_disallow)

    def commit(self, token_id: int):
        token_bytes = self.id_to_token[token_id]
        terminals_map, state_map = {}, {}
        for tsid in self.state:
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tsid)
            if end_state is not None:
                state_map[tsid] = end_state
            terminals_map[tsid] = RangeSet.from_indices([m[0] for m in matches])

        @_acc_memoize()
        def mutator(acc: PyAcc) -> Optional[PyAcc]:
            for tsid, matched in terminals_map.items():
                if acc.terminals_union.get(tsid, RangeSet.empty()).intersects(matched):
                    return None
            new_bvs = collections.defaultdict(RangeSet.empty)
            for old, new in state_map.items():
                if old in acc.terminals_union:
                    new_bvs[new] |= acc.terminals_union[old]
            return PyAcc(dict(new_bvs), acc.llm_mask)

        cache = {}
        current = {tsid: g.apply_and_prune(mutator, cache) for tsid, g in self.state.items()}
        current = {tsid: g for tsid, g in current.items() if not g.is_empty()}

        new_states = collections.defaultdict(list)
        work = {(0, tsid): gss for tsid, gss in current.items()}
        q = collections.deque(work.keys())

        while q:
            offset, tsid = q.popleft()
            gss = work.pop((offset, tsid))
            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tsid)

            for term_id, width in matches:
                proc_gss = gss
                dummy_id = self.original_to_dummy_map.get(term_id)
                if dummy_id is not None:
                    proc_gss = self._process_token(proc_gss, dummy_id)
                proc_gss = self._process_token(proc_gss, term_id)

                if end_state is not None and term_id in self.tokenizer.states[end_state].possible_future_group_ids:
                    proc_gss = self._disallow_terminal_in_state(proc_gss, end_state, term_id)
                if not proc_gss.is_empty():
                    new_offset, next_tsid = offset + width, self.tokenizer.initial_state_id()
                    if new_offset == len(token_bytes):
                        new_states[next_tsid].append(proc_gss)
                    else:
                        key = (new_offset, next_tsid)
                        if key in work:
                            work[key] = work[key].merge(proc_gss)
                        else:
                            work[key] = proc_gss
                            q.append(key)
            if end_state is not None:
                new_states[end_state].append(gss)

        merged = {sid: GSS.merge_many(gssl) for sid, gssl in new_states.items() if gssl}
        self.state = {sid: g for sid, g in merged.items() if not g.is_empty()}

    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        if self.ignore_terminal_id == terminal_id:
            return gss

        heads_by_state = collections.defaultdict(list)
        for state_id in gss.peek():
            heads_by_state[state_id].append(gss.isolate(state_id))

        shifted = []
        while heads_by_state:
            state_id, gss_list = heads_by_state.popitem()
            state_gss = GSS.merge_many(gss_list)
            row = self.parser_table.table.get(state_id)
            if not row:
                continue
            action = row.actions.get(terminal_id)
            if action is None:
                continue

            if isinstance(action, int):
                shifted.append(state_gss.push(action))
            elif isinstance(action, Reduce):
                popped = state_gss.popn(action.len)
                for from_id in popped.peek():
                    goto_id = self.parser_table.table[from_id].gotos[action.nonterminal_id]
                    heads_by_state[goto_id].append(popped.isolate(from_id).push(goto_id))
            elif isinstance(action, Split):
                if action.shift is not None:
                    shifted.append(state_gss.push(action.shift))
                for length, nts in action.reduces.items():
                    popped = state_gss.popn(length)
                    for nt_id in nts:
                        for from_id in popped.peek():
                            goto_id = self.parser_table.table[from_id].gotos[nt_id]
                            heads_by_state[goto_id].append(popped.isolate(from_id).push(goto_id))
        return GSS.merge_many(shifted)

    def _apply_weight(self, gss: GSS, weight: LLMTokenSet) -> GSS:
        @_acc_memoize(use_value_cache=False)
        def apply_fn(acc: PyAcc) -> Optional[PyAcc]:
            new_mask = acc.llm_mask.intersection(weight)
            if new_mask.is_empty():
                return None
            return PyAcc(acc.terminals_union, new_mask)
        return gss.apply_and_prune(apply_fn)

    def _merge_into_queue(self, queue: Dict[int, Dict[int, GSS]], gss: GSS, target_state_id: int):
        depth = gss.max_depth()
        if target_state_id in queue[depth]:
            queue[depth][target_state_id] = queue[depth][target_state_id].merge(gss)
        else:
            queue[depth][target_state_id] = gss

    def get_mask(self) -> Union[RangeSetOut, Dict]:
        print(f"GSSs")
        for dwa_id, gss in self.state.items():
            print(f"  DWA state {dwa_id}:")
            print(f"    {gss}")
        stats = Stats.get()
        stats.start('get_mask')
        
        all_ones = self.all_internal_llm_tokens_bitset
        final_mask = RangeSet.empty()
        
        # Queue: depth -> {dwa_state_id: GSS}
        queue: Dict[int, Dict[int, GSS]] = collections.defaultdict(dict)
        
        # 1. Seed initial states
        start_node = self.dwa.states[self.dwa.start_state]

        @_acc_memoize(use_value_cache=False)
        def initialize_acc(acc: PyAcc) -> PyAcc:
            disallowed = RangeSet.empty()
            for tsid, terms in acc.terminals_union.items():
                if tsid in self.possible_matches_cache:
                    term_map = self.possible_matches_cache[tsid]
                    for term_id in terms.iter_indices():
                        if term_id in term_map:
                            disallowed |= term_map[term_id]
            return PyAcc({}, all_ones.difference(disallowed))

        init_cache = {}
        for sid, gss in self.state.items():
            gss_init = gss.apply(initialize_acc, init_cache)
            if gss_init.is_empty():
                continue
                
            # Transition on tokenizer state id
            target = start_node.transitions.get(sid)
            if target is not None:
                target_state_id, weight = target
                gss_next = self._apply_weight(gss_init, weight)
                if not gss_next.is_empty():
                    self._merge_into_queue(queue, gss_next, target_state_id)

        print(f"Initial items")
        for depth, states in queue.items():
            for dwa_id, gss in states.items():
                print(f"  Depth {depth}, DWA state {dwa_id}:")
                print(f"    {gss}")

        # 2. Main worklist loop
        while queue:
            max_depth = max(queue.keys())
            states_at_depth = queue.pop(max_depth)
            
            for dwa_id, gss in states_at_depth.items():
                dwa_state = self.dwa.states[dwa_id]

                # Check for final state
                if dwa_state.final_weight is not None:
                    print("Checking final state at DWA state", dwa_id)
                    acc = gss.reduce_acc()
                    print("  Accumulator at final state:", acc)
                    if acc:
                        final_tokens = acc.llm_mask.intersection(dwa_state.final_weight)
                        print("  Final tokens after intersection:", final_tokens)
                        if not final_tokens.is_empty():
                            print("  Merging into final mask.")
                            final_mask |= final_tokens
                            
                # Process transitions
                peeked = gss.peek()
                if not peeked:
                    continue
                    
                for edge in peeked:
                    parser_state_id = edge
                    
                    # 1. Specific transition
                    t = dwa_state.transitions.get(parser_state_id)
                    if t:
                        target_id, weight = t
                        isolated = gss.isolate(edge)
                        popped = isolated.pop()
                        if not popped.is_empty():
                            final_gss = self._apply_weight(popped, weight)
                            if not final_gss.is_empty():
                                self._merge_into_queue(queue, final_gss, target_id)
                                
                    # 2. Default transition
                    t_def = dwa_state.transitions.get(DEFAULT_TRANSITION_SYMBOL)
                    if t_def:
                        target_id, weight = t_def
                        isolated = gss.isolate(edge)
                        popped = isolated.pop()
                        if not popped.is_empty():
                            final_gss = self._apply_weight(popped, weight)
                            if not final_gss.is_empty():
                                self._merge_into_queue(queue, final_gss, target_id)

        stats.start('get_mask.teardown.final_conversion')
        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        stats.stop('get_mask.teardown.final_conversion')
        stats.stop('get_mask')

        if not self.suppress_stats_report:
            Stats.get().report(sort_by='alpha')

        print(f"self.internal_to_original_map: {self.internal_to_original_map}")
        print(f"final mask: {final_mask}")
        print(f"get_mask() returning {original_indices}")
        return original_indices

    def finalize(self):
        print("\n--- Final Stats Report from Model ---")
        Stats.get().report(sort_by='alpha')
