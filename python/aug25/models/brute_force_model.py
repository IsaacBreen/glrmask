from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set

import _sep1 as ffi
from tqdm import tqdm

from python.gss_tester.implementations.leveled_rs_impl import LeveledRSGSS as GSS
from ..common_interface import GraphProvider
from ..stats import Stats
from ..range_set import FFIRangeSet as RangeSet
from ..range_set import SetRangeSet as RangeSetOut

# Type Aliases
LLMTokenSet = RangeSet
TerminalIdSet = RangeSet

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
        done = False

        # Check for initial matches (epsilon)
        for group_id in self.states[current_state].finalizers:
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
            for group_id in self.states[current_state].finalizers:
                if group_id in self.non_greedy_finalizers:
                    matches.setdefault(group_id, i + 1)
                else:
                    matches[group_id] = i + 1

        end_state = None if done else current_state
        return end_state, [(gid, width) for gid, width in matches.items() if width > 0]

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

@dataclass
class Model(GraphProvider):
    stats = Stats.get()
    stats.add_group('get_mask')
    stats.add_group('commit')

    parser_table: ParserTable
    tokenizer: PyTokenizer
    tokenizer_initial_state: int
    id_to_token: Dict[int, bytes]
    internal_to_original_map: Dict[int, RangeSetOut]
    all_internal_llm_tokens_bitset: LLMTokenSet
    ignore_terminal_id: Optional[int]
    state: Dict[int, GSS]
    suppress_stats_report: bool = True

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        Stats.get().reset()
        data = json.loads(s)

        # Tokenizer
        dfa_data = data['tokenizer']['dfa']
        dfa_states = [DFAState(transitions={int(k): v for k, v in s['transitions'].get('data', {}).items()}, finalizers=set(s['finalizers']), possible_future_group_ids=set(s['possible_future_group_ids'])) for s in dfa_data['states']]
        tokenizer = PyTokenizer(dfa_states, dfa_data['start_state'], set(dfa_data['non_greedy_finalizers']))

        # Parser Table
        parser_data = data['parser']
        py_table: Dict[int, Row] = {}
        for state_id_str, row_data in parser_data['stage_7_table']:
            state_id, py_row = int(state_id_str), Row()
            for term_id_str, action_data in row_data['shifts_and_reduces_full']:
                term_id, variant = int(term_id_str), action_data['variant']
                if variant == 'Shift': py_row.actions[term_id] = action_data['state_id']
                elif variant == 'Reduce': py_row.actions[term_id] = Reduce(action_data['nonterminal_id'], action_data['len'], tuple(sorted(action_data['production_ids'])))
                elif variant == 'Split':
                    reduces = {int(l): {int(n): tuple(sorted(p)) for n, p in nd} for l, nd in action_data['reduces']}
                    py_row.actions[term_id] = Split(action_data['shift'], reduces)
            py_row.gotos = {int(nt): goto['state_id'] for nt, goto in row_data['gotos'] if goto['state_id'] is not None}
            py_table[state_id] = py_row
        parser_table = ParserTable(parser_data['start_state_id'], py_table)

        # Vocab and maps
        constraint = ffi.GrammarConstraint.from_json_string(s)
        vocab = data['precompute3_vocab']
        all_internal_llm_tokens_bitset = RangeSet.from_ranges([(0, vocab['internal_max_llm_token'])])

        # Initial state
        initial_acc = PyAcc({}, all_internal_llm_tokens_bitset)
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(parser_table.start_state_id)

        model = Model(
            parser_table=parser_table,
            tokenizer=tokenizer, tokenizer_initial_state=tokenizer.initial_state_id(),
            id_to_token={v: bytes(k) for k, v in data['llm_token_map']},
            internal_to_original_map={int(k): RangeSetOut.from_indices(v) for k, v in dict(vocab['internal_to_original']).items()},
            all_internal_llm_tokens_bitset=all_internal_llm_tokens_bitset,
            ignore_terminal_id=constraint.glr_parser().ignore_terminal_id,
            state={tokenizer.initial_state_id(): initial_gss},
        )
        return model

    def make_initial_state(self) -> Dict[int, GSS]:
        """Create a fresh initial state without reloading models."""
        initial_acc = PyAcc({}, self.all_internal_llm_tokens_bitset)
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(self.parser_table.start_state_id)
        return {self.tokenizer_initial_state: initial_gss}

    def reset_state(self) -> None:
        """Reset to initial state."""
        self.state = self.make_initial_state()

    def clone_sharing_structure(self) -> "Model":
        """Create a new Model instance sharing static structures and with a fresh initial state."""
        return Model(
            parser_table=self.parser_table,
            tokenizer=self.tokenizer,
            tokenizer_initial_state=self.tokenizer_initial_state,
            id_to_token=self.id_to_token,
            internal_to_original_map=self.internal_to_original_map,
            all_internal_llm_tokens_bitset=self.all_internal_llm_tokens_bitset,
            ignore_terminal_id=self.ignore_terminal_id,
            state=self.make_initial_state(),
            suppress_stats_report=self.suppress_stats_report,
        )

    def commit(self, token_id: int):
        stats = Stats.get()
        stats.start('commit')
        token_bytes = self.id_to_token[token_id]

        stats.start('commit.build_tokenizer_maps')
        stats.inc('commit.tokenizer_states_in', len(self.state))
        terminals_map, state_map = {}, {}
        for tsid in self.state:
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tsid)
            if end_state is not None: state_map[tsid] = end_state
            terminals_map[tsid] = RangeSet.from_indices([m[0] for m in matches])
        stats.stop('commit.build_tokenizer_maps')

        stats.start('commit.prune_and_map_gss')
        @_acc_memoize()
        def mutator(acc: PyAcc) -> Optional[PyAcc]:
            for tsid, matched in terminals_map.items():
                if acc.terminals_union.get(tsid, RangeSet.empty()).intersects(matched): return None
            new_bvs = collections.defaultdict(RangeSet.empty)
            for old, new in state_map.items():
                if old in acc.terminals_union: new_bvs[new] |= acc.terminals_union[old]
            return PyAcc(dict(new_bvs), acc.llm_mask)

        cache = {}
        current = {tsid: g.apply_and_prune(mutator, cache) for tsid, g in self.state.items()}
        current = {tsid: g for tsid, g in current.items() if not g.is_empty()}
        stats.stop('commit.prune_and_map_gss')

        stats.start('commit.main_loop')
        new_states = collections.defaultdict(list)
        work = {(0, tsid): gss for tsid, gss in current.items()}
        q = collections.deque(work.keys())

        while q:
            offset, tsid = q.popleft()
            gss = work.pop((offset, tsid))
            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tsid)

            for term_id, width in matches:
                proc_gss = self._process_token(gss, term_id)
                if end_state is not None and term_id in self.tokenizer.states[end_state].possible_future_group_ids:
                    proc_gss = self._disallow_terminal_in_state(proc_gss, end_state, term_id)
                if not proc_gss.is_empty():
                    new_offset, next_tsid = offset + width, self.tokenizer.initial_state_id()
                    if new_offset == len(token_bytes): new_states[next_tsid].append(proc_gss)
                    else:
                        key = (new_offset, next_tsid)
                        if key in work: work[key] = work[key].merge(proc_gss)
                        else:
                            work[key] = proc_gss
                            q.append(key)
            if end_state is not None: new_states[end_state].append(gss)
        stats.stop('commit.main_loop')

        stats.start('commit.merge_states')
        merged = {sid: GSS.merge_many(gssl) for sid, gssl in new_states.items() if gssl}
        merged = {sid: g for sid, g in merged.items() if not g.is_empty()}
        stats.stop('commit.merge_states')

        stats.inc('commit.tokenizer_states_out', len(merged))
        self.state = merged
        stats.stop('commit')

    def _process_token(self, gss: GSS, terminal_id: int) -> GSS:
        stats = Stats.get()
        p = 'commit.main_loop._process_token'
        stats.start(f'{p}.total')

        if self.ignore_terminal_id == terminal_id:
            stats.stop(f'{p}.total')
            return gss

        heads_by_state = collections.defaultdict(list)
        for state_id in gss.peek(): heads_by_state[state_id].append(gss.isolate(state_id))

        shifted = []
        while heads_by_state:
            state_id, gss_list = heads_by_state.popitem()
            state_gss = GSS.merge_many(gss_list)
            row = self.parser_table.table.get(state_id)
            if not row: continue
            action = row.actions.get(terminal_id)
            if action is None: continue

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

        result = GSS.merge_many(shifted)
        stats.stop(f'{p}.total')
        return result

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        term_rs = RangeSet.from_indices([terminal_id])
        @_acc_memoize(use_value_cache=False)
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current = acc.terminals_union.get(state_id, RangeSet.empty())
            if current.contains(terminal_id): return acc
            new_map = acc.terminals_union.copy()
            new_map[state_id] = current.union(term_rs)
            return PyAcc(new_map, acc.llm_mask)
        return gss.apply(apply_disallow)

    def get_mask(self) -> RangeSetOut:
        """
        Calculates the allowed token mask by brute force.
        For every possible token, it clones the current state and runs `commit`.
        If the resulting state is not empty, the token is added to the mask.
        """
        stats = Stats.get()
        stats.start('get_mask')

        allowed_original_tokens = RangeSetOut.empty()
        
        # Iterate over all known internal tokens
        token_ids_to_check = list(self.id_to_token.keys())

        for token_id in tqdm(token_ids_to_check, desc="[BruteForce] Checking tokens", disable=True):
            # Create a temporary model instance with a copy of the current state.
            # This is slow but simple and correct.
            temp_model = self.clone_sharing_structure()
            temp_model.state = self.state.copy() # Shallow copy is sufficient

            # Run commit on the temporary model
            temp_model.commit(token_id)

            # If the commit results in a valid (non-empty) state, the token is allowed
            if temp_model.state:
                original_ids = self.internal_to_original_map.get(token_id)
                if original_ids:
                    allowed_original_tokens |= original_ids

        stats.stop('get_mask')
        return allowed_original_tokens

    # --- Dummy/Simple implementations for the GraphProvider interface ---
    def get_last_get_mask_cost(self) -> int: return 0
    def get_last_get_mask_metrics(self) -> Dict[str, float]: return {}
    def default_variations(self) -> List: return []
    def select_hot_steps(self, aggregated_costs: List[float], k: int = 1) -> List[int]: return []
    def get_benchmark_config(self) -> Dict: return {}
    def iter_edges(self, node, token): return iter([])
    def get_root(self, state_id: int): return 0
    def is_end(self, node) -> bool: return False

    def finalize(self):
        """Called at the end of a benchmark run to perform any final actions."""
        if not self.suppress_stats_report:
            print("\n--- Final Stats Report from BruteForce Model ---")
            Stats.get().report()
