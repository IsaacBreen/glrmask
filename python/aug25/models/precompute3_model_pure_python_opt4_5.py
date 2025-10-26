from __future__ import annotations

"""
Precompute3 model (pure Python, optimized structure).

This revision replaces the original "edge list + state bitset" trie representation with a
state-indexed adjacency map that eliminates bitset intersection on the hot path.

New arena structure per node u:
  by_pop: Dict[int, Dict[int, Dict[int, List[LLMTokenSet]]]]
    - For each pop distance p (how many heads to pop in the GSS):
      - For each LR parser state s (an integer), a map:
          dest_node_id -> list of LLMTokenSet (tokens admitted along (u --p,s--> dest)).
    - Each LLMTokenSet in the list corresponds to an original edge’s llm_bv. Multiple entries
      arise if there were distinct original edges with different llm_bv leading to the same dest
      for that source state.

We also precompute:
  - pop_llm_union[p]: union of all tokens admitted by any (s, dest) in bucket p.
  - llm_bv_union: union of pop_llm_union[p] over all p.
  - llm_bv_descendant: a monotone fixed-point upper bound of tokens reachable from this node
    to some clean_end via the trie (see proof sketch below).

Hot-path traversal: Given a node u, a GSS snapshot g, and its peeked heads S = peek(g.popn(p)),
we no longer intersect a bitset "S ∩ state_bv". Instead, we loop over s in S and perform a
constant-time dictionary lookup by_pop[p].get(s). This avoids iterating through full bitsets,
eliminating the combinatorics of iterating state ranges and then intersecting. Next, we group
per (dest, llm_bv) all s that share the same route. We perform at most one apply_and_prune per
unique llm_bv and one isolate_many per (dest, llm_bv) group—exactly matching the original
semantics where llm_bv is edge-local.

Correctness sketch (semantics preservation):

Let the original trie contain entries of the form (u, p, M, D, S) where:
  - u is the source node,
  - p is pop distance,
  - M is the token set (llm_bv),
  - D is a destination node,
  - S is a set of LR states (represented earlier as a bitset) allowed for this edge.

Original get_mask transition condition: from node u with GSS heads H, one may traverse (u -> D)
via that edge iff H contains some s in S; the accumulator mask transforms by intersecting with M,
and then only those heads s in S survive for continuation to D. The new structure instead stores,
for each s ∈ S, an entry by_pop[p][s][D] that includes the list of token sets {M_i} for edges with
this exact D and p. During traversal, given peeked heads H, we test each s ∈ H via the dictionary
lookup. For each (D, M_i), we accumulate s into the (D, M_i)-bucket. Then:
  - We compute source_after_apply = apply_and_prune(intersect with M_i) ONCE per M_i, reusing
    its result for all D that share M_i, exactly as the original algorithm did when multiple
    edges had the same llm_bv.
  - For each (D, M_i), we isolate_many only the states s that route to D via M_i, producing
    the exact same successor as the edge-list version (which did "apply per edge, then isolate"
    to the edge’s S ∩ peeked heads).
Thus, any path admitted in the original model is admitted here, and none is added: we never
union distinct M_i prior to apply; we keep them separate per edge token set. This precludes
spurious tokens that could otherwise arise if one coalesced token sets across states or edges.

Descendant closure correctness (upper bound of reachable tokens to an end):
Define F on maps C: Node -> LLMTokenSet by
  F(C)(u) = I[u is clean_end] * U + ⋃_{edges u -> v with llm M over any s}
                            ( M ∩ C(v) )
where U is "all internal tokens" and I[.] is 1 if true, else 0; edges are represented by entries
by_pop[p][s][v] with tokenset list. Our implementation computes the least fixed point by
monotone iteration:
  - Initialize C0(u) = U if u.clean_end else ∅.
  - Repeat: C_{k+1}(u) = C_k(u) ∪ ⋃_{u->v, tokenset M} ( M ∩ C_k(v) ).
Monotonicity in the complete lattice of LLMTokenSet w.r.t. ⊆ ensures convergence to the least
fixed point C* that satisfies C* = F(C*). The closure we compute therefore safely upper bounds
all tokens that can flow from u to some end node. We use this mask for pruning and dynamic
prioritization.

Time/space notes:
  - Hot path no longer needs bitset intersection per edge. For each pop p, we iterate only the
    heads actually present (peeked states) and perform O(1) map lookups for edges. The total
    number of apply_and_prune calls is at most the number of distinct llm_bv observed for the
    current (p, u) visit.
  - Memory increases when expanding S bitsets into explicit per-state maps. This is the tradeoff
    for consistent sub‑millisecond traversal on typical inputs where peeked heads are few and
    the map lookup is constant-time.
"""

import collections
import functools
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set, Generator, Any

from ..stats import Stats
import _sep1 as ffi
from tqdm import tqdm

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

RangeSet.union = _patched_union
RangeSet.intersection = _patched_intersection
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

# --- Tokenizer and Parser Table ---

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
        matches: Dict[int, int] = {}
        done = False

        # Epsilon matches
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

    def tokens_accessible_from_state(self, state_id: int) -> List[int]:
        return list(self.states[state_id].possible_future_group_ids)

    def initial_state_id(self) -> int:
        return self.start_state

    def max_state(self) -> int:
        return len(self.states)

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

# --- PyAcc ---

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
        return PyAcc(new_terminals_union, self.llm_mask.union(other.llm_mask))

    def is_empty(self):
        return self.llm_mask.is_empty()

# --- Work items ---

@dataclass
class WorkItemNew:
    node_id: NodeID
    gss: GSS
    depth: int

# --- New Arena Node structure ---

@dataclass
class ArenaNode:
    # by_pop[pop][state_id][dest_id] -> list of token sets (LLMTokenSet) for edges
    by_pop: Dict[int, Dict[int, Dict[int, List[LLMTokenSet]]]] = field(default_factory=dict)
    pop_llm_union: Dict[int, LLMTokenSet] = field(default_factory=dict)
    llm_bv_union: LLMTokenSet = field(default_factory=RangeSet.empty)
    llm_bv_descendant: LLMTokenSet = field(default_factory=RangeSet.empty)
    clean_end: bool = False

# --- Model ---

@dataclass
class Model(GraphProvider):
    stats = Stats.get()
    stats.add_group('get_mask')
    stats.add_group('commit')

    arena: Dict[NodeID, ArenaNode]
    roots_map: Dict[int, NodeID]
    max_depth: Dict[NodeID, int]
    parser_table: ParserTable
    tokenizer: PyTokenizer
    tokenizer_initial_state: int
    possible_matches_cache: Dict[int, Dict[int, LLMTokenSet]]
    id_to_token: Dict[int, bytes]
    internal_to_original_map: Dict[int, RangeSetOut]
    all_internal_llm_tokens_bitset: LLMTokenSet
    ignore_terminal_id: Optional[int]
    state: Dict[int, GSS]

    # Runtime tunables and results
    gm_max_dests: int = 4096
    last_get_mask_cost: int = 0
    last_get_mask_metrics: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        Stats.get().reset()
        data = json.loads(s)

        # Roots map
        roots_map = {int(src): int(root) for src, root in data["precomputed3"]}

        # Arena construction (state-indexed map)
        arena_dict = {int(k): v for k, v in data["trie3_god"].get("values", [])}
        max_depth: Dict[NodeID, int] = {}
        dumps, bs_from_json = json.dumps, ffi.Bitset.from_json_string

        arena: Dict[NodeID, ArenaNode] = {}

        for uid, node_data in tqdm(arena_dict.items(), desc="Building state-indexed arena"):
            max_depth[uid] = int(node_data.get("max_depth", 0) or 0)
            children_data = node_data.get("children") or []
            by_pop: Dict[int, Dict[int, Dict[int, List[LLMTokenSet]]]] = {}
            pop_llm_union: Dict[int, LLMTokenSet] = collections.defaultdict(RangeSet.empty)

            for (pop, llm_json), dest_map_json in children_data:
                pop = int(pop)
                llm_bv = RangeSet.from_ranges(bs_from_json(dumps(llm_json)).to_ranges())
                # Expand to per-state mapping, preserving per-edge llm_bv (no pre-union across edges)
                for dest_idx, state_json in dest_map_json:
                    dest_idx = int(dest_idx)
                    state_bv = RangeSetStates.from_ranges(bs_from_json(dumps(state_json)).to_ranges())
                    # Iterate concrete state indices in this bitset
                    for start, end in state_bv.to_ranges():
                        for sid in range(int(start), int(end) + 1):
                            # by_pop[pop][sid][dest_idx].append(llm_bv)
                            pop_map = by_pop.get(pop)
                            if pop_map is None:
                                pop_map = {}
                                by_pop[pop] = pop_map
                            sid_map = pop_map.get(sid)
                            if sid_map is None:
                                sid_map = {}
                                pop_map[sid] = sid_map
                            lst = sid_map.get(dest_idx)
                            if lst is None:
                                lst = []
                                sid_map[dest_idx] = lst
                            # Avoid accidental duplicates
                            if not lst or lst[-1] != llm_bv:
                                lst.append(llm_bv)
                            pop_llm_union[pop] = pop_llm_union[pop].union(llm_bv)

            # Compute unions
            llm_bv_union = RangeSet.empty()
            for m in pop_llm_union.values():
                llm_bv_union = llm_bv_union.union(m)

            clean_end = bool(node_data.get("value", {}).get("clean_end", False))
            arena[uid] = ArenaNode(
                by_pop=by_pop,
                pop_llm_union=dict(pop_llm_union),
                llm_bv_union=llm_bv_union,
                llm_bv_descendant=RangeSet.empty(),  # filled below
                clean_end=clean_end,
            )

        # Tokenizer
        dfa_data = data['tokenizer']['dfa']
        dfa_states = [
            DFAState(
                transitions={int(k): v for k, v in s['transitions'].get('data', {}).items()},
                finalizers=set(s['finalizers']),
                possible_future_group_ids=set(s['possible_future_group_ids'])
            )
            for s in dfa_data['states']
        ]
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
                    py_row.actions[term_id] = Reduce(
                        action_data['nonterminal_id'],
                        action_data['len'],
                        tuple(sorted(action_data['production_ids']))
                    )
                elif variant == 'Split':
                    reduces = {int(l): {int(n): tuple(sorted(p)) for n, p in nd} for l, nd in action_data['reduces']}
                    py_row.actions[term_id] = Split(action_data['shift'], reduces)
            py_row.gotos = {int(nt): goto['state_id'] for nt, goto in row_data['gotos'] if goto['state_id'] is not None}
            py_table[state_id] = py_row
        parser_table = ParserTable(parser_data['start_state_id'], py_table)

        # Misc data (via FFI)
        constraint = ffi.GrammarConstraint.from_json_string(s)
        pmc_ffi = constraint.possible_matches()
        possible_matches_cache = {
            int(t): {int(i): RangeSet.from_ranges(b.to_ranges()) for i, b in inner.items()}
            for t, inner in pmc_ffi.items()
        }
        vocab = data['precompute3_vocab']
        all_internal_llm_tokens_bitset = RangeSet.from_ranges([(0, vocab['internal_max_llm_token'])])

        # Initial state (GSS)
        initial_acc = PyAcc({}, all_internal_llm_tokens_bitset)
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(parser_table.start_state_id)

        model = Model(
            arena=arena,
            roots_map=roots_map,
            max_depth=max_depth,
            parser_table=parser_table,
            tokenizer=tokenizer,
            tokenizer_initial_state=tokenizer.initial_state_id(),
            possible_matches_cache=possible_matches_cache,
            id_to_token={v: bytes(k) for k, v in data['llm_token_map']},
            internal_to_original_map={int(k): RangeSetOut.from_indices(v) for k, v in dict(vocab['internal_to_original']).items()},
            all_internal_llm_tokens_bitset=all_internal_llm_tokens_bitset,
            ignore_terminal_id=constraint.glr_parser().ignore_terminal_id,
            state={tokenizer.initial_state_id(): initial_gss},
        )

        # Descendant closure for pruning
        model._compute_descendant_llm_closure()

        # Optional stats
        model._compute_and_print_stats(minimal=True)

        return model

    # --- Basic GraphProvider interface ---

    def get_root(self, state_id: int) -> NodeID:
        return self.roots_map[int(state_id)]

    def is_end(self, node: NodeID) -> bool:
        n = self.arena.get(int(node))
        return bool(n and n.clean_end)

    def iter_edges(self, node: NodeID, token: int):
        # Slow path iterator for tests/diagnostics: enumerate all triples (pop, state, dest) that admit 'token'
        a = self.arena.get(int(node))
        if not a:
            return
        for pop, sid_map in a.by_pop.items():
            for sid, dest_map in sid_map.items():
                for dest, tok_list in dest_map.items():
                    for m in tok_list:
                        if m.contains(token):
                            yield (int(pop), int(sid), int(dest))

    # --- Token commits / parser actions (unchanged) ---

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
                proc_gss = self._process_token(gss, term_id)
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

    # --- Descendant closure ---

    def _compute_descendant_llm_closure(self) -> None:
        """Monotone fixed-point: for each node u,
           C(u) = all_ones if u.clean_end else ⋃_{edges u->v with tokens M} (M ∩ C(v))."""
        all_ones = self.all_internal_llm_tokens_bitset
        # Initialize
        for u, node in self.arena.items():
            node.llm_bv_descendant = all_ones if node.clean_end else RangeSet.empty()
        # Iterate to fixed-point
        changed = True
        while changed:
            changed = False
            for u, node in self.arena.items():
                if not node.by_pop and not node.clean_end:
                    continue
                accum = RangeSet.empty()
                # For each destination, first union tokens across all states (safe due to distributivity)
                dest_to_tokens: Dict[int, LLMTokenSet] = {}
                for pop, sid_map in node.by_pop.items():
                    for sid, dest_map in sid_map.items():
                        for dest, tok_list in dest_map.items():
                            if not tok_list:
                                continue
                            M_union = RangeSet.empty()
                            for M in tok_list:
                                M_union = M_union.union(M)
                            prev = dest_to_tokens.get(int(dest))
                            dest_to_tokens[int(dest)] = M_union if prev is None else prev.union(M_union)
                for dest, M_union in dest_to_tokens.items():
                    child = self.arena.get(int(dest))
                    if not child:
                        continue
                    thru = M_union.intersection(child.llm_bv_descendant)
                    if not thru.is_empty():
                        accum = accum.union(thru)
                if node.clean_end:
                    accum = accum.union(all_ones)
                if accum != node.llm_bv_descendant:
                    node.llm_bv_descendant = accum
                    changed = True

    # --- Main mask computation ---

    def _get_mask_run(self) -> Dict:
        stats = Stats.get()
        stats.start('get_mask')
        stats.start('get_mask.seeding')

        # Initialize accumulators at roots: disallow terminals implied by initial tokenizer states
        all_ones = self.all_internal_llm_tokens_bitset

        @_acc_memoize()
        def initialize_acc(acc: PyAcc) -> PyAcc:
            disallowed = RangeSet.empty()
            for tsid, terms in acc.terminals_union.items():
                if tsid in self.possible_matches_cache:
                    term_map = self.possible_matches_cache[tsid]
                    for term_id in terms.iter_indices():
                        if term_id in term_map:
                            disallowed = disallowed.union(term_map[term_id])
            return PyAcc({}, all_ones.difference(disallowed))

        init_cache = {}
        q: collections.deque[WorkItemNew] = collections.deque()
        for sid, gss in self.state.items():
            r = self.roots_map[int(sid)]
            gss_init = gss.apply(initialize_acc, init_cache)
            if not gss_init.is_empty():
                q.append(WorkItemNew(int(r), gss_init, 0))
        stats.stop('get_mask.seeding')

        stats.start('get_mask.main_loop')
        final_mask = RangeSet.empty()
        total_routes = 0

        while q:
            work = q.popleft()
            node_id, gss_node, depth = work.node_id, work.gss, work.depth

            # Reduce accumulators at this node
            stats.start('get_mask.reduce_acc')
            gss_acc = gss_node.reduce_acc()
            stats.stop('get_mask.reduce_acc')

            # End node handling
            if self.is_end(node_id):
                stats.start('get_mask.final_mask_union')
                final_mask = final_mask.union(gss_acc.llm_mask)
                stats.stop('get_mask.final_mask_union')
                continue

            a_node = self.arena.get(int(node_id))
            if not a_node or not a_node.by_pop:
                continue

            # Closure-based pruning
            work_llm_mask = a_node.llm_bv_descendant.intersection(gss_acc.llm_mask)
            if work_llm_mask.is_empty():
                continue

            # Per-node caches
            global_apply_cache: Dict[Tuple[int, int], GSS] = {}

            for pop, sid_map in a_node.by_pop.items():
                # popn and peek the heads only once per pop bucket
                popped = gss_node.popn(int(pop))
                if popped.is_empty():
                    continue
                popped_acc = popped.reduce_acc()
                if not popped_acc or popped_acc.llm_mask.is_empty():
                    continue

                # Bucket-level token pruning: if no overlap, skip bucket
                pop_union = a_node.pop_llm_union.get(int(pop), RangeSet.empty())
                if popped_acc.llm_mask.isdisjoint(pop_union):
                    continue
                if pop_union.isdisjoint(work_llm_mask):
                    continue

                peeked = popped.peek()
                if not peeked:
                    continue

                # Accumulate per (dest, token-set) the list of source states to isolate
                bucket_groups: Dict[Tuple[int, int], List[int]] = {}
                # Cache per-token-set apply results
                per_token_after_apply: Dict[int, Optional[GSS]] = {}

                for sid in peeked:
                    dest_map = sid_map.get(int(sid))
                    if not dest_map:
                        continue
                    for dest, tok_list in dest_map.items():
                        if not tok_list:
                            continue
                        for M in tok_list:
                            # Early pruning per token-set
                            if popped_acc.llm_mask.isdisjoint(M) or work_llm_mask.isdisjoint(M):
                                continue
                            key_apply = (id(popped), id(M))
                            if key_apply in global_apply_cache:
                                after_apply = global_apply_cache[key_apply]
                            else:
                                @_acc_memoize(use_value_cache=False)
                                def intersect(acc: PyAcc):
                                    nm = acc.llm_mask.intersection(M)
                                    return None if nm.is_empty() else PyAcc(acc.terminals_union, nm)
                                tmp = popped.apply_and_prune(intersect)
                                if tmp.is_empty():
                                    after_apply = None
                                else:
                                    after_apply = tmp
                                global_apply_cache[key_apply] = after_apply if after_apply is not None else GSS.empty()
                            if after_apply is None or after_apply.is_empty():
                                continue
                            per_token_after_apply[id(M)] = after_apply
                            # Group by (dest, token_id)
                            key = (int(dest), id(M))
                            lst = bucket_groups.get(key)
                            if lst is None:
                                bucket_groups[key] = [int(sid)]
                            else:
                                lst.append(int(sid))

                # Emit children
                for (dest, token_id), states_list in bucket_groups.items():
                    after_apply = per_token_after_apply.get(token_id)
                    if after_apply is None or after_apply.is_empty():
                        continue
                    # If all heads survive, reuse after_apply; else isolate_many
                    child_gss = after_apply if len(states_list) == len(peeked) else after_apply.isolate_many(states_list)
                    if child_gss.is_empty():
                        continue
                    q.append(WorkItemNew(int(dest), child_gss, depth + 1))
                    total_routes += 1
                    if total_routes >= int(self.gm_max_dests):
                        # Optional yield point for extremely dense nodes
                        total_routes = 0

        stats.stop('get_mask.main_loop')

        # Convert to original token IDs
        stats.start('get_mask.final_conversion')
        original_indices = RangeSetOut.empty()
        for i in final_mask.iter_indices():
            if i in self.internal_to_original_map:
                original_indices |= self.internal_to_original_map[i]
        stats.stop('get_mask.final_conversion')

        stats.stop('get_mask')

        # Metrics
        self.last_get_mask_cost = int(total_routes)
        self.last_get_mask_metrics = {
            "routes_emitted": float(total_routes),
            "main_loop_ms": float(Stats.get().times.get('get_mask.main_loop', 0.0) * 1000.0),
            "total_ms": float(Stats.get().times.get('get_mask', 0.0) * 1000.0),
        }

        # Optional internal stats printout
        if Stats.get().times['get_mask.main_loop'] * 1000 > 1:
            Stats.get().report()

        return {
            "type": "timed_output",
            "output": original_indices,
            "time_sec": Stats.get().times.get('get_mask.main_loop', 0.0),
        }

    def get_mask(self) -> Dict:
        Stats.get().reset()
        return self._get_mask_run()

    # --- Minimal stats report for the new structure ---

    def _compute_and_print_stats(self, minimal: bool = True):
        try:
            import numpy as np  # noqa: F401
        except Exception:
            np = None  # noqa: F841

        num_nodes = len(self.arena)
        num_end = sum(1 for n in self.arena.values() if n.clean_end)
        num_pops = sum(len(n.by_pop) for n in self.arena.values())
        approx_entries = 0
        approx_states_indexed = 0
        for n in self.arena.values():
            for sid_map in n.by_pop.values():
                approx_entries += sum(len(dest_map) for dest_map in sid_map.values())
                approx_states_indexed += len(sid_map)

        print("\n--- Arena (state-indexed) Stats ---")
        print(f"Total nodes: {num_nodes:,}")
        print(f"Clean-end nodes: {num_end:,}")
        print(f"Pop buckets (sum over nodes): {num_pops:,}")
        print(f"Per-(pop,state) entries (dest maps): {approx_states_indexed:,}")
        print(f"Per-(pop,state,dest) entries (token lists): {approx_entries:,}")
        print("-------------------\n")

    def finalize(self):
        print("\n--- Final Stats Report from Model ---")
        Stats.get().report()
