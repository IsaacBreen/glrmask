from __future__ import annotations
import json
import heapq
import collections
import os
import time
from typing import Dict, List, Tuple, Optional, Union, Callable, Iterable, Set, Type, Any
from dataclasses import dataclass, field
from ..common_interface import GraphProvider, RangeSet
import _sep1 as ffi


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


@dataclass(frozen=True)
class PyAcc:
    terminals_union: ffi.HybridL2Bitset

    def __hash__(self) -> int:
        return hash(self.terminals_union)

    def merge(self, other: "PyAcc") -> "PyAcc":
        return PyAcc(self.terminals_union.union(other.terminals_union))


# -----------------------------
# Debug/profiling infrastructure
# -----------------------------

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v not in ("", "0", "false", "False", "no", "NO")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


@dataclass
class GSSStats:
    heads: int
    empty_stacks: int
    unique_stacks: int
    accs: int
    depth_min: int
    depth_max: int
    depth_mean: float

    def brief(self) -> str:
        return f"heads={self.heads} unique_stacks={self.unique_stacks} empty={self.empty_stacks} depth[min/mean/max]={self.depth_min}/{round(self.depth_mean,2)}/{self.depth_max} accs={self.accs}"


class _CommitProfiler:
    """
    Lightweight, opt-in per-commit profiler and tracer.
    Controlled by environment variables:
      - PROFILE_COMMIT=1: enable profiling
      - PROFILE_THRESHOLD_MS=float (default 5.0): only print detailed stats if commit exceeds this
      - DEBUG_COMMIT_VERBOSE=1: always print detailed trace
      - DEBUG_PROCESS_TOKEN_TRACE=1: print details from _process_token
      - DEBUG_GSS_SAMPLES=int: print example stacks for the largest GSS (limit count)
    """

    def __init__(self, step_index: int, token_bytes: bytes) -> None:
        self.step_index = step_index
        self.token_bytes = token_bytes
        self.start_time = time.perf_counter()
        self.threshold_ms = _env_float("PROFILE_THRESHOLD_MS", 5.0)
        self.verbose = _env_flag("DEBUG_COMMIT_VERBOSE", False)
        self.trace_process = _env_flag("DEBUG_PROCESS_TOKEN_TRACE", False)
        self.sample_limit = int(os.environ.get("DEBUG_GSS_SAMPLES", "0") or "0")

        # Pre/post state stats
        self.pre_stats: Dict[int, GSSStats] = {}
        self.post_stats: Dict[int, GSSStats] = {}

        # Queue traversal
        self.queue_enqueued = 0
        self.queue_popped = 0
        self.offsets_explored: Set[int] = set()
        self.max_offset = 0

        # Tokenizer branching
        self.matches_total = 0
        self.matches_by_offset: Dict[int, int] = collections.Counter()
        self.width_hist: Dict[int, int] = collections.Counter()

        # _process_token counters
        self.process_calls = 0
        self.process_shifts = 0
        self.process_reduces = 0
        self.process_splits = 0
        self.process_gotos = 0

        # GSS merges
        self.merge_calls = 0
        self.merge_inputs_gss = 0
        self.merge_input_stacks = 0
        self.merge_output_stacks = 0

        # Others
        self.carry_forward_count = 0  # times we carried gss to end_state (no consumption)
        self.enqueue_after_match = 0  # times we enqueued next offset
        self.end_state_prunes = 0     # times we explicitly disallowed terminal at end_state

        # Hook into GSS.merge
        self.prev_profiler = GSS._active_profiler
        GSS._active_profiler = self

    def on_finalize(self) -> None:
        GSS._active_profiler = self.prev_profiler

    def add_pre_stats(self, sid: int, stats: GSSStats) -> None:
        self.pre_stats[int(sid)] = stats

    def add_post_stats(self, sid: int, stats: GSSStats) -> None:
        self.post_stats[int(sid)] = stats

    def on_queue_enqueued(self, n: int) -> None:
        self.queue_enqueued += n

    def on_queue_pop(self, offset: int) -> None:
        self.queue_popped += 1
        self.offsets_explored.add(offset)
        if offset > self.max_offset:
            self.max_offset = offset

    def on_matches(self, offset: int, matches: List[Tuple[int, int]]) -> None:
        self.matches_total += len(matches)
        self.matches_by_offset[offset] += len(matches)
        for _, width in matches:
            self.width_hist[width] += 1

    def on_end_state_prune(self) -> None:
        self.end_state_prunes += 1

    def on_enqueue_after_match(self) -> None:
        self.enqueue_after_match += 1

    def on_carry_forward(self) -> None:
        self.carry_forward_count += 1

    def on_process_token(self, terminal_id: int, shifts: int, reduces: int, splits: int, gotos: int) -> None:
        self.process_calls += 1
        self.process_shifts += shifts
        self.process_reduces += reduces
        self.process_splits += splits
        self.process_gotos += gotos
        if self.trace_process:
            print(f"[process_token] term={terminal_id} shifts={shifts} reduces={reduces} splits={splits} gotos={gotos}")

    def on_gss_merge(self, inputs_gss: int, input_stacks: int, output_stacks: int) -> None:
        self.merge_calls += 1
        self.merge_inputs_gss += inputs_gss
        self.merge_input_stacks += input_stacks
        self.merge_output_stacks += output_stacks

    def summarize_gss_map(self, m: Dict[int, GSSStats]) -> str:
        if not m:
            return "0 states"
        sids = sorted(m.keys())
        total_heads = sum(st.heads for st in m.values())
        total_stacks = sum(st.unique_stacks for st in m.values())
        max_depth = max(st.depth_max for st in m.values())
        return f"{len(m)} states, heads={total_heads}, stacks={total_stacks}, max_depth={max_depth}"

    def maybe_print(self) -> None:
        end = time.perf_counter()
        elapsed_ms = (end - self.start_time) * 1000.0
        should_print = self.verbose or elapsed_ms >= self.threshold_ms or _env_flag("PROFILE_COMMIT", False)
        if not should_print:
            return

        tb = self.token_bytes
        token_preview = tb[:80]
        token_repr = token_preview.decode('utf-8', 'replace')
        if len(tb) > 80:
            token_repr += "…"

        print("=== COMMIT PROFILE START ===")
        print(f"step={self.step_index} token='{token_repr}' time_ms={round(elapsed_ms, 3)}")
        print(f"pre:  {self.summarize_gss_map(self.pre_stats)}")
        print(f"post: {self.summarize_gss_map(self.post_stats)}")
        print(f"queue: enq={self.queue_enqueued} pop={self.queue_popped} offsets={len(self.offsets_explored)} max_offset={self.max_offset}")
        print(f"tokenizer: matches_total={self.matches_total} width_hist={dict(self.width_hist)}")
        print(f"process_token: calls={self.process_calls} shifts={self.process_shifts} reduces={self.process_reduces} splits={self.process_splits} gotos={self.process_gotos}")
        print(f"merges: calls={self.merge_calls} inputs_gss={self.merge_inputs_gss} input_stacks={self.merge_input_stacks} output_stacks={self.merge_output_stacks} dedup_saved={self.merge_input_stacks - self.merge_output_stacks}")
        print(f"end_state: carry_forward={self.carry_forward_count} prunes={self.end_state_prunes} enqueue_after_match={self.enqueue_after_match}")

        if self.verbose and self.pre_stats:
            largest_sid = max(self.pre_stats.items(), key=lambda kv: kv[1].unique_stacks)[0]
            pre = self.pre_stats[largest_sid]
            post = self.post_stats.get(largest_sid)
            print(f"largest pre-state sid={largest_sid} {pre.brief()}")
            if post:
                print(f"same sid post-state {post.brief()}")

        print("=== COMMIT PROFILE END ===")

    def finalize(self) -> None:
        self.maybe_print()
        self.on_finalize()


# GSS implementation (simplified)

class GSS:
    """
    A compact Graph-Structured Stack representation.

    Conceptually, it's just a multiset of stacks (each a tuple of state IDs),
    each annotated with a PyAcc accumulator. We canonicalize by merging
    accumulators per identical stack, so internally it's:
        Dict[Tuple[int, ...], PyAcc]

    The empty stack is represented by the empty tuple ().
    """
    # Active profiler for merge instrumentation (set by _CommitProfiler)
    _active_profiler: Optional[_CommitProfiler] = None

    def __init__(self, heads: Optional[Dict[Tuple[int, ...], PyAcc]] = None) -> None:
        self._heads: Dict[Tuple[int, ...], PyAcc] = heads if heads is not None else {}

    @classmethod
    def from_stacks(cls: Type["GSS"], stacks: List[Tuple[List[int], PyAcc]], node_factory: Optional[Any] = None) -> "GSS":
        """
        Build a GSS from explicit stacks.

        node_factory is ignored in this simplified implementation; we keep it in the
        signature for compatibility with existing call sites.
        """
        m: Dict[Tuple[int, ...], PyAcc] = {}
        for vals, acc in stacks:
            key = tuple(vals)
            if key in m:
                m[key] = m[key].merge(acc)
            else:
                m[key] = acc
        return cls(m)

    def _clone_with(self, heads: Optional[Dict[Tuple[int, ...], PyAcc]] = None) -> "GSS":
        return GSS(dict(self._heads) if heads is None else heads)

    # --- Introspection helpers (used by profiler) ---

    def head_count(self) -> int:
        # Non-empty stacks only
        return sum(1 for k in self._heads.keys() if len(k) > 0)

    def acc_count(self) -> int:
        # One accumulator per unique stack
        return len(self._heads)

    def stack_count(self) -> int:
        # Unique stacks (including empty if present)
        return len(self._heads)

    def _iter_stack_tuples(self, limit: Optional[int] = None) -> Iterable[Tuple[int, ...]]:
        n = 0
        for k in self._heads.keys():
            yield k
            n += 1
            if limit is not None and n >= limit:
                break

    def compute_stats(self) -> GSSStats:
        if not self._heads:
            return GSSStats(0, 0, 0, 0, 0, 0, 0.0)
        depths = [len(k) for k in self._heads.keys()]
        dmin = min(depths) if depths else 0
        dmax = max(depths) if depths else 0
        dmean = (sum(depths) / len(depths)) if depths else 0.0
        empty_present = 1 if () in self._heads else 0
        heads_nonempty = sum(1 for k in self._heads.keys() if k)
        return GSSStats(
            heads=heads_nonempty,
            empty_stacks=empty_present,
            unique_stacks=len(self._heads),
            accs=len(self._heads),
            depth_min=dmin,
            depth_max=dmax,
            depth_mean=dmean,
        )

    # --- Core operations ---

    def push(self, value: int) -> "GSS":
        new_heads: Dict[Tuple[int, ...], PyAcc] = {}
        for stack, acc in self._heads.items():
            new_stack = stack + (value,)
            if new_stack in new_heads:
                new_heads[new_stack] = new_heads[new_stack].merge(acc)
            else:
                new_heads[new_stack] = acc
        return GSS(new_heads)

    def pop(self) -> "GSS":
        return self.popn(1)

    def popn(self, n: int) -> "GSS":
        if n <= 0:
            return self
        new_heads: Dict[Tuple[int, ...], PyAcc] = {}
        for stack, acc in self._heads.items():
            if len(stack) <= n:
                new_stack: Tuple[int, ...] = ()
            else:
                new_stack = stack[:-n]
            if new_stack in new_heads:
                new_heads[new_stack] = new_heads[new_stack].merge(acc)
            else:
                new_heads[new_stack] = acc
        return GSS(new_heads)

    def is_empty(self) -> bool:
        return not self._heads

    def isolate(self, value: Optional[int]) -> "GSS":
        if value is None:
            acc = self._heads.get(())
            return GSS({(): acc} if acc is not None else {})
        new_heads: Dict[Tuple[int, ...], PyAcc] = {}
        for stack, acc in self._heads.items():
            if stack and stack[-1] == value:
                new_heads[stack] = acc
        return GSS(new_heads)

    def _partition_by_top(self) -> Dict[int, Dict[Tuple[int, ...], PyAcc]]:
        groups: Dict[int, Dict[Tuple[int, ...], PyAcc]] = {}
        for stack, acc in self._heads.items():
            if not stack:
                continue
            top = stack[-1]
            bucket = groups.get(top)
            if bucket is None:
                bucket = {}
                groups[top] = bucket
            bucket[stack] = acc
        return groups

    def apply(self, func: Callable[[PyAcc], PyAcc]) -> "GSS":
        return GSS({stack: func(acc) for stack, acc in self._heads.items()})

    def prune(self, predicate: Callable[[PyAcc], bool]) -> "GSS":
        return GSS({stack: acc for stack, acc in self._heads.items() if predicate(acc)})

    def peek(self) -> Set[int]:
        return {stack[-1] for stack in self._heads.keys() if stack}

    def reduce_acc(self) -> Optional[PyAcc]:
        combined: Optional[PyAcc] = None
        for acc in self._heads.values():
            combined = acc if combined is None else combined.merge(acc)
        return combined

    @staticmethod
    def merge(gss_list: Iterable["GSS"]) -> "GSS":
        merged: Dict[Tuple[int, ...], PyAcc] = {}
        num_inputs = 0
        input_stacks_total = 0

        gss_list = list(gss_list)

        for gss in gss_list:
            num_inputs += 1
            input_stacks_total += gss.stack_count()
            for stack, acc in gss._heads.items():
                if stack in merged:
                    merged[stack] = merged[stack].merge(acc)
                else:
                    merged[stack] = acc

        output_stacks = len(merged)
        if GSS._active_profiler is not None:
            GSS._active_profiler.on_gss_merge(num_inputs, input_stacks_total, output_stacks)

        return GSS(merged)


def get_disallowed_terminals_py(gss: GSS) -> ffi.HybridL2Bitset:
    acc = gss.reduce_acc()
    return ffi.HybridL2Bitset.all() if acc is None else acc.terminals_union.complement()


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
        self._commit_step: int = 0  # for profiling

        dumps = json.dumps
        bs_from_json = ffi.Bitset.from_json_string

        for uid, node in self.arena.items():
            self.max_depth[int(uid)] = int(node.get("max_depth", 0) or 0)
            children = node.get("children") or []
            if not children:
                node["children"] = []
                continue
            new_children = []
            for (pop, llm_bv_json), dest_map in children:
                llm_bv = bs_from_json(dumps(llm_bv_json))
                new_dest_map = []
                for dest_idx, state_bv_json in dest_map:
                    new_dest_map.append((int(dest_idx), bs_from_json(dumps(state_bv_json))))
                new_children.append(((int(pop), llm_bv), new_dest_map))
            node["children"] = new_children

    @staticmethod
    def from_json_string(s: str) -> 'Model':
        data = json.loads(s)
        roots_map = data["precomputed3"]
        arena_json = data["trie3_god"]
        arena = {int(k): v for k, v in arena_json.get("values", [])}
        model = Model(roots_map, arena)

        constraint = ffi.GrammarConstraint.from_json_string(s)
        model.tokenizer = constraint.tokenizer()
        model.glr_parser = constraint.glr_parser()
        model.ignore_terminal_id = model.glr_parser.ignore_terminal_id
        model.tokenizer_initial_state = model.tokenizer.initial_state_id()

        parser_data = data['parser']
        table_data = parser_data['stage_7_table']
        start_state_id = parser_data['start_state_id']

        def _parse_action(ad):
            v = ad['variant']
            if v == 'Shift':
                return ad['state_id']
            if v == 'Reduce':
                return Reduce(ad['nonterminal_id'], ad['len'], tuple(sorted(ad['production_ids'])))
            if v == 'Split':
                reduces: Dict[int, Dict[int, Tuple[int, ...]]] = {
                    int(length): {int(nt): tuple(sorted(pids)) for nt, pids in nts}
                    for length, nts in ad['reduces']
                }
                return Split(ad['shift'], reduces)
            return None

        py_table: Dict[int, Row] = {}
        for state_id_str, row in table_data:
            state_id = int(state_id_str)
            actions = {int(tid): act for tid, ad in row['shifts_and_reduces_full'] if (act := _parse_action(ad)) is not None}
            gotos = {int(nt): gd['state_id'] for nt, gd in row['gotos'] if gd['state_id'] is not None}
            py_table[state_id] = Row(actions, gotos)
        model.parser_table = ParserTable(start_state_id, py_table)

        initial_acc = PyAcc(terminals_union=ffi.HybridL2Bitset.all())
        initial_gss = GSS.from_stacks([([], initial_acc)]).push(model.parser_table.start_state_id)
        model.state = {model.tokenizer_initial_state: initial_gss}

        model.id_to_token = {v: bytes(k) for k, v in data['llm_token_map']}
        raw_pm = constraint.possible_matches()
        model.possible_matches_cache = {int(tsid): {int(tid): bv for tid, bv in inner.items()} for tsid, inner in raw_pm.items()}
        model.internal_to_original_map = constraint.internal_to_original_map()
        model.all_internal_llm_tokens_bitset = constraint.all_internal_llm_tokens_bitset()
        return model

    # --- Internal helpers for commit() profiling ---

    def _gss_stats_map(self, m: Dict[int, GSS]) -> Dict[int, GSSStats]:
        return {sid: gss.compute_stats() for sid, gss in m.items()}

    def _prune_disallowed_terminals(self, gss: GSS, terminals_map: Dict[int, ffi.Bitset]) -> GSS:
        def predicate(acc: PyAcc) -> bool:
            allowed_l2 = acc.terminals_union
            for state_id, matched_bv in terminals_map.items():
                if not matched_bv.is_subset(allowed_l2.get_l2_bitset(state_id)):
                    return False
            return True
        return gss.prune(predicate)

    def _map_allowed_terminals_tokenizer_states(self, gss: GSS, state_map: Dict[int, int]) -> GSS:
        def apply_map(acc: PyAcc) -> PyAcc:
            old_l2 = acc.terminals_union
            new_bvs: Dict[int, ffi.Bitset] = collections.defaultdict(ffi.Bitset.zeros)
            for old_sid, new_sid in state_map.items():
                new_bvs[new_sid] = new_bvs[new_sid].union(old_l2.get_l2_bitset(old_sid))
            new_l2 = ffi.HybridL2Bitset.all()
            for new_sid, bv in new_bvs.items():
                new_l2.insert_l2_bitset(new_sid, bv)
            return PyAcc(new_l2)
        return gss.apply(apply_map)

    def _disallow_terminal_in_state(self, gss: GSS, state_id: int, terminal_id: int) -> GSS:
        def apply_disallow(acc: PyAcc) -> PyAcc:
            current_l2 = acc.terminals_union
            new_l2 = current_l2.union(current_l2)  # clone
            curr_bv = current_l2.get_l2_bitset(state_id)
            if curr_bv.contains(terminal_id):
                new_l2.insert_l2_bitset(state_id, curr_bv.difference(ffi.Bitset.from_indices([terminal_id])))
            return PyAcc(new_l2)
        return gss.apply(apply_disallow)

    def get_root(self, state_id: int) -> int:
        return self.roots_map[int(state_id)]

    def is_end(self, node: int) -> bool:
        return bool((self.arena.get(node, {}).get("value") or {}).get("clean_end", False))

    def iter_edges(self, node: int, token: int):
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
        """
        Core incremental update. This method is the main focus for performance analysis.
        Optional profiling/tracing is controlled by environment variables (see _CommitProfiler).
        """
        self._commit_step += 1
        t0 = time.perf_counter()
        token_bytes = self.id_to_token[token_id]

        profiler: Optional[_CommitProfiler] = None
        if _env_flag("PROFILE_COMMIT", False) or _env_flag("DEBUG_COMMIT_VERBOSE", False):
            profiler = _CommitProfiler(self._commit_step, token_bytes)
            # Pre-state summary
            for sid, st in self._gss_stats_map(self.state).items():
                profiler.add_pre_stats(sid, st)

        # Build tokenizer maps
        terminals_map: Dict[int, ffi.Bitset] = {}
        state_map: Dict[int, int] = {}
        for tokenizer_sid in self.state.keys():
            end_state, matches = self.tokenizer.execute_from_state(token_bytes, tokenizer_sid)
            if end_state is not None:
                state_map[tokenizer_sid] = end_state
            b = ffi.Bitset.zeros()
            for terminal_id, _ in matches:
                b.insert(terminal_id)
            terminals_map[tokenizer_sid] = b

        # Prune and map per-state GSS
        current_state_for_processing: Dict[int, GSS] = {}
        for tokenizer_sid, gss in self.state.items():
            pruned = self._prune_disallowed_terminals(gss, terminals_map)
            if not pruned.is_empty():
                current_state_for_processing[tokenizer_sid] = self._map_allowed_terminals_tokenizer_states(pruned, state_map)

        # Accumulate results incrementally per tokenizer state (avoid large lists)
        new_states: Dict[int, GSS] = {}
        q = collections.deque((0, sid, gss) for sid, gss in current_state_for_processing.items())
        visited_q_items: set = set()
        if profiler is not None:
            profiler.on_queue_enqueued(len(q))

        # Small cache for tokens accessible from tokenizer end states
        accessible_cache: Dict[int, Set[int]] = {}

        while q:
            offset, tokenizer_sid, gss = q.popleft()
            if profiler is not None:
                profiler.on_queue_pop(offset)
            q_key = (offset, tokenizer_sid, id(gss))
            if q_key in visited_q_items:
                continue
            visited_q_items.add(q_key)

            end_state, matches = self.tokenizer.execute_from_state(token_bytes[offset:], tokenizer_sid)
            if profiler is not None:
                profiler.on_matches(offset, matches)

            for terminal_id, width in matches:
                # Process parser transitions for this terminal
                processed_gss = gss if terminal_id == self.ignore_terminal_id else self._process_token(gss, terminal_id, profiler=profiler)

                # Prune the same terminal from the tokenizer end_state to avoid double-consume
                if end_state is not None:
                    if end_state not in accessible_cache:
                        accessible_cache[end_state] = set(self.tokenizer.tokens_accessible_from_state(end_state))
                    if terminal_id in accessible_cache[end_state]:
                        processed_gss = self._disallow_terminal_in_state(processed_gss, end_state, terminal_id)
                        if profiler is not None:
                            profiler.on_end_state_prune()

                if not processed_gss.is_empty():
                    new_offset = offset + width
                    next_sid = self.tokenizer_initial_state
                    if new_offset == len(token_bytes):
                        # Merge incrementally
                        existing = new_states.get(next_sid)
                        if existing is None:
                            new_states[next_sid] = processed_gss
                        else:
                            new_states[next_sid] = GSS.merge([existing, processed_gss])
                    else:
                        q.append((new_offset, next_sid, processed_gss))
                        if profiler is not None:
                            profiler.on_enqueue_after_match()

            if end_state is not None:
                # carry-forward without consumption
                existing = new_states.get(end_state)
                if existing is None:
                    new_states[end_state] = gss
                else:
                    new_states[end_state] = GSS.merge([existing, gss])
                if profiler is not None:
                    profiler.on_carry_forward()

        # Prune empty GSS states
        self.state = {sid: st for sid, st in new_states.items() if not st.is_empty()}

        t1 = time.perf_counter()

        if profiler is not None:
            for sid, st in self._gss_stats_map(self.state).items():
                profiler.add_post_stats(sid, st)
            profiler.finalize()

        if os.environ.get("REPORT_COMMIT_TIME") == "1":
            print(f"commit (ms): {round((t1 - t0) * 1000, 2)}")

    def _process_token(self, gss: GSS, terminal_id: int, profiler: Optional[_CommitProfiler] = None) -> GSS:
        # Seed by partitioning once (avoid O(H^2) from repeated isolate)
        heads_by_state: Dict[int, List[GSS]] = collections.defaultdict(list)
        for state_id, heads in gss._partition_by_top().items():
            heads_by_state[state_id].append(GSS(heads))

        shifted_gsses: List[GSS] = []
        c_shifts = 0
        c_reduces = 0
        c_splits = 0
        c_gotos = 0

        while heads_by_state:
            state_id, state_gsss = heads_by_state.popitem()
            state_gss = GSS.merge(state_gsss)
            row = self.parser_table.table.get(state_id)
            if not row:
                continue
            action = row.actions.get(terminal_id)
            if not action:
                continue

            if isinstance(action, int):  # Shift
                shifted_gsses.append(state_gss.push(action))
                c_shifts += 1
                continue

            if isinstance(action, Reduce):
                c_reduces += 1
                popped = state_gss.popn(action.len)
                # Partition once by from_state_id after reduction
                groups = popped._partition_by_top()
                for from_state_id, heads in groups.items():
                    goto_state_id = self.parser_table.table[from_state_id].gotos[action.nonterminal_id]
                    heads_by_state[goto_state_id].append(GSS(heads).push(goto_state_id))
                    c_gotos += 1
                continue

            # Split
            c_splits += 1
            if action.shift is not None:
                shifted_gsses.append(state_gss.push(action.shift))
                c_shifts += 1
            for length, nts in action.reduces.items():
                popped = state_gss.popn(length)
                groups = popped._partition_by_top()
                for from_state_id, heads in groups.items():
                    table_row = self.parser_table.table[from_state_id]
                    for nt_id in nts.keys():
                        goto_state_id = table_row.gotos[nt_id]
                        heads_by_state[goto_state_id].append(GSS(heads).push(goto_state_id))
                        c_gotos += 1

        if profiler is not None:
            profiler.on_process_token(terminal_id, shifts=c_shifts, reduces=c_reduces, splits=c_splits, gotos=c_gotos)

        return GSS() if not shifted_gsses else GSS.merge(shifted_gsses)

    def get_mask(self) -> RangeSet:
        """
        Compute the final LLM token mask by traversing the precomputed trie with the current GSS.
        """
        active_states = self.state
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
        for sid, gss in active_states.items():
            root_idx = roots_map.get(int(sid))
            if root_idx is None:
                continue
            root_idx = int(root_idx)
            existing = values.get(root_idx)
            if existing is not None:
                egss, emask = existing
                values[root_idx] = (GSS.merge([egss, gss]), emask.union(all_ones_mask))
            else:
                values[root_idx] = (gss, all_ones_mask)
            d = max_depth[root_idx]
            bucket = todo.get(d)
            if bucket is None:
                todo[d] = {root_idx}
                heappush(depth_heap, d)
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
                    disallowed_l2 = get_disallowed_terminals_py(gss_node)
                    possible_matches = self.possible_matches_cache

                    for (start, end), disallowed_bv in disallowed_l2.range_values():
                        if disallowed_bv.is_empty():
                            continue
                        end = min(end, self.tokenizer.max_state())
                        for tsid in range(start, end + 1):
                            pm = possible_matches.get(tsid)
                            if not pm:
                                continue
                            for terminal_id, llm_tokens_for_terminal in pm.items():
                                if disallowed_bv.contains(terminal_id):
                                    forbidden_llm_tokens = forbidden_llm_tokens.union(llm_tokens_for_terminal)

                    allowed_tokens = llm_mask.difference(forbidden_llm_tokens)
                    if not allowed_tokens.is_empty():
                        final_mask = final_mask.union(allowed_tokens)

                if llm_mask.is_empty():
                    stopped.add(node_idx)
                    continue

                # Transitions
                for (pop, llm_bv), dests in arena.get(node_idx, {}).get("children") or []:
                    popped = gss_node.popn(pop)
                    llm_empty = llm_bv.is_empty()
                    for dest_idx, state_bv in dests:
                        matched: List[GSS] = []
                        if not state_bv.is_empty():
                            for sid_val in popped.peek():
                                if state_bv.contains(sid_val):
                                    matched.append(popped.isolate(sid_val))
                        if not matched:
                            continue
                        child_gss = GSS.merge(matched)
                        child_mask = llm_mask if llm_empty else llm_mask.intersection(llm_bv)
                        d = int(dest_idx)
                        existing = values.get(d)
                        if existing is not None:
                            egss, emask = existing
                            values[d] = (GSS.merge([egss, child_gss]), emask.union(child_mask))
                        else:
                            values[d] = (child_gss, child_mask)
                        enqueue(max_depth[d], d)

        # Convert internal mask back to original IDs
        original_mask = ffi.Bitset.zeros()
        for internal_id in final_mask.to_indices():
            orig = self.internal_to_original_map.get(internal_id)
            if orig is not None:
                original_mask.insert(orig)
        return RangeSet.from_ranges(original_mask.to_ranges())
