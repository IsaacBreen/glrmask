from __future__ import annotations

import time
import tracemalloc
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple, Optional, Type, Iterable

from ..interface import GSS, MergeableInt, T, Acc


@dataclass
class WorkloadResult:
    name: str
    preset: str
    params: Dict[str, Any]
    phases: List[Dict[str, Any]]
    totals: Dict[str, Any]
    error: Optional[str] = None
    timed_out: bool = False


class TimingCollector:
    """Collects timing statistics for GSS method calls."""
    def __init__(self):
        self.stats: Dict[str, Dict[str, Any]] = {}

    def record(self, method_name: str, ms: float):
        if method_name not in self.stats:
            self.stats[method_name] = {'calls': 0, 'total_ms': 0.0}
        self.stats[method_name]['calls'] += 1
        self.stats[method_name]['total_ms'] += ms

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        return self.stats

    def reset(self):
        self.stats = {}


class BenchmarkContext:
    """Manages state for a single benchmark run, including timing and phases."""
    def __init__(self, mem_profile: bool):
        self.collector = TimingCollector()
        self.phases: List[Dict[str, Any]] = []
        self.mem_profile = mem_profile
        self.start_ms = _now_ms()
        self.timed_out = False
        self._memctx = _with_mem_tracking(mem_profile)

    def __enter__(self):
        self._memctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._memctx.__exit__(exc_type, exc_val, exc_tb)

    @contextmanager
    def phase(self, name: str, **kwargs):
        self.collector.reset()
        p_start_ms = _now_ms()
        phase_data = {"phase": name, **kwargs}
        yield phase_data
        phase_data['ms'] = _now_ms() - p_start_ms
        phase_data['method_stats'] = self.collector.get_stats()
        self.phases.append(phase_data)
        self._memctx.checkpoint()

    @property
    def peak_mem_bytes(self) -> int:
        return self._memctx.peak_bytes


def create_timed_gss_class(gss_class: Type[GSS], collector: TimingCollector) -> Type[GSS]:
    """Dynamically creates a GSS wrapper class that records method timings."""

    class TimedGSS(GSS[T, Acc]):
        _collector_ref = collector
        _wrapped_class_ref = gss_class

        def __init__(self, wrapped_instance: GSS[T, Acc]):
            self._wrapped = wrapped_instance

        @classmethod
        def from_stacks(cls: Type, stacks: List[Tuple[List[T], Acc]]) -> TimedGSS[T, Acc]:
            start_ms = _now_ms()
            result = cls._wrapped_class_ref.from_stacks(stacks)
            cls._collector_ref.record('from_stacks', _now_ms() - start_ms)
            return cls(result)

        def to_stacks(self) -> List[Tuple[List[T], Acc]]:
            start_ms = _now_ms()
            result = self._wrapped.to_stacks()
            self._collector_ref.record('to_stacks', _now_ms() - start_ms)
            return result

        def push(self: TimedGSS, value: T) -> TimedGSS[T, Acc]:
            start_ms = _now_ms()
            result = self._wrapped.push(value)
            self._collector_ref.record('push', _now_ms() - start_ms)
            return TimedGSS(result)

        def pop(self: TimedGSS) -> TimedGSS[T, Acc]:
            start_ms = _now_ms()
            result = self._wrapped.pop()
            self._collector_ref.record('pop', _now_ms() - start_ms)
            return TimedGSS(result)
        
        def popn(self: TimedGSS, n: int) -> TimedGSS[T, Acc]:
            start_ms = _now_ms()
            result = self._wrapped.popn(n)
            self._collector_ref.record('popn', _now_ms() - start_ms)
            return TimedGSS(result)

        def is_empty(self) -> bool:
            start_ms = _now_ms()
            result = self._wrapped.is_empty()
            self._collector_ref.record('is_empty', _now_ms() - start_ms)
            return result

        def isolate(self: TimedGSS, value: Optional[T]) -> TimedGSS[T, Acc]:
            start_ms = _now_ms()
            result = self._wrapped.isolate(value)
            self._collector_ref.record('isolate', _now_ms() - start_ms)
            return TimedGSS(result)

        def apply(self: TimedGSS, func: Callable[[Acc], Acc]) -> TimedGSS[T, Acc]:
            start_ms = _now_ms()
            result = self._wrapped.apply(func)
            self._collector_ref.record('apply', _now_ms() - start_ms)
            return TimedGSS(result)

        def prune(self: TimedGSS, predicate: Callable[[Acc], bool]) -> TimedGSS[T, Acc]:
            start_ms = _now_ms()
            result = self._wrapped.prune(predicate)
            self._collector_ref.record('prune', _now_ms() - start_ms)
            return TimedGSS(result)

        def merge(self: TimedGSS, other: GSS[T, Acc]) -> TimedGSS[T, Acc]:
            # Unwrap the other GSS if it's also a TimedGSS
            other_wrapped = other._wrapped if isinstance(other, TimedGSS) else other
            start_ms = _now_ms()
            result = self._wrapped.merge(other_wrapped)
            self._collector_ref.record('merge', _now_ms() - start_ms)
            return TimedGSS(result)

        @classmethod
        def merge_many(cls: Type, gss_list: Iterable[GSS[T, Acc]]) -> TimedGSS[T, Acc]:
            # Unwrap all GSS instances in the list
            unwrapped_list = [g._wrapped if isinstance(g, TimedGSS) else g for g in gss_list]
            start_ms = _now_ms()
            result = cls._wrapped_class_ref.merge_many(unwrapped_list)
            cls._collector_ref.record('merge_many', _now_ms() - start_ms)
            return cls(result)

        def peek(self) -> Set[T]:
            start_ms = _now_ms()
            result = self._wrapped.peek()
            self._collector_ref.record('peek', _now_ms() - start_ms)
            return result

        def reduce_acc(self) -> Optional[Acc]:
            start_ms = _now_ms()
            result = self._wrapped.reduce_acc()
            self._collector_ref.record('reduce_acc', _now_ms() - start_ms)
            return result
            
    return TimedGSS


@dataclass
class Workload:
    """
    Represents a benchmark workload.

    - name: stable name used in filtering and reports
    - description: human-readable description
    - param_presets: mapping preset name -> param dict
    - runner: callable that executes the workload and returns WorkloadResult
    """
    name: str
    description: str
    param_presets: Dict[str, Dict[str, Any]]
    runner: Callable[[Type[GSS], Dict[str, Any], str, int, BenchmarkContext], WorkloadResult]


# Registry for discovery/filtering
WORKLOADS: Dict[str, Workload] = {}


def register_workload(w: Workload):
    if w.name in WORKLOADS:
        raise ValueError(f"Duplicate workload name: {w.name}")
    WORKLOADS[w.name] = w


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _with_mem_tracking(enabled: bool):
    class MemCtx:
        def __enter__(self):
            if enabled:
                tracemalloc.start()
                self._start = tracemalloc.get_traced_memory()[1]  # peak so far
            else:
                self._start = 0
            self._peak = 0
            return self

        def checkpoint(self):
            if enabled:
                current, peak = tracemalloc.get_traced_memory()
                self._peak = max(self._peak, peak)
                return current, peak
            return 0, 0

        def __exit__(self, exc_type, exc, tb):
            if enabled:
                _, peak = tracemalloc.get_traced_memory()
                self._peak = max(self._peak, peak)
                tracemalloc.stop()
            return False

        @property
        def peak_bytes(self):
            return int(self._peak)

    return MemCtx()


def _maybe_timeout(start_ms: float, max_ms: Optional[float]) -> bool:
    if max_ms is None:
        return False
    return (_now_ms() - start_ms) > max_ms


def _merge_many(cls: Type[GSS], lst: Iterable[GSS]) -> GSS:
    # use provided interface method
    return cls.merge_many(lst)


def _count_stacks_safely(gss: GSS) -> int:
    """
    Counts stacks via to_stacks. This can be expensive; use sparingly.
    """
    try:
        stacks = gss.to_stacks()
        return len(stacks)
    except Exception:
        return -1


def _default_presets(
    tiny: Dict[str, Any],
    small: Dict[str, Any],
    medium: Dict[str, Any],
    large: Dict[str, Any],
    max_ms_tiny: float = 2_000,
    max_ms_small: float = 10_000,
    max_ms_medium: float = 30_000,
    max_ms_large: float = 120_000,
) -> Dict[str, Dict[str, Any]]:
    """
    Utility to add per-preset defaults including max_ms budget.
    """
    tiny = dict(tiny)
    small = dict(small)
    medium = dict(medium)
    large = dict(large)
    tiny.setdefault("max_ms", max_ms_tiny)
    small.setdefault("max_ms", max_ms_small)
    medium.setdefault("max_ms", max_ms_medium)
    large.setdefault("max_ms", max_ms_large)
    return {
        "tiny": tiny,
        "small": small,
        "medium": medium,
        "large": large,
    }


def _build_wide_dag(
    gss_class: Type[GSS],
    depth: int,
    branching: int,
    seed: int,
    acc0: MergeableInt = MergeableInt(0),
) -> Tuple[GSS, Dict[str, Any]]:
    """
    Builds a wide DAG via repeated push and merge_many on clones.

    At each level l, produce 'branching' clones of the current GSS, each pushed with a distinct
    value that encodes (level, branch). Then merge_many these clones.
    The number of stacks grows like branching**depth; the underlying structure should share heavily.

    Returns:
      gss: constructed GSS
      info: dict with per-level timing and totals
    """
    rng = random.Random(seed)
    info: Dict[str, Any] = {
        "levels": [],
        "depth": depth,
        "branching": branching
    }

    t0 = _now_ms()
    g = gss_class.from_stacks([([], acc0)])
    t1 = _now_ms()
    info["init_ms"] = t1 - t0

    for lvl in range(depth):
        level_info: Dict[str, Any] = {"level": lvl}
        push_start = _now_ms()
        clones: List[GSS] = []
        # Use integer labels to keep lightweight values
        for b in range(branching):
            val = (lvl << 20) | b
            clones.append(g.push(val))
        push_ms = _now_ms() - push_start
        level_info["push_clones_ms"] = push_ms
        level_info["num_clones"] = len(clones)

        merge_start = _now_ms()
        g = _merge_many(gss_class, clones)
        merge_ms = _now_ms() - merge_start
        level_info["merge_many_ms"] = merge_ms

        info["levels"].append(level_info)

    info["total_ms"] = _now_ms() - t0
    return g, info


def _workload_split_modify_merge_shared(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    context: BenchmarkContext,
) -> WorkloadResult:
    """
    Build a wide DAG with substantial sharing, then create k shallowly-modified clones and merge them all.
    The shallow modification is a single push with a per-clone tag. This should exercise the ability to
    merge variants without revisiting the entire base structure if sharing is preserved.
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    depth = int(params["depth"])
    branching = int(params["branching"])
    clones_n = int(params["clones"])
    totals: Dict[str, Any] = {}

    try:
        with context:
            with context.phase("build_base") as phase:
                base, build_info = _build_wide_dag(gss_class, depth, branching, seed)
                phase["detail"] = build_info
                if measure_counts:
                    c0 = _now_ms()
                    phase["base_stack_count"] = _count_stacks_safely(base)
                    phase["count_ms"] = _now_ms() - c0
                if _maybe_timeout(context.start_ms, max_ms):
                    context.timed_out = True

            if not context.timed_out:
                with context.phase("create_modified_clones", num_clones=clones_n) as phase:
                    clones: List[GSS] = []
                    for i in range(clones_n):
                        tag = (depth + 1) << 20 | i
                        clones.append(base.push(tag))
                        if _maybe_timeout(context.start_ms, max_ms):
                            context.timed_out = True
                            break

            if not context.timed_out:
                with context.phase("merge_clones") as phase:
                    merged = _merge_many(gss_class, clones)
                    if measure_counts:
                        c0 = _now_ms()
                        phase["merged_stack_count"] = _count_stacks_safely(merged)
                        phase["count_ms"] = _now_ms() - c0

            totals["total_ms"] = _now_ms() - context.start_ms
            totals["peak_mem_bytes"] = context.peak_mem_bytes
            totals["timed_out"] = context.timed_out
            totals["base_depth"] = depth
            totals["base_branching"] = branching
            totals["num_clones"] = clones_n

        return WorkloadResult(
            name="split_modify_merge_shared", preset=preset, params=params,
            phases=context.phases, totals=totals, error=None, timed_out=context.timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="split_modify_merge_shared", preset=preset, params=params,
            phases=context.phases, totals=totals, error=f"{type(e).__name__}: {e}", timed_out=context.timed_out,
        )


def _workload_pairwise_merge_pop(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    context: BenchmarkContext,
) -> WorkloadResult:
    """
    Build a moderately wide base, then produce several pairs of shallowly different clones,
    merge each pair, and immediately pop.
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    depth = int(params["depth"])
    branching = int(params["branching"])
    pairs = int(params["pairs"])
    totals: Dict[str, Any] = {}

    try:
        with context:
            with context.phase("build_base") as phase:
                base, info = _build_wide_dag(gss_class, depth, branching, seed)
                phase["detail"] = info
                if measure_counts:
                    c0 = _now_ms()
                    phase["base_stack_count"] = _count_stacks_safely(base)
                    phase["count_ms"] = _now_ms() - c0

            with context.phase("pairwise_merge_and_pop") as phase:
                for i in range(pairs):
                    if _maybe_timeout(context.start_ms, max_ms):
                        context.timed_out = True
                        break
                    tag_a = (depth + 1) << 20 | (2 * i)
                    tag_b = (depth + 1) << 20 | (2 * i + 1)
                    g_a = base.push(tag_a)
                    g_b = base.push(tag_b)
                    merged = g_a.merge(g_b)
                    popped = merged.pop()
                phase['num_pairs'] = i + 1

            totals["total_ms"] = _now_ms() - context.start_ms
            totals["peak_mem_bytes"] = context.peak_mem_bytes
            totals["timed_out"] = context.timed_out
            totals["base_depth"] = depth
            totals["base_branching"] = branching
            totals["pairs"] = pairs

        return WorkloadResult(
            name="pairwise_merge_pop", preset=preset, params=params,
            phases=context.phases, totals=totals, error=None, timed_out=context.timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="pairwise_merge_pop", preset=preset, params=params,
            phases=context.phases, totals=totals, error=f"{type(e).__name__}: {e}", timed_out=context.timed_out,
        )


def _workload_isolate_rare_top(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    context: BenchmarkContext,
) -> WorkloadResult:
    """
    Build a wide base with many distinct top-of-stack values, then isolate on a value
    that is present in only a tiny fraction of stacks.
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    depth = int(params["depth"])
    branching = int(params["branching"])
    repeats = int(params["repeats"])
    target_index = int(params.get("target_index", 0))
    totals: Dict[str, Any] = {}

    try:
        with context:
            with context.phase("build_base") as phase:
                base, info = _build_wide_dag(gss_class, depth, branching, seed)
                phase["detail"] = info
                if measure_counts:
                    c0 = _now_ms()
                    phase["base_stack_count"] = _count_stacks_safely(base)
                    phase["count_ms"] = _now_ms() - c0

            rare_val = ((depth - 1) << 20) | target_index
            with context.phase("isolate_rare_top", repeats=repeats, rare_val=rare_val) as phase:
                post_counts = []
                for i in range(repeats):
                    isolated = base.isolate(rare_val)
                    if measure_counts:
                        post_counts.append(_count_stacks_safely(isolated))
                    if _maybe_timeout(context.start_ms, max_ms):
                        context.timed_out = True
                        break
                if measure_counts:
                    phase["isolated_stack_count_stats"] = _stats(post_counts)

            totals["total_ms"] = _now_ms() - context.start_ms
            totals["peak_mem_bytes"] = context.peak_mem_bytes
            totals["timed_out"] = context.timed_out
            totals["base_depth"] = depth
            totals["base_branching"] = branching
            totals["rare_val"] = rare_val

        return WorkloadResult(
            name="isolate_rare_top", preset=preset, params=params,
            phases=context.phases, totals=totals, error=None, timed_out=context.timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="isolate_rare_top", preset=preset, params=params,
            phases=context.phases, totals=totals, error=f"{type(e).__name__}: {e}", timed_out=context.timed_out,
        )


def _workload_apply_peek_reduce(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    context: BenchmarkContext,
) -> WorkloadResult:
    """
    Build a wide base; then measure apply() over the whole structure,
    followed by peek() and reduce_acc().
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    depth = int(params["depth"])
    branching = int(params["branching"])
    apply_steps = int(params["apply_steps"])
    apply_inc = int(params["apply_increment"])
    totals: Dict[str, Any] = {}

    try:
        with context:
            with context.phase("build_base") as phase:
                base, info = _build_wide_dag(gss_class, depth, branching, seed)
                phase["detail"] = info
                if measure_counts:
                    c0 = _now_ms()
                    phase["base_stack_count"] = _count_stacks_safely(base)
                    phase["count_ms"] = _now_ms() - c0
            
            g = base
            with context.phase("apply_steps", steps=apply_steps) as phase:
                for _ in range(apply_steps):
                    g = g.apply(lambda acc, inc=apply_inc: MergeableInt(int(acc) + inc))
                    if _maybe_timeout(context.start_ms, max_ms):
                        context.timed_out = True
                        break

            if not context.timed_out:
                with context.phase("peek_and_reduce") as phase:
                    peek_result = g.peek()
                    red = g.reduce_acc()
                    phase["peek_cardinality"] = len(peek_result)
                    phase["reduce_result"] = int(red) if red is not None else None

            totals["total_ms"] = _now_ms() - context.start_ms
            totals["peak_mem_bytes"] = context.peak_mem_bytes
            totals["timed_out"] = context.timed_out

        return WorkloadResult(
            name="apply_peek_reduce", preset=preset, params=params,
            phases=context.phases, totals=totals, error=None, timed_out=context.timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="apply_peek_reduce", preset=preset, params=params,
            phases=context.phases, totals=totals, error=f"{type(e).__name__}: {e}", timed_out=context.timed_out,
        )


def _workload_popn_scaling(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    context: BenchmarkContext,
) -> WorkloadResult:
    """
    Build several chains of different depths merged together, then measure popn(k) for various k.
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    chain_depths: List[int] = list(params["chain_depths"])
    popn_values: List[int] = list(params["popn_values"])
    totals: Dict[str, Any] = {}

    try:
        with context:
            with context.phase("build_chains") as phase:
                chains: List[GSS] = []
                for d in chain_depths:
                    g = gss_class.from_stacks([([], MergeableInt(0))])
                    for i in range(d):
                        g = g.push(i)
                    chains.append(g)
                    if _maybe_timeout(context.start_ms, max_ms):
                        context.timed_out = True
                        break
            
            merged = None
            if not context.timed_out:
                with context.phase("merge_chains") as phase:
                    merged = _merge_many(gss_class, chains)
                    if measure_counts:
                        c0 = _now_ms()
                        phase["stack_count"] = _count_stacks_safely(merged)
                        phase["count_ms"] = _now_ms() - c0

            if not context.timed_out and merged is not None:
                with context.phase("popn_runs") as phase:
                    for k in popn_values:
                        res = merged.popn(k)
                        if _maybe_timeout(context.start_ms, max_ms):
                            context.timed_out = True
                            break

            totals["total_ms"] = _now_ms() - context.start_ms
            totals["peak_mem_bytes"] = context.peak_mem_bytes
            totals["timed_out"] = context.timed_out
            totals["chain_depths"] = chain_depths
            totals["popn_values"] = popn_values

        return WorkloadResult(
            name="popn_scaling", preset=preset, params=params,
            phases=context.phases, totals=totals, error=None, timed_out=context.timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="popn_scaling", preset=preset, params=params,
            phases=context.phases, totals=totals, error=f"{type(e).__name__}: {e}", timed_out=context.timed_out,
        )


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "stdev": None}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 0 else 0
    stdev = math.sqrt(var)
    return {
        "count": n,
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "stdev": stdev,
    }


# Register workloads with presets designed to be reasonable on ReferenceGSS for tiny/small,
# and increasingly larger for medium/large presets intended for optimized implementations.

register_workload(Workload(
    name="split_modify_merge_shared",
    description=(
        "Build a wide DAG, create multiple clones with a single additional push (shallow change), "
        "then merge all clones together. This probes whether merges can exploit structural sharing."
    ),
    param_presets=_default_presets(
        tiny={"depth": 3, "branching": 6, "clones": 8, "measure_counts": False},
        small={"depth": 4, "branching": 6, "clones": 16, "measure_counts": False},
        medium={"depth": 5, "branching": 8, "clones": 40, "measure_counts": False},
        large={"depth": 6, "branching": 10, "clones": 200, "measure_counts": False},
        max_ms_tiny=4_000,
        max_ms_small=15_000,
        max_ms_medium=60_000,
        max_ms_large=240_000,
    ),
    runner=_workload_split_modify_merge_shared
))

register_workload(Workload(
    name="pairwise_merge_pop",
    description=(
        "From a common base, produce pairs of shallowly different clones, merge each pair, and pop. "
        "Stresses many merges to a shared parent and subsequent pops."
    ),
    param_presets=_default_presets(
        tiny={"depth": 2, "branching": 8, "pairs": 20, "measure_counts": False},
        small={"depth": 3, "branching": 8, "pairs": 40, "measure_counts": False},
        medium={"depth": 4, "branching": 10, "pairs": 100, "measure_counts": False},
        large={"depth": 5, "branching": 12, "pairs": 300, "measure_counts": False},
        max_ms_tiny=3_000,
        max_ms_small=10_000,
        max_ms_medium=45_000,
        max_ms_large=180_000,
    ),
    runner=_workload_pairwise_merge_pop
))

register_workload(Workload(
    name="isolate_rare_top",
    description=(
        "Construct many distinct top-of-stack values then isolate a single rare value, "
        "repeatedly. Detects isolate performance and indexing."
    ),
    param_presets=_default_presets(
        tiny={"depth": 3, "branching": 10, "repeats": 10, "target_index": 0, "measure_counts": False},
        small={"depth": 4, "branching": 10, "repeats": 20, "target_index": 1, "measure_counts": False},
        medium={"depth": 5, "branching": 12, "repeats": 40, "target_index": 2, "measure_counts": False},
        large={"depth": 6, "branching": 14, "repeats": 80, "target_index": 3, "measure_counts": False},
        max_ms_tiny=3_000,
        max_ms_small=10_000,
        max_ms_medium=30_000,
        max_ms_large=120_000,
    ),
    runner=_workload_isolate_rare_top
))

register_workload(Workload(
    name="apply_peek_reduce",
    description=(
        "Apply a small accumulator increment multiple times across a wide base, then run peek() and reduce_acc(). "
        "Measures iteration costs over the active stacks."
    ),
    param_presets=_default_presets(
        tiny={"depth": 3, "branching": 8, "apply_steps": 3, "apply_increment": 1, "measure_counts": False},
        small={"depth": 4, "branching": 8, "apply_steps": 5, "apply_increment": 2, "measure_counts": False},
        medium={"depth": 5, "branching": 10, "apply_steps": 6, "apply_increment": 3, "measure_counts": False},
        large={"depth": 6, "branching": 12, "apply_steps": 8, "apply_increment": 4, "measure_counts": False},
        max_ms_tiny=3_000,
        max_ms_small=12_000,
        max_ms_medium=30_000,
        max_ms_large=90_000,
    ),
    runner=_workload_apply_peek_reduce
))

register_workload(Workload(
    name="popn_scaling",
    description=(
        "Merge multiple chains of varying depths and then benchmark popn(k) for several k values. "
        "Targets sensitivity of popn to k vs total depth/size."
    ),
    param_presets=_default_presets(
        tiny={"chain_depths": [20, 40, 80], "popn_values": [0, 1, 5, 10], "measure_counts": False},
        small={"chain_depths": [32, 64, 96, 128], "popn_values": [0, 1, 8, 16, 24], "measure_counts": False},
        medium={"chain_depths": [64, 128, 192, 256], "popn_values": [0, 2, 16, 32, 64], "measure_counts": False},
        large={"chain_depths": [128, 256, 384, 512], "popn_values": [0, 4, 32, 64, 128], "measure_counts": False},
        max_ms_tiny=2_500,
        max_ms_small=8_000,
        max_ms_medium=24_000,
        max_ms_large=80_000,
    ),
    runner=_workload_popn_scaling
))


def list_workloads() -> List[Tuple[str, str]]:
    """Returns [(name, description), ...]"""
    return [(w.name, w.description) for w in WORKLOADS.values()]


def resolve_workloads(include: Optional[List[str]], exclude: Optional[List[str]]) -> List[Workload]:
    names = sorted(WORKLOADS.keys())
    selected = names

    if include:
        incl_set = set(name.strip() for name in include if name.strip())
        selected = [n for n in names if n in incl_set]

    if exclude:
        excl_set = set(name.strip() for name in exclude if name.strip())
        selected = [n for n in selected if n not in excl_set]

    return [WORKLOADS[n] for n in selected]
