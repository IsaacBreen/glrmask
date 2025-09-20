from __future__ import annotations

import time
import tracemalloc
import math
import random
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
    runner: Callable[[Type[GSS], Dict[str, Any], str, int, bool], WorkloadResult]


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
    mem_profile: bool,
) -> WorkloadResult:
    """
    Build a wide DAG with substantial sharing, then create k shallowly-modified clones and merge them all.
    The shallow modification is a single push with a per-clone tag. This should exercise the ability to
    merge variants without revisiting the entire base structure if sharing is preserved.

    Params:
      depth: number of levels for _build_wide_dag
      branching: number of clones per level for _build_wide_dag
      clones: number of clones to create from the base
      measure_counts: whether to compute to_stacks() size (expensive)
      max_ms: soft budget for the whole workload
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    depth = int(params["depth"])
    branching = int(params["branching"])
    clones_n = int(params["clones"])

    phases: List[Dict[str, Any]] = []
    timed_out = False
    start_ms = _now_ms()
    totals: Dict[str, Any] = {}

    try:
        with _with_mem_tracking(mem_profile) as memctx:
            # Phase 1: Build base
            phase = {"phase": "build_base"}
            p0 = _now_ms()
            base, build_info = _build_wide_dag(gss_class, depth, branching, seed)
            phase["ms"] = _now_ms() - p0
            phase["detail"] = build_info
            if measure_counts:
                c0 = _now_ms()
                phase["base_stack_count"] = _count_stacks_safely(base)
                phase["count_ms"] = _now_ms() - c0
            phases.append(phase)
            memctx.checkpoint()
            if _maybe_timeout(start_ms, max_ms):
                timed_out = True

            # Phase 2: Shallow modifications on clones
            if not timed_out:
                phase = {"phase": "create_modified_clones", "num_clones": clones_n}
                clone_start = _now_ms()
                clones: List[GSS] = []
                for i in range(clones_n):
                    # push a per-clone tag value shallowly
                    tag = (depth + 1) << 20 | i  # distinct top labels
                    clones.append(base.push(tag))
                    if _maybe_timeout(start_ms, max_ms):
                        timed_out = True
                        break
                phase["ms"] = _now_ms() - clone_start
                phases.append(phase)
                memctx.checkpoint()

            # Phase 3: merge_many clones
            if not timed_out:
                phase = {"phase": "merge_clones"}
                m0 = _now_ms()
                merged = _merge_many(gss_class, clones)
                phase["ms"] = _now_ms() - m0
                if measure_counts:
                    c0 = _now_ms()
                    phase["merged_stack_count"] = _count_stacks_safely(merged)
                    phase["count_ms"] = _now_ms() - c0
                phases.append(phase)
                memctx.checkpoint()

            totals["total_ms"] = _now_ms() - start_ms
            totals["peak_mem_bytes"] = memctx.peak_bytes
            totals["timed_out"] = timed_out
            totals["base_depth"] = depth
            totals["base_branching"] = branching
            totals["num_clones"] = clones_n

        return WorkloadResult(
            name="split_modify_merge_shared",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=None,
            timed_out=timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="split_modify_merge_shared",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=f"{type(e).__name__}: {e}",
            timed_out=timed_out,
        )


def _workload_pairwise_merge_pop(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    mem_profile: bool,
) -> WorkloadResult:
    """
    Build a moderately wide base, then produce several pairs of shallowly different clones,
    merge each pair, and immediately pop. This stresses both merge performance on shared
    parents and correctness of pop-to-shared-parent at scale.

    Params:
      depth: depth for base builder
      branching: branching for base builder
      pairs: number of pairs to create
      measure_counts: whether to compute to_stacks() counts
      max_ms: time budget (soft)
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    depth = int(params["depth"])
    branching = int(params["branching"])
    pairs = int(params["pairs"])

    phases: List[Dict[str, Any]] = []
    timed_out = False
    start_ms = _now_ms()
    totals: Dict[str, Any] = {}

    try:
        with _with_mem_tracking(mem_profile) as memctx:
            # Build base
            phase = {"phase": "build_base"}
            p0 = _now_ms()
            base, info = _build_wide_dag(gss_class, depth, branching, seed)
            phase["ms"] = _now_ms() - p0
            phase["detail"] = info
            if measure_counts:
                c0 = _now_ms()
                phase["base_stack_count"] = _count_stacks_safely(base)
                phase["count_ms"] = _now_ms() - c0
            phases.append(phase)
            memctx.checkpoint()

            # Create and merge pairs
            merged_times_ms: List[float] = []
            popped_times_ms: List[float] = []
            for i in range(pairs):
                if _maybe_timeout(start_ms, max_ms):
                    timed_out = True
                    break
                tag_a = (depth + 1) << 20 | (2 * i)
                tag_b = (depth + 1) << 20 | (2 * i + 1)
                g_a = base.push(tag_a)
                g_b = base.push(tag_b)

                m0 = _now_ms()
                merged = g_a.merge(g_b)
                merged_times_ms.append(_now_ms() - m0)

                p0 = _now_ms()
                popped = merged.pop()
                popped_times_ms.append(_now_ms() - p0)

            phases.append({
                "phase": "pairwise_merge_and_pop",
                "num_pairs": len(merged_times_ms),
                "merge_ms_stats": _stats(merged_times_ms),
                "pop_ms_stats": _stats(popped_times_ms),
            })
            memctx.checkpoint()

            totals["total_ms"] = _now_ms() - start_ms
            totals["peak_mem_bytes"] = memctx.peak_bytes
            totals["timed_out"] = timed_out
            totals["base_depth"] = depth
            totals["base_branching"] = branching
            totals["pairs"] = pairs

        return WorkloadResult(
            name="pairwise_merge_pop",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=None,
            timed_out=timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="pairwise_merge_pop",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=f"{type(e).__name__}: {e}",
            timed_out=timed_out,
        )


def _workload_isolate_rare_top(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    mem_profile: bool,
) -> WorkloadResult:
    """
    Build a wide base with many distinct top-of-stack values, then isolate on a value
    that is present in only a tiny fraction of stacks.

    Params:
      depth: number of levels
      branching: number of clones per level
      target_index: pick which branch value to isolate at the last level (rare)
      repeats: how many isolate repetitions (averaged)
      measure_counts: if True, count stacks before/after
      max_ms: soft budget
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    depth = int(params["depth"])
    branching = int(params["branching"])
    repeats = int(params["repeats"])
    target_index = int(params.get("target_index", 0))

    phases: List[Dict[str, Any]] = []
    timed_out = False
    start_ms = _now_ms()
    totals: Dict[str, Any] = {}

    try:
        with _with_mem_tracking(mem_profile) as memctx:
            # Build base
            p0 = _now_ms()
            base, info = _build_wide_dag(gss_class, depth, branching, seed)
            build_ms = _now_ms() - p0
            phases.append({
                "phase": "build_base",
                "ms": build_ms,
                "detail": info,
            })
            memctx.checkpoint()
            if measure_counts:
                c0 = _now_ms()
                base_count = _count_stacks_safely(base)
                phases[-1]["base_stack_count"] = base_count
                phases[-1]["count_ms"] = _now_ms() - c0

            # Determine a rare top value. We know last level used tags (depth-1, b)
            rare_val = ((depth - 1) << 20) | target_index

            # Repeat isolates
            iso_times: List[float] = []
            post_counts: List[int] = []
            for _ in range(repeats):
                i0 = _now_ms()
                isolated = base.isolate(rare_val)
                iso_times.append(_now_ms() - i0)
                if measure_counts:
                    post_counts.append(_count_stacks_safely(isolated))
                if _maybe_timeout(start_ms, max_ms):
                    timed_out = True
                    break

            phase = {
                "phase": "isolate_rare_top",
                "repeats": len(iso_times),
                "isolate_ms_stats": _stats(iso_times),
                "rare_val": rare_val,
            }
            if measure_counts:
                phase["isolated_stack_count_stats"] = _stats(post_counts)
            phases.append(phase)
            memctx.checkpoint()

            totals["total_ms"] = _now_ms() - start_ms
            totals["peak_mem_bytes"] = memctx.peak_bytes
            totals["timed_out"] = timed_out
            totals["base_depth"] = depth
            totals["base_branching"] = branching
            totals["rare_val"] = rare_val

        return WorkloadResult(
            name="isolate_rare_top",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=None,
            timed_out=timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="isolate_rare_top",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=f"{type(e).__name__}: {e}",
            timed_out=timed_out,
        )


def _workload_apply_peek_reduce(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    mem_profile: bool,
) -> WorkloadResult:
    """
    Build a wide base; then measure apply() over the whole structure,
    followed by peek() and reduce_acc().

    The accumulator work is uniform due to interface constraints, but this still measures
    per-stack iteration costs. This is helpful to compare naive list-based vs. optimized DAG.

    Params:
      depth, branching
      apply_steps: number of times to call apply
      apply_increment: the increment each apply() adds to the accumulator
      measure_counts: if True, count stacks once
      max_ms: soft budget
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    depth = int(params["depth"])
    branching = int(params["branching"])
    apply_steps = int(params["apply_steps"])
    apply_inc = int(params["apply_increment"])

    phases: List[Dict[str, Any]] = []
    timed_out = False
    start_ms = _now_ms()
    totals: Dict[str, Any] = {}

    try:
        with _with_mem_tracking(mem_profile) as memctx:
            # Build base
            p0 = _now_ms()
            base, info = _build_wide_dag(gss_class, depth, branching, seed)
            build_ms = _now_ms() - p0
            phases.append({"phase": "build_base", "ms": build_ms, "detail": info})
            memctx.checkpoint()
            if measure_counts:
                c0 = _now_ms()
                phases[-1]["base_stack_count"] = _count_stacks_safely(base)
                phases[-1]["count_ms"] = _now_ms() - c0

            # Apply steps
            g = base
            apply_times: List[float] = []
            for _ in range(apply_steps):
                a0 = _now_ms()
                g = g.apply(lambda acc, inc=apply_inc: MergeableInt(int(acc) + inc))
                apply_times.append(_now_ms() - a0)
                if _maybe_timeout(start_ms, max_ms):
                    timed_out = True
                    break
            phases.append({
                "phase": "apply_steps",
                "steps": len(apply_times),
                "apply_ms_stats": _stats(apply_times),
            })
            memctx.checkpoint()

            # peek and reduce_acc
            if not timed_out:
                k0 = _now_ms()
                peek_result = g.peek()
                peek_ms = _now_ms() - k0

                r0 = _now_ms()
                red = g.reduce_acc()
                reduce_ms = _now_ms() - r0

                phases.append({
                    "phase": "peek_and_reduce",
                    "peek_ms": peek_ms,
                    "peek_cardinality": len(peek_result),
                    "reduce_ms": reduce_ms,
                    "reduce_result": int(red) if red is not None else None,
                })
                memctx.checkpoint()

            totals["total_ms"] = _now_ms() - start_ms
            totals["peak_mem_bytes"] = memctx.peak_bytes
            totals["timed_out"] = timed_out

        return WorkloadResult(
            name="apply_peek_reduce",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=None,
            timed_out=timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="apply_peek_reduce",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=f"{type(e).__name__}: {e}",
            timed_out=timed_out,
        )


def _workload_popn_scaling(
    gss_class: Type[GSS],
    params: Dict[str, Any],
    preset: str,
    seed: int,
    mem_profile: bool,
) -> WorkloadResult:
    """
    Build several chains of different depths merged together, then measure popn(k) for various k.
    In an efficient structure with sharing, popn(k) should be roughly O(k) and not the total depth.

    Params:
      chain_depths: list of ints
      popn_values: list of ints
      measure_counts: whether to count stacks occasionally
      max_ms: time budget (soft)
    """
    max_ms = params.get("max_ms", None)
    measure_counts = params.get("measure_counts", False)
    chain_depths: List[int] = list(params["chain_depths"])
    popn_values: List[int] = list(params["popn_values"])

    phases: List[Dict[str, Any]] = []
    timed_out = False
    start_ms = _now_ms()
    totals: Dict[str, Any] = {}

    try:
        with _with_mem_tracking(mem_profile) as memctx:
            # Build chains
            chains: List[GSS] = []
            build_stats: List[Dict[str, Any]] = []
            for d in chain_depths:
                b0 = _now_ms()
                g = gss_class.from_stacks([([], MergeableInt(0))])
                for i in range(d):
                    g = g.push(i)
                build_stats.append({"depth": d, "ms": _now_ms() - b0})
                chains.append(g)
                if _maybe_timeout(start_ms, max_ms):
                    timed_out = True
                    break
            phases.append({"phase": "build_chains", "chains": build_stats})
            memctx.checkpoint()

            if not timed_out:
                # Merge chains into a single GSS with many stacks of varying lengths
                m0 = _now_ms()
                merged = _merge_many(gss_class, chains)
                m_ms = _now_ms() - m0
                ph = {"phase": "merge_chains", "ms": m_ms}
                if measure_counts:
                    c0 = _now_ms()
                    ph["stack_count"] = _count_stacks_safely(merged)
                    ph["count_ms"] = _now_ms() - c0
                phases.append(ph)
                memctx.checkpoint()

                # popn for various k
                popn_stats: List[Dict[str, Any]] = []
                for k in popn_values:
                    p0 = _now_ms()
                    res = merged.popn(k)
                    popn_ms = _now_ms() - p0
                    item = {"k": k, "ms": popn_ms}
                    if measure_counts:
                        c0 = _now_ms()
                        item["stack_count"] = _count_stacks_safely(res)
                        item["count_ms"] = _now_ms() - c0
                    popn_stats.append(item)
                    if _maybe_timeout(start_ms, max_ms):
                        timed_out = True
                        break
                phases.append({"phase": "popn_runs", "runs": popn_stats})
                memctx.checkpoint()

            totals["total_ms"] = _now_ms() - start_ms
            totals["peak_mem_bytes"] = memctx.peak_bytes
            totals["timed_out"] = timed_out
            totals["chain_depths"] = chain_depths
            totals["popn_values"] = popn_values

        return WorkloadResult(
            name="popn_scaling",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=None,
            timed_out=timed_out,
        )
    except Exception as e:
        return WorkloadResult(
            name="popn_scaling",
            preset=preset,
            params=params,
            phases=phases,
            totals=totals,
            error=f"{type(e).__name__}: {e}",
            timed_out=timed_out,
        )


def _stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "stdev": None}
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
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
