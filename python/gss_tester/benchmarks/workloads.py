from __future__ import annotations

import itertools
import random
import time
import tracemalloc
import gc
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from gss_tester.interface import GSS, MergeableInt


@dataclass
class WorkloadTiming:
    # By-op timings (seconds) for critical ops in this workload
    per_op: Dict[str, List[float]]
    # Aggregate wall times for phases
    phases: Dict[str, float]
    # Peak memory usage in KiB during the workload
    peak_mem_kib: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_op": {k: list(v) for k, v in self.per_op.items()},
            "phases": dict(self.phases),
            "peak_mem_kib": self.peak_mem_kib,
        }


@dataclass
class WorkloadResult:
    name: str
    params: Dict[str, Any]
    # Final GSS state (for structural introspection)
    final_state: GSS
    # Operational metrics
    operations_executed: int
    timings: WorkloadTiming

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "params": dict(self.params),
            "operations_executed": self.operations_executed,
            "timings": self.timings.to_dict(),
        }
        return d


def _measure_peak_memory(fn: Callable[[], Any]) -> Tuple[Any, int, float]:
    """
    Run fn() under tracemalloc to capture peak memory (KiB) and elapsed time.
    Returns (result, peak_mem_kib, elapsed_seconds).
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kib = int(peak / 1024)
    return result, peak_kib, elapsed


def workload_linear_push(gss_class: Type[GSS], length: int) -> WorkloadResult:
    """
    Push-only linear growth from a single empty stack. Measures push scaling.
    """
    per_op: Dict[str, List[float]] = {"push": [], "to_stacks": []}
    phases: Dict[str, float] = {}

    def run() -> GSS:
        state = gss_class.from_stacks([([], MergeableInt(0))])
        for i in range(length):
            t0 = time.perf_counter()
            state = state.push(i)
            per_op["push"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _ = state.to_stacks()
        per_op["to_stacks"].append(time.perf_counter() - t0)
        return state

    final_state, peak_kib, elapsed = _measure_peak_memory(run)
    phases["total"] = elapsed
    return WorkloadResult(
        name="linear_push",
        params={"length": length},
        final_state=final_state,
        operations_executed=length + 1,  # pushes + one to_stacks
        timings=WorkloadTiming(per_op=per_op, phases=phases, peak_mem_kib=peak_kib),
    )


def workload_product_tree(
    gss_class: Type[GSS],
    depth: int,
    k: int,
    base_value: int = 1000,
) -> WorkloadResult:
    """
    Build k^depth stacks by constructing each sequence independently from the same base,
    then merge them all. This stresses merge_many and tests prefix sharing.
    """
    per_op: Dict[str, List[float]] = {"push": [], "merge": [], "to_stacks": []}
    phases: Dict[str, float] = {}

    levels: List[List[int]] = [
        [base_value + d * 100 + j for j in range(k)] for d in range(depth)
    ]
    sequences = list(itertools.product(*levels))
    base = gss_class.from_stacks([([], MergeableInt(0))])

    # Construct all leaf GSS states
    def build_all() -> List[GSS]:
        states: List[GSS] = []
        for seq in sequences:
            s = base
            for v in seq:
                t0 = time.perf_counter()
                s = s.push(v)
                per_op["push"].append(time.perf_counter() - t0)
            states.append(s)
        return states

    def run() -> GSS:
        leaves = build_all()
        t0 = time.perf_counter()
        merged = gss_class.merge_many(leaves)
        per_op["merge"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _ = merged.to_stacks()
        per_op["to_stacks"].append(time.perf_counter() - t0)
        return merged

    final_state, peak_kib, elapsed = _measure_peak_memory(run)
    phases["total"] = elapsed
    return WorkloadResult(
        name="product_tree",
        params={"depth": depth, "k": k},
        final_state=final_state,
        operations_executed=(len(sequences) * depth) + 2,  # pushes + merge + to_stacks
        timings=WorkloadTiming(per_op=per_op, phases=phases, peak_mem_kib=peak_kib),
    )


def workload_diamond_repeat(
    gss_class: Type[GSS],
    repeats: int,
    base_value: int = 2000,
) -> WorkloadResult:
    """
    Repeated diamond pattern:
      s0 -> push(a) -> s1
      s0 -> push(b) -> s2
      merged = s1.merge(s2)
    Optionally push+pop to exercise collapse after merge.
    """
    per_op: Dict[str, List[float]] = {"push": [], "merge": [], "pop": [], "to_stacks": []}
    phases: Dict[str, float] = {}

    def run() -> GSS:
        s0 = gss_class.from_stacks([([], MergeableInt(0))])
        merged = s0
        for i in range(repeats):
            a = base_value + i * 2
            b = base_value + i * 2 + 1
            t0 = time.perf_counter()
            s1 = s0.push(a)
            per_op["push"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            s2 = s0.push(b)
            per_op["push"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            merged = s1.merge(s2)
            per_op["merge"].append(time.perf_counter() - t0)

            # Collapse
            t0 = time.perf_counter()
            merged = merged.pop()
            per_op["pop"].append(time.perf_counter() - t0)

            # Keep s0 advancing to simulate stages on the same base chain
            s0 = s0.push(a)  # linear growth for context

        t0 = time.perf_counter()
        _ = merged.to_stacks()
        per_op["to_stacks"].append(time.perf_counter() - t0)
        return merged

    final_state, peak_kib, elapsed = _measure_peak_memory(run)
    phases["total"] = elapsed

    return WorkloadResult(
        name="diamond_repeat",
        params={"repeats": repeats},
        final_state=final_state,
        operations_executed=(repeats * 4) + 1,  # push, push, merge, pop per repeat + to_stacks
        timings=WorkloadTiming(per_op=per_op, phases=phases, peak_mem_kib=peak_kib),
    )


def workload_merge_many(
    gss_class: Type[GSS],
    count: int,
    stack_len: int,
    seed: int = 123,
) -> WorkloadResult:
    """
    Create 'count' stacks of length 'stack_len' with pseudo-random values and merge them.
    """
    rng = random.Random(seed)
    per_op: Dict[str, List[float]] = {"push": [], "merge": [], "to_stacks": []}
    phases: Dict[str, float] = {}

    def build_one() -> GSS:
        s = gss_class.from_stacks([([], MergeableInt(0))])
        for _ in range(stack_len):
            v = rng.randint(0, 100000)
            t0 = time.perf_counter()
            s = s.push(v)
            per_op["push"].append(time.perf_counter() - t0)
        return s

    def run() -> GSS:
        states = [build_one() for _ in range(count)]
        t0 = time.perf_counter()
        merged = gss_class.merge_many(states)
        per_op["merge"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        _ = merged.to_stacks()
        per_op["to_stacks"].append(time.perf_counter() - t0)
        return merged

    final_state, peak_kib, elapsed = _measure_peak_memory(run)
    phases["total"] = elapsed
    return WorkloadResult(
        name="merge_many",
        params={"count": count, "stack_len": stack_len, "seed": seed},
        final_state=final_state,
        operations_executed=(count * stack_len) + 2,  # pushes + merge + to_stacks
        timings=WorkloadTiming(per_op=per_op, phases=phases, peak_mem_kib=peak_kib),
    )


def workload_pop_collapse(
    gss_class: Type[GSS],
    base_depth: int,
    branch_count: int,
) -> WorkloadResult:
    """
    Build a base path of length base_depth-1, branch into 'branch_count' last-step variants,
    merge them, then pop once to collapse to the common parent.
    """
    per_op: Dict[str, List[float]] = {"push": [], "merge": [], "pop": [], "to_stacks": []}
    phases: Dict[str, float] = {}

    def run() -> GSS:
        base = gss_class.from_stacks([([], MergeableInt(0))])
        for i in range(max(base_depth - 1, 0)):
            t0 = time.perf_counter()
            base = base.push(i)
            per_op["push"].append(time.perf_counter() - t0)

        last_states: List[GSS] = []
        for j in range(branch_count):
            t0 = time.perf_counter()
            s = base.push(10_000 + j)
            per_op["push"].append(time.perf_counter() - t0)
            last_states.append(s)

        t0 = time.perf_counter()
        merged = gss_class.merge_many(last_states)
        per_op["merge"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        popped = merged.pop()
        per_op["pop"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = popped.to_stacks()
        per_op["to_stacks"].append(time.perf_counter() - t0)
        return popped

    final_state, peak_kib, elapsed = _measure_peak_memory(run)
    phases["total"] = elapsed
    pushes = max(base_depth - 1, 0) + branch_count
    return WorkloadResult(
        name="pop_collapse",
        params={"base_depth": base_depth, "branch_count": branch_count},
        final_state=final_state,
        operations_executed=pushes + 3,  # pushes + merge + pop + to_stacks
        timings=WorkloadTiming(per_op=per_op, phases=phases, peak_mem_kib=peak_kib),
    )


def workload_apply_prune(
    gss_class: Type[GSS],
    breadth: int,
    depth: int,
    apply_amount: int = 7,
    prune_threshold: int = 20,
) -> WorkloadResult:
    """
    Build a breadth x depth structure (approx) via product-tree merging,
    then apply a function to all accumulators and prune by a predicate.
    """
    per_op: Dict[str, List[float]] = {"push": [], "merge": [], "apply": [], "prune": [], "to_stacks": []}
    phases: Dict[str, float] = {}

    # We'll emulate a shallow product tree to get roughly breadth^depth stacks,
    # but cap breadth to avoid blow-ups.
    k = max(1, min(8, breadth))
    levels: List[List[int]] = [
        [5000 + d * 100 + j for j in range(k)] for d in range(depth)
    ]
    sequences = list(itertools.product(*levels))
    base = gss_class.from_stacks([([], MergeableInt(0))])

    def run() -> GSS:
        leaves: List[GSS] = []
        for seq in sequences:
            s = base
            for v in seq:
                t0 = time.perf_counter()
                s = s.push(v)
                per_op["push"].append(time.perf_counter() - t0)
            leaves.append(s)

        t0 = time.perf_counter()
        merged = gss_class.merge_many(leaves)
        per_op["merge"].append(time.perf_counter() - t0)

        # apply
        t0 = time.perf_counter()
        merged = merged.apply(lambda acc, amt=apply_amount: acc + amt)
        per_op["apply"].append(time.perf_counter() - t0)

        # prune (keep only accumulators greater than threshold)
        t0 = time.perf_counter()
        merged = merged.prune(lambda acc, thr=prune_threshold: acc > thr)
        per_op["prune"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        _ = merged.to_stacks()
        per_op["to_stacks"].append(time.perf_counter() - t0)

        return merged

    final_state, peak_kib, elapsed = _measure_peak_memory(run)
    phases["total"] = elapsed

    return WorkloadResult(
        name="apply_prune",
        params={"breadth": breadth, "depth": depth, "apply_amount": apply_amount, "prune_threshold": prune_threshold},
        final_state=final_state,
        operations_executed=(len(sequences) * depth) + 4,
        timings=WorkloadTiming(per_op=per_op, phases=phases, peak_mem_kib=peak_kib),
    )


# Registry of workloads for ease of selection
WORKLOADS: Dict[str, Callable[..., WorkloadResult]] = {
    "linear_push": workload_linear_push,
    "product_tree": workload_product_tree,
    "diamond_repeat": workload_diamond_repeat,
    "merge_many": workload_merge_many,
    "pop_collapse": workload_pop_collapse,
    "apply_prune": workload_apply_prune,
}


def default_specs(preset: str = "small") -> Dict[str, List[Dict[str, Any]]]:
    """
    Preset parameter grids for workloads.
    - small: fast to run, suitable for quick comparisons
    - medium: moderate sizes to observe scaling trends
    - large: heavier runs (take longer)
    """
    if preset not in {"small", "medium", "large"}:
        preset = "small"

    if preset == "small":
        return {
            "linear_push": [{"length": 500}, {"length": 2_000}],
            "product_tree": [{"depth": 3, "k": 3}, {"depth": 4, "k": 2}],
            "diamond_repeat": [{"repeats": 300}],
            "merge_many": [{"count": 50, "stack_len": 8}],
            "pop_collapse": [{"base_depth": 20, "branch_count": 200}],
            "apply_prune": [{"breadth": 4, "depth": 4, "apply_amount": 7, "prune_threshold": 20}],
        }
    elif preset == "medium":
        return {
            "linear_push": [{"length": 5_000}, {"length": 20_000}],
            "product_tree": [{"depth": 5, "k": 3}],
            "diamond_repeat": [{"repeats": 2_000}],
            "merge_many": [{"count": 200, "stack_len": 16}],
            "pop_collapse": [{"base_depth": 60, "branch_count": 1_000}],
            "apply_prune": [{"breadth": 6, "depth": 5, "apply_amount": 11, "prune_threshold": 50}],
        }
    else:  # large
        return {
            "linear_push": [{"length": 50_000}],
            "product_tree": [{"depth": 6, "k": 4}],
            "diamond_repeat": [{"repeats": 10_000}],
            "merge_many": [{"count": 1_000, "stack_len": 24}],
            "pop_collapse": [{"base_depth": 150, "branch_count": 5_000}],
            "apply_prune": [{"breadth": 8, "depth": 6, "apply_amount": 17, "prune_threshold": 200}],
        }
