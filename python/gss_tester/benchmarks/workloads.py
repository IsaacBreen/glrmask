from __future__ import annotations

import math
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from ..interface import MergeableInt, GSS  # Type hints only
from .instrumentation import TimingRecorder, GSSFactory, TimedGSS

JsonDict = Dict[str, Any]

# ----------------------------
# Builders (shared initializers)
# ----------------------------

def build_balanced_fanout(factory: GSSFactory, depth: int, branching: int) -> TimedGSS:
    """
    Builds a balanced b-ary DAG of depth 'depth', by repeated fanout: g = merge_{j}(g.push(val_j)).
    This produces b^depth stacks. Value labels are unique per (layer, branch).
    """
    base = factory.from_stacks([([], MergeableInt(0))])
    g = base
    for i in range(depth):
        vals = [i * 10_000 + j for j in range(branching)]
        # Fan-out from g to each value and then merge them back
        candidates = [g.push(v) for v in vals]
        merged = candidates[0]
        for c in candidates[1:]:
            merged = merged.merge(c)
        g = merged
    return g


def build_prefix_then_branch(factory: GSSFactory, prefix_depth: int, width: int) -> TimedGSS:
    """
    Builds a structure that has a long shared prefix (hidden complexity) and then
    a single fanout to 'width' leaves (surface width), to stress "surface-only" operations.
    """
    g = factory.from_stacks([([], MergeableInt(0))])
    # Long shared prefix
    for i in range(prefix_depth):
        g = g.push(("p", i))
    # Single fanout layer
    vals = [("w", j) for j in range(width)]
    candidates = [g.push(v) for v in vals]
    merged = candidates[0]
    for c in candidates[1:]:
        merged = merged.merge(c)
    return merged


# ----------------------------
# Mutators (surface edits)
# ----------------------------

def mutate_push(g: TimedGSS, tag: int) -> TimedGSS:
    return g.push(("m", tag))

def mutate_apply(g: TimedGSS, amount: int) -> TimedGSS:
    func = lambda acc, amt=amount: acc + amt
    return g.apply(func)

def mutate_prune_half(g: TimedGSS) -> TimedGSS:
    # Keep roughly half: threshold randomized to prune about half for MergeableInt accumulators that start at 0.
    # We annotate stacks randomly by applying a function first
    rng = random.Random(12345)
    # Apply: add small random to acc to create some variance
    g = g.apply(lambda acc: acc + rng.randint(0, 1))
    threshold = 0  # keep those > 0
    predicate = lambda acc, thr=threshold: acc.real > thr
    return g.prune(predicate)

def mutate_isolate_nonexistent(g: TimedGSS) -> TimedGSS:
    # Isolate a value that's almost surely not present at the top; should return empty.
    return g.isolate(("nonexistent", -1))


# ----------------------------
# Workload base
# ----------------------------

@dataclass
class WorkloadConfig:
    name: str
    params: Dict[str, Any]
    max_seconds: float  # maximum wall time for this single workload invocation


# ----------------------------
# Workloads
# ----------------------------

def workload_merge_surface_changes(factory: GSSFactory, cfg: WorkloadConfig) -> JsonDict:
    """
    Build a balanced fanout DAG and then clone it N times, apply a surface change to each clone,
    and merge them all. Measures whether merge scales with the change surface rather than the hidden size.
    Params:
      - depth: int
      - branching: int
      - clones: int
      - mutation: str in {"push", "apply", "prune_half", "isolate_none"}
      - apply_amount: int (used if mutation == "apply")
    """
    params = cfg.params
    depth = int(params.get("depth", 6))
    branching = int(params.get("branching", 2))
    clones = int(params.get("clones", 4))
    mutation: str = str(params.get("mutation", "push"))
    apply_amount = int(params.get("apply_amount", 1))

    recorder = factory.recorder
    workload_result: JsonDict = {
        "workload": "merge_surface_changes",
        "params": params,
        "phases": [],
        "outcome": "ok",
        "notes": [],
        "derived": {},
    }

    tracemalloc.start()
    t_workload_start = time.perf_counter_ns()
    try:
        # Build phase
        recorder.start_phase("build")
        base = build_balanced_fanout(factory, depth=depth, branching=branching)
        recorder.end_phase()

        # Clone & mutate phase
        recorder.start_phase("clone_and_mutate")
        clones_list: List[TimedGSS] = [base] * clones
        mutated: List[TimedGSS] = []
        for i, g in enumerate(clones_list):
            if mutation == "push":
                mutated.append(mutate_push(g, i))
            elif mutation == "apply":
                mutated.append(mutate_apply(g, apply_amount))
            elif mutation == "prune_half":
                mutated.append(mutate_prune_half(g))
            elif mutation == "isolate_none":
                mutated.append(g.isolate(None))
            else:
                mutated.append(mutate_push(g, i))
        recorder.end_phase()

        # Merge phase
        recorder.start_phase("merge")
        merged = factory.merge_many(mutated)
        recorder.end_phase()

        # Postcheck - keep lightweight ops
        recorder.start_phase("postcheck")
        _ = merged.is_empty()
        _ = merged.peek()
        _ = merged.reduce_acc()
        recorder.end_phase()

    except Exception as e:
        workload_result["outcome"] = "error"
        workload_result["error"] = f"{e.__class__.__name__}: {e}"
        workload_result["traceback"] = traceback.format_exc()
    finally:
        t_workload_end = time.perf_counter_ns()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        workload_result["wall_time_ns"] = t_workload_end - t_workload_start
        workload_result["memory"] = {"current_bytes": current, "peak_bytes": peak}
        workload_result["phases"] = [p.to_json() for p in recorder.phases]
        workload_result["methods"] = recorder.to_json()["overall_methods"]

    # Derived metrics (without calling to_stacks to avoid heavy cost)
    # Theoretical leaf count for balanced b-ary tree: b^d
    try:
        b = branching
        d = depth
        leaves = b ** d
        nodes = (b ** (d + 1) - 1) // (b - 1) if b > 1 else d + 1
        workload_result["derived"]["theoretical_leaves"] = leaves
        workload_result["derived"]["theoretical_nodes"] = nodes
        workload_result["derived"]["clones"] = clones
    except Exception:
        pass

    # Enforce max_seconds cut-off: if exceeded, mark as aborted (but keep results)
    if workload_result["wall_time_ns"] / 1e9 > cfg.max_seconds:
        workload_result["outcome"] = "aborted"
        workload_result["notes"].append(f"Exceeded max_seconds={cfg.max_seconds}")

    return workload_result


def workload_push_scaling(factory: GSSFactory, cfg: WorkloadConfig) -> JsonDict:
    """
    Build a long shared prefix and a fixed surface width, then measure a single push() on the result.
    Params:
      - prefix_depth: int
      - surface_width: int
    """
    params = cfg.params
    prefix_depth = int(params.get("prefix_depth", 100))
    surface_width = int(params.get("surface_width", 64))

    recorder = factory.recorder
    workload_result: JsonDict = {
        "workload": "push_scaling",
        "params": params,
        "phases": [],
        "outcome": "ok",
        "notes": [],
        "derived": {},
    }

    tracemalloc.start()
    t_workload_start = time.perf_counter_ns()
    try:
        # Build
        recorder.start_phase("build")
        g = build_prefix_then_branch(factory, prefix_depth=prefix_depth, width=surface_width)
        recorder.end_phase()

        # Single push
        recorder.start_phase("push")
        g2 = g.push(("push", 1))
        recorder.end_phase()

        # Postcheck
        recorder.start_phase("postcheck")
        _ = g2.peek()
        _ = g2.is_empty()
        recorder.end_phase()
    except Exception as e:
        workload_result["outcome"] = "error"
        workload_result["error"] = f"{e.__class__.__name__}: {e}"
        workload_result["traceback"] = traceback.format_exc()
    finally:
        t_workload_end = time.perf_counter_ns()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        workload_result["wall_time_ns"] = t_workload_end - t_workload_start
        workload_result["memory"] = {"current_bytes": current, "peak_bytes": peak}
        workload_result["phases"] = [p.to_json() for p in recorder.phases]
        workload_result["methods"] = recorder.to_json()["overall_methods"]

    workload_result["derived"]["hidden_prefix_depth"] = prefix_depth
    workload_result["derived"]["surface_width"] = surface_width

    if workload_result["wall_time_ns"] / 1e9 > cfg.max_seconds:
        workload_result["outcome"] = "aborted"
        workload_result["notes"].append(f"Exceeded max_seconds={cfg.max_seconds}")

    return workload_result


def workload_merge_after_prefix_mutations(factory: GSSFactory, cfg: WorkloadConfig) -> JsonDict:
    """
    Use the prefix-then-branch builder to create large hidden complexity and fixed width.
    Clone the GSS C times, push a unique tag to each clone, then merge.
    Params:
      - prefix_depth: int
      - surface_width: int
      - clones: int
    """
    params = cfg.params
    prefix_depth = int(params.get("prefix_depth", 200))
    surface_width = int(params.get("surface_width", 64))
    clones = int(params.get("clones", 8))

    recorder = factory.recorder
    workload_result: JsonDict = {
        "workload": "merge_after_prefix_mutations",
        "params": params,
        "phases": [],
        "outcome": "ok",
        "notes": [],
        "derived": {},
    }

    tracemalloc.start()
    t_workload_start = time.perf_counter_ns()
    try:
        # Build
        recorder.start_phase("build")
        base = build_prefix_then_branch(factory, prefix_depth=prefix_depth, width=surface_width)
        recorder.end_phase()

        # Mutate clones
        recorder.start_phase("mutate_clones")
        clones_list = [base] * clones
        mutated = [mutate_push(g, i) for i, g in enumerate(clones_list)]
        recorder.end_phase()

        # Merge
        recorder.start_phase("merge")
        merged = factory.merge_many(mutated)
        recorder.end_phase()

        # Postcheck
        recorder.start_phase("postcheck")
        _ = merged.peek()
        _ = merged.reduce_acc()
        recorder.end_phase()

    except Exception as e:
        workload_result["outcome"] = "error"
        workload_result["error"] = f"{e.__class__.__name__}: {e}"
        workload_result["traceback"] = traceback.format_exc()
    finally:
        t_workload_end = time.perf_counter_ns()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        workload_result["wall_time_ns"] = t_workload_end - t_workload_start
        workload_result["memory"] = {"current_bytes": current, "peak_bytes": peak}
        workload_result["phases"] = [p.to_json() for p in recorder.phases]
        workload_result["methods"] = recorder.to_json()["overall_methods"]

    workload_result["derived"]["hidden_prefix_depth"] = prefix_depth
    workload_result["derived"]["surface_width"] = surface_width
    workload_result["derived"]["clones"] = clones

    if workload_result["wall_time_ns"] / 1e9 > cfg.max_seconds:
        workload_result["outcome"] = "aborted"
        workload_result["notes"].append(f"Exceeded max_seconds={cfg.max_seconds}")

    return workload_result


def workload_pop_common_parent(factory: GSSFactory, cfg: WorkloadConfig) -> JsonDict:
    """
    Build a base with one level of many siblings, then pop back to parent.
    This tests whether pop merges efficiently and scales with sibling count rather than hidden size.
    Params:
      - siblings: int
      - parent_prefix_depth: int
    """
    params = cfg.params
    siblings = int(params.get("siblings", 64))
    prefix_depth = int(params.get("parent_prefix_depth", 50))

    recorder = factory.recorder
    workload_result: JsonDict = {
        "workload": "pop_common_parent",
        "params": params,
        "phases": [],
        "outcome": "ok",
        "notes": [],
        "derived": {},
    }

    tracemalloc.start()
    t_workload_start = time.perf_counter_ns()
    try:
        recorder.start_phase("build")
        base = factory.from_stacks([([], MergeableInt(0))])
        # Deep shared prefix
        for i in range(prefix_depth):
            base = base.push(("p", i))
        # One fanout
        vals = [("sib", j) for j in range(siblings)]
        candidates = [base.push(v) for v in vals]
        merged = candidates[0]
        for c in candidates[1:]:
            merged = merged.merge(c)
        recorder.end_phase()

        recorder.start_phase("pop")
        popped = merged.pop()
        recorder.end_phase()

        recorder.start_phase("postcheck")
        _ = popped.peek()
        _ = popped.is_empty()
        recorder.end_phase()
    except Exception as e:
        workload_result["outcome"] = "error"
        workload_result["error"] = f"{e.__class__.__name__}: {e}"
        workload_result["traceback"] = traceback.format_exc()
    finally:
        t_workload_end = time.perf_counter_ns()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        workload_result["wall_time_ns"] = t_workload_end - t_workload_start
        workload_result["memory"] = {"current_bytes": current, "peak_bytes": peak}
        workload_result["phases"] = [p.to_json() for p in recorder.phases]
        workload_result["methods"] = recorder.to_json()["overall_methods"]

    workload_result["derived"]["siblings"] = siblings
    workload_result["derived"]["hidden_prefix_depth"] = prefix_depth

    if workload_result["wall_time_ns"] / 1e9 > cfg.max_seconds:
        workload_result["outcome"] = "aborted"
        workload_result["notes"].append(f"Exceeded max_seconds={cfg.max_seconds}")

    return workload_result


def workload_apply_prune(factory: GSSFactory, cfg: WorkloadConfig) -> JsonDict:
    """
    Build a balanced fanout and then apply and prune to stress accumulator transforms.
    Params:
      - depth: int
      - branching: int
      - apply_amount: int
      - prune_threshold: int
    """
    params = cfg.params
    depth = int(params.get("depth", 6))
    branching = int(params.get("branching", 2))
    apply_amount = int(params.get("apply_amount", 5))
    prune_threshold = int(params.get("prune_threshold", 10))

    recorder = factory.recorder
    workload_result: JsonDict = {
        "workload": "apply_prune",
        "params": params,
        "phases": [],
        "outcome": "ok",
        "notes": [],
        "derived": {},
    }

    tracemalloc.start()
    t_workload_start = time.perf_counter_ns()
    try:
        recorder.start_phase("build")
        g = build_balanced_fanout(factory, depth=depth, branching=branching)
        recorder.end_phase()

        recorder.start_phase("apply")
        g2 = g.apply(lambda acc, amt=apply_amount: acc + amt)
        recorder.end_phase()

        recorder.start_phase("prune")
        g3 = g2.prune(lambda acc, thr=prune_threshold: acc.real > thr)
        recorder.end_phase()

        recorder.start_phase("postcheck")
        _ = g3.peek()
        _ = g3.reduce_acc()
        recorder.end_phase()
    except Exception as e:
        workload_result["outcome"] = "error"
        workload_result["error"] = f"{e.__class__.__name__}: {e}"
        workload_result["traceback"] = traceback.format_exc()
    finally:
        t_workload_end = time.perf_counter_ns()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        workload_result["wall_time_ns"] = t_workload_end - t_workload_start
        workload_result["memory"] = {"current_bytes": current, "peak_bytes": peak}
        workload_result["phases"] = [p.to_json() for p in recorder.phases]
        workload_result["methods"] = recorder.to_json()["overall_methods"]

    try:
        workload_result["derived"]["theoretical_leaves"] = branching ** depth
        workload_result["derived"]["theoretical_nodes"] = (branching ** (depth + 1) - 1) // (branching - 1) if branching > 1 else depth + 1
    except Exception:
        pass

    if workload_result["wall_time_ns"] / 1e9 > cfg.max_seconds:
        workload_result["outcome"] = "aborted"
        workload_result["notes"].append(f"Exceeded max_seconds={cfg.max_seconds}")

    return workload_result


def workload_fuzz(factory: GSSFactory, cfg: WorkloadConfig) -> JsonDict:
    """
    A short, deterministic fuzz run to capture general behavior and per-method mix.
    Params:
      - seed: int
      - steps: int
      - max_gss_states: int
    """
    from ..tests.fuzzer import run_fuzz_test

    params = cfg.params
    seed = int(params.get("seed", 42))
    steps = int(params.get("steps", 200))
    max_states = int(params.get("max_gss_states", 10))

    recorder = factory.recorder
    workload_result: JsonDict = {
        "workload": "fuzz",
        "params": params,
        "phases": [],
        "outcome": "ok",
        "notes": [],
        "derived": {},
    }

    tracemalloc.start()
    t_workload_start = time.perf_counter_ns()
    try:
        recorder.start_phase("fuzz")
        # Note: fuzzer expects a class; we wrap states ourselves via GSSFactory
        # Seed run: we build states using the concrete class, not the wrapper.
        # For measurement, we rebuild/wrap the yielded states into TimedGSS by reconstructing from stacks.
        gss_class = factory.gss_class

        # Start with an initial wrapped GSS
        _ = factory.from_stacks([([], MergeableInt(0))])

        for state in run_fuzz_test(
            gss_class=gss_class,
            seed=seed,
            num_steps=steps,
            max_gss_states=max_states,
            value_pool=None
        ):
            # Convert state to wrapped instance by round-tripping through to_stacks; keep modest steps for tiny/small
            try:
                stacks = state.to_stacks()
            except Exception as e:
                recorder.record_error("to_stacks", f"error during fuzz: {e}")
                continue
            _ = factory.from_stacks(stacks)
        recorder.end_phase()

        recorder.start_phase("postcheck")
        recorder.end_phase()
    except Exception as e:
        workload_result["outcome"] = "error"
        workload_result["error"] = f"{e.__class__.__name__}: {e}"
        workload_result["traceback"] = traceback.format_exc()
    finally:
        t_workload_end = time.perf_counter_ns()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        workload_result["wall_time_ns"] = t_workload_end - t_workload_start
        workload_result["memory"] = {"current_bytes": current, "peak_bytes": peak}
        workload_result["phases"] = [p.to_json() for p in recorder.phases]
        workload_result["methods"] = recorder.to_json()["overall_methods"]

    if workload_result["wall_time_ns"] / 1e9 > cfg.max_seconds:
        workload_result["outcome"] = "aborted"
        workload_result["notes"].append(f"Exceeded max_seconds={cfg.max_seconds}")

    return workload_result


# ----------------------------
# Presets and registry
# ----------------------------

def tiny_preset() -> List[WorkloadConfig]:
    # Target sizes that run quickly even on ReferenceGSS.
    return [
        WorkloadConfig("merge_surface_changes", {"depth": 4, "branching": 2, "clones": 4, "mutation": "push"}, max_seconds=5.0),
        WorkloadConfig("push_scaling", {"prefix_depth": 50, "surface_width": 32}, max_seconds=3.0),
        WorkloadConfig("merge_after_prefix_mutations", {"prefix_depth": 75, "surface_width": 32, "clones": 6}, max_seconds=5.0),
        WorkloadConfig("pop_common_parent", {"siblings": 64, "parent_prefix_depth": 50}, max_seconds=3.0),
        WorkloadConfig("apply_prune", {"depth": 4, "branching": 2, "apply_amount": 5, "prune_threshold": 3}, max_seconds=5.0),
        WorkloadConfig("fuzz", {"seed": 7, "steps": 100, "max_gss_states": 10}, max_seconds=5.0),
    ]


def small_preset() -> List[WorkloadConfig]:
    # Still safe for ReferenceGSS but a bit larger.
    return [
        WorkloadConfig("merge_surface_changes", {"depth": 5, "branching": 3, "clones": 6, "mutation": "push"}, max_seconds=12.0),
        WorkloadConfig("push_scaling", {"prefix_depth": 150, "surface_width": 64}, max_seconds=10.0),
        WorkloadConfig("merge_after_prefix_mutations", {"prefix_depth": 200, "surface_width": 64, "clones": 8}, max_seconds=15.0),
        WorkloadConfig("pop_common_parent", {"siblings": 128, "parent_prefix_depth": 150}, max_seconds=10.0),
        WorkloadConfig("apply_prune", {"depth": 5, "branching": 3, "apply_amount": 7, "prune_threshold": 10}, max_seconds=12.0),
        WorkloadConfig("fuzz", {"seed": 42, "steps": 200, "max_gss_states": 10}, max_seconds=10.0),
    ]


def medium_preset() -> List[WorkloadConfig]:
    # Designed for more efficient GSS implementations; reference may struggle.
    return [
        WorkloadConfig("merge_surface_changes", {"depth": 7, "branching": 3, "clones": 12, "mutation": "push"}, max_seconds=20.0),
        WorkloadConfig("push_scaling", {"prefix_depth": 500, "surface_width": 128}, max_seconds=20.0),
        WorkloadConfig("merge_after_prefix_mutations", {"prefix_depth": 800, "surface_width": 128, "clones": 16}, max_seconds=25.0),
        WorkloadConfig("pop_common_parent", {"siblings": 256, "parent_prefix_depth": 500}, max_seconds=20.0),
        WorkloadConfig("apply_prune", {"depth": 6, "branching": 4, "apply_amount": 10, "prune_threshold": 20}, max_seconds=20.0),
        WorkloadConfig("fuzz", {"seed": 1337, "steps": 500, "max_gss_states": 15}, max_seconds=20.0),
    ]


def large_preset() -> List[WorkloadConfig]:
    # Stressful for advanced implementations; large hidden complexity and surface sizes.
    return [
        WorkloadConfig("merge_surface_changes", {"depth": 8, "branching": 4, "clones": 20, "mutation": "push"}, max_seconds=45.0),
        WorkloadConfig("push_scaling", {"prefix_depth": 2000, "surface_width": 256}, max_seconds=45.0),
        WorkloadConfig("merge_after_prefix_mutations", {"prefix_depth": 3000, "surface_width": 256, "clones": 24}, max_seconds=60.0),
        WorkloadConfig("pop_common_parent", {"siblings": 512, "parent_prefix_depth": 2000}, max_seconds=45.0),
        WorkloadConfig("apply_prune", {"depth": 7, "branching": 5, "apply_amount": 20, "prune_threshold": 40}, max_seconds=45.0),
        WorkloadConfig("fuzz", {"seed": 2024, "steps": 1000, "max_gss_states": 20}, max_seconds=40.0),
    ]


PRESETS: Dict[str, Callable[[], List[WorkloadConfig]]] = {
    "tiny": tiny_preset,
    "small": small_preset,
    "medium": medium_preset,
    "large": large_preset,
}


WORKLOAD_FUNCS: Dict[str, Callable[[GSSFactory, WorkloadConfig], JsonDict]] = {
    "merge_surface_changes": workload_merge_surface_changes,
    "push_scaling": workload_push_scaling,
    "merge_after_prefix_mutations": workload_merge_after_prefix_mutations,
    "pop_common_parent": workload_pop_common_parent,
    "apply_prune": workload_apply_prune,
    "fuzz": workload_fuzz,
}
