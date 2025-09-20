from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from .workloads import resolve_workloads, list_workloads, WorkloadResult, WORKLOADS, BenchmarkContext, create_timed_gss_class
from ..interface import GSS


def _load_impl(module_name: str, class_name: str) -> Type[GSS]:
    module = importlib.import_module(module_name)
    gss_class = getattr(module, class_name)
    return gss_class


def _parse_workload_filters(only: Optional[str], exclude: Optional[str]) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    incl = [s for s in only.split(",")] if only else None
    excl = [s for s in exclude.split(",")] if exclude else None
    if incl:
        incl = [s.strip() for s in incl if s.strip()]
    if excl:
        excl = [s.strip() for s in excl if s.strip()]
    return incl, excl


def _env_metadata() -> Dict[str, Any]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "time": datetime.utcnow().isoformat() + "Z",
    }


def _run_for_impl(
    gss_class: Type[GSS],
    impl_name: str,
    preset: str,
    workloads_to_run: List[str],
    seed: int,
    mem: bool,
    repeat: int,
) -> Dict[str, Any]:
    results: List[WorkloadResult] = []

    for wname in workloads_to_run:
        w = WORKLOADS[wname]
        params = dict(w.param_presets[preset])
        # repeat each workload and capture all repeats
        for rep in range(repeat):
            # Variation of seed per repeat for deterministic variety
            rep_seed = seed + rep
            context = BenchmarkContext(mem_profile=mem)
            timed_gss_class = create_timed_gss_class(gss_class, context.collector)
            res = w.runner(timed_gss_class, params, preset, rep_seed, context)

            # Tag repetition index
            res_dict = {
                "name": res.name,
                "preset": res.preset,
                "params": res.params,
                "phases": res.phases,
                "totals": res.totals,
                "error": res.error,
                "timed_out": res.timed_out,
                "repeat_index": rep,
            }
            results.append(res)

    # Summarize results by workload name (averaging across repeats)
    summary = {}
    # Group results by (workload_name, preset_name)
    tmp_agg_results: Dict[Tuple[str, str], List[WorkloadResult]] = {}
    for r in results:
        tmp_agg_results.setdefault((r.name, r.preset), []).append(r)

    for (wname, preset_name), runs in tmp_agg_results.items():
        # Aggregate totals
        totals_list = [run.totals for run in runs]
        keys = set().union(*(d.keys() for d in totals_list))
        agg: Dict[str, Any] = {"count": len(runs)}
        for k in keys:
            vals = [d[k] for d in totals_list if k in d and isinstance(d[k], (int, float))]
            if vals:
                agg[k + "_mean"] = sum(vals) / len(vals)
                agg[k + "_min"] = min(vals)
                agg[k + "_max"] = max(vals)

        # Aggregate phases
        phases_by_name: Dict[str, List[Dict[str, Any]]] = {}
        for run in runs:
            for phase_result in run.phases:
                phase_name = phase_result.get("phase", "unknown")
                phases_by_name.setdefault(phase_name, []).append(phase_result)

        aggregated_phases = []
        for phase_name, phase_runs in sorted(phases_by_name.items()):
            # Average 'ms'
            ms_vals = [p['ms'] for p in phase_runs if 'ms' in p]
            ms_mean = sum(ms_vals) / len(ms_vals) if ms_vals else 0.0

            # Average method_stats
            method_stats_agg: Dict[str, Dict[str, float]] = {}
            all_method_names = set()
            for p in phase_runs:
                all_method_names.update(p.get('method_stats', {}).keys())

            for method in sorted(all_method_names):
                calls_vals = [p.get('method_stats', {}).get(method, {}).get('calls', 0) for p in phase_runs]
                total_ms_vals = [p.get('method_stats', {}).get(method, {}).get('total_ms', 0.0) for p in phase_runs]

                calls_mean = sum(calls_vals) / len(calls_vals) if calls_vals else 0.0
                total_ms_mean = sum(total_ms_vals) / len(total_ms_vals) if total_ms_vals else 0.0

                method_stats_agg[method] = {
                    'calls_mean': calls_mean,
                    'total_ms_mean': total_ms_mean,
                }

            aggregated_phases.append({
                "phase": phase_name,
                "ms_mean": ms_mean,
                "method_stats_mean": method_stats_agg
            })

        agg["phases_mean"] = aggregated_phases
        summary[wname] = agg

    # Construct JSON data
    json_results = []
    # Convert WorkloadResult objects to plain dicts
    for r in results:
        json_results.append({
            "name": r.name,
            "preset": r.preset,
            "params": r.params,
            "phases": r.phases,
            "totals": r.totals,
            "error": r.error,
            "timed_out": r.timed_out,
        })

    return {
        "implementation": impl_name,
        "preset": preset,
        "metadata": _env_metadata(),
        "runner_config": {
            "seed": seed,
            "memory_profiled": mem,
            "repeat": repeat,
            "workloads": workloads_to_run,
        },
        "results": json_results,
        "summary": summary,
        "version": "1.0.0",
    }


def main():
    parser = argparse.ArgumentParser(description="Run GSS benchmarks against an implementation.")
    parser.add_argument("implementation_module", help="Python module path for the GSS implementation (e.g., 'gss_tester.implementations.reference_impl').")
    parser.add_argument("implementation_class", help="Class name of the GSS implementation (e.g., 'ReferenceGSS').")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON file path.")
    parser.add_argument("-p", "--preset", choices=["tiny", "small", "medium", "large"], default="tiny", help="Size preset.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated list of workload names to include.")
    parser.add_argument("--exclude", type=str, default=None, help="Comma-separated list of workload names to exclude.")
    parser.add_argument("--list-workloads", action="store_true", help="List available workloads and exit.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed baseline.")
    parser.add_argument("--mem", action="store_true", help="Enable memory profiling using tracemalloc.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each workload this many times and aggregate.")
    args = parser.parse_args()

    if args.list_workloads:
        print("Available workloads:")
        for name, desc in list_workloads():
            print(f"  - {name}: {desc}")
        return

    try:
        gss_class = _load_impl(args.implementation_module, args.implementation_class)
    except Exception as e:
        print(f"Error: Could not load implementation: {e}", file=sys.stderr)
        sys.exit(1)

    incl, excl = _parse_workload_filters(args.only, args.exclude)
    workloads = resolve_workloads(incl, excl)
    if not workloads:
        print("No workloads selected. Use --list-workloads to see options.", file=sys.stderr)
        sys.exit(1)

    workload_names = [w.name for w in workloads]
    impl_name = f"{args.implementation_module}.{args.implementation_class}"

    data = _run_for_impl(
        gss_class=gss_class,
        impl_name=impl_name,
        preset=args.preset,
        workloads_to_run=workload_names,
        seed=args.seed,
        mem=args.mem,
        repeat=args.repeat,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote benchmark results to: {args.output}")


if __name__ == "__main__":
    main()
