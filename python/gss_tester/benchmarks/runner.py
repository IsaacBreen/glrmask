from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

from gss_tester.interface import GSS
from .workloads import WORKLOADS, default_specs, WorkloadResult
from .introspect import summarize_structure


def _load_impl(descriptor: str) -> Tuple[str, Type[GSS]]:
    """
    Load an implementation from 'module:Class' or 'module.Class'.
    Returns (pretty_name, class_obj).
    """
    if ":" in descriptor:
        module_name, class_name = descriptor.split(":", 1)
    elif "." in descriptor:
        # Split on last dot to allow dotted module paths.
        parts = descriptor.split(".")
        module_name, class_name = ".".join(parts[:-1]), parts[-1]
    else:
        raise ValueError(f"Invalid implementation descriptor: {descriptor}. Use module:Class")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    pretty = f"{module_name}.{class_name}"
    return pretty, cls


def _summarize_per_op_timings(per_op: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    For each op, compute count, mean, median, p95, max. Times are in seconds.
    """
    summary: Dict[str, Dict[str, float]] = {}
    for op, vals in per_op.items():
        if not vals:
            summary[op] = {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
            continue
        vals_sorted = sorted(vals)
        n = len(vals)
        p95_idx = min(n - 1, int(n * 0.95))
        summary[op] = {
            "count": float(n),
            "mean": float(sum(vals) / n),
            "median": float(statistics.median(vals)),
            "p95": float(vals_sorted[p95_idx]),
            "max": float(vals_sorted[-1]),
        }
    return summary


def _run_workloads_for_impl(
    impl_name: str,
    impl_class: Type[GSS],
    selected: List[str],
    specs: Dict[str, List[Dict[str, Any]]],
    repeats: int,
) -> Dict[str, Any]:
    """
    Execute selected workloads with given parameter specs and repeats.
    Returns a dict of results for JSON emission.
    """
    results: List[Dict[str, Any]] = []
    for wl_name in selected:
        if wl_name not in WORKLOADS:
            print(f"Warning: workload '{wl_name}' not found. Skipping.", file=sys.stderr)
            continue
        workload_fn = WORKLOADS[wl_name]
        param_sets = specs.get(wl_name, [{}])  # default to empty param-set if unspecified

        for params in param_sets:
            for r in range(repeats):
                run_label = f"{wl_name} params={params} repeat={r+1}/{repeats}"
                print(f"[{impl_name}] Running {run_label} ...", flush=True)
                t0 = time.perf_counter()
                wl_result: WorkloadResult = workload_fn(impl_class, **params)
                elapsed = time.perf_counter() - t0

                # Structural summary derived from final state
                summary = summarize_structure(wl_result.final_state).to_dict()

                per_op_summary = _summarize_per_op_timings(wl_result.timings.per_op)

                results.append({
                    "workload": wl_result.name,
                    "params": wl_result.params,
                    "repeat_index": r,
                    "operations_executed": wl_result.operations_executed,
                    "timing": {
                        "total_wall_time": elapsed,
                        "peak_mem_kib": wl_result.timings.peak_mem_kib,
                        "phases": wl_result.timings.phases,
                        "per_op_summary": per_op_summary,
                    },
                    "structure": summary,
                })
    return {
        "implementation": impl_name,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark GSS implementations for performance and scaling.")
    parser.add_argument(
        "--implementations",
        nargs="*",
        default=[
            "gss_tester.reference_impl:ReferenceGSS",
            "gss_tester.leveled_impl:LeveledGSS",
        ],
        help="List of implementations in the form module:Class. Default runs the bundled ReferenceGSS and LeveledGSS.",
    )
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=list(WORKLOADS.keys()),
        help=f"Subset of workloads to run. Available: {', '.join(sorted(WORKLOADS.keys()))}",
    )
    parser.add_argument(
        "--preset",
        choices=["tiny", "small", "medium", "large"],
        default="tiny",
        help="Parameter preset for workloads: tiny, small, medium, large."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat each workload-param combination.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("gss_bench_results.json"),
        help="Path to write JSON results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file specifying workload parameter sets. Structure: {workload_name: [ {param: value, ...}, ... ]}",
    )
    args = parser.parse_args()

    # Resolve workload specs
    if args.config and args.config.exists():
        try:
            specs = json.loads(args.config.read_text())
            if not isinstance(specs, dict):
                raise ValueError("Config must be a JSON object mapping workload -> list of param dicts")
        except Exception as e:
            print(f"Error reading config {args.config}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        specs = default_specs(args.preset)

    # Load implementations
    loaded_impls: List[Tuple[str, Type[GSS]]] = []
    for desc in args.implementations:
        try:
            pretty, cls = _load_impl(desc)
            loaded_impls.append((pretty, cls))
        except Exception as e:
            print(f"Error loading implementation '{desc}': {e}", file=sys.stderr)

    if not loaded_impls:
        print("No valid implementations loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Environment info
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }

    all_out: Dict[str, Any] = {
        "env": env,
        "workloads_selected": args.workloads,
        "preset": args.preset,
        "timestamp": time.time(),
        "implementations": [],
    }

    for impl_name, impl_class in loaded_impls:
        impl_out = _run_workloads_for_impl(
            impl_name=impl_name,
            impl_class=impl_class,
            selected=args.workloads,
            specs=specs,
            repeats=max(1, args.repeats),
        )
        all_out["implementations"].append(impl_out)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_out, indent=2))
    print(f"\nBenchmark complete. Results written to {args.output}")


if __name__ == "__main__":
    main()
