import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import platform
import socket

from .instrumentation import TimingRecorder, GSSFactory
from .workloads import PRESETS, WORKLOAD_FUNCS, WorkloadConfig


def _filter_workloads(configs: List[WorkloadConfig], include: List[str], exclude: List[str]) -> List[WorkloadConfig]:
    if include:
        configs = [c for c in configs if any(p.lower() in c.name.lower() for p in include)]
    if exclude:
        configs = [c for c in configs if all(p.lower() not in c.name.lower() for p in exclude)]
    return configs


def main():
    parser = argparse.ArgumentParser(description="Run GSS benchmark workloads for a given implementation.")
    parser.add_argument("implementation_module", help="Python module containing the GSS class (e.g., gss_tester.implementations.reference_impl)")
    parser.add_argument("implementation_class", help="Class name of the GSS implementation (e.g., ReferenceGSS)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Path to output JSON file for results.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="small", help="Workload size preset.")
    parser.add_argument("--include", nargs="*", default=[], help="Only run workloads whose names contain one of these substrings.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Exclude workloads whose names contain one of these substrings.")
    parser.add_argument("--list", action="store_true", help="List workloads for the selected preset and exit.")
    args = parser.parse_args()

    # Load implementation
    try:
        module = importlib.import_module(args.implementation_module)
        gss_class = getattr(module, args.implementation_class)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not load GSS implementation. {e}", file=sys.stderr)
        sys.exit(1)

    # Build workload list based on preset
    preset_func = PRESETS[args.preset]
    workloads = preset_func()
    workloads = _filter_workloads(workloads, args.include, args.exclude)

    if args.list:
        print(f"Preset '{args.preset}' workloads ({len(workloads)}):")
        for cfg in workloads:
            print(f"  - {cfg.name}: {cfg.params} (max_seconds={cfg.max_seconds})")
        return

    # Prepare output structure
    header = {
        "benchmark_version": 1,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "implementation": f"{args.implementation_module}.{args.implementation_class}",
        "preset": args.preset,
        "platform": {
            "python": sys.version,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
        "workloads": [],
    }

    # Run workloads sequentially; errors are captured per workload.
    for cfg in workloads:
        print(f">>> Running workload '{cfg.name}' with params {cfg.params} (max {cfg.max_seconds}s)")
        # Fresh recorder per workload to get clean per-phase stats
        recorder = TimingRecorder()
        factory = GSSFactory(gss_class=gss_class, recorder=recorder)
        func = WORKLOAD_FUNCS.get(cfg.name)
        if func is None:
            print(f"Warning: Unknown workload '{cfg.name}', skipping.")
            continue

        try:
            result = func(factory, cfg)
        except Exception as e:
            # Shouldn't reach here; workloads already catch errors. But be resilient.
            result = {
                "workload": cfg.name,
                "params": cfg.params,
                "outcome": "error",
                "error": f"{e.__class__.__name__}: {e}",
            }
        header["workloads"].append(result)
        print(f"<<< Finished workload '{cfg.name}' outcome={result.get('outcome','ok')} in {result.get('wall_time_ns',0)/1e9:.3f}s")

    # Write JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(header, f, indent=2)
    print(f"Benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
