import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

def load_results(files: List[Path]) -> List[Dict[str, Any]]:
    results = []
    for p in files:
        try:
            data = json.loads(p.read_text())
            # Basic validation
            if "implementation" in data and "workloads" in data:
                data["_source_file"] = str(p)
                results.append(data)
            else:
                print(f"Warning: Skipping {p} (missing required keys).")
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}")
    return results


def summarize(results: List[Dict[str, Any]], out_dir: Path, make_plots: bool = True):
    if not results:
        print("No results to analyze.")
        return

    # Group by workload name and implementation
    grouped: DefaultDict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    impls: set[str] = set()
    for doc in results:
        impl_name = doc["implementation"]
        impls.add(impl_name)
        for w in doc["workloads"]:
            grouped[w["workload"]][impl_name].append(w)

    print("--- Benchmark Summary ---")
    print(f"Loaded {len(results)} result file(s) across {len(impls)} implementation(s).")
    for workload, impl_map in grouped.items():
        print(f"\nWorkload: {workload}")
        for impl, runs in impl_map.items():
            ok = sum(1 for r in runs if r.get("outcome") == "ok")
            aborted = sum(1 for r in runs if r.get("outcome") == "aborted")
            errored = sum(1 for r in runs if r.get("outcome") == "error")
            times = [r.get("wall_time_ns", 0)/1e9 for r in runs if r.get("outcome") in ("ok", "aborted")]
            if times:
                print(f"  - {impl}: {len(runs)} run(s), ok={ok}, aborted={aborted}, error={errored}, time range: {min(times):.3f}s..{max(times):.3f}s")
            else:
                print(f"  - {impl}: {len(runs)} run(s), ok={ok}, aborted={aborted}, error={errored}")

    # Try to plot scaling curves if matplotlib is available
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import math

        # Helper to extract x (complexity) and y (phase times)
        def extract_xy(runs: List[Dict[str, Any]], workload: str) -> Tuple[List[float], Dict[str, List[float]]]:
            xs: List[float] = []
            phase_times: Dict[str, List[float]] = defaultdict(list)
            for r in runs:
                # Pick a heuristic complexity axis depending on workload
                if workload in ("merge_surface_changes", "apply_prune"):
                    derived = r.get("derived", {})
                    x = float(derived.get("theoretical_nodes", 0))
                    if x <= 0:
                        x = float(derived.get("theoretical_leaves", 0))
                elif workload in ("push_scaling", "merge_after_prefix_mutations", "pop_common_parent"):
                    derived = r.get("derived", {})
                    x = float(derived.get("hidden_prefix_depth", 0))
                    if x <= 0:
                        x = float(derived.get("siblings", 0))
                else:
                    x = float(r.get("wall_time_ns", 0))
                xs.append(x)
                for phase in r.get("phases", []):
                    phase_times[phase["name"]].append(phase.get("elapsed_ns", 0)/1e9)
            return xs, phase_times

        # For each workload, produce per-impl scaling plots
        for workload, impl_map in grouped.items():
            plt.figure(figsize=(8, 5))
            for impl, runs in impl_map.items():
                xs, phase_times = extract_xy(runs, workload)
                if not xs:
                    continue
                # Sort by x
                paired = list(zip(xs, *phase_times.values()))
                paired.sort(key=lambda t: t[0])
                xs_sorted = [t[0] for t in paired]
                # Sum phases to get total per run
                totals = [sum(t[1:]) for t in paired]
                plt.plot(xs_sorted, totals, marker='o', label=impl.split(".")[-1])
            plt.xlabel("Complexity (heuristic)")
            plt.ylabel("Total time (s) [sum of phases]")
            plt.title(f"Scaling: {workload}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = plots_dir / f"{workload}_scaling.png"
            plt.savefig(out_path)
            plt.close()
            print(f"Saved plot: {out_path}")

            # Per-phase stacked bars for the last run of each impl
            plt.figure(figsize=(9, 5))
            labels = []
            bottoms = None
            phases_order: List[str] = []
            # Determine a union of phase names
            union_phases = set()
            for impl, runs in impl_map.items():
                if not runs:
                    continue
                for ph in runs[-1].get("phases", []):
                    union_phases.add(ph["name"])
            phases_order = sorted(union_phases)
            values_by_phase: Dict[str, List[float]] = {p: [] for p in phases_order}
            for impl, runs in impl_map.items():
                if not runs:
                    continue
                labels.append(impl.split(".")[-1])
                last = runs[-1]
                phmap = {ph["name"]: ph.get("elapsed_ns", 0)/1e9 for ph in last.get("phases", [])}
                for p in phases_order:
                    values_by_phase[p].append(phmap.get(p, 0.0))
            import numpy as np
            x = np.arange(len(labels))
            width = 0.65
            bottoms = np.zeros(len(labels))
            for p in phases_order:
                data = np.array(values_by_phase[p])
                plt.bar(x, data, width, bottom=bottoms, label=p)
                bottoms += data
            plt.xticks(x, labels, rotation=45, ha="right")
            plt.ylabel("Time (s)")
            plt.title(f"Phase breakdown (last run): {workload}")
            plt.legend()
            plt.tight_layout()
            out_path = plots_dir / f"{workload}_phase_breakdown.png"
            plt.savefig(out_path)
            plt.close()
            print(f"Saved plot: {out_path}")

    except Exception as e:
        print(f"Note: Plotting skipped or failed ({e}). You can install matplotlib and numpy for plots.")


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark JSON files and produce summaries/plots.")
    parser.add_argument("result_files", nargs="+", type=Path, help="Paths to benchmark JSON files.")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("bench_analysis"), help="Output directory for analyses and plots.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    docs = load_results(args.result_files)
    summarize(docs, args.outdir, make_plots=True)


if __name__ == "__main__":
    main()
