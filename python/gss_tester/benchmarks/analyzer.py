import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict
import math

from .workloads import get_scaling_expectations

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

def _loglog_fit(xs: List[float], ys: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Fit ys ~ a * xs^b via log-log linear regression and return (b, a, r2).
    Returns (None, None, None) if not enough valid points.
    """
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x > 0 and y > 0]
    n = len(pairs)
    if n < 2:
        return None, None, None
    lx = [math.log(x) for x, _ in pairs]
    ly = [math.log(y) for _, y in pairs]
    sumx = sum(lx)
    sumy = sum(ly)
    sumx2 = sum(v*v for v in lx)
    sumxy = sum(a*b for a, b in zip(lx, ly))
    den = n * sumx2 - sumx * sumx
    if den <= 0:
        return None, None, None
    b = (n * sumxy - sumx * sumy) / den
    a_log = (sumy - b * sumx) / n
    # r^2
    y_mean = sumy / n
    ss_tot = sum((v - y_mean) ** 2 for v in ly)
    ss_res = sum((lyi - (a_log + b*lxi)) ** 2 for lxi, lyi in zip(lx, ly))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None
    a = math.exp(a_log)
    return b, a, r2

def analyze_sweeps(results: List[Dict[str, Any]], out_dir: Path):
    """
    Detect sweep-mode benchmark documents and estimate scaling exponents per phase.
    Compare measured exponents to ideal expectations (if available), and plot.
    """
    sweeps = [doc for doc in results if "sweep" in doc]
    if not sweeps:
        return

    # Group sweeps by (workload, axis) for comparative analysis
    grouped_sweeps: DefaultDict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for doc in sweeps:
        sweep_meta = doc["sweep"]
        key = (sweep_meta["workload"], sweep_meta["axis"])
        grouped_sweeps[key].append(doc)

    expectations = get_scaling_expectations()
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\nScaling Analysis (sweeps)")
    for (workload, axis), docs in sorted(grouped_sweeps.items()):
        print(f"\nSweep: workload={workload}, axis={axis}")
        ideal_map = expectations.get(workload, {}).get(axis, {})

        # Header for the results table
        print(f"  {'Implementation':<55} {'Phase':<20} {'Slope':>7} {'R^2':>6} {'Ideal':>7} {'Status'}")
        print(f"  {'-'*55} {'-'*20} {'-'*7} {'-'*6} {'-'*7} {'-'*8}")

        all_plot_data = []

        for doc in sorted(docs, key=lambda d: d.get("implementation", "")):
            impl = doc.get("implementation", "(unknown)")
            runs = [w for w in doc["workloads"] if w.get("workload") == workload]
            if not runs:
                continue

            phases_union = set()
            for r in runs:
                for ph in r.get("phases", []):
                    phases_union.add(ph["name"])
            phases = sorted([p for p in phases_union if p not in ("build", "postcheck")])

            x = []
            phase_ys: Dict[str, List[float]] = {p: [] for p in phases}
            for r in runs:
                pv = r.get("params", {}).get(axis)
                if pv is None:
                    pv = r.get("derived", {}).get(axis)
                try:
                    xval = float(pv)
                except Exception:
                    x = []
                    break
                x.append(xval)
                phmap = {ph["name"]: ph.get("elapsed_ns", 0) / 1e9 for ph in r.get("phases", [])}
                for p in phases:
                    phase_ys[p].append(phmap.get(p, 0.0))

            if len(x) < 2:
                continue

            all_plot_data.append({"impl": impl, "x": x, "phase_ys": phase_ys, "phases": phases})

            for p in phases:
                b, a, r2 = _loglog_fit(x, phase_ys[p])
                if b is None:
                    print(f"  {impl:<55.55} {p:<20} {'N/A':>7} {'N/A':>6} {'N/A':>7} {'NO DATA'}")
                    continue
                ideal = ideal_map.get(p)
                if ideal is None:
                    status = "(no ideal)"
                    ideal_str = "N/A"
                else:
                    delta = abs(b - ideal)
                    status = "OK" if delta <= 0.25 else "DRIFT"
                    ideal_str = f"{ideal:.2f}"
                print(f"  {impl:<55.55} {p:<20} {b:7.3f} {r2:6.3f} {ideal_str:>7} {status}")

        # Plotting for this sweep group
        plotted_anything = False
        try:
            import matplotlib.pyplot as plt
            for plot_data in all_plot_data:
                impl, x, phase_ys, phases = plot_data["impl"], plot_data["x"], plot_data["phase_ys"], plot_data["phases"]
                for p in phases:
                    y = phase_ys[p]
                    valid = [(xi, yi) for xi, yi in zip(x, y) if xi > 0 and yi > 0]
                    if len(valid) < 2:
                        continue
                    if not plotted_anything:
                        print("\n  Plots:")
                        plotted_anything = True
                    xs, ys = zip(*valid)
                    b, a, r2 = _loglog_fit(list(xs), list(ys))
                    plt.figure(figsize=(6, 4))
                    plt.loglog(xs, ys, marker='o', linestyle='-', label=f"measured (r2={r2:.2f})")
                    if b is not None and a is not None:
                        xs_sorted = sorted(xs)
                        ys_fit = [a * (xx ** b) for xx in xs_sorted]
                        plt.loglog(xs_sorted, ys_fit, linestyle='--', label=f"fit: y~{a:.2e}x^{b:.2f}")
                    ideal = ideal_map.get(p)
                    if ideal is not None:
                        x0, y0 = xs[0], ys[0]
                        ref = [y0 * ((xx / x0) ** ideal) for xx in xs_sorted]
                        plt.loglog(xs_sorted, ref, linestyle=':', label=f'ideal slope {ideal:.2f}')
                    plt.xlabel(axis)
                    plt.ylabel("Time (s)")
                    plt.title(f"{impl.split('.')[-1]} | {workload} | phase={p}")
                    plt.legend()
                    plt.grid(True, which="both", alpha=0.3)
                    plt.tight_layout()
                    impl_fname_part = impl.replace('.', '_')
                    out_path = plots_dir / f"sweep_{workload}_{axis}_{impl_fname_part}_{p}.png"
                    plt.savefig(out_path)
                    plt.close()
                    try:
                        rel_path = out_path.relative_to(Path.cwd())
                        print(f"    - Saved: {rel_path}")
                    except ValueError:
                        print(f"    - Saved: {out_path}")
        except Exception as e:
            print(f"  Note: Sweep plotting skipped ({e}).")

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

    print("Benchmark Summary")
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
            try:
                rel_path = out_path.relative_to(Path.cwd())
                print(f"  - Saved plot: {rel_path}")
            except ValueError:
                print(f"  - Saved plot: {out_path}")

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
            try:
                rel_path = out_path.relative_to(Path.cwd())
                print(f"  - Saved plot: {rel_path}")
            except ValueError:
                print(f"  - Saved plot: {out_path}")

    except Exception as e:
        print(f"Note: Plotting skipped or failed ({e}). You can install matplotlib and numpy for plots.")

    # After general plots, run sweep-specific scaling analysis
    analyze_sweeps(results, out_dir)


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
