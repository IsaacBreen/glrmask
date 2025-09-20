import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict
import textwrap


KEY_PHASES = {
    "merge_surface_changes": ["merge"],
    "push_scaling": ["push"],
    "merge_after_prefix_mutations": ["merge"],
    "pop_common_parent": ["pop"],
    "apply_prune": ["apply", "prune"],
    "fuzz": ["fuzz"],
}


def _load_results(files: List[Path]) -> List[Dict[str, Any]]:
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


def _format_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f} ms"
    return f"{ns/1_000_000_000:.2f} s"


def summarize(results: List[Dict[str, Any]], out_dir: Path):
    if not results:
        print("No results to analyze.")
        return

    grouped: DefaultDict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    impls: set[str] = set()
    for doc in results:
        impl_name = doc["implementation"]
        impls.add(impl_name)
        for w in doc["workloads"]:
            # Sort runs by complexity value to ensure proper plotting order
            grouped[w["workload"]][impl_name].append(w)
    for workload, impl_map in grouped.items():
        for impl, runs in impl_map.items():
            runs.sort(key=lambda r: r.get("complexity_val", 0))

    print("--- Benchmark Summary ---")
    print(f"Loaded {len(results)} result file(s) across {len(impls)} implementation(s).")

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nWarning: matplotlib not found. Skipping plot generation. `pip install matplotlib`")
        plt = None

    for workload, impl_map in sorted(grouped.items()):
        print(f"\n{'='*10} Workload: {workload} {'='*10}")

        first_run = next(iter(next(iter(impl_map.values()), [])), None)
        if not first_run: continue
        
        complexity_param = first_run.get("complexity_param")
        key_phases = KEY_PHASES.get(workload, [])

        if not complexity_param or not key_phases:
            print("  (Could not determine complexity parameter or key phases, skipping scaling analysis.)")
            continue
        
        print(f"  Scaling analysis against: '{complexity_param}' | Key phase(s): {key_phases}")

        # --- Data Extraction for Table and Plot ---
        plot_data = defaultdict(lambda: defaultdict(list))
        for impl, runs in sorted(impl_map.items()):
            print(f"\n  Implementation: {impl}")
            headers = [complexity_param] + [f"{p}_time" for p in key_phases] + ["peak_mem_mb"]
            col_widths = [len(h) for h in headers]
            rows = [headers]

            for r in runs:
                x = r.get("complexity_val", "N/A")
                row = [x]
                
                phase_map = {p["name"]: p.get("elapsed_ns", 0) for p in r.get("phases", [])}
                for p_name in key_phases:
                    y_ns = phase_map.get(p_name, 0)
                    row.append(y_ns)
                    if plt:
                        plot_data[impl][p_name].append((x, y_ns / 1e9))
                
                mem_mb = r.get("memory", {}).get("peak_bytes", 0) / (1024*1024)
                row.append(mem_mb)
                
                # Format for printing
                fmt_row = [f"{row[0]}"]
                fmt_row.extend([_format_ns(t) for t in row[1:-1]])
                fmt_row.append(f"{row[-1]:.2f}")
                
                for i, item in enumerate(fmt_row):
                    col_widths[i] = max(col_widths[i], len(item))
                rows.append(fmt_row)

            # Print table
            header_line = " | ".join(h.ljust(w) for h, w in zip(rows[0], col_widths))
            print("  " + header_line)
            print("  " + "-" * len(header_line))
            for r in rows[1:]:
                line = " | ".join(item.ljust(w) for item, w in zip(r, col_widths))
                print("  " + line)

        # --- Plotting ---
        if plt:
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 6))
            
            has_data = False
            for impl, phase_data in sorted(plot_data.items()):
                for phase_name, xy_pairs in sorted(phase_data.items()):
                    if not xy_pairs: continue
                    has_data = True
                    xs = [p[0] for p in xy_pairs]
                    ys = [p[1] for p in xy_pairs]
                    label = f"{impl.split('.')[-1]} ({phase_name})"
                    ax.plot(xs, ys, marker='o', linestyle='-', label=label)

            if has_data:
                ax.set_xlabel(f"Complexity: {complexity_param}")
                ax.set_ylabel("Key Phase Time (s)")
                ax.set_title(f"Scaling Analysis: {workload}", fontsize=14, pad=15)
                ax.legend(title="Implementation (Phase)", bbox_to_anchor=(1.02, 1), loc='upper left')
                ax.set_yscale('log')
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
                
                out_path = plots_dir / f"{workload}_scaling.png"
                fig.savefig(out_path, dpi=120)
                plt.close(fig)
                plot_paths.append(out_path)

    if plot_paths:
        print("\n--- Plots Generated ---")
        for p in plot_paths:
            print(f"  - {p}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze benchmark JSON files and produce scaling summaries and plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        This analyzer is designed to process benchmark results where workloads have been
        run over a range of complexity parameters. It identifies the key computational
        phase of each workload (e.g., 'merge', 'push') and plots its execution time
        against the complexity parameter that was varied.

        This helps visualize and quantify the scaling performance of different GSS
        implementations.
        """)
    )
    parser.add_argument("result_files", nargs="+", type=Path, help="Paths to benchmark JSON files.")
    parser.add_argument("-o", "--outdir", type=Path, default=Path("bench_analysis"), help="Output directory for analyses and plots.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    docs = _load_results(args.result_files)
    summarize(docs, args.outdir)


if __name__ == "__main__":
    main()
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
