from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

from .plotting import Plotter


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _coalesce(files: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns mapping implementation -> list of run dicts (one per file).
    """
    impl_map: Dict[str, List[Dict[str, Any]]] = {}
    for p in files:
        try:
            data = _load_json(p)
            impl = data.get("implementation", str(p))
            impl_map.setdefault(impl, []).append(data)
        except Exception as e:
            print(f"Warning: Skipping invalid bench file {p}: {e}")
    return impl_map


def _pretty_ms(ms: float) -> str:
    return f"{ms:,.2f} ms"


def _print_summary(impl_map: Dict[str, List[Dict[str, Any]]]):
    print("--- Benchmark Summary ---")
    for impl, runs in impl_map.items():
        presets = ", ".join(sorted({r.get("preset", "?") for r in runs}))
        print(f"Implementation: {impl} (presets: {presets})")
        # Gather per-workload summaries across runs (potentially different presets)
        per_workload: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        for r in runs:
            summ = r.get("summary", {})
            for wname, agg in summ.items():
                per_workload.setdefault(wname, []).append((r.get("preset", "?"), agg))

        for wname, lst in sorted(per_workload.items()):
            # Print basic metrics
            # Choose a few key fields
            # Use total_ms_mean if present
            metrics_lines = []
            for preset, agg in sorted(lst, key=lambda x: x[0]):
                t_mean = agg.get("total_ms_mean", None)
                mem_mean = agg.get("peak_mem_bytes_mean", None)
                t_str = _pretty_ms(t_mean) if t_mean is not None else "n/a"
                if mem_mean is not None:
                    mem_str = f"{mem_mean/1024.0:,.1f} KiB"
                else:
                    mem_str = "n/a"
                metrics_lines.append(f"[{preset}] time={t_str}, peak_mem={mem_str}")
            print(f"  - {wname}: " + "; ".join(metrics_lines))
        print()


def _collect_for_plotting(files: List[Path]) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Returns:
      - workloads list
      - metrics dict: metrics[workload][preset][impl] -> total_ms_mean
    """
    impl_map = _coalesce(files)
    workloads: set = set()
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for impl, runs in impl_map.items():
        for r in runs:
            preset = r.get("preset", "?")
            for wname, agg in r.get("summary", {}).items():
                workloads.add(wname)
                metrics.setdefault(wname, {}).setdefault(preset, {})[impl] = float(agg.get("total_ms_mean", 0.0))
    return sorted(workloads), metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze GSS benchmark results and generate summaries/plots.")
    parser.add_argument("bench_files", nargs='+', type=Path, help="Paths to benchmark JSON result files.")
    parser.add_argument("-o", "--out-dir", type=Path, default=None, help="Directory for analysis artifacts (plots, summaries).")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    args = parser.parse_args()

    files = [p for p in args.bench_files if p.exists() and p.is_file()]
    if not files:
        print("Error: No valid benchmark result files provided.")
        return

    impl_map = _coalesce(files)
    _print_summary(impl_map)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path("gss_bench_analysis") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        workloads, metrics = _collect_for_plotting(files)
        plotter = Plotter(out_dir)
        for w in workloads:
            # one bar plot per preset for this workload
            for preset, impl_to_time in metrics.get(w, {}).items():
                plotter.plot_bar_for_workload(workload=w, preset=preset, impl_to_value=impl_to_time,
                                              metric_name="total_ms_mean", title=f"{w} [{preset}] total time (mean)",
                                              filename=f"{w}__{preset}.png")
        # Combined summary index.json
        index = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "artifacts": sorted([p.name for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]),
        }
        (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    # Write a textual summary file
    summary_txt = []
    for impl, runs in impl_map.items():
        summary_txt.append(f"Implementation: {impl}")
        for r in runs:
            summary_txt.append(f"  Preset: {r.get('preset')}")
            for wname, agg in r.get("summary", {}).items():
                t_mean = agg.get("total_ms_mean", None)
                mem_mean = agg.get("peak_mem_bytes_mean", None)
                t_str = f"{t_mean:.2f} ms" if t_mean is not None else "n/a"
                mem_str = f"{mem_mean/1024.0:.1f} KiB" if mem_mean is not None else "n/a"
                summary_txt.append(f"    - {wname}: time={t_str}, peak_mem={mem_str}")
        summary_txt.append("")
    (out_dir / "summary.txt").write_text("\n".join(summary_txt))

    print(f"\nAnalysis artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
