from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone

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
    if ms < 1:
        return f"{ms:,.3f} ms"
    return f"{ms:,.2f} ms"


def _format_summary_lines(impl_map: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Generates formatted lines for both console and file output."""
    lines = []
    for impl, runs in sorted(impl_map.items()):
        lines.append(f"Implementation: {impl}")
        
        per_preset: Dict[str, List[Dict[str, Any]]] = {}
        for r in runs:
            per_preset.setdefault(r.get("preset", "?"), []).append(r)

        for preset, preset_runs in sorted(per_preset.items()):
            lines.append(f"  Preset: {preset}")
            
            # All runs for a given preset should have the same summary structure. Take the first.
            summary = preset_runs[0].get("summary", {})
            
            for wname, w_summary in sorted(summary.items()):
                t_mean = w_summary.get("total_ms_mean", 0.0)
                mem_mean = w_summary.get("peak_mem_bytes_mean", 0.0)
                mem_str = f"{mem_mean/1024.0:,.1f} KiB"
                lines.append(f"    Workload: {wname} (Total: {_pretty_ms(t_mean)}, Peak Mem: {mem_str})")

                phases_mean = w_summary.get("phases_mean", [])
                if not phases_mean:
                    lines.append("      - No phase data available.")
                    continue

                for phase in phases_mean:
                    phase_name = phase.get("phase", "unknown")
                    phase_ms = phase.get("ms_mean", 0.0)
                    lines.append(f"      - Phase '{phase_name}' ({_pretty_ms(phase_ms)})")
                    
                    method_stats = phase.get("method_stats_mean", {})
                    if not method_stats:
                        lines.append("          - No method stats.")
                        continue
                    
                    lines.append("          Methods:")
                    for method, stats in sorted(method_stats.items()):
                        calls = stats.get('calls_mean', 0.0)
                        total_ms = stats.get('total_ms_mean', 0.0)
                        avg_ms = total_ms / calls if calls > 0 else 0.0
                        calls_str = f"{calls:,.1f}" if not calls.is_integer() else f"{int(calls)}"
                        lines.append(f"            - {method:<15}: {calls_str} calls, {_pretty_ms(total_ms)} total, {_pretty_ms(avg_ms)} avg")
        lines.append("")
    return lines


def _print_summary(impl_map: Dict[str, List[Dict[str, Any]]]):
    print("--- Benchmark Summary ---")
    lines = _format_summary_lines(impl_map)
    for line in lines:
        print(line)


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
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": sorted([p.name for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]),
        }
        (out_dir / "index.json").write_text(json.dumps(index, indent=2))

    # Write a textual summary file
    summary_lines = _format_summary_lines(impl_map)
    (out_dir / "summary.txt").write_text("\n".join(summary_lines))

    print(f"\nAnalysis artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
