from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


class Plotter:
    """
    Lightweight plotting helper that gracefully degrades if matplotlib is unavailable.
    """
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            self._plt = plt
        except Exception:
            self._plt = None

    def plot_bar_for_workload(
        self,
        workload: str,
        preset: str,
        impl_to_value: Dict[str, float],
        metric_name: str,
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        if not self._plt:
            print(f"(plotting) matplotlib not available, skipping plot for {workload} [{preset}]")
            return
        if not impl_to_value:
            return

        plt = self._plt
        labels = list(impl_to_value.keys())
        values = [impl_to_value[k] for k in labels]

        fig, ax = plt.subplots(figsize=(max(6, min(14, len(labels) * 0.8)), 4.5))
        bars = ax.bar(range(len(labels)), values, color="#4C78A8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(metric_name)
        ax.set_title(title or f"{workload} [{preset}] {metric_name}")
        # Annotate bars
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{v:.1f}", ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        if not filename:
            filename = f"{workload}__{preset}__{metric_name}.png"
        out_path = self.out_dir / filename
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
