from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir


PANELS = [
    ("MovieLens", "movielens_100k_svdpp_optimized_20seeds"),
    ("Flixster", "flixster_svdpp_tuned_20seeds_20cycles"),
    ("Douban", "douban_svdpp_tuned_epochs6_lambda10_20seeds"),
]

POLICIES = ["random", "myopic", "entropy", "du_net"]
LABELS = {"random": "Random", "myopic": "Myopic", "entropy": "Entropy", "du_net": "DU-Net"}


def plot_metric(ax: plt.Axes, metrics: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    for policy in POLICIES:
        cur = metrics[metrics["policy"] == policy]
        if cur.empty:
            continue
        grouped = cur.groupby("cycle")[metric].agg(["mean", "std", "count"]).reset_index()
        ci = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
        x = grouped["cycle"].to_numpy(dtype=float)
        y = grouped["mean"].to_numpy(dtype=float)
        ci_arr = ci.to_numpy(dtype=float)
        ax.plot(x, y, marker="o", markersize=2.8, linewidth=1.8, label=LABELS.get(policy, policy))
        if grouped["count"].max() > 1:
            ax.fill_between(x, y - ci_arr, y + ci_arr, alpha=0.11)
    ax.set_title(title)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)


def main() -> None:
    table_dir = ROOT / "outputs" / "tables"
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    fig, axes = plt.subplots(len(PANELS), 2, figsize=(12, 12))
    for row, (dataset, prefix) in enumerate(PANELS):
        metrics = pd.read_csv(table_dir / f"{prefix}_metrics.csv")
        plot_metric(axes[row, 0], metrics, "w_rmse", f"{dataset}: implementation error", "w-RMSE")
        plot_metric(axes[row, 1], metrics, "cumulative_reward", f"{dataset}: acquisition reward", "Cumulative reward")
    axes[0, 1].legend(loc="best", frameon=False)
    fig.suptitle("Three-dataset DU-Net reproduction", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_dir / "three_dataset_main_reproduction.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
