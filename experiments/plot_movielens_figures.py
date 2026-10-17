from __future__ import annotations

import argparse
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


POLICY_LABELS = {
    "random": "Random",
    "myopic": "Myopic",
    "entropy": "Max entropy",
    "linear_ucb": "Linear UCB",
    "low_rank_bandit": "Low-rank bandit",
    "du_net": "DU-Net",
    "du_net_no_dls": "w/o DLS",
    "du_net_static_sigma": "Static sigma",
    "du_net_dynamic_sigma": "Dynamic sigma",
    "du_net_no_network": "w/o network",
}


def _mean_ci(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = frame.groupby("cycle")[metric].agg(["mean", "std", "count"]).reset_index()
    out["ci95"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def _plot_cycle_panel(ax: plt.Axes, metrics: pd.DataFrame, policies: list[str], metric: str, ylabel: str) -> None:
    for policy in policies:
        cur = metrics[metrics["policy"] == policy]
        if cur.empty:
            continue
        stats = _mean_ci(cur, metric)
        x = stats["cycle"].to_numpy(dtype=float)
        y = stats["mean"].to_numpy(dtype=float)
        ci = stats["ci95"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=2, label=POLICY_LABELS.get(policy, policy))
        if stats["count"].max() > 1:
            ax.fill_between(x, y - ci, y + ci, alpha=0.14)
    ax.set_xlabel("Acquisition cycle")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.24)


def figure4(main_csv: Path, output: Path) -> None:
    metrics = pd.read_csv(main_csv)
    policies = ["random", "myopic", "entropy", "du_net"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _plot_cycle_panel(axes[0], metrics, policies, "w_rmse", "w-RMSE")
    _plot_cycle_panel(axes[1], metrics, policies, "cumulative_reward", "Cumulative centered reward")
    axes[0].set_title("Implementation error")
    axes[1].set_title("Acquisition reward")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Figure 4 style: MovieLens 100K homogeneous response", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure5(robustness_csv: Path, output: Path) -> None:
    metrics = pd.read_csv(robustness_csv)
    if "alpha" not in metrics.columns:
        raise ValueError(f"{robustness_csv} does not contain an alpha column")
    alpha = float(metrics["alpha"].iloc[0])
    policies = ["random", "myopic", "entropy", "du_net"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _plot_cycle_panel(axes[0], metrics, policies, "w_rmse", "w-RMSE")
    _plot_cycle_panel(axes[1], metrics, policies, "cumulative_reward", "Cumulative centered reward")
    axes[0].set_title("Implementation error")
    axes[1].set_title("Acquisition reward")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle(f"Figure 5 style: MovieLens heterogeneous response (alpha={alpha:.2f})", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def comparison_figure(csv_path: Path, output: Path, title: str, policies: list[str] | None = None) -> None:
    metrics = pd.read_csv(csv_path)
    if policies is None:
        policies = list(metrics["policy"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _plot_cycle_panel(axes[0], metrics, policies, "w_rmse", "w-RMSE")
    _plot_cycle_panel(axes[1], metrics, policies, "cumulative_reward", "Cumulative centered reward")
    axes[0].set_title("Implementation error")
    axes[1].set_title("Acquisition reward")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-prefix", default="movielens_100k_svdpp_optimized_20seeds")
    parser.add_argument("--robustness-prefix", default="movielens_100k_svdpp_robustness_20seeds")
    parser.add_argument("--baseline-prefix", default="movielens_100k_svdpp_baselines_20seeds")
    parser.add_argument("--ablation-prefix", default="movielens_100k_svdpp_dls_ablation_20seeds")
    args = parser.parse_args()

    table_dir = ROOT / "outputs" / "tables"
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    figure4(table_dir / f"{args.main_prefix}_metrics.csv", figure_dir / "figure4_movielens_main_20seeds.png")
    figure5(
        table_dir / f"{args.robustness_prefix}_metrics.csv",
        figure_dir / "figure5_movielens_heterogeneous_20seeds.png",
    )

    baseline_csv = table_dir / f"{args.baseline_prefix}_metrics.csv"
    if baseline_csv.exists():
        comparison_figure(
            baseline_csv,
            figure_dir / "movielens_bandit_baselines_20seeds.png",
            "MovieLens 100K bandit baselines",
        )

    ablation_csv = table_dir / f"{args.ablation_prefix}_metrics.csv"
    if ablation_csv.exists():
        comparison_figure(
            ablation_csv,
            figure_dir / "movielens_dls_ablation_20seeds.png",
            "MovieLens 100K DLS and network ablation",
        )


if __name__ == "__main__":
    main()
