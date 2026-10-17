from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, run_policy
from src.baselines import ABLATION_POLICIES
from src.data_loader import generate_synthetic_dataset, split_ratings
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


def plot_ablation(metrics: pd.DataFrame, metric: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(9, 5.5))
    for policy in ABLATION_POLICIES:
        cur = metrics[metrics["policy"] == policy]
        summary = cur.groupby("cycle", as_index=False)[metric].mean()
        plt.plot(summary["cycle"], summary[metric], marker="o", linewidth=1.8, label=policy)
    plt.xlabel("Acquisition cycle")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_ablation_report(metrics: pd.DataFrame, path: Path) -> None:
    final = metrics.sort_values("cycle").groupby(["policy", "seed"], as_index=False).tail(1)
    summary = (
        final.groupby("policy", as_index=False)
        .agg(
            cumulative_reward=("cumulative_reward", "mean"),
            rmse=("rmse", "mean"),
            w_rmse=("w_rmse", "mean"),
            sigma=("sigma", "mean"),
            avg_effective_neighborhood=("avg_effective_neighborhood", "mean"),
        )
        .sort_values("w_rmse")
    )
    lines = [
        "# Ablation Report",
        "",
        "Synthetic reproduction of the DU-Net ablation plan.",
        "",
        "| policy | cumulative_reward | rmse | w_rmse | sigma | avg_effective_neighborhood |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.policy} | {row.cumulative_reward:.4f} | {row.rmse:.4f} | {row.w_rmse:.4f} | "
            f"{row.sigma:.4f} | {row.avg_effective_neighborhood:.4f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `du_net_no_dls` removes DLS-smoothed user factors.",
            "- `du_net_no_entropy` uses reward only.",
            "- `du_net_no_reward` uses information value only.",
            "- `du_net_no_network` removes neighbor reward and network entropy amplification.",
            "- `du_net_dynamic_sigma` chooses sigma from the configured grid each cycle using observed reconstruction error.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "ablation_synthetic.json"))
    args = parser.parse_args()
    config = load_json(args.config)

    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    report_dir = ensure_dir(ROOT / "reports")

    mf_config = MFConfig(**config["mf"])
    acquisition_config = AcquisitionConfig(**config["acquisition"])
    all_metrics = []
    for seed in config["seeds"]:
        dataset = generate_synthetic_dataset(seed=seed, **config["synthetic_data"])
        split = split_ratings(dataset.ratings, seed=seed, **config["split"])
        for policy in ABLATION_POLICIES:
            metrics = run_policy(
                policy=policy,
                split=split,
                adjacency=dataset.adjacency,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=mf_config,
                acquisition_config=acquisition_config,
                seed=seed,
            )
            metrics.insert(0, "seed", seed)
            all_metrics.append(metrics)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_path = table_dir / "ablation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    final = metrics_df.sort_values("cycle").groupby(["policy", "seed"], as_index=False).tail(1)
    final_summary = final.groupby("policy", as_index=False).agg(
        cumulative_reward=("cumulative_reward", "mean"),
        rmse=("rmse", "mean"),
        w_rmse=("w_rmse", "mean"),
        avg_effective_neighborhood=("avg_effective_neighborhood", "mean"),
    )
    final_summary.to_csv(table_dir / "ablation_final_summary.csv", index=False)
    plot_ablation(metrics_df, "w_rmse", figure_dir / "ablation_w_rmse.png", "w-RMSE")
    plot_ablation(metrics_df, "cumulative_reward", figure_dir / "ablation_cumulative_reward.png", "Cumulative centered reward")
    write_ablation_report(metrics_df, report_dir / "ablation_report.md")
    print(f"Wrote metrics: {metrics_path}")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
