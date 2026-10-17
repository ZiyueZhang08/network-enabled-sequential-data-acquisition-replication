from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, run_policy
from src.baselines import MVP_POLICIES
from src.data_loader import generate_synthetic_dataset, split_ratings
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


def plot_alpha_summary(summary: pd.DataFrame, metric: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for policy in MVP_POLICIES:
        cur = summary[summary["policy"] == policy].sort_values("alpha")
        plt.plot(cur["alpha"], cur[metric], marker="o", linewidth=2, label=policy)
    plt.xlabel("Response strength alpha")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_robustness_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Heterogeneous Response Robustness Report",
        "",
        "Response model: `P(response)=clip(0.8 + alpha * (rating - 3), 0, 1)`.",
        "",
        "| alpha | policy | cumulative_reward | rmse | w_rmse | response_rate | observed_size |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["alpha", "policy"]).itertuples(index=False):
        lines.append(
            f"| {row.alpha:.4f} | {row.policy} | {row.cumulative_reward:.4f} | {row.rmse:.4f} | "
            f"{row.w_rmse:.4f} | {row.response_rate:.4f} | {row.observed_size:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "robustness_synthetic.json"))
    args = parser.parse_args()
    config = load_json(args.config)

    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    report_dir = ensure_dir(ROOT / "reports")

    mf_config = MFConfig(**config["mf"])
    base_acquisition_config = AcquisitionConfig(**config["acquisition"])
    all_metrics = []
    for alpha in config["response_alphas"]:
        acquisition_config = replace(base_acquisition_config, response_alpha=alpha)
        for seed in config["seeds"]:
            dataset = generate_synthetic_dataset(seed=seed, **config["synthetic_data"])
            split = split_ratings(dataset.ratings, seed=seed, **config["split"])
            for policy in MVP_POLICIES:
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
                metrics.insert(0, "alpha", alpha)
                metrics.insert(0, "seed", seed)
                all_metrics.append(metrics)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_path = table_dir / "robustness_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    final = metrics_df.sort_values("cycle").groupby(["alpha", "policy", "seed"], as_index=False).tail(1)
    summary = final.groupby(["alpha", "policy"], as_index=False).agg(
        cumulative_reward=("cumulative_reward", "mean"),
        rmse=("rmse", "mean"),
        w_rmse=("w_rmse", "mean"),
        response_rate=("response_rate", "mean"),
        observed_size=("observed_size", "mean"),
    )
    summary.to_csv(table_dir / "robustness_final_summary.csv", index=False)
    plot_alpha_summary(summary, "w_rmse", figure_dir / "robustness_w_rmse_by_alpha.png", "Final w-RMSE")
    plot_alpha_summary(summary, "cumulative_reward", figure_dir / "robustness_reward_by_alpha.png", "Final cumulative centered reward")
    write_robustness_report(summary, report_dir / "robustness_report.md")
    print(f"Wrote metrics: {metrics_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
