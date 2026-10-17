from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, run_policy
from src.baselines import MVP_POLICIES
from src.data_loader import load_dataset_from_config, split_dataset
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


def summarize_final(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for policy, group in final.groupby("policy", sort=False):
        row: dict[str, float | str] = {"policy": policy}
        for metric in [
            "cumulative_reward",
            "rmse",
            "w_rmse",
            "sigmoid_reward_w_rmse",
            "mild_reward_w_rmse",
            "medium_reward_w_rmse",
            "linear_reward_norm_w_rmse",
            "positive_reward_norm_w_rmse",
            "observed_size",
        ]:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[metric] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric(metrics: pd.DataFrame, policies: list[str], metric: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for policy in policies:
        cur = metrics[metrics["policy"] == policy]
        grouped = cur.groupby("cycle")[metric]
        summary = grouped.agg(["mean", "std", "count"]).reset_index()
        ci95 = 1.96 * summary["std"].fillna(0.0) / np.sqrt(summary["count"].clip(lower=1))
        x = summary["cycle"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        ci = ci95.to_numpy(dtype=float)
        plt.plot(x, y, marker="o", linewidth=2, label=policy)
        if summary["count"].max() > 1:
            plt.fill_between(x, y - ci, y + ci, alpha=0.12)
    plt.xlabel("Acquisition cycle")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_report(config: dict, dataset_name: str, metrics: pd.DataFrame, path: Path) -> None:
    final = metrics.sort_values("cycle").groupby(["policy", "seed"], as_index=False).tail(1)
    summary = summarize_final(final)
    lines = [
        f"# {dataset_name} Reproduction Report",
        "",
        f"Dataset: `{dataset_name}` loaded from `{config['dataset']['raw_dir']}`.",
        "",
        "## Configuration",
        "",
        f"- Seeds: {config['seeds']}",
        f"- Initial fraction of train: {config['split']['initial_fraction_of_train']}",
        f"- Cycles: {config['acquisition']['cycles']}",
        f"- Batch size: {config['acquisition']['batch_size']}",
        f"- Policies: {', '.join(config.get('policies', MVP_POLICIES))}",
        f"- Entropy mode: {config['acquisition'].get('entropy_mode', 'coverage')}",
        "",
        "## Final-Cycle Summary",
        "",
        "| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("policy").itertuples(index=False):
        lines.append(
            f"| {row.policy} | {row.cumulative_reward:.4f} | {row.w_rmse:.4f} | "
            f"{row.w_rmse_ci95:.4f} | {row.rmse:.4f} | {row.observed_size:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- This is a {len(config['seeds'])}-seed real-data reproduction run.",
            "- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "movielens_100k.json"))
    args = parser.parse_args()
    config = load_json(args.config)

    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    report_dir = ensure_dir(ROOT / "reports")

    dataset = load_dataset_from_config(config)
    mf_config = MFConfig(**config["mf"])
    acquisition_config = AcquisitionConfig(**config["acquisition"])
    policies = config.get("policies", MVP_POLICIES)
    policy_overrides = config.get("policy_overrides", {})
    all_metrics = []
    for seed in config["seeds"]:
        split = split_dataset(dataset, seed=seed, **config["split"])
        for policy in policies:
            policy_acquisition_config = acquisition_config
            if policy in policy_overrides:
                policy_acquisition_config = replace(acquisition_config, **policy_overrides[policy])
            metrics = run_policy(
                policy=policy,
                split=split,
                adjacency=dataset.adjacency,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=mf_config,
                acquisition_config=policy_acquisition_config,
                seed=seed,
            )
            metrics.insert(0, "seed", seed)
            all_metrics.append(metrics)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    prefix = config.get("output_prefix", "movielens_100k")
    metrics_path = table_dir / f"{prefix}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    final = metrics_df.sort_values("cycle").groupby(["policy", "seed"], as_index=False).tail(1)
    final_summary = summarize_final(final)
    final_summary.to_csv(table_dir / f"{prefix}_final_summary.csv", index=False)
    plot_metric(metrics_df, policies, "w_rmse", figure_dir / f"{prefix}_w_rmse.png", "w-RMSE")
    plot_metric(metrics_df, policies, "cumulative_reward", figure_dir / f"{prefix}_cumulative_reward.png", "Cumulative centered reward")
    write_report(config, dataset.name, metrics_df, report_dir / f"{prefix}_report.md")
    print(f"Wrote metrics: {metrics_path}")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
