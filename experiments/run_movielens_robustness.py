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
    group_cols = ["alpha", "variant"] if "variant" in final.columns else ["alpha", "policy"]
    for keys, group in final.groupby(group_cols, sort=False):
        if len(group_cols) == 2:
            alpha, label = keys
        else:
            alpha, label = keys[0], str(group["policy"].iloc[0])
        row: dict[str, float | str] = {"alpha": float(alpha), group_cols[1]: label}
        row["policy"] = str(group["policy"].iloc[0])
        for metric in ["cumulative_reward", "rmse", "w_rmse", "response_rate", "observed_size"]:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[metric] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95
        rows.append(row)
    return pd.DataFrame(rows)


def plot_alpha_metric(summary: pd.DataFrame, labels: list[str], metric: str, path: Path, ylabel: str) -> None:
    label_col = "variant" if "variant" in summary.columns else "policy"
    plt.figure(figsize=(8, 5))
    for label in labels:
        cur = summary[summary[label_col] == label].sort_values("alpha")
        x = cur["alpha"].to_numpy(dtype=float)
        y = cur[metric].to_numpy(dtype=float)
        ci = cur[f"{metric}_ci95"].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", linewidth=2, label=label)
        plt.fill_between(x, y - ci, y + ci, alpha=0.12)
    plt.xlabel("Response strength alpha")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_report(config: dict, summary: pd.DataFrame, path: Path) -> None:
    label_col = "variant" if "variant" in summary.columns else "policy"
    dataset_name = config.get("dataset", {}).get("name", "dataset")
    lines = [
        f"# {dataset_name} Heterogeneous Response Robustness",
        "",
        "Response model: `P(response)=clip(0.8 + alpha * (rating - 3), 0, 1)`.",
        "",
        "## Configuration",
        "",
        f"- Seeds: {config['seeds']}",
        f"- Response alphas: {config['response_alphas']}",
        f"- Policies: {', '.join(config.get('policies', [variant.get('policy', 'du_net') for variant in config.get('variants', [])]) or MVP_POLICIES)}",
        f"- Factorization: {config['mf'].get('algorithm', 'mf')}",
        "",
        "## Final Summary",
        "",
        f"| alpha | {label_col} | policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate | observed_size |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["alpha", label_col]).itertuples(index=False):
        lines.append(
            f"| {row.alpha:.4f} | {getattr(row, label_col)} | {row.policy} | {row.cumulative_reward:.4f} | "
            f"{row.cumulative_reward_ci95:.4f} | {row.w_rmse:.4f} | {row.w_rmse_ci95:.4f} | "
            f"{row.response_rate:.4f} | {row.observed_size:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "movielens_100k_svdpp_robustness_10seeds.json"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)

    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    report_dir = ensure_dir(ROOT / "reports")

    dataset = load_dataset_from_config(config)
    mf_config = MFConfig(**config["mf"])
    base_acquisition_config = AcquisitionConfig(**config["acquisition"])
    variants = config.get("variants")
    policies = config.get("policies", MVP_POLICIES)
    policy_overrides = config.get("policy_overrides", {})
    all_metrics = []
    prefix = config.get("output_prefix", "movielens_100k_svdpp_robustness_10seeds")
    metrics_path = table_dir / f"{prefix}_metrics.csv"
    completed: set[tuple[float, str, int]] = set()
    if args.resume and metrics_path.exists():
        existing_metrics = pd.read_csv(metrics_path)
        all_metrics.append(existing_metrics)
        existing_label_col = "variant" if "variant" in existing_metrics.columns else "policy"
        completed = {
            (float(row.alpha), str(getattr(row, existing_label_col)), int(row.seed))
            for row in existing_metrics[["alpha", existing_label_col, "seed"]].drop_duplicates().itertuples(index=False)
        }
    wrote_metrics_header = args.resume and metrics_path.exists()

    for alpha in config["response_alphas"]:
        acquisition_config = replace(base_acquisition_config, response_alpha=float(alpha))
        for seed in config["seeds"]:
            split = split_dataset(dataset, seed=seed, **config["split"])
            run_items = variants or [{"label": policy, "policy": policy, "acquisition_overrides": policy_overrides.get(policy, {})} for policy in policies]
            for item in run_items:
                label = item["label"]
                policy = item.get("policy", "du_net")
                if (float(alpha), str(label), int(seed)) in completed:
                    continue
                overrides = item.get("acquisition_overrides", {})
                policy_acquisition_config = replace(acquisition_config, **overrides) if overrides else acquisition_config
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
                metrics.insert(0, "alpha", alpha)
                metrics.insert(0, "seed", seed)
                if variants:
                    metrics.insert(0, "variant", label)
                all_metrics.append(metrics)
                metrics.to_csv(metrics_path, mode="w" if not wrote_metrics_header else "a", header=not wrote_metrics_header, index=False)
                wrote_metrics_header = True

    metrics_df = pd.concat(all_metrics, ignore_index=True)

    final = metrics_df.sort_values("cycle").groupby(["alpha", "policy", "seed"], as_index=False).tail(1)
    if variants:
        final = metrics_df.sort_values("cycle").groupby(["alpha", "variant", "seed"], as_index=False).tail(1)
    summary = summarize_final(final)
    summary.to_csv(table_dir / f"{prefix}_final_summary.csv", index=False)
    labels = [variant["label"] for variant in variants] if variants else policies
    plot_alpha_metric(summary, labels, "w_rmse", figure_dir / f"{prefix}_w_rmse_by_alpha.png", "Final w-RMSE")
    plot_alpha_metric(
        summary,
        labels,
        "cumulative_reward",
        figure_dir / f"{prefix}_reward_by_alpha.png",
        "Final cumulative centered reward",
    )
    write_report(config, summary, report_dir / f"{prefix}_report.md")
    print(f"Wrote metrics: {metrics_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
