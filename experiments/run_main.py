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
from src.baselines import MVP_POLICIES
from src.data_loader import generate_synthetic_dataset, split_ratings
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


def _plot_metric(df: pd.DataFrame, metric: str, out_path: Path, ylabel: str) -> None:
    plt.figure(figsize=(8, 5))
    summary = df.groupby(["policy", "cycle"], as_index=False)[metric].agg(["mean", "std"]).reset_index()
    for policy in MVP_POLICIES:
        cur = summary[summary["policy"] == policy]
        x = cur["cycle"].to_numpy()
        y = cur["mean"].to_numpy()
        sd = cur["std"].fillna(0).to_numpy()
        plt.plot(x, y, marker="o", linewidth=2, label=policy)
        if len(df["seed"].unique()) > 1:
            plt.fill_between(x, y - sd, y + sd, alpha=0.12)
    plt.xlabel("Acquisition cycle")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_report(config: dict, metrics: pd.DataFrame, report_path: Path) -> None:
    final = metrics.sort_values("cycle").groupby(["policy", "seed"], as_index=False).tail(1)
    summary = (
        final.groupby("policy", as_index=False)
        .agg(
            cumulative_reward=("cumulative_reward", "mean"),
            rmse=("rmse", "mean"),
            w_rmse=("w_rmse", "mean"),
            observed_size=("observed_size", "mean"),
        )
        .sort_values("policy")
    )
    best_reward = summary.sort_values("cumulative_reward", ascending=False).iloc[0]
    best_wrmse = summary.sort_values("w_rmse").iloc[0]
    table_lines = ["| policy | cumulative_reward | rmse | w_rmse | observed_size |", "|---|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        table_lines.append(
            f"| {row.policy} | {row.cumulative_reward:.4f} | {row.rmse:.4f} | {row.w_rmse:.4f} | {row.observed_size:.0f} |"
        )

    lines = [
        "# MVP Reproduction Report",
        "",
        "This is a Level 1 reproduction run using a synthetic low-rank rating matrix with a homophilous user network.",
        "It validates the end-to-end acquisition loop before replacing the synthetic loader with MovieLens/Flixster/Douban.",
        "",
        "## Configuration",
        "",
        f"- Seeds: {config['seeds']}",
        f"- Cycles: {config['acquisition']['cycles']}",
        f"- Batch size: {config['acquisition']['batch_size']}",
        f"- Policies: {', '.join(MVP_POLICIES)}",
        "",
        "## Final-Cycle Summary",
        "",
        "\n".join(table_lines),
        "",
        "## Quick Read",
        "",
        f"- Highest final cumulative reward: `{best_reward['policy']}` ({best_reward['cumulative_reward']:.4f}).",
        f"- Lowest final w-RMSE: `{best_wrmse['policy']}` ({best_wrmse['w_rmse']:.4f}).",
        "- Figures are saved under `outputs/figures/` and tables under `outputs/tables/`.",
        "",
        "## Known MVP Gaps",
        "",
        "- Uses synthetic data because no raw MovieLens/Flixster/Douban files are present in the workspace.",
        "- Uses spectral graph embedding as a dependency-free proxy for Node2Vec/ELPH.",
        "- Uses coverage/network uncertainty as an entropy proxy rather than the full observational entropy derivation.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "mvp_synthetic.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    output_dir = ensure_dir(ROOT / "outputs")
    figure_dir = ensure_dir(output_dir / "figures")
    table_dir = ensure_dir(output_dir / "tables")
    report_dir = ensure_dir(ROOT / "reports")

    mf_config = MFConfig(**config["mf"])
    acquisition_config = AcquisitionConfig(**config["acquisition"])

    all_metrics = []
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
            metrics.insert(0, "seed", seed)
            all_metrics.append(metrics)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    metrics_path = table_dir / "mvp_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    final_summary = (
        metrics_df.sort_values("cycle")
        .groupby(["policy", "seed"], as_index=False)
        .tail(1)
        .groupby("policy", as_index=False)
        .agg(cumulative_reward=("cumulative_reward", "mean"), rmse=("rmse", "mean"), w_rmse=("w_rmse", "mean"))
    )
    final_summary.to_csv(table_dir / "mvp_final_summary.csv", index=False)

    _plot_metric(metrics_df, "w_rmse", figure_dir / "mvp_w_rmse.png", "w-RMSE")
    _plot_metric(metrics_df, "rmse", figure_dir / "mvp_rmse.png", "RMSE")
    _plot_metric(metrics_df, "cumulative_reward", figure_dir / "mvp_cumulative_reward.png", "Cumulative centered reward")
    _write_report(config, metrics_df, report_dir / "reproduction_report.md")

    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote report: {report_dir / 'reproduction_report.md'}")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
