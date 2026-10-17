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
from src.data_loader import load_dataset_from_config, split_dataset
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "movielens_100k_lambda_grid.json"))
    args = parser.parse_args()
    config = load_json(args.config)

    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    report_dir = ensure_dir(ROOT / "reports")

    dataset = load_dataset_from_config(config)
    mf_config = MFConfig(**config["mf"])
    base_acq = AcquisitionConfig(**config["acquisition"])
    all_metrics = []
    grid_param = config.get("grid_param", "lambda_entropy")
    grid_values = config.get("grid_values", config.get("lambda_grid", []))
    grid_target = config.get("grid_target", "acquisition")
    for seed in config["seeds"]:
        split = split_dataset(dataset, seed=seed, **config["split"])
        for value in grid_values:
            mf = mf_config
            acq = base_acq
            if grid_target == "mf":
                current = getattr(mf_config, grid_param)
                cast_value = int(value) if isinstance(current, int) and not isinstance(current, bool) else value
                mf = replace(mf_config, **{grid_param: cast_value})
            else:
                current = getattr(base_acq, grid_param)
                if isinstance(current, bool):
                    cast_value = bool(value)
                elif isinstance(current, int) and not isinstance(current, bool):
                    cast_value = int(value)
                elif isinstance(current, float):
                    cast_value = float(value)
                else:
                    cast_value = str(value)
                acq = replace(base_acq, **{grid_param: cast_value})
            metrics = run_policy(
                policy=config.get("policy", "du_net"),
                split=split,
                adjacency=dataset.adjacency,
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                mf_config=mf,
                acquisition_config=acq,
                seed=seed,
            )
            if grid_param in metrics.columns:
                metrics[grid_param] = value
            else:
                metrics.insert(0, grid_param, value)
            metrics.insert(0, "seed", seed)
            all_metrics.append(metrics)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    prefix = config.get("output_prefix", "movielens_100k_lambda_grid")
    metrics_df.to_csv(table_dir / f"{prefix}_metrics.csv", index=False)
    final = metrics_df.sort_values("cycle").groupby([grid_param, "seed"], as_index=False).tail(1)
    summary = final.groupby(grid_param, as_index=False).agg(
        cumulative_reward=("cumulative_reward", "mean"),
        rmse=("rmse", "mean"),
        w_rmse=("w_rmse", "mean"),
    )
    summary.to_csv(table_dir / f"{prefix}_final_summary.csv", index=False)

    x_values = summary[grid_param]
    string_axis = x_values.dtype == object
    x_plot = list(range(len(summary))) if string_axis else x_values
    plt.figure(figsize=(7, 4.5))
    plt.plot(x_plot, summary["w_rmse"], marker="o", linewidth=2)
    if string_axis:
        plt.xticks(x_plot, x_values.astype(str), rotation=25, ha="right")
    plt.xlabel(grid_param)
    plt.ylabel("Final w-RMSE")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / f"{prefix}_w_rmse.png", dpi=180)
    plt.close()

    lines = [
        f"# {dataset.name} {grid_param} Grid",
        "",
        f"| {grid_param} | cumulative_reward | rmse | w_rmse |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("w_rmse").itertuples(index=False):
        value = getattr(row, grid_param)
        value_text = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
        lines.append(f"| {value_text} | {row.cumulative_reward:.4f} | {row.rmse:.4f} | {row.w_rmse:.4f} |")
    report_path = report_dir / f"{prefix}_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.sort_values("w_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
