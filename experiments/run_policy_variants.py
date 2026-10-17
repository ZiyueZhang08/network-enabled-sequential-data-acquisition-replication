from __future__ import annotations

import argparse
import json
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
from src.data_loader import load_dataset_from_config, split_dataset
from src.graph_embedding import user_embedding
from src.graph_transforms import transform_user_graph
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


def table_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def summarize_final(final: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for variant, group in final.groupby("variant", sort=False):
        all_rows = metrics[metrics["variant"] == variant]
        row: dict[str, float | str] = {
            "variant": variant,
            "policy": str(group["policy"].iloc[0]),
        }
        for col in [
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
            if col not in group.columns:
                continue
            values = group[col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[col] = np.nan
                row[f"{col}_ci95"] = np.nan
                continue
            row[col] = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        for col in [
            "selected_avg_reward_percentile",
            "selected_avg_entropy_percentile",
            "selected_avg_user_degree",
            "selected_cold_user_share",
            "avg_dls_shrinkage",
            "graph_edge_count",
            "graph_density",
        ]:
            if col not in all_rows.columns:
                continue
            values = all_rows[col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                row[col] = np.nan
                row[f"{col}_ci95"] = np.nan
                continue
            row[col] = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{col}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_top_variants(metrics: pd.DataFrame, summary: pd.DataFrame, prefix: str, figure_dir: Path) -> None:
    top = summary.sort_values("w_rmse").head(8)["variant"].tolist()
    for metric, ylabel in [("w_rmse", "w-RMSE"), ("cumulative_reward", "Cumulative centered reward")]:
        plt.figure(figsize=(9, 5))
        for variant in top:
            cur = metrics[metrics["variant"] == variant]
            grouped = cur.groupby("cycle")[metric]
            cycle_summary = grouped.agg(["mean", "std", "count"]).reset_index()
            ci95 = 1.96 * cycle_summary["std"].fillna(0.0) / np.sqrt(cycle_summary["count"].clip(lower=1))
            x = cycle_summary["cycle"].to_numpy(dtype=float)
            y = cycle_summary["mean"].to_numpy(dtype=float)
            ci = ci95.to_numpy(dtype=float)
            plt.plot(x, y, linewidth=2, label=variant)
            if cycle_summary["count"].max() > 1:
                plt.fill_between(x, y - ci, y + ci, alpha=0.10)
        plt.xlabel("Acquisition cycle")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_dir / f"{prefix}_{metric}.png", dpi=180)
        plt.close()


def write_report(config: dict, summary: pd.DataFrame, path: Path) -> None:
    lines = [
        f"# {config.get('output_prefix', 'policy_variants')} Policy-Variant Grid",
        "",
        f"Dataset: `{config['dataset']['name']}` from `{config['dataset']['raw_dir']}`.",
        "",
        "## Top Variants by w-RMSE",
        "",
        "| variant | policy | reward | w-RMSE | reward pct | entropy pct |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("w_rmse").itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.policy} | "
            f"{getattr(row, 'cumulative_reward', np.nan):.4f} | "
            f"{getattr(row, 'w_rmse', np.nan):.4f} | "
            f"{getattr(row, 'selected_avg_reward_percentile', np.nan):.4f} | "
            f"{getattr(row, 'selected_avg_entropy_percentile', np.nan):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)

    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    report_dir = ensure_dir(ROOT / "reports")

    prefix = config.get("output_prefix", "policy_variants")
    metrics_path = table_dir / f"{prefix}_metrics.csv"
    if args.summarize_only:
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        metrics_df = pd.read_csv(metrics_path)
    else:
        dataset = load_dataset_from_config(config)
        mf_config = MFConfig(**config["mf"])
        base_acquisition = AcquisitionConfig(**config["acquisition"])
        variants = config["variants"]
        override_keys = sorted(
            {
                key
                for variant in variants
                for key in variant.get("acquisition_overrides", {}).keys()
            }
        )

        all_metrics = []
        completed: set[tuple[str, int]] = set()
        if args.resume and metrics_path.exists():
            existing_metrics = pd.read_csv(metrics_path)
            all_metrics.append(existing_metrics)
            completed = {
                (str(row.variant), int(row.seed))
                for row in existing_metrics[["variant", "seed"]].drop_duplicates().itertuples(index=False)
            }
        wrote_metrics_header = args.resume and metrics_path.exists()
        for seed in config["seeds"]:
            split = split_dataset(dataset, seed=seed, **config["split"])
            embedding_cache: dict[tuple[str, float, str, int], np.ndarray] = {}
            for variant in variants:
                label = variant["label"]
                if (str(label), int(seed)) in completed:
                    continue
                policy = variant.get("policy", "du_net")
                overrides = variant.get("acquisition_overrides", {})
                acquisition = replace(base_acquisition, **overrides)
                graph_options = variant.get("graph_transform", {"mode": "none", "rate": 0.0})
                graph_mode = graph_options.get("mode", "none")
                graph_rate = float(graph_options.get("rate", 0.0))
                graph_seed = seed * 1009 + sum(ord(ch) for ch in label)
                graph = transform_user_graph(
                    dataset.adjacency,
                    mode=graph_mode,
                    rate=graph_rate,
                    seed=graph_seed,
                )
                embedding_key = (
                    graph_mode,
                    graph_rate,
                    graph_seed,
                    acquisition.graph_embedding_method,
                    mf_config.rank,
                )
                if embedding_key not in embedding_cache:
                    embedding_cache[embedding_key] = user_embedding(
                        graph,
                        dim=mf_config.rank,
                        method=acquisition.graph_embedding_method,
                        seed=seed,
                    )
                metrics = run_policy(
                    policy=policy,
                    split=split,
                    adjacency=graph,
                    num_users=dataset.num_users,
                    num_items=dataset.num_items,
                    mf_config=mf_config,
                    acquisition_config=acquisition,
                    seed=seed,
                    network_embedding=embedding_cache[embedding_key],
                )
                metrics.insert(0, "variant", label)
                metrics.insert(1, "seed", seed)
                metrics["graph_transform"] = graph_mode
                metrics["graph_rate"] = graph_rate
                metrics["graph_edge_count"] = int(np.triu(graph, k=1).sum())
                metrics["graph_density"] = float(np.triu(graph, k=1).sum() / max(graph.shape[0] * (graph.shape[0] - 1) / 2, 1))
                for key in override_keys:
                    metrics[key] = table_value(overrides.get(key, np.nan))
                all_metrics.append(metrics)
                metrics.to_csv(metrics_path, mode="w" if not wrote_metrics_header else "a", header=not wrote_metrics_header, index=False)
                wrote_metrics_header = True

        metrics_df = pd.concat(all_metrics, ignore_index=True)
    final = metrics_df.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)
    summary = summarize_final(final, metrics_df)
    summary.to_csv(table_dir / f"{prefix}_final_summary.csv", index=False)
    plot_top_variants(metrics_df, summary, prefix, figure_dir)
    write_report(config, summary, report_dir / f"{prefix}_report.md")
    print(f"Wrote metrics: {metrics_path}")
    print(summary.sort_values("w_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
