"""Run a small MovieLens backend-fidelity screen for Figure 4a."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, run_policy
from src.data_loader import RatingDataset, load_dataset_from_config, split_dataset
from src.graph_embedding import user_embedding
from src.matrix_factorization import MFConfig
from src.utils import load_json


TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

CONFIG_PATH = ROOT / "configs" / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds.json"
OUT_METRICS = TABLE_DIR / "figure4a_backend_fidelity_screen_metrics.csv"
OUT_SUMMARY = TABLE_DIR / "figure4a_backend_fidelity_screen.csv"
OUT_GAPS = TABLE_DIR / "figure4a_backend_fidelity_screen_paired_gaps.csv"
OUT_REPORT = REPORT_DIR / "figure4a_backend_fidelity_screen.md"

SEEDS = [0, 1, 2]
POLICIES = [
    {"label": "du_net", "policy": "du_net", "acquisition_overrides": {}},
    {"label": "mf_linear_ucb_a002", "policy": "mf_linear_ucb", "acquisition_overrides": {"ucb_alpha": 0.02}},
]


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def drop_sparsest_item(dataset: RatingDataset) -> tuple[RatingDataset, int, int]:
    counts = dataset.ratings.groupby("item_id").size().sort_values(kind="mergesort")
    dropped_item = int(counts.index[0])
    dropped_count = int(counts.iloc[0])
    ratings = dataset.ratings[dataset.ratings["item_id"] != dropped_item].copy()
    remaining_items = sorted(int(item) for item in ratings["item_id"].unique())
    item_map = {old: new for new, old in enumerate(remaining_items)}
    ratings["item_id"] = ratings["item_id"].map(item_map).astype(int)
    return (
        RatingDataset(
            ratings=ratings.reset_index(drop=True),
            adjacency=dataset.adjacency.copy(),
            num_users=dataset.num_users,
            num_items=len(remaining_items),
            name=f"{dataset.name}-drop-sparsest-item",
        ),
        dropped_item,
        dropped_count,
    )


def scenario_specs(base_mf: dict[str, object]) -> list[dict[str, object]]:
    stronger = dict(base_mf)
    stronger.update({"rank": 16, "epochs": 8, "learning_rate": 0.02, "regularization": 0.06})
    return [
        {
            "scenario": "current_rank12_ep6",
            "item_filter": "none",
            "mf": dict(base_mf),
            "diagnosis": "Current Figure 4a local backend.",
        },
        {
            "scenario": "item1681_drop_sparsest",
            "item_filter": "drop_sparsest",
            "mf": dict(base_mf),
            "diagnosis": "Forces MovieLens item count to the paper Table 1 value.",
        },
        {
            "scenario": "stronger_rank16_ep8",
            "item_filter": "none",
            "mf": stronger,
            "diagnosis": "Small stronger-SVD++ probe without changing preprocessing.",
        },
        {
            "scenario": "item1681_stronger_rank16_ep8",
            "item_filter": "drop_sparsest",
            "mf": stronger,
            "diagnosis": "Combines paper item count with the stronger-SVD++ probe.",
        },
    ]


def dataset_for_scenario(base: RatingDataset, scenario: dict[str, object]) -> tuple[RatingDataset, int | str, int | str]:
    if scenario["item_filter"] == "drop_sparsest":
        return drop_sparsest_item(base)
    return base, "", ""


def summarize(metrics: pd.DataFrame) -> list[dict[str, object]]:
    final = metrics.sort_values("cycle").groupby(["scenario", "variant", "seed"], as_index=False).tail(1)
    rows: list[dict[str, object]] = []
    metric_cols = ["cumulative_reward", "rmse", "w_rmse", "sigmoid_reward_w_rmse", "observed_size"]
    for (scenario, variant), group in final.groupby(["scenario", "variant"], sort=False):
        first = group.iloc[0]
        row: dict[str, object] = {
            "scenario": scenario,
            "variant": variant,
            "policy": first["policy"],
            "seed_count": int(group["seed"].nunique()),
            "num_users": int(first["num_users"]),
            "num_items": int(first["num_items"]),
            "dropped_item_id": first["dropped_item_id"],
            "dropped_item_count": first["dropped_item_count"],
            "mf_rank": int(first["mf_rank"]),
            "mf_epochs": int(first["mf_epochs"]),
            "mf_learning_rate": float(first["mf_learning_rate"]),
            "mf_regularization": float(first["mf_regularization"]),
        }
        for col in metric_cols:
            values = group[col].to_numpy(dtype=float)
            row[col] = float(np.mean(values))
            row[f"{col}_ci95"] = ci95([float(value) for value in values])
        rows.append(row)
    return rows


def paired_gaps(metrics: pd.DataFrame) -> list[dict[str, object]]:
    final = metrics.sort_values("cycle").groupby(["scenario", "variant", "seed"], as_index=False).tail(1)
    rows: list[dict[str, object]] = []
    for (scenario, seed), group in final.groupby(["scenario", "seed"], sort=False):
        by_variant = {str(row.variant): row for row in group.itertuples(index=False)}
        if "du_net" not in by_variant or "mf_linear_ucb_a002" not in by_variant:
            continue
        du = by_variant["du_net"]
        linear = by_variant["mf_linear_ucb_a002"]
        rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "linear_minus_du_rmse": float(linear.rmse - du.rmse),
                "linear_minus_du_w_rmse": float(linear.w_rmse - du.w_rmse),
                "linear_minus_du_sigmoid_reward_w_rmse": float(
                    linear.sigmoid_reward_w_rmse - du.sigmoid_reward_w_rmse
                ),
                "du_minus_linear_reward": float(du.cumulative_reward - linear.cumulative_reward),
            }
        )
    return rows


def gap_summary(gaps: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame = pd.DataFrame(gaps)
    for scenario, group in frame.groupby("scenario", sort=False):
        row: dict[str, object] = {"scenario": scenario, "seed_count": int(group["seed"].nunique())}
        for col in ["linear_minus_du_rmse", "linear_minus_du_w_rmse", "linear_minus_du_sigmoid_reward_w_rmse"]:
            values = group[col].to_numpy(dtype=float)
            row[col] = float(np.mean(values))
            row[f"{col}_ci95"] = ci95([float(value) for value in values])
            row[f"{col}_du_wins"] = int(np.sum(values > 0))
        values = group["du_minus_linear_reward"].to_numpy(dtype=float)
        row["du_minus_linear_reward"] = float(np.mean(values))
        row["du_minus_linear_reward_ci95"] = ci95([float(value) for value in values])
        row["du_reward_wins"] = int(np.sum(values > 0))
        rows.append(row)
    return rows


def run_screen(resume: bool) -> pd.DataFrame:
    config = load_json(CONFIG_PATH)
    base_dataset = load_dataset_from_config(config)
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    scenarios = scenario_specs(config["mf"])
    completed: set[tuple[str, str, int]] = set()
    all_metrics: list[pd.DataFrame] = []
    if resume and OUT_METRICS.exists():
        existing = pd.read_csv(OUT_METRICS)
        all_metrics.append(existing)
        completed = {
            (str(row.scenario), str(row.variant), int(row.seed))
            for row in existing[["scenario", "variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
    wrote_header = resume and OUT_METRICS.exists()
    for scenario in scenarios:
        dataset, dropped_item, dropped_count = dataset_for_scenario(base_dataset, scenario)
        mf_config = MFConfig(**scenario["mf"])
        for seed in SEEDS:
            split = split_dataset(dataset, seed=seed, **config["split"])
            embedding = user_embedding(
                dataset.adjacency,
                dim=mf_config.rank,
                method=base_acquisition.graph_embedding_method,
                seed=seed,
            )
            for variant in POLICIES:
                label = str(variant["label"])
                key = (str(scenario["scenario"]), label, int(seed))
                if key in completed:
                    continue
                acquisition = replace(base_acquisition, **variant["acquisition_overrides"])
                metrics = run_policy(
                    policy=str(variant["policy"]),
                    split=split,
                    adjacency=dataset.adjacency,
                    num_users=dataset.num_users,
                    num_items=dataset.num_items,
                    mf_config=mf_config,
                    acquisition_config=acquisition,
                    seed=seed,
                    network_embedding=embedding,
                )
                metrics.insert(0, "scenario", scenario["scenario"])
                metrics.insert(1, "variant", label)
                metrics.insert(2, "seed", seed)
                metrics["num_users"] = dataset.num_users
                metrics["num_items"] = dataset.num_items
                metrics["dropped_item_id"] = dropped_item
                metrics["dropped_item_count"] = dropped_count
                metrics["mf_rank"] = mf_config.rank
                metrics["mf_epochs"] = mf_config.epochs
                metrics["mf_learning_rate"] = mf_config.learning_rate
                metrics["mf_regularization"] = mf_config.regularization
                all_metrics.append(metrics)
                metrics.to_csv(OUT_METRICS, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)
                wrote_header = True
                print(f"Finished {scenario['scenario']} / {label} / seed {seed}")
    return pd.concat(all_metrics, ignore_index=True)


def summary_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["scenario"]), str(row["variant"])): row for row in rows}


def write_report(summary_rows: list[dict[str, object]], gap_rows: list[dict[str, object]]) -> None:
    by_key = summary_lookup(summary_rows)
    lines = [
        "# Figure 4a Backend-Fidelity Screen",
        "",
        "Generated by `experiments/build_figure4a_backend_fidelity_screen.py`.",
        "",
        "This screen tests whether small paper-plausible backend changes move the MovieLens Figure 4a ordering.",
        "Each row uses seeds 0-2, 20 acquisition cycles, and compares DU-Net with MF-centered Linear UCB.",
        "",
        "Gap convention: positive RMSE gaps mean Linear UCB is higher error than DU-Net, so positive favors DU-Net.",
        "",
        "## Scenario Summary",
        "",
        "| scenario | items | MF | DU RMSE | Linear RMSE | RMSE gap | DU sigmoid | Linear sigmoid | sigmoid gap | DU local w | Linear local w | local gap | reward gap |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for gap in gap_rows:
        scenario = str(gap["scenario"])
        du = by_key[(scenario, "du_net")]
        linear = by_key[(scenario, "mf_linear_ucb_a002")]
        mf = f"r{du['mf_rank']}/e{du['mf_epochs']}/reg{fmt(du['mf_regularization'], 2)}"
        lines.append(
            f"| {scenario} | {du['num_items']} | {mf} | "
            f"{fmt(du['rmse'])} | {fmt(linear['rmse'])} | {fmt(gap['linear_minus_du_rmse'])} | "
            f"{fmt(du['sigmoid_reward_w_rmse'])} | {fmt(linear['sigmoid_reward_w_rmse'])} | "
            f"{fmt(gap['linear_minus_du_sigmoid_reward_w_rmse'])} | "
            f"{fmt(du['w_rmse'])} | {fmt(linear['w_rmse'])} | {fmt(gap['linear_minus_du_w_rmse'])} | "
            f"{fmt(gap['du_minus_linear_reward'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )
    best_sigmoid = max(gap_rows, key=lambda row: float(row["linear_minus_du_sigmoid_reward_w_rmse"]))
    best_rmse = max(gap_rows, key=lambda row: float(row["linear_minus_du_rmse"]))
    if float(best_sigmoid["linear_minus_du_sigmoid_reward_w_rmse"]) > 0:
        lines.append(
            f"- Best sigmoid-weighted ordering is `{best_sigmoid['scenario']}` with gap "
            f"{fmt(best_sigmoid['linear_minus_du_sigmoid_reward_w_rmse'])}; this favors DU-Net on the paper-like reward weighting."
        )
    else:
        lines.append(
            f"- No tested scenario restores DU-Net over Linear UCB on sigmoid-weighted RMSE. "
            f"The least-negative gap is `{best_sigmoid['scenario']}` at {fmt(best_sigmoid['linear_minus_du_sigmoid_reward_w_rmse'])}."
        )
    if float(best_rmse["linear_minus_du_rmse"]) > 0:
        lines.append(
            f"- Best conventional-RMSE ordering is `{best_rmse['scenario']}` with gap "
            f"{fmt(best_rmse['linear_minus_du_rmse'])}; this favors DU-Net."
        )
    else:
        lines.append(
            f"- No tested scenario restores DU-Net over Linear UCB on conventional RMSE. "
            f"The least-negative gap is `{best_rmse['scenario']}` at {fmt(best_rmse['linear_minus_du_rmse'])}."
        )
    lines.extend(
        [
            "- Dropping one sparse item changes the item count but is not, by itself, a credible explanation of the Figure 4a gap.",
            "- The next useful branch should focus on the acquisition/evaluation mechanism rather than another broad SVD++ hyperparameter sweep.",
            "",
            "CSV artifacts:",
            "",
            f"- `{OUT_SUMMARY.relative_to(ROOT)}`",
            f"- `{OUT_GAPS.relative_to(ROOT)}`",
            f"- `{OUT_METRICS.relative_to(ROOT)}`",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        metrics = pd.read_csv(OUT_METRICS)
    else:
        metrics = run_screen(resume=args.resume)
    summary_rows = summarize(metrics)
    seed_gap_rows = paired_gaps(metrics)
    gap_rows = gap_summary(seed_gap_rows)
    write_csv(summary_rows, OUT_SUMMARY)
    write_csv(seed_gap_rows, OUT_GAPS)
    write_report(summary_rows, gap_rows)
    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_GAPS}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
