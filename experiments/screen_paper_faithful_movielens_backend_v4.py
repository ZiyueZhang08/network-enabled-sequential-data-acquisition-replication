"""Screen paper-faithful MovieLens backends under the v4 acquisition protocol."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_paper_faithful_movielens import make_u1_fallback_split, summarize_final
from src.acquisition import AcquisitionConfig, run_policy
from src.data_loader import load_movielens_100k
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
DEFAULT_CONFIG = ROOT / "configs" / "paper_faithful_movielens_v1.json"

SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "scenario": "v4_rank12_ep6_lr0025_reg004",
        "mf": {"rank": 12, "epochs": 6, "learning_rate": 0.025, "regularization": 0.04, "algorithm": "svdpp"},
        "rationale": "Current v4 backend; included as the matched baseline.",
    },
    {
        "scenario": "rank20_ep12_lr002_reg004",
        "mf": {"rank": 20, "epochs": 12, "learning_rate": 0.02, "regularization": 0.04, "algorithm": "svdpp"},
        "rationale": "Moderately stronger SVD++ with conservative learning rate.",
    },
    {
        "scenario": "rank20_ep12_lr0015_reg006",
        "mf": {"rank": 20, "epochs": 12, "learning_rate": 0.015, "regularization": 0.06, "algorithm": "svdpp"},
        "rationale": "More regularized stronger SVD++ to avoid overfitting sparse early cycles.",
    },
    {
        "scenario": "rank30_ep12_lr0015_reg006",
        "mf": {"rank": 30, "epochs": 12, "learning_rate": 0.015, "regularization": 0.06, "algorithm": "svdpp"},
        "rationale": "Higher-rank SVD++ with the same regularized training budget.",
    },
    {
        "scenario": "rank20_ep20_lr0015_reg006",
        "mf": {"rank": 20, "epochs": 20, "learning_rate": 0.015, "regularization": 0.06, "algorithm": "svdpp"},
        "rationale": "Longer training branch from the same regularized setting.",
    },
)


def ci95(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(1.96 * statistics.stdev([float(value) for value in values]) / math.sqrt(len(values)))


def scenario_by_name(names: set[str] | None) -> list[dict[str, Any]]:
    if not names:
        return [dict(scenario) for scenario in SCENARIOS]
    selected = [dict(scenario) for scenario in SCENARIOS if str(scenario["scenario"]) in names]
    missing = sorted(names - {str(scenario["scenario"]) for scenario in selected})
    if missing:
        raise ValueError(f"Unknown scenarios: {missing}")
    return selected


def comparator_gap_summary(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = ["pdf_literal_norm_rmse", "standard_mask_rmse", "cumulative_reward", "reward_tail_proxy_rmse", "w_rmse"]
    comparators = ["linucb_concat", "linucb_mf_interaction"]
    for scenario, group in final.groupby("scenario", sort=False):
        for comparator in comparators:
            seed_rows: list[dict[str, float]] = []
            for seed, seed_group in group.groupby("seed"):
                by_variant = {str(row.variant): row for row in seed_group.itertuples(index=False)}
                if "paper_faithful_dunet" not in by_variant or comparator not in by_variant:
                    continue
                du = by_variant["paper_faithful_dunet"]
                comp = by_variant[comparator]
                seed_rows.append(
                    {
                        "seed": float(seed),
                        "du_minus_comp_pdf_literal_norm_rmse": float(
                            du.pdf_literal_norm_rmse - comp.pdf_literal_norm_rmse
                        ),
                        "du_minus_comp_standard_mask_rmse": float(du.standard_mask_rmse - comp.standard_mask_rmse),
                        "du_minus_comp_cumulative_reward": float(du.cumulative_reward - comp.cumulative_reward),
                        "du_minus_comp_reward_tail_proxy_rmse": float(
                            du.reward_tail_proxy_rmse - comp.reward_tail_proxy_rmse
                        ),
                        "du_minus_comp_w_rmse": float(du.w_rmse - comp.w_rmse),
                    }
                )
            if not seed_rows:
                continue
            seed_frame = pd.DataFrame(seed_rows)
            row: dict[str, Any] = {
                "scenario": scenario,
                "comparator": comparator,
                "seed_count": int(seed_frame["seed"].nunique()),
            }
            for col in [c for c in seed_frame.columns if c != "seed"]:
                values = seed_frame[col].to_numpy(dtype=float)
                row[col] = float(values.mean())
                row[f"{col}_ci95"] = ci95(values)
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, gaps: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Paper-faithful MovieLens v4 backend screen",
        "",
        "This screen changes only the SVD++ backend while keeping the v4 paper-faithful acquisition protocol:",
        "user-feature full DLS, Eq. 3.6-style Schur entropy, full-pool exponential sampling, and MovieLens `p=0`.",
        "The same backend is used for DU-Net and all Linear UCB comparators.",
        "",
        "## DU-Net summary",
        "",
        "| scenario | seeds | rank | epochs | lr | reg | PDF RMSE | standard RMSE | reward | tail RMSE | local w-RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    du_rows = summary[summary["variant"] == "paper_faithful_dunet"].copy()
    for row in du_rows.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {int(row.seed_count)} | {int(row.mf_rank)} | {int(row.mf_epochs)} | "
            f"{float(row.mf_learning_rate):.4f} | {float(row.mf_regularization):.4f} | "
            f"{float(row.pdf_literal_norm_rmse):.6f} | {float(row.standard_mask_rmse):.6f} | "
            f"{float(row.cumulative_reward):.3f} | {float(row.reward_tail_proxy_rmse):.6f} | "
            f"{float(row.w_rmse):.6f} |"
        )
    lines.extend(
        [
            "",
            "## DU-Net minus comparator gaps",
            "",
            "Negative RMSE gaps favor DU-Net; positive reward gaps favor DU-Net.",
            "",
            "| scenario | comparator | standard RMSE gap | PDF RMSE gap | reward gap | local w-RMSE gap |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in gaps.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.comparator} | "
            f"{float(row.du_minus_comp_standard_mask_rmse):.6f} | "
            f"{float(row.du_minus_comp_pdf_literal_norm_rmse):.6f} | "
            f"{float(row.du_minus_comp_cumulative_reward):.3f} | "
            f"{float(row.du_minus_comp_w_rmse):.6f} |"
        )
    if not du_rows.empty:
        best = du_rows.sort_values("standard_mask_rmse", kind="mergesort").iloc[0]
        lines.extend(
            [
                "",
                "## Interim interpretation",
                "",
                f"- Best DU-Net standard RMSE in this screen: `{best['scenario']}` at "
                f"{float(best['standard_mask_rmse']):.6f}.",
                "- Backend changes are promotion candidates only if the same-backend comparator gaps also stay consistent with the paper direction.",
                "- If stronger backends improve Linear UCB as much as DU-Net, the remaining Figure 4a gap is more likely exact split/source/backend-detail uncertainty than a tunable DU-Net mechanism.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_screen(
    config_path: Path,
    seeds: list[int],
    scenario_names: set[str] | None,
    variant_names: set[str] | None,
    selection_override: str | None,
    prefix: str,
    resume: bool,
) -> None:
    config = load_json(config_path)
    table_dir = ensure_dir(TABLE_DIR)
    report_dir = ensure_dir(REPORT_DIR)
    metrics_path = table_dir / f"{prefix}_metrics.csv"
    summary_path = table_dir / f"{prefix}_final_summary.csv"
    gaps_path = table_dir / f"{prefix}_gaps.csv"
    report_path = report_dir / f"{prefix}.md"

    dataset_cfg = config["dataset"]
    dataset = load_movielens_100k(
        ROOT / dataset_cfg["raw_dir"],
        network_mode=dataset_cfg.get("network_mode", "gcmc_features"),
    )
    embedding_mode = str(dataset_cfg.get("network_embedding_mode", "graph")).lower()
    network_embedding = None
    if embedding_mode in {"user_features", "metadata_features", "monti_features", "gcmc_features"}:
        if dataset.user_features is None:
            raise ValueError(f"Dataset {dataset.name} does not expose user_features.")
        network_embedding = dataset.user_features

    split_cfg = config["split"]
    num_items = int(split_cfg.get("num_items", 1681))
    num_users = int(split_cfg.get("num_users", dataset.num_users))
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    variants = config["variants"]
    if variant_names:
        variants = [variant for variant in variants if str(variant["label"]) in variant_names]
        missing = sorted(variant_names - {str(variant["label"]) for variant in variants})
        if missing:
            raise ValueError(f"Unknown variants: {missing}")
    scenarios = scenario_by_name(scenario_names)

    completed: set[tuple[str, str, int]] = set()
    all_metrics: list[pd.DataFrame] = []
    wrote_header = False
    if resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        all_metrics.append(existing)
        completed = {
            (str(row.scenario), str(row.variant), int(row.seed))
            for row in existing[["scenario", "variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True

    for scenario in scenarios:
        mf_config = MFConfig(**scenario["mf"])
        for seed in seeds:
            split = make_u1_fallback_split(
                ROOT / dataset_cfg["raw_dir"],
                seed=int(seed),
                initial_size=int(split_cfg.get("initial_size", 1364)),
                num_items=num_items,
            )
            for variant in variants:
                label = str(variant["label"])
                key = (str(scenario["scenario"]), label, int(seed))
                if key in completed:
                    continue
                acquisition = base_acquisition
                if variant.get("acquisition_overrides"):
                    acquisition = replace(base_acquisition, **variant["acquisition_overrides"])
                if selection_override:
                    acquisition = replace(acquisition, selection=selection_override)
                metrics = run_policy(
                    policy=str(variant["policy"]),
                    split=split,
                    adjacency=dataset.adjacency,
                    num_users=num_users,
                    num_items=num_items,
                    mf_config=mf_config,
                    acquisition_config=acquisition,
                    seed=int(seed),
                    network_embedding=network_embedding,
                )
                metrics.insert(0, "scenario", str(scenario["scenario"]))
                metrics.insert(1, "scenario_rationale", str(scenario["rationale"]))
                metrics.insert(2, "protocol", str(split_cfg.get("mode", "movielens_u1_fallback_table1_density")))
                metrics.insert(3, "variant", label)
                metrics.insert(4, "seed", int(seed))
                metrics.insert(5, "initial_size", len(split.initial))
                metrics.insert(6, "test_size", len(split.test))
                metrics["mf_rank"] = mf_config.rank
                metrics["mf_epochs"] = mf_config.epochs
                metrics["mf_learning_rate"] = mf_config.learning_rate
                metrics["mf_regularization"] = mf_config.regularization
                all_metrics.append(metrics)
                metrics.to_csv(metrics_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
                wrote_header = True
                print(f"Finished {scenario['scenario']} / {label} / seed {seed}", flush=True)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    final = metrics_df.sort_values("cycle").groupby(["scenario", "variant", "seed"], as_index=False).tail(1)
    summary_frames: list[pd.DataFrame] = []
    for scenario, group in final.groupby("scenario", sort=False):
        summary = summarize_final(group)
        summary.insert(0, "scenario", scenario)
        first = group.iloc[0]
        summary["mf_rank"] = int(first["mf_rank"])
        summary["mf_epochs"] = int(first["mf_epochs"])
        summary["mf_learning_rate"] = float(first["mf_learning_rate"])
        summary["mf_regularization"] = float(first["mf_regularization"])
        summary_frames.append(summary)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    gaps_df = comparator_gap_summary(final)
    summary_df.to_csv(summary_path, index=False)
    gaps_df.to_csv(gaps_path, index=False)
    write_report(summary_df, gaps_df, report_path)
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote gaps: {gaps_path}")
    print(f"Wrote report: {report_path}")
    print(summary_df[summary_df["variant"] == "paper_faithful_dunet"].to_string(index=False))
    print(gaps_df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--scenario", action="append", help="Scenario name to run; may be repeated.")
    parser.add_argument("--variant", action="append", help="Variant label to run; may be repeated.")
    parser.add_argument("--override-selection", choices=["topk", "exp"], help="Force selection mode for selected variants.")
    parser.add_argument("--output-prefix", default="paper_faithful_movielens_backend_v4_screen_3seeds")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_screen(
        config_path=Path(args.config),
        seeds=[int(seed) for seed in args.seeds],
        scenario_names=set(args.scenario) if args.scenario else None,
        variant_names=set(args.variant) if args.variant else None,
        selection_override=args.override_selection,
        prefix=str(args.output_prefix),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
