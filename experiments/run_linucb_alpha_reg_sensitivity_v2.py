from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

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


PREFIX = "linucb_alpha_reg_sensitivity_v2"
V2_CONFIG_PATH = ROOT / "configs" / "paper_exact_attempt_movielens_v2.json"
V2_METRICS_PATH = ROOT / "outputs" / "tables" / "paper_exact_attempt_movielens_v2_metrics.csv"
MAINLINE_VARIANT = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
POLICY = "linucb_mf_interaction"
DEFAULT_ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.8, 1.0)
DEFAULT_REGULARIZATIONS = (0.1, 1.0, 10.0)
METRIC_COLUMNS = [
    "pdf_literal_norm_rmse",
    "standard_mask_rmse",
    "reward_tail_proxy_rmse",
    "rmse",
    "w_rmse",
    "cumulative_reward",
    "centered_cumulative_reward",
]


def float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def variant_label(alpha: float, regularization: float) -> str:
    return f"linucb_mf_alpha_{float_token(alpha)}_reg_{float_token(regularization)}"


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.6g}" if pd.notna(value) else "")
        else:
            view[col] = view[col].fillna("").astype(str)

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(col) for col in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(clean(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def load_network_embedding(config: dict) -> tuple[object, np.ndarray | None]:
    dataset_cfg = config["dataset"]
    dataset = load_movielens_100k(
        ROOT / dataset_cfg["raw_dir"],
        network_mode=dataset_cfg.get("network_mode", "gcmc_features"),
    )
    embedding_mode = str(dataset_cfg.get("network_embedding_mode", "graph")).lower()
    if embedding_mode in {"user_features", "metadata_features", "monti_features", "gcmc_features"}:
        if dataset.user_features is None:
            raise ValueError(f"Dataset {dataset.name} does not expose user_features for {embedding_mode}.")
        return dataset, dataset.user_features
    return dataset, None


def read_completed(metrics_path: Path) -> tuple[pd.DataFrame | None, set[tuple[str, int]]]:
    if not metrics_path.exists():
        return None, set()
    existing = pd.read_csv(metrics_path)
    completed = {
        (str(row.variant), int(row.seed))
        for row in existing[["variant", "seed"]].drop_duplicates().itertuples(index=False)
    }
    return existing, completed


def load_mainline_seed_matched(seeds: list[int]) -> pd.Series:
    if not V2_METRICS_PATH.exists():
        return pd.Series(dtype=float)
    metrics = pd.read_csv(V2_METRICS_PATH)
    mainline = metrics[
        (metrics["variant"].astype(str) == MAINLINE_VARIANT)
        & (metrics["seed"].astype(int).isin([int(seed) for seed in seeds]))
    ].copy()
    if mainline.empty:
        return pd.Series(dtype=float)
    final = mainline.sort_values("cycle").groupby("seed", as_index=False).tail(1)
    return final[METRIC_COLUMNS].mean(numeric_only=True)


def add_mainline_gaps(summary: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    baseline = load_mainline_seed_matched(seeds)
    out = summary.copy()
    for metric in METRIC_COLUMNS:
        if metric not in out.columns or metric not in baseline.index:
            continue
        if metric in {"cumulative_reward", "centered_cumulative_reward"}:
            out[f"linear_minus_tau1_du_{metric}"] = out[metric].astype(float) - float(baseline[metric])
        else:
            out[f"linear_minus_tau1_du_{metric}"] = out[metric].astype(float) - float(baseline[metric])
    if "linear_minus_tau1_du_standard_mask_rmse" in out.columns:
        out["figure4a_standard_direction"] = np.where(
            out["linear_minus_tau1_du_standard_mask_rmse"] > 0,
            "du_lower_rmse",
            "linear_lower_or_tie_rmse",
        )
    if "linear_minus_tau1_du_reward_tail_proxy_rmse" in out.columns:
        out["figure4a_reward_tail_direction"] = np.where(
            out["linear_minus_tau1_du_reward_tail_proxy_rmse"] > 0,
            "du_lower_rmse",
            "linear_lower_or_tie_rmse",
        )
    if "linear_minus_tau1_du_cumulative_reward" in out.columns:
        out["figure4b_reward_direction"] = np.where(
            out["linear_minus_tau1_du_cumulative_reward"] > 0,
            "linear_higher_reward",
            "du_higher_or_tie_reward",
        )
    return out


def write_report(summary: pd.DataFrame, seeds: list[int], alphas: list[float], regularizations: list[float]) -> None:
    report_dir = ensure_dir(ROOT / "reports")
    report_path = report_dir / f"{PREFIX}.md"
    complete_points = int(summary[["ucb_alpha", "ucb_regularization"]].drop_duplicates().shape[0])
    expected_points = len(alphas) * len(regularizations)
    min_seed_count = int(summary["seed_count"].min()) if not summary.empty else 0

    best_reward_tail = summary.sort_values("reward_tail_proxy_rmse").head(5)
    best_reward = summary.sort_values("cumulative_reward", ascending=False).head(5)
    current = summary[
        np.isclose(summary["ucb_alpha"].astype(float), 0.8)
        & np.isclose(summary["ucb_regularization"].astype(float), 1.0)
    ]

    reward_tail_range = (
        float(summary["reward_tail_proxy_rmse"].min()),
        float(summary["reward_tail_proxy_rmse"].max()),
    )
    reward_range = (
        float(summary["cumulative_reward"].min()),
        float(summary["cumulative_reward"].max()),
    )
    standard_range = (
        float(summary["standard_mask_rmse"].min()),
        float(summary["standard_mask_rmse"].max()),
    )

    lines = [
        f"# {PREFIX}",
        "",
        "Purpose: close the local Linear UCB alpha/regularization sensitivity gate without promoting any proxy to a paper-exact comparator.",
        "",
        "## Protocol",
        "",
        f"- Source config: `{V2_CONFIG_PATH.relative_to(ROOT).as_posix()}`",
        f"- Policy: `{POLICY}`",
        f"- Seeds: `{seeds}`",
        f"- Alpha grid: `{alphas}`",
        f"- Regularization grid: `{regularizations}`",
        f"- Completed grid points: `{complete_points}/{expected_points}`",
        f"- Minimum seed count per point: `{min_seed_count}`",
        "",
        "## Sensitivity Ranges",
        "",
        f"- reward-tail proxy RMSE range: `{reward_tail_range[0]:.6g}` to `{reward_tail_range[1]:.6g}`",
        f"- standard-mask RMSE range: `{standard_range[0]:.6g}` to `{standard_range[1]:.6g}`",
        f"- cumulative reward range: `{reward_range[0]:.6g}` to `{reward_range[1]:.6g}`",
        "",
        "## Current Local Default",
        "",
        md_table(current) if not current.empty else "_Missing alpha=0.8/reg=1.0 row._",
        "",
        "## Lowest Reward-Tail Proxy RMSE",
        "",
        md_table(best_reward_tail),
        "",
        "## Highest Cumulative Reward",
        "",
        md_table(best_reward),
        "",
        "## Claim Boundary",
        "",
        "This grid is sensitivity evidence only. The paper does not expose the Linear UCB feature map, alpha, regularization, reward transform, or update protocol, so no row is promoted to `paper_linear_ucb_reconstructed`.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(metrics_df: pd.DataFrame, seeds: list[int], alphas: list[float], regularizations: list[float]) -> pd.DataFrame:
    final = metrics_df.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)
    summary = summarize_final(final)
    params = (
        final.groupby("variant", as_index=False)[["ucb_alpha", "ucb_regularization"]]
        .first()
        .astype({"ucb_alpha": float, "ucb_regularization": float})
    )
    summary = summary.merge(params, on="variant", how="left")
    summary["source"] = "local_clean_room_proxy"
    summary["feature_map"] = "mf_interaction"
    summary["paper_linear_ucb_reconstructed"] = "no"
    summary = add_mainline_gaps(summary, seeds)
    ordered = [
        "variant",
        "policy",
        "source",
        "feature_map",
        "ucb_alpha",
        "ucb_regularization",
        "seed_count",
        "paper_linear_ucb_reconstructed",
    ]
    remaining = [col for col in summary.columns if col not in ordered]
    summary = summary[ordered + remaining].sort_values(["ucb_regularization", "ucb_alpha"]).reset_index(drop=True)
    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    summary.to_csv(table_dir / f"{PREFIX}.csv", index=False)
    summary.to_csv(table_dir / f"{PREFIX}_final_summary.csv", index=False)
    write_report(summary, seeds, alphas, regularizations)
    return summary


def run_grid(
    config: dict,
    seeds: list[int],
    alphas: list[float],
    regularizations: list[float],
    resume: bool,
) -> pd.DataFrame:
    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    metrics_path = table_dir / f"{PREFIX}_metrics.csv"
    existing, completed = read_completed(metrics_path) if resume else (None, set())
    all_metrics: list[pd.DataFrame] = []
    if existing is not None:
        all_metrics.append(existing)
    wrote_header = bool(existing is not None and not existing.empty)

    dataset, network_embedding = load_network_embedding(config)
    split_cfg = config["split"]
    num_items = int(split_cfg.get("num_items", 1681))
    num_users = int(split_cfg.get("num_users", dataset.num_users))
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])

    for seed in seeds:
        split = make_u1_fallback_split(
            ROOT / config["dataset"]["raw_dir"],
            seed=int(seed),
            initial_size=int(split_cfg.get("initial_size", 1364)),
            num_items=num_items,
        )
        for regularization in regularizations:
            for alpha in alphas:
                label = variant_label(alpha, regularization)
                if (label, int(seed)) in completed:
                    continue
                acquisition = replace(
                    base_acquisition,
                    selection="topk",
                    ucb_alpha=float(alpha),
                    ucb_regularization=float(regularization),
                )
                metrics = run_policy(
                    policy=POLICY,
                    split=split,
                    adjacency=dataset.adjacency,
                    num_users=num_users,
                    num_items=num_items,
                    mf_config=mf_config,
                    acquisition_config=acquisition,
                    seed=int(seed),
                    network_embedding=network_embedding,
                )
                metrics.insert(0, "protocol", str(split_cfg.get("mode", "movielens_u1_fallback_table1_density")))
                metrics.insert(1, "variant", label)
                metrics.insert(2, "seed", int(seed))
                metrics.insert(3, "initial_size", len(split.initial))
                metrics.insert(4, "test_size", len(split.test))
                metrics.insert(5, "ucb_alpha", float(alpha))
                metrics.insert(6, "ucb_regularization", float(regularization))
                all_metrics.append(metrics)
                metrics.to_csv(metrics_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
                wrote_header = True
    if not all_metrics:
        raise RuntimeError("No metrics were produced and no resumable metrics table exists.")
    return pd.concat(all_metrics, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(V2_CONFIG_PATH))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    parser.add_argument("--regularizations", nargs="+", type=float, default=list(DEFAULT_REGULARIZATIONS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    seeds = [int(seed) for seed in args.seeds]
    alphas = [float(alpha) for alpha in args.alphas]
    regularizations = [float(reg) for reg in args.regularizations]
    metrics = run_grid(config, seeds, alphas, regularizations, resume=bool(args.resume))
    summary = summarize(metrics, seeds, alphas, regularizations)
    print(ROOT / "outputs" / "tables" / f"{PREFIX}_metrics.csv")
    print(ROOT / "outputs" / "tables" / f"{PREFIX}.csv")
    print(ROOT / "reports" / f"{PREFIX}.md")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
