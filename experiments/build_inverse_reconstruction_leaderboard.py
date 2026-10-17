from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import RATING_COLUMNS


PREFIX = "inverse_reconstruction_v1"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
MASK_DIR = ROOT / "outputs" / "masks"

TARGET_PATH = (
    TABLE_DIR
    / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
    / "tau1_figure4_digitized_targets.csv"
)
TARGET_ENDPOINT_PATH = TABLE_DIR / "figure4_digitization_precision_v2_targets.csv"
METRIC_ALIGNMENT_PATH = TABLE_DIR / "metric_curve_alignment_v4.csv"
METRIC_COMPARISON_PATH = TABLE_DIR / "metric_reconstruction_v4_comparison.csv"
LINUCB_METRICS_PATH = TABLE_DIR / "linucb_alpha_reg_sensitivity_v2_metrics.csv"
SCHUR_AUDIT_PATH = TABLE_DIR / "paper_exact_attempt_movielens_v2_schur_shortlist_vs_full_audit.csv"
LAMBDA_AUDIT_PATH = TABLE_DIR / "paper_exact_attempt_movielens_v2_lambda_noise_sensitivity_audit.csv"
TUNING_AUDIT_PATH = TABLE_DIR / "paper_exact_attempt_movielens_v2_tuning_frequency_objective_audit.csv"
SAME_CONFIG_PATH = TABLE_DIR / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_one_config_figure4a_figure4b.csv"
ROW_LEVEL_DIR = ROOT / "outputs" / "row_level_predictions" / "paper_exact_attempt_movielens_v2"

DU_VARIANT = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
MAINLINE_METRIC = "reward_tail_proxy_rmse"
COMPARATORS = ["linucb_concat", "linucb_mf_interaction", "linucb_bias_interaction"]
REQUIRED_MASK_USERS = 943
REQUIRED_MASK_ITEMS = 1681
DEFAULT_MASK_COUNT = 50
DEFAULT_MASK_BASE_SEED = 20260712
INITIAL_SIZE = 1364
TEST_SIZE = 20000


def out_path(name: str) -> Path:
    return TABLE_DIR / f"{PREFIX}_{name}.csv"


def report_path() -> Path:
    return REPORT_DIR / f"{PREFIX}_report.md"


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def md_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
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


def rmse(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    return float(np.sqrt(np.mean(arr * arr)))


def normalized(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return math.inf
    return float(value) / max(float(scale), 1e-12)


def protocol_penalty(*parts: str) -> float:
    penalty = 0.0
    joined = " ".join(str(part).lower() for part in parts)
    if "visual" in joined or "digitized" in joined:
        penalty += 0.05
    if "proxy" in joined:
        penalty += 0.15
    if "source_blocked" in joined or "unresolved" in joined:
        penalty += 0.30
    if "queued" in joined or "not_run" in joined:
        penalty += 0.25
    return penalty


def load_target_endpoints(targets: pd.DataFrame) -> pd.DataFrame:
    if not TARGET_ENDPOINT_PATH.exists() and not targets.empty:
        endpoints = targets.sort_values("cycle").groupby(["figure", "method"], as_index=False).tail(1)
        rows: list[dict[str, object]] = []
        for figure, group in endpoints.groupby("figure", sort=False):
            du = group[group["method"] == "DU-Net"]
            lin = group[group["method"] == "Linear UCB"]
            if du.empty or lin.empty:
                continue
            rows.append(
                {
                    "figure": figure,
                    "paper_metric": str(group["paper_metric"].iloc[0]),
                    "du_endpoint_visual": float(du["paper_value"].iloc[0]),
                    "linear_endpoint_visual": float(lin["paper_value"].iloc[0]),
                    "linear_minus_du_visual_gap": float(lin["paper_value"].iloc[0] - du["paper_value"].iloc[0]),
                    "digitization_status": str(group["digitization_status"].iloc[0]),
                }
            )
        return pd.DataFrame(rows)
    return read_csv_or_empty(TARGET_ENDPOINT_PATH)


def build_metric_top20(alignment: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    if alignment.empty:
        return pd.DataFrame()
    curves = alignment[alignment["status"].astype(str) == "computed"].copy()
    fig4a = curves[curves["figure"].astype(str) == "Figure 4a"].copy()
    if fig4a.empty:
        return pd.DataFrame()
    if "endpoint_abs_error_mean" not in fig4a.columns:
        fig4a["endpoint_abs_error_mean"] = np.nan
    if "gap_abs_error" not in fig4a.columns:
        fig4a["gap_abs_error"] = np.nan
    fig4a["metric_family_score"] = (
        fig4a["curve_rmse"].astype(float)
        + fig4a["endpoint_abs_error_mean"].astype(float)
        + 5.0 * fig4a["gap_abs_error"].astype(float)
    )
    fig4a["metric"] = fig4a["metric"].astype(str)
    comp = comparison.copy()
    if not comp.empty:
        comp["metric"] = comp["metric"].astype(str)
        comp_metric = comp.rename(
            columns={
                "DU-Net endpoint": "du_endpoint",
                "Linear UCB endpoint": "linear_endpoint",
                "scale_distance_to_paper": "scale_distance",
                "gap_distance_to_paper": "gap_distance",
                "definition_defensible_from_text": "text_defensible",
                "paper_wrmse_reconstructed": "paper_wrmse_reconstructed",
            }
        )
        comp_metric["metric"] = comp_metric["metric"].str.replace("_rmse$", "", regex=True)
        metric_alias = {
            "standard_mask": "standard",
            "w": "current_w",
        }
        comp_metric["metric_join_key"] = comp_metric["metric"].replace(metric_alias)
        fig4a["metric_join_key"] = fig4a["metric"].str.replace("_rmse$", "", regex=True).replace(metric_alias)
        merged = fig4a.merge(
            comp_metric[
                [
                    "metric_join_key",
                    "comparator",
                    "du_endpoint",
                    "linear_endpoint",
                    "gap",
                    "direction",
                    "scale_distance",
                    "gap_distance",
                    "text_defensible",
                    "row_level_seed_count",
                    "paper_wrmse_reconstructed",
                ]
            ],
            on=["metric_join_key", "comparator"],
            how="left",
        )
    else:
        merged = fig4a
    if "text_defensible" not in merged.columns:
        merged["text_defensible"] = False
    merged["implausibility_penalty"] = np.where(merged["text_defensible"].fillna(False).astype(bool), 0.0, 0.2)
    merged["metric_family_score"] = merged["metric_family_score"].astype(float) + merged["implausibility_penalty"]
    merged["rank"] = merged["metric_family_score"].rank(method="first").astype(int)
    cols = [
        "rank",
        "metric",
        "comparator",
        "curve_rmse",
        "endpoint_abs_error_mean",
        "gap_abs_error",
        "metric_family_score",
        "direction",
        "scale_distance",
        "gap_distance",
        "text_defensible",
        "row_level_seed_count",
        "paper_wrmse_reconstructed",
        "implausibility_penalty",
    ]
    available = [col for col in cols if col in merged.columns]
    return merged.sort_values("metric_family_score")[available].head(20).reset_index(drop=True)


def safe_weight(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    if float(arr.sum()) <= 0.0:
        return np.ones_like(arr, dtype=float)
    return arr


def metric_value(err: np.ndarray, weight: np.ndarray, aggregation: str) -> float:
    weight = safe_weight(weight)
    if aggregation == "sqrt_mean_e2":
        return float(np.sqrt(np.mean(err * err)))
    if aggregation == "sqrt_sum_e2_over_n":
        return float(np.sqrt(np.sum(err * err)) / max(len(err), 1))
    if aggregation == "sqrt_weighted_mean_e2":
        return float(np.sqrt(np.sum(weight * err * err) / np.sum(weight)))
    if aggregation == "sqrt_mean_weighted_e2":
        return float(np.sqrt(np.mean((weight * err) ** 2)))
    raise ValueError(f"Unknown aggregation: {aggregation}")


def load_row_level_predictions() -> pd.DataFrame:
    if not ROW_LEVEL_DIR.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    wanted = {DU_VARIANT, *COMPARATORS}
    for path in sorted(ROW_LEVEL_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "method" not in frame.columns:
            continue
        method = str(frame["method"].iloc[0])
        if method not in wanted:
            continue
        frame["source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def weight_candidates(group: pd.DataFrame) -> list[tuple[str, str, np.ndarray, float]]:
    truth = group["y_true"].to_numpy(dtype=float)
    reward = 1.0 / (1.0 + np.exp(-(truth - 3.0)))
    item_pop = group.get("item_popularity", pd.Series(np.ones(len(group)))).to_numpy(dtype=float)
    positive = np.maximum(truth - 3.0, 0.0)
    rows: list[tuple[str, str, np.ndarray, float]] = [
        ("unit", "1", np.ones_like(truth), 0.0),
        ("rating", "rating", truth, 0.05),
        ("rating_mean_norm", "rating / mean(rating)", truth / max(float(np.mean(truth)), 1e-12), 0.05),
        ("sigmoid_reward", "sigmoid(rating - 3)", reward, 0.08),
        ("positive_tail_eps", "max(rating - 3, 0) + 0.1", positive + 0.1, 0.08),
        ("item_popularity", "item popularity", item_pop, 0.15),
        ("inverse_item_popularity", "1 / max(item popularity, 1)", 1.0 / np.maximum(item_pop, 1.0), 0.15),
    ]
    for alpha in [0.25, 0.5, 1.0, 2.0, 5.0]:
        rows.append(
            (
                f"unit_plus_{alpha:g}_sigmoid",
                f"1 + {alpha:g} * sigmoid(rating - 3)",
                1.0 + alpha * reward,
                0.12 + 0.02 * alpha,
            )
        )
        rows.append(
            (
                f"unit_plus_{alpha:g}_positive_tail",
                f"1 + {alpha:g} * max(rating - 3, 0)",
                1.0 + alpha * positive,
                0.12 + 0.02 * alpha,
            )
        )
    for base in [0.1, 0.5, 1.0]:
        for slope in [0.5, 1.0, 2.0]:
            rows.append(
                (
                    f"affine_tail_{base:g}_{slope:g}",
                    f"{base:g} + {slope:g} * max(rating - 3, 0)",
                    base + slope * positive,
                    0.20,
                )
            )
    return rows


def endpoint_metric_family_top20(predictions: pd.DataFrame, target_endpoints: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or target_endpoints.empty:
        return pd.DataFrame()
    test = predictions[predictions["selected_or_test"].astype(str).str.lower() == "test"].copy()
    if test.empty:
        return pd.DataFrame()
    cycles = sorted(int(cycle) for cycle in test["cycle"].dropna().unique())
    target_a = target_endpoints[target_endpoints["figure"] == "Figure 4a"]
    if target_a.empty:
        return pd.DataFrame()
    paper_du = float(target_a["du_endpoint_visual"].iloc[0])
    paper_linear = float(target_a["linear_endpoint_visual"].iloc[0])
    paper_gap = float(target_a["linear_minus_du_visual_gap"].iloc[0])

    metric_rows: list[dict[str, object]] = []
    aggregations = [
        ("sqrt_mean_e2", 0.0),
        ("sqrt_sum_e2_over_n", 0.05),
        ("sqrt_weighted_mean_e2", 0.05),
        ("sqrt_mean_weighted_e2", 0.12),
    ]
    error_specs = [
        ("clipped_prediction_error", "err_clipped", 1.0, 0.0),
        ("unclipped_prediction_error", "err_raw", 1.0, 0.05),
        ("normalized_clipped_error", "err_clipped", 0.25, 0.05),
    ]
    for (method, seed, cycle), group in test.groupby(["method", "seed", "cycle"], sort=False):
        for error_name, err_col, err_scale, err_penalty in error_specs:
            if err_col not in group.columns:
                continue
            err = group[err_col].to_numpy(dtype=float) * float(err_scale)
            for weight_name, weight_definition, weight, weight_penalty in weight_candidates(group):
                for aggregation, aggregation_penalty in aggregations:
                    name = f"{error_name}|{aggregation}|{weight_name}"
                    definition = f"{aggregation}; error={error_name}; weight={weight_definition}"
                    metric_rows.append(
                        {
                            "method": method,
                            "seed": int(seed),
                            "cycle": int(cycle),
                            "metric": name,
                            "definition": definition,
                            "value": metric_value(err, weight, aggregation),
                            "implausibility_penalty": err_penalty + weight_penalty + aggregation_penalty,
                        }
                    )
    metric_frame = pd.DataFrame(metric_rows)
    if metric_frame.empty:
        return pd.DataFrame()
    summary = (
        metric_frame.groupby(["method", "metric", "definition"], as_index=False)
        .agg(value=("value", "mean"), seed_count=("seed", "nunique"), implausibility_penalty=("implausibility_penalty", "mean"))
    )
    rows: list[dict[str, object]] = []
    for comparator in COMPARATORS:
        for metric, group in summary.groupby("metric", sort=False):
            du = group[group["method"] == DU_VARIANT]
            base = group[group["method"] == comparator]
            if du.empty or base.empty:
                continue
            du_value = float(du["value"].iloc[0])
            base_value = float(base["value"].iloc[0])
            local_gap = base_value - du_value
            scale_distance = float(np.mean([abs(du_value - paper_du) / paper_du, abs(base_value - paper_linear) / paper_linear]))
            gap_distance = float(abs(local_gap - paper_gap) / max(abs(paper_gap), 1e-12))
            penalty = float(max(du["implausibility_penalty"].iloc[0], base["implausibility_penalty"].iloc[0]))
            score = scale_distance + 0.25 * gap_distance + penalty
            rows.append(
                {
                    "metric": metric,
                    "definition": str(group["definition"].iloc[0]),
                    "comparator": comparator,
                    "du_endpoint": du_value,
                    "linear_endpoint": base_value,
                    "local_gap": local_gap,
                    "scale_distance": scale_distance,
                    "gap_distance": gap_distance,
                    "direction": "DU-Net < Linear UCB" if local_gap > 0 else "not_matched",
                    "seed_count": int(min(du["seed_count"].iloc[0], base["seed_count"].iloc[0])),
                    "row_level_scope": "final_cycle_only" if cycles == [20] else "multi_cycle",
                    "curve_search_status": "blocked_requires_all_cycle_row_level_predictions" if cycles == [20] else "available",
                    "implausibility_penalty": penalty,
                    "endpoint_family_score": score,
                    "paper_wrmse_reconstructed": "no",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["endpoint_family_score", "scale_distance", "gap_distance"]).head(20).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_linucb_ranking(targets: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if targets.empty or metrics.empty:
        return pd.DataFrame()
    local = metrics.groupby(["variant", "cycle"], as_index=False).agg(
        ucb_alpha=("ucb_alpha", "first"),
        ucb_regularization=("ucb_regularization", "first"),
        seed_count=("seed", "nunique"),
        fig4a_local=("reward_tail_proxy_rmse", "mean"),
        fig4a_standard=("standard_mask_rmse", "mean"),
        fig4b_local=("cumulative_reward", "mean"),
    )
    target_a = targets[(targets["figure"] == "Figure 4a") & (targets["method"] == "Linear UCB")][
        ["cycle", "paper_value"]
    ].rename(columns={"paper_value": "fig4a_paper"})
    target_b = targets[(targets["figure"] == "Figure 4b") & (targets["method"] == "Linear UCB")][
        ["cycle", "paper_value"]
    ].rename(columns={"paper_value": "fig4b_paper"})
    rows: list[dict[str, object]] = []
    for variant, group in local.groupby("variant", sort=False):
        a = group[["cycle", "fig4a_local"]].merge(target_a, on="cycle", how="inner")
        b = group[["cycle", "fig4b_local"]].merge(target_b, on="cycle", how="inner")
        if a.empty and b.empty:
            continue
        fig4a_curve_error = rmse(a["fig4a_local"] - a["fig4a_paper"]) if not a.empty else math.nan
        fig4b_curve_error = rmse(b["fig4b_local"] - b["fig4b_paper"]) if not b.empty else math.nan
        fig4a_endpoint_error = (
            float(abs(a.sort_values("cycle").tail(1)["fig4a_local"].iloc[0] - a.sort_values("cycle").tail(1)["fig4a_paper"].iloc[0]))
            if not a.empty
            else math.nan
        )
        fig4b_endpoint_error = (
            float(abs(b.sort_values("cycle").tail(1)["fig4b_local"].iloc[0] - b.sort_values("cycle").tail(1)["fig4b_paper"].iloc[0]))
            if not b.empty
            else math.nan
        )
        proto_penalty = protocol_penalty("local_clean_room_proxy")
        rows.append(
            {
                "candidate": variant,
                "feature": "mf_interaction",
                "alpha": float(group["ucb_alpha"].iloc[0]),
                "reg": float(group["ucb_regularization"].iloc[0]),
                "target": "rating",
                "update": "batch_update_per_cycle",
                "seed_count": int(group["seed_count"].max()),
                "fig4a_linear_curve_error": fig4a_curve_error,
                "fig4a_endpoint_error": fig4a_endpoint_error,
                "fig4b_linear_curve_error": fig4b_curve_error,
                "fig4b_endpoint_error": fig4b_endpoint_error,
                "protocol_penalty": proto_penalty,
                "linear_ucb_score": normalized(fig4a_curve_error, 1.0) + normalized(fig4b_curve_error, 100.0) + proto_penalty,
                "paper_linear_ucb_reconstructed": "no",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("linear_ucb_score").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def read_movielens_ratings(raw_dir: Path, num_items: int = REQUIRED_MASK_ITEMS) -> pd.DataFrame:
    path = raw_dir / "u.data"
    frame = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    frame["source_row_id"] = np.arange(len(frame), dtype=int)
    frame["user_id"] = frame["user_id"].astype(int) - 1
    frame["item_id"] = frame["item_id"].astype(int) - 1
    frame = frame[frame["item_id"] < int(num_items)].copy()
    return frame[["source_row_id", *RATING_COLUMNS]].reset_index(drop=True)


def distribution_distance(mask_values: pd.Series, reference_values: pd.Series) -> float:
    bins = np.arange(0.5, 6.0, 1.0)
    mask_hist, _ = np.histogram(mask_values.to_numpy(dtype=float), bins=bins)
    ref_hist, _ = np.histogram(reference_values.to_numpy(dtype=float), bins=bins)
    mask_prob = mask_hist / max(mask_hist.sum(), 1)
    ref_prob = ref_hist / max(ref_hist.sum(), 1)
    return float(np.abs(mask_prob - ref_prob).sum() / 2.0)


def generate_random_masks(raw_dir: Path, count: int, base_seed: int, force: bool = False) -> tuple[pd.DataFrame, Path]:
    mask_dir = MASK_DIR / PREFIX
    mask_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path("mask_candidates_50")
    npz_path = mask_dir / "table1_random_masks_v1.npz"
    if manifest_path.exists() and npz_path.exists() and not force:
        return pd.read_csv(manifest_path), npz_path

    ratings = read_movielens_ratings(raw_dir)
    rng_master = np.random.default_rng(base_seed)
    all_pos = np.arange(len(ratings), dtype=np.int64)
    rows: list[dict[str, object]] = []
    initial_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []
    candidate_hash_parts: list[str] = []
    full_density = float(len(ratings) / (REQUIRED_MASK_USERS * REQUIRED_MASK_ITEMS))
    for idx in range(count):
        seed = int(base_seed + idx)
        rng = np.random.default_rng(seed)
        test_pos = np.sort(rng.choice(all_pos, size=TEST_SIZE, replace=False))
        remaining = np.setdiff1d(all_pos, test_pos, assume_unique=False)
        initial_pos = np.sort(rng.choice(remaining, size=INITIAL_SIZE, replace=False))
        candidate_pos = np.setdiff1d(remaining, initial_pos, assume_unique=False)
        initial = ratings.iloc[initial_pos]
        candidate = ratings.iloc[candidate_pos]
        test = ratings.iloc[test_pos]
        initial_indices.append(ratings.iloc[initial_pos]["source_row_id"].to_numpy(dtype=np.int64))
        test_indices.append(ratings.iloc[test_pos]["source_row_id"].to_numpy(dtype=np.int64))
        candidate_hash = int(np.bitwise_xor.reduce(ratings.iloc[candidate_pos]["source_row_id"].to_numpy(dtype=np.int64), initial=0))
        candidate_hash_parts.append(str(candidate_hash))
        rows.append(
            {
                "mask_id": f"table1_random_{idx:03d}",
                "type": "random_table1_global",
                "seed": seed,
                "users": REQUIRED_MASK_USERS,
                "items": REQUIRED_MASK_ITEMS,
                "initial_size": int(len(initial)),
                "candidate_size": int(len(candidate)),
                "test_size": int(len(test)),
                "matrix_density": full_density,
                "initial_rating_mean": float(initial["rating"].mean()),
                "candidate_rating_mean": float(candidate["rating"].mean()),
                "test_rating_mean": float(test["rating"].mean()),
                "initial_user_coverage": int(initial["user_id"].nunique()),
                "candidate_user_coverage": int(candidate["user_id"].nunique()),
                "test_user_coverage": int(test["user_id"].nunique()),
                "initial_item_coverage": int(initial["item_id"].nunique()),
                "candidate_item_coverage": int(candidate["item_id"].nunique()),
                "test_item_coverage": int(test["item_id"].nunique()),
                "rating_distribution_l1_to_full": distribution_distance(test["rating"], ratings["rating"]),
                "density_match": "pass" if len(initial) == INITIAL_SIZE and len(test) == TEST_SIZE else "fail",
                "selected": "queued_for_3seed_screen",
                "candidate_xor_hash": candidate_hash,
            }
        )
    manifest = pd.DataFrame(rows)
    np.savez_compressed(
        npz_path,
        initial_source_row_id=np.vstack(initial_indices),
        test_source_row_id=np.vstack(test_indices),
        candidate_xor_hash=np.asarray(candidate_hash_parts, dtype="U32"),
        base_seed=np.asarray([base_seed], dtype=np.int64),
    )
    manifest.to_csv(manifest_path, index=False)
    return manifest, npz_path


def build_mask_run_queue(mask_manifest: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "schur_loglambda_tuned_full_pool_softmax_tau1_literal",
        "linucb_concat",
        "linucb_mf_interaction",
    ]
    rows: list[dict[str, object]] = []
    for mask_id in mask_manifest["mask_id"].head(DEFAULT_MASK_COUNT):
        for method in methods:
            rows.append(
                {
                    "mask_id": mask_id,
                    "method": method,
                    "seeds": "0 1 2",
                    "cycles": 20,
                    "status": "queued_not_run",
                    "reason": "requires mask-aware acquisition runner; generated mask indices are available",
                    "suggested_output_prefix": f"{PREFIX}_{mask_id}_{method}_3seeds",
                }
            )
    return pd.DataFrame(rows)


def build_schur_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary = rows.groupby("cycle", as_index=False).agg(
        seeds=("seed", "nunique"),
        min_top100_overlap=("top100_overlap", "min"),
        min_selected_batch_overlap=("selected_batch_overlap", "min"),
        max_selected_changed_count=("selected_changed_count", "max"),
        max_abs_one_step_rmse_diff=(
            "one_step_standard_rmse_diff_full_minus_short",
            lambda values: float(np.max(np.abs(pd.Series(values).dropna().to_numpy(dtype=float)))) if len(pd.Series(values).dropna()) else math.nan,
        ),
        mean_abs_rating_diff=("selected_rating_diff_full_minus_short", lambda values: float(np.mean(np.abs(pd.Series(values).dropna().to_numpy(dtype=float))))),
    )
    summary["selection"] = "shortlist20k_vs_all_candidate_tiled"
    summary["status"] = np.where(
        (summary["min_selected_batch_overlap"] >= 0.95) & (summary["max_abs_one_step_rmse_diff"].fillna(0.0) <= 0.001),
        "screen_pass_low_effect",
        "live_exactness_risk",
    )
    return summary


def build_lambda_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary = rows.groupby("lambda_noise", as_index=False).agg(
        seeds=("seed", "nunique"),
        min_top100_overlap=("top100_overlap_vs_base", "min"),
        min_selected_batch_overlap=("selected_batch_overlap_vs_base", "min"),
        max_selected_changed_count=("selected_changed_count_vs_base", "max"),
        mean_selected_rating=("selected_mean_rating", "mean"),
        mean_selected_entropy=("selected_mean_entropy", "mean"),
    )
    summary["status"] = np.where(
        summary["min_selected_batch_overlap"] >= 0.99,
        "cycle0_low_effect",
        "cycle0_selection_sensitive",
    )
    return summary.sort_values(["status", "lambda_noise"])


def build_du_knob_summary(schur: pd.DataFrame, lambdas: pd.DataFrame, tuning: pd.DataFrame, same_config: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not schur.empty:
        rows.append(
            {
                "knob_family": "selection_pool",
                "candidate": "shortlist20k_vs_all_candidate",
                "best_available_evidence": f"min selected overlap={schur['min_selected_batch_overlap'].min():.4g}; max changed={schur['max_selected_changed_count'].max():.4g}",
                "fig4a_effect": "one_step_rmse_screen_only",
                "fig4b_effect": "selected_rating_screen_only",
                "status": "screened_not_end_to_end_full_pool",
                "protocol_penalty": protocol_penalty("source_blocked", "not_run_end_to_end"),
            }
        )
    if not lambdas.empty:
        best_lambda = lambdas.sort_values(["min_selected_batch_overlap", "lambda_noise"], ascending=[False, True]).head(1).iloc[0]
        rows.append(
            {
                "knob_family": "lambda_noise",
                "candidate": f"lambda_noise={float(best_lambda['lambda_noise']):g}",
                "best_available_evidence": f"cycle0 selected overlap={float(best_lambda['min_selected_batch_overlap']):.4g}; changed={float(best_lambda['max_selected_changed_count']):.4g}",
                "fig4a_effect": "cycle0_selection_screen_only",
                "fig4b_effect": "selected_rating_screen_only",
                "status": "screened_not_curve_confirmed",
                "protocol_penalty": protocol_penalty("source_blocked"),
            }
        )
    if not tuning.empty:
        for row in tuning.itertuples(index=False):
            rows.append(
                {
                    "knob_family": "tuning_objective",
                    "candidate": getattr(row, "audit_item"),
                    "best_available_evidence": getattr(row, "evidence"),
                    "fig4a_effect": "audit",
                    "fig4b_effect": "audit",
                    "status": getattr(row, "status"),
                    "protocol_penalty": protocol_penalty(str(getattr(row, "status")), str(getattr(row, "exactness_risk"))),
                }
            )
    if not same_config.empty:
        evidence = "; ".join(
            f"{row.metric}: gap={float(row.gap):.6g}" for row in same_config.itertuples(index=False) if hasattr(row, "gap")
        )
        rows.append(
            {
                "knob_family": "same_config_figure4a_figure4b",
                "candidate": "tau0p3_current_best_same_config",
                "best_available_evidence": evidence,
                "fig4a_effect": "strong_rmse_recovery",
                "fig4b_effect": "comparator_dependent_reward_gap",
                "status": "not_jointly_closed",
                "protocol_penalty": protocol_penalty("unresolved"),
            }
        )
    return pd.DataFrame(rows)


def best_value(frame: pd.DataFrame, column: str, default: float = math.inf) -> float:
    if frame.empty or column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else default


def build_joint_leaderboard(
    metric_top20: pd.DataFrame,
    linucb: pd.DataFrame,
    mask_manifest: pd.DataFrame,
    du_knobs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_candidates = metric_top20.head(5) if not metric_top20.empty else pd.DataFrame()
    linear_candidates = linucb.head(5) if not linucb.empty else pd.DataFrame()
    mask_candidates = mask_manifest.sort_values("rating_distribution_l1_to_full").head(5) if not mask_manifest.empty else pd.DataFrame()
    du_candidates = du_knobs[du_knobs["knob_family"].isin(["same_config_figure4a_figure4b", "selection_pool", "lambda_noise"])].head(5)
    if metric_candidates.empty:
        metric_candidates = pd.DataFrame([{"metric": "unresolved", "comparator": "unresolved", "curve_rmse": math.inf, "gap_abs_error": math.inf, "implausibility_penalty": 0.3}])
    if linear_candidates.empty:
        linear_candidates = pd.DataFrame([{"candidate": "unresolved", "fig4a_linear_curve_error": math.inf, "fig4b_linear_curve_error": math.inf, "protocol_penalty": 0.3}])
    if mask_candidates.empty:
        mask_candidates = pd.DataFrame([{"mask_id": "unresolved", "type": "source_blocked", "rating_distribution_l1_to_full": math.inf, "selected": "not_run"}])
    if du_candidates.empty:
        du_candidates = pd.DataFrame([{"candidate": "unresolved", "knob_family": "unresolved", "protocol_penalty": 0.3, "status": "unresolved"}])

    for _, metric in metric_candidates.iterrows():
        for _, linear in linear_candidates.iterrows():
            for _, mask in mask_candidates.iterrows():
                for _, du in du_candidates.iterrows():
                    fig4a_du_error = float(metric.get("curve_rmse", math.inf))
                    fig4a_linear_error = float(linear.get("fig4a_linear_curve_error", math.inf))
                    fig4b_du_error = 0.0 if str(du.get("fig4b_effect", "")).startswith("comparator_dependent") else math.inf
                    fig4b_linear_error = float(linear.get("fig4b_linear_curve_error", math.inf))
                    penalty = (
                        float(metric.get("implausibility_penalty", 0.0) or 0.0)
                        + float(linear.get("protocol_penalty", 0.0) or 0.0)
                        + float(du.get("protocol_penalty", 0.0) or 0.0)
                        + protocol_penalty(str(mask.get("selected", "")), str(mask.get("type", "")))
                    )
                    total = (
                        0.35 * normalized(fig4a_du_error, 1.0)
                        + 0.35 * normalized(fig4a_linear_error, 1.0)
                        + 0.15 * normalized(fig4b_du_error, 100.0)
                        + 0.15 * normalized(fig4b_linear_error, 100.0)
                        + penalty
                    )
                    rows.append(
                        {
                            "pipeline_id": f"candidate_{len(rows) + 1:04d}",
                            "metric": str(metric.get("metric", "")),
                            "mask": str(mask.get("mask_id", "")),
                            "linear_ucb": str(linear.get("candidate", "")),
                            "du_net_tuning": str(du.get("candidate", "")),
                            "selection": str(du.get("knob_family", "")),
                            "fig4a_du_error": fig4a_du_error,
                            "fig4a_linear_error": fig4a_linear_error,
                            "fig4b_du_error": fig4b_du_error,
                            "fig4b_linear_error": fig4b_linear_error,
                            "protocol_penalty": penalty,
                            "total_error": total,
                            "claim_status": "candidate_not_confirmed_20seed" if np.isfinite(total) else "blocked",
                        }
                    )
    out = pd.DataFrame(rows).sort_values("total_error").head(50).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_stop_rules(metric_top20: pd.DataFrame, linucb: pd.DataFrame, mask_queue: pd.DataFrame, du_knobs: pd.DataFrame, leaderboard: pd.DataFrame) -> pd.DataFrame:
    metric_closed = bool((metric_top20.get("paper_wrmse_reconstructed", pd.Series(dtype=str)).astype(str) == "yes").any())
    linucb_closed = bool((linucb.get("paper_linear_ucb_reconstructed", pd.Series(dtype=str)).astype(str) == "yes").any())
    masks_screened = bool(mask_queue.empty or (mask_queue["status"].astype(str) != "queued_not_run").all())
    same_config_confirmed = bool((leaderboard.get("claim_status", pd.Series(dtype=str)).astype(str) == "20seed_confirmed").any())
    rows = [
        {
            "stop_rule": "best_attainable_paper_only_reconstruction",
            "status": "not_met",
            "evidence": "metric, Linear UCB, mask, and DU-Net latent-variable gates are represented, but at least one remains source-blocked or queued.",
            "required_next": "run queued mask screens and promote only defensible candidates",
        },
        {
            "stop_rule": "near_perfect_empirical_reconstruction",
            "status": "met" if metric_closed and linucb_closed and masks_screened and same_config_confirmed else "not_met",
            "evidence": f"metric_closed={metric_closed}; linucb_closed={linucb_closed}; masks_screened={masks_screened}; same_config_confirmed={same_config_confirmed}",
            "required_next": "top 5 joint candidates require 20-seed confirmation under the same config",
        },
        {
            "stop_rule": "exact_replication",
            "status": "not_claimable",
            "evidence": "author metric code, exact masks, Linear UCB protocol, and numeric Figure 4 data are still absent",
            "required_next": "recover source artifacts or keep exactness claim disabled",
        },
    ]
    return pd.DataFrame(rows)


def write_report(
    target_endpoints: pd.DataFrame,
    metric_endpoint_top20: pd.DataFrame,
    metric_top20: pd.DataFrame,
    linucb: pd.DataFrame,
    mask_manifest: pd.DataFrame,
    mask_queue: pd.DataFrame,
    schur: pd.DataFrame,
    lambdas: pd.DataFrame,
    du_knobs: pd.DataFrame,
    leaderboard: pd.DataFrame,
    stop_rules: pd.DataFrame,
    mask_npz: Path,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Inverse Reconstruction Leaderboard v1",
        "",
        "Purpose: convert the Figure 4 replication gap into a latent-variable reconstruction surface. This report ranks what can be ranked from existing artifacts, generates Table-1-compatible random masks, and keeps unrun/source-blocked gates explicit.",
        "",
        "## Inputs",
        "",
        f"- Digitized target curves: `{TARGET_PATH.relative_to(ROOT).as_posix()}`",
        f"- Metric alignment: `{METRIC_ALIGNMENT_PATH.relative_to(ROOT).as_posix()}`",
        f"- Linear UCB alpha/reg metrics: `{LINUCB_METRICS_PATH.relative_to(ROOT).as_posix()}`",
        f"- Generated mask index archive: `{mask_npz.relative_to(ROOT).as_posix()}`",
        "",
        "## Target Endpoints",
        "",
        md_table(target_endpoints),
        "",
        "## Task 1: Row-Level Metric Family Endpoint Top 20",
        "",
        md_table(metric_endpoint_top20, max_rows=20),
        "",
        "## Task 1b: Existing Curve-Level Metric Alignment Top 20",
        "",
        md_table(metric_top20, max_rows=20),
        "",
        "## Task 2: Linear UCB Candidate Family",
        "",
        md_table(linucb.head(20)),
        "",
        "## Task 3: Random Table-1-Compatible Masks",
        "",
        md_table(mask_manifest.head(10)),
        "",
        "### Mask Run Queue",
        "",
        md_table(mask_queue.head(20)),
        "",
        "## Task 4: Schur Shortlist vs Full",
        "",
        md_table(schur),
        "",
        "## Task 5: Lambda Noise and Tuning Objective",
        "",
        md_table(lambdas),
        "",
        md_table(du_knobs),
        "",
        "## Task 6: Joint Candidate Leaderboard",
        "",
        md_table(leaderboard.head(20)),
        "",
        "## Stop Rules",
        "",
        md_table(stop_rules),
        "",
        "## Decision Boundary",
        "",
        "No row is promoted to exact replication. Current outputs support a paper-only reconstruction workflow and a mask-screening queue; they do not close the missing author metric, mask, Linear UCB, backend, or numeric Figure 4 gates.",
    ]
    report_path().write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all(mask_count: int, mask_seed: int, force_masks: bool = False) -> dict[str, pd.DataFrame | Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    targets = read_csv_or_empty(TARGET_PATH)
    target_endpoints = load_target_endpoints(targets)
    metric_endpoint_top20 = endpoint_metric_family_top20(load_row_level_predictions(), target_endpoints)
    metric_top20 = build_metric_top20(read_csv_or_empty(METRIC_ALIGNMENT_PATH), read_csv_or_empty(METRIC_COMPARISON_PATH))
    linucb = build_linucb_ranking(targets, read_csv_or_empty(LINUCB_METRICS_PATH))
    mask_manifest, mask_npz = generate_random_masks(ROOT / "data" / "raw" / "ml-100k", mask_count, mask_seed, force=force_masks)
    mask_queue = build_mask_run_queue(mask_manifest)
    schur = build_schur_summary(read_csv_or_empty(SCHUR_AUDIT_PATH))
    lambdas = build_lambda_summary(read_csv_or_empty(LAMBDA_AUDIT_PATH))
    du_knobs = build_du_knob_summary(
        schur,
        lambdas,
        read_csv_or_empty(TUNING_AUDIT_PATH),
        read_csv_or_empty(SAME_CONFIG_PATH),
    )
    leaderboard = build_joint_leaderboard(metric_top20, linucb, mask_manifest, du_knobs)
    stop_rules = build_stop_rules(metric_top20, linucb, mask_queue, du_knobs, leaderboard)

    outputs: dict[str, pd.DataFrame | Path] = {
        "target_endpoints": target_endpoints,
        "metric_endpoint_family_top20": metric_endpoint_top20,
        "metric_top20": metric_top20,
        "linucb_candidate_ranking": linucb,
        "mask_candidates_50": mask_manifest,
        "mask_run_queue": mask_queue,
        "schur_shortlist_summary": schur,
        "lambda_noise_summary": lambdas,
        "du_knob_summary": du_knobs,
        "joint_candidate_leaderboard": leaderboard,
        "stop_rules": stop_rules,
        "mask_npz": mask_npz,
    }
    for name, frame in outputs.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(out_path(name), index=False)
    write_report(
        target_endpoints,
        metric_endpoint_top20,
        metric_top20,
        linucb,
        mask_manifest,
        mask_queue,
        schur,
        lambdas,
        du_knobs,
        leaderboard,
        stop_rules,
        mask_npz,
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-count", type=int, default=DEFAULT_MASK_COUNT)
    parser.add_argument("--mask-seed", type=int, default=DEFAULT_MASK_BASE_SEED)
    parser.add_argument("--force-masks", action="store_true")
    args = parser.parse_args()
    outputs = build_all(mask_count=int(args.mask_count), mask_seed=int(args.mask_seed), force_masks=bool(args.force_masks))
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            print(out_path(name))
        else:
            print(value)
    print(report_path())


if __name__ == "__main__":
    main()
