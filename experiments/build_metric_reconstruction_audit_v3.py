from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_exact_attempt_movielens_v1"
ROW_LEVEL_DIR = ROOT / "outputs" / "row_level_predictions" / PREFIX
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

DU = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
COMPARATORS = ["linucb_concat", "linucb_mf_interaction"]
REQUIRED_METHODS = [
    DU,
    "linucb_concat",
    "linucb_mf_interaction",
    "schur_loglambda_tuned_full_pool_softmax_tau0p3",
    "schur_tuned_topk",
    "lambda0p3_diagnostic",
]

PAPER_FIG4A_DU = 1.788
PAPER_FIG4A_LINEAR = 1.814


def ci95(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: f"{float(value):.6g}" if pd.notna(value) else "")
        else:
            formatted[col] = formatted[col].fillna("").astype(str)
    lines = [
        "| " + " | ".join(str(col) for col in formatted.columns) + " |",
        "| " + " | ".join(["---"] * len(formatted.columns)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in formatted.columns) + " |")
    return "\n".join(lines)


def load_predictions() -> pd.DataFrame:
    files = sorted(ROW_LEVEL_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No row-level prediction files under {ROW_LEVEL_DIR}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "method",
        "seed",
        "cycle",
        "selected_or_test",
        "y_true",
        "y_pred_raw",
        "y_pred_clipped",
        "err_raw",
        "err_clipped",
        "rating",
        "reward",
        "item_popularity",
        "user_degree",
        "test_mask_id",
    }
    missing = sorted(required.difference(out.columns))
    if missing:
        raise ValueError(f"Row-level predictions missing required v3 columns: {missing}")
    return out


def completion_status(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in REQUIRED_METHODS:
        subset = predictions[predictions["method"] == method]
        seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique())
        cycles = sorted(int(cycle) for cycle in subset["cycle"].dropna().unique())
        test_rows = subset[subset["selected_or_test"].astype(str).str.lower() == "test"]
        rows.append(
            {
                "method": method,
                "available_seed_count": len(seeds),
                "available_seeds": " ".join(str(seed) for seed in seeds),
                "available_cycles": " ".join(str(cycle) for cycle in cycles),
                "test_row_count": int(len(test_rows)),
                "required_seed_count": 20,
                "row_level_scope": "final_cycle_only" if cycles == [20] else "multi_cycle",
                "status": "complete_20seed" if len(seeds) >= 20 else "partial",
            }
        )
    return pd.DataFrame(rows)


def safe_weight(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    if float(np.sum(arr)) <= 0.0:
        return np.ones_like(arr, dtype=float)
    return arr


def add_metric(rows: list[dict[str, object]], group: pd.DataFrame, metric: str, definition: str, value: float) -> None:
    rows.append(
        {
            "method": str(group["method"].iloc[0]),
            "seed": int(group["seed"].iloc[0]),
            "cycle": int(group["cycle"].max()),
            "metric": metric,
            "definition": definition,
            "value": float(value),
            "rows": int(len(group)),
        }
    )


def metric_rows(group: pd.DataFrame) -> list[dict[str, object]]:
    truth = group["y_true"].to_numpy(dtype=float)
    pred_raw = group["y_pred_raw"].to_numpy(dtype=float)
    pred = group["y_pred_clipped"].to_numpy(dtype=float)
    err_raw = group["err_raw"].to_numpy(dtype=float)
    err = group["err_clipped"].to_numpy(dtype=float)
    n = max(len(group), 1)
    rows: list[dict[str, object]] = []

    add_metric(rows, group, "standard_rmse", "sqrt(mean(err_clipped^2))", np.sqrt(np.mean(err * err)))
    add_metric(rows, group, "pdf_literal", "sqrt(sum(err_clipped^2)) / N", np.sqrt(np.sum(err * err)) / n)
    add_metric(rows, group, "plain_l2", "sqrt(sum(err_clipped^2))", np.sqrt(np.sum(err * err)))
    add_metric(rows, group, "centered_rating_rmse", "sqrt(mean(((rating-3)-(pred-3))^2))", np.sqrt(np.mean(((truth - 3.0) - (pred - 3.0)) ** 2)))
    add_metric(rows, group, "raw_unclipped_rmse", "sqrt(mean(err_raw^2))", np.sqrt(np.mean(err_raw * err_raw)))
    add_metric(rows, group, "clipped_rmse", "sqrt(mean(err_clipped^2))", np.sqrt(np.mean(err * err)))
    add_metric(rows, group, "rating_0_1_normalized_rmse", "sqrt(mean((err_clipped/4)^2))", np.sqrt(np.mean((err / 4.0) ** 2)))

    weights = {
        "rating_weighted_rmse": safe_weight(group["rating"]),
        "normalized_rating_weighted_rmse": safe_weight(group["rating"] / max(float(group["rating"].mean()), 1e-12)),
        "sigmoid_reward_weighted_rmse": safe_weight(group["reward"]),
        "max_rating_minus_3_eps_weighted_rmse": safe_weight(np.maximum(truth - 3.0, 0.0) + 0.1),
        "item_popularity_weighted_rmse": safe_weight(group["item_popularity"]),
        "inverse_item_popularity_weighted_rmse": safe_weight(1.0 / np.maximum(group["item_popularity"].to_numpy(dtype=float), 1.0)),
        "user_degree_weighted_rmse": safe_weight(group["user_degree"]),
    }
    for metric, weight in weights.items():
        add_metric(rows, group, metric, f"sqrt(sum(w*err_clipped^2)/sum(w)); w={metric.removesuffix('_weighted_rmse')}", np.sqrt(np.sum(weight * err * err) / np.sum(weight)))
        add_metric(rows, group, f"sqrt_mean_squared_weighted_{metric}", f"sqrt(mean((w*err_clipped)^2)); w={metric.removesuffix('_weighted_rmse')}", np.sqrt(np.mean((weight * err) ** 2)))
    return rows


def compute_metric_grid(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[predictions["selected_or_test"].astype(str).str.lower() == "test"].copy()
    rows = []
    for _, group in test.groupby(["method", "seed", "cycle"], sort=False):
        rows.extend(metric_rows(group))
    return pd.DataFrame(rows)


def summarize(metric_grid: pd.DataFrame) -> pd.DataFrame:
    return (
        metric_grid.groupby(["method", "metric", "definition"], as_index=False)
        .agg(value=("value", "mean"), value_ci95=("value", ci95), seed_count=("seed", "nunique"), rows=("rows", "mean"))
        .sort_values(["metric", "method"])
    )


def compare_to_paper(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    paper_gap = PAPER_FIG4A_LINEAR - PAPER_FIG4A_DU
    defensible_metrics = {
        "standard_rmse",
        "pdf_literal",
        "rating_weighted_rmse",
        "normalized_rating_weighted_rmse",
        "sigmoid_reward_weighted_rmse",
        "max_rating_minus_3_eps_weighted_rmse",
        "centered_rating_rmse",
        "raw_unclipped_rmse",
        "clipped_rmse",
        "rating_0_1_normalized_rmse",
    }
    for comparator in COMPARATORS:
        for metric, group in summary.groupby("metric", sort=False):
            du = group[group["method"] == DU]
            base = group[group["method"] == comparator]
            if du.empty or base.empty:
                continue
            du_value = float(du["value"].iloc[0])
            base_value = float(base["value"].iloc[0])
            local_gap = base_value - du_value
            scale_distance = float(
                np.mean(
                    [
                        abs(du_value - PAPER_FIG4A_DU) / PAPER_FIG4A_DU,
                        abs(base_value - PAPER_FIG4A_LINEAR) / PAPER_FIG4A_LINEAR,
                    ]
                )
            )
            gap_distance = float(abs(local_gap - paper_gap) / max(abs(paper_gap), 1e-12))
            direction = "DU-Net < Linear UCB" if local_gap > 0 else "not_matched"
            defensible = metric in defensible_metrics
            pass_gate = scale_distance <= 0.15 and gap_distance <= 0.20 and local_gap > 0 and defensible
            rows.append(
                {
                    "metric": metric,
                    "comparator": comparator,
                    "DU-Net endpoint": du_value,
                    "Linear UCB endpoint": base_value,
                    "gap": local_gap,
                    "direction": direction,
                    "scale_distance_to_paper": scale_distance,
                    "gap_distance_to_paper": gap_distance,
                    "definition_defensible_from_text": defensible,
                    "raw_match": "pass" if pass_gate else "fail",
                    "affine_match": "not_testable_from_final_only_row_level",
                    "paper_wrmse_reconstructed": "yes" if pass_gate else "no",
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["paper_wrmse_reconstructed", "scale_distance_to_paper", "gap_distance_to_paper"], ascending=[False, True, True])


def curve_evidence() -> pd.DataFrame:
    candidates = [
        ROOT / "outputs" / "tables" / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1" / "tau1_figure4_curve_alignment_summary.csv",
        ROOT / "outputs" / "tables" / PREFIX / "tau1_figure4_curve_alignment_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            frame["curve_source"] = str(path.relative_to(ROOT)).replace("\\", "/")
            return frame
    return pd.DataFrame()


def write_report(completion: pd.DataFrame, comparison: pd.DataFrame, curves: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    status = "pass" if (comparison["paper_wrmse_reconstructed"] == "yes").any() else "fail"
    lines = [
        "# Metric Reconstruction Audit v3",
        "",
        f"- Row-level source: `{ROW_LEVEL_DIR.relative_to(ROOT)}`",
        f"- Paper Figure 4a visual endpoints: DU-Net `{PAPER_FIG4A_DU}`, Linear UCB `{PAPER_FIG4A_LINEAR}`",
        f"- Gate status: **{status}**",
        "",
        "## Row-Level Completion",
        "",
        md_table(completion),
        "",
        "## Best Metric Candidates",
        "",
        md_table(comparison.head(20)),
        "",
        "## Curve Evidence",
        "",
        md_table(curves.head(12)),
        "",
        "## Decision",
        "",
        "No metric is promoted to `paper_wrmse_reconstructed` unless endpoint scale, direction, gap, curve shape, and text-defensible definition all pass. With current final-cycle row-level evidence, no candidate passes. Exact w-RMSE remains blocked.",
    ]
    (REPORT_DIR / f"{PREFIX}_metric_reconstruction_audit_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    completion = completion_status(predictions)
    metric_grid = compute_metric_grid(predictions)
    summary = summarize(metric_grid)
    comparison = compare_to_paper(summary)
    curves = curve_evidence()

    completion.to_csv(TABLE_DIR / f"{PREFIX}_row_level_completion_v3.csv", index=False)
    metric_grid.to_csv(TABLE_DIR / f"{PREFIX}_metric_reconstruction_v3_long.csv", index=False)
    summary.to_csv(TABLE_DIR / f"{PREFIX}_metric_reconstruction_v3_summary.csv", index=False)
    comparison.to_csv(TABLE_DIR / f"{PREFIX}_metric_reconstruction_v3_comparison.csv", index=False)
    curves.to_csv(TABLE_DIR / f"{PREFIX}_metric_reconstruction_v3_curve_evidence.csv", index=False)
    write_report(completion, comparison, curves)
    print(REPORT_DIR / f"{PREFIX}_metric_reconstruction_audit_v3.md")
    print(TABLE_DIR / f"{PREFIX}_metric_reconstruction_v3_comparison.csv")


if __name__ == "__main__":
    main()
