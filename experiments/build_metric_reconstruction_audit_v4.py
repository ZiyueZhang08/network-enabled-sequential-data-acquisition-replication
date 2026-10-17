from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_exact_attempt_movielens_v2"
ROW_LEVEL_DIR = ROOT / "outputs" / "row_level_predictions" / PREFIX
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"

DU = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
COMPARATORS = ["linucb_concat", "linucb_mf_interaction", "linucb_bias_interaction"]
EXACT_REQUIRED_METHODS = [
    DU,
    *COMPARATORS,
]
DIAGNOSTIC_METHODS = [
    "schur_loglambda_tuned_full_pool_softmax_tau0p3",
    "schur_tuned_topk",
    "lambda0p3_diagnostic",
]
REQUIRED_METHODS = [
    *EXACT_REQUIRED_METHODS,
    *DIAGNOSTIC_METHODS,
]
REQUIRED_SEEDS = list(range(20))

PAPER_FIG4A_DU = 1.788
PAPER_FIG4A_LINEAR = 1.814
PAPER_GAP = PAPER_FIG4A_LINEAR - PAPER_FIG4A_DU

COMPLETENESS_PATH = TABLE_DIR / "row_level_completeness_v2.csv"
METRIC_LONG_PATH = TABLE_DIR / "metric_reconstruction_v4_allseeds.csv"
METRIC_SUMMARY_PATH = TABLE_DIR / "metric_reconstruction_v4_summary.csv"
COMPARISON_PATH = TABLE_DIR / "metric_reconstruction_v4_comparison.csv"
CURVE_PATH = TABLE_DIR / "metric_curve_alignment_v4.csv"
FIGURE_PATH = FIGURE_DIR / "metric_curve_alignment_v4.png"
REPORT_PATH = REPORT_DIR / "metric_reconstruction_audit_v4.md"


def md_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    if max_rows is not None:
        view = view.head(max_rows)
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


def ci95(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def safe_weight(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    if float(np.sum(arr)) <= 0.0:
        return np.ones_like(arr, dtype=float)
    return arr


def weighted_rmse(err: np.ndarray, weight: np.ndarray) -> float:
    weight = safe_weight(weight)
    return float(np.sqrt(np.sum(weight * err * err) / np.sum(weight)))


def weighted_norm_rmse(err: np.ndarray, weight: np.ndarray) -> float:
    weight = safe_weight(weight)
    return float(np.sqrt(np.mean((weight * err) ** 2)))


def load_predictions() -> pd.DataFrame:
    files = sorted(ROW_LEVEL_DIR.glob("*.csv"))
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True, sort=False)
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
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Row-level predictions missing required v4 columns: {missing}")
    return predictions


def completion_status(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in REQUIRED_METHODS:
        required_for_metric_gate = method in EXACT_REQUIRED_METHODS
        subset = predictions[predictions["method"] == method] if not predictions.empty else pd.DataFrame()
        seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique()) if not subset.empty else []
        cycles = sorted(int(cycle) for cycle in subset["cycle"].dropna().unique()) if not subset.empty else []
        test_rows = (
            subset[subset["selected_or_test"].astype(str).str.lower() == "test"]
            if not subset.empty
            else pd.DataFrame()
        )
        missing_seeds = [seed for seed in REQUIRED_SEEDS if seed not in seeds]
        if len(seeds) >= len(REQUIRED_SEEDS) and 20 in cycles:
            status = "complete_20seed"
        elif len(seeds) > 0:
            status = "partial"
        else:
            status = "missing"
        rows.append(
            {
                "method": method,
                "gate_role": "metric_gate_required" if required_for_metric_gate else "diagnostic_tracked",
                "required_for_metric_gate": required_for_metric_gate,
                "available_seed_count": len(seeds),
                "available_seeds": " ".join(str(seed) for seed in seeds),
                "missing_seeds": " ".join(str(seed) for seed in missing_seeds),
                "available_cycles": " ".join(str(cycle) for cycle in cycles),
                "test_row_count": int(len(test_rows)),
                "required_seed_count": len(REQUIRED_SEEDS),
                "row_level_scope": "final_cycle_only" if cycles == [20] else ("multi_cycle" if cycles else "none"),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def add_metric(
    rows: list[dict[str, object]],
    group: pd.DataFrame,
    metric: str,
    definition: str,
    value: float,
) -> None:
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
    reward = group["reward"].to_numpy(dtype=float)
    rating_pos_eps = np.maximum(truth - 3.0, 0.0) + 0.1
    rating_pos = np.maximum(truth - 3.0, 0.0)

    rows: list[dict[str, object]] = []
    add_metric(rows, group, "standard_rmse", "sqrt(mean(err_clipped^2))", np.sqrt(np.mean(err * err)))
    add_metric(rows, group, "pdf_literal_norm_rmse", "sqrt(sum(err_clipped^2)) / N", np.sqrt(np.sum(err * err)) / n)
    add_metric(rows, group, "plain_l2_norm", "sqrt(sum(err_clipped^2))", np.sqrt(np.sum(err * err)))
    add_metric(rows, group, "current_w_rmse", "sqrt(sum((max(y-3,0)+0.1)*err_clipped^2)/sum(w))", weighted_rmse(err, rating_pos_eps))
    add_metric(rows, group, "mild_reward_w_rmse", "sqrt(sum((1+0.5*max(y-3,0))*err_clipped^2)/sum(w))", weighted_rmse(err, 1.0 + 0.5 * rating_pos))
    add_metric(rows, group, "medium_reward_w_rmse", "sqrt(sum((1+max(y-3,0))*err_clipped^2)/sum(w))", weighted_rmse(err, 1.0 + rating_pos))
    add_metric(rows, group, "sigmoid_reward_w_rmse", "sqrt(sum(sigmoid(y-3)*err_clipped^2)/sum(w))", weighted_rmse(err, reward))
    add_metric(rows, group, "reward_tail_proxy", "sqrt(mean(((max(y-3,0)+0.1)*err_clipped)^2))", weighted_norm_rmse(err, rating_pos_eps))
    add_metric(rows, group, "rating_weighted_rmse", "sqrt(sum(y*err_clipped^2)/sum(y))", weighted_rmse(err, truth))
    add_metric(rows, group, "rating_normalized_weighted_rmse", "sqrt(sum((y/mean(y))*err_clipped^2)/sum(w))", weighted_rmse(err, truth / max(float(np.mean(truth)), 1e-12)))
    add_metric(rows, group, "centered_rating_rmse", "sqrt(mean(((rating-3)-(pred-3))^2))", np.sqrt(np.mean(((truth - 3.0) - (pred - 3.0)) ** 2)))
    add_metric(rows, group, "unclipped_standard_rmse", "sqrt(mean(err_raw^2))", np.sqrt(np.mean(err_raw * err_raw)))
    add_metric(rows, group, "0_1_normalized_rmse", "sqrt(mean((err_clipped/4)^2))", np.sqrt(np.mean((err / 4.0) ** 2)))
    add_metric(rows, group, "item_popularity_weighted_rmse", "sqrt(sum(pop*err_clipped^2)/sum(pop))", weighted_rmse(err, group["item_popularity"]))
    add_metric(rows, group, "inverse_item_popularity_weighted_rmse", "sqrt(sum((1/max(pop,1))*err_clipped^2)/sum(w))", weighted_rmse(err, 1.0 / np.maximum(group["item_popularity"].to_numpy(dtype=float), 1.0)))
    add_metric(rows, group, "user_degree_weighted_rmse", "sqrt(sum(user_degree*err_clipped^2)/sum(w))", weighted_rmse(err, group["user_degree"]))
    add_metric(rows, group, "raw_prediction_standard_rmse", "sqrt(mean((y-y_pred_raw)^2))", np.sqrt(np.mean((truth - pred_raw) ** 2)))
    return rows


def compute_metric_grid(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["method", "seed", "cycle", "metric", "definition", "value", "rows"])
    test = predictions[predictions["selected_or_test"].astype(str).str.lower() == "test"].copy()
    test = test[test["cycle"] == test["cycle"].max()] if not test.empty else test
    rows: list[dict[str, object]] = []
    for _, group in test.groupby(["method", "seed", "cycle"], sort=False):
        rows.extend(metric_rows(group))
    return pd.DataFrame(rows)


def summarize(metric_grid: pd.DataFrame) -> pd.DataFrame:
    if metric_grid.empty:
        return pd.DataFrame(columns=["method", "metric", "definition", "value", "value_ci95", "seed_count", "rows"])
    return (
        metric_grid.groupby(["method", "metric", "definition"], as_index=False)
        .agg(value=("value", "mean"), value_ci95=("value", ci95), seed_count=("seed", "nunique"), rows=("rows", "mean"))
        .sort_values(["metric", "method"])
    )


def compare_to_paper(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "comparator",
                "DU-Net endpoint",
                "Linear UCB endpoint",
                "gap",
                "direction",
                "scale_distance_to_paper",
                "gap_distance_to_paper",
                "definition_defensible_from_text",
                "row_level_seed_count",
                "raw_match",
                "paper_wrmse_reconstructed",
            ]
        )

    defensible_metrics = {
        "standard_rmse",
        "pdf_literal_norm_rmse",
        "current_w_rmse",
        "mild_reward_w_rmse",
        "medium_reward_w_rmse",
        "sigmoid_reward_w_rmse",
        "reward_tail_proxy",
        "rating_weighted_rmse",
        "rating_normalized_weighted_rmse",
        "centered_rating_rmse",
        "unclipped_standard_rmse",
        "0_1_normalized_rmse",
        "item_popularity_weighted_rmse",
        "inverse_item_popularity_weighted_rmse",
    }
    rows: list[dict[str, object]] = []
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
            gap_distance = float(abs(local_gap - PAPER_GAP) / max(abs(PAPER_GAP), 1e-12))
            defensible = metric in defensible_metrics
            seed_count = int(min(du["seed_count"].iloc[0], base["seed_count"].iloc[0]))
            pass_gate = scale_distance <= 0.15 and gap_distance <= 0.20 and local_gap > 0 and defensible and seed_count >= 20
            rows.append(
                {
                    "metric": metric,
                    "comparator": comparator,
                    "DU-Net endpoint": du_value,
                    "Linear UCB endpoint": base_value,
                    "gap": local_gap,
                    "direction": "DU-Net < Linear UCB" if local_gap > 0 else "not_matched",
                    "scale_distance_to_paper": scale_distance,
                    "gap_distance_to_paper": gap_distance,
                    "definition_defensible_from_text": defensible,
                    "row_level_seed_count": seed_count,
                    "raw_match": "pass" if pass_gate else "fail",
                    "paper_wrmse_reconstructed": "yes" if pass_gate else "no",
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["paper_wrmse_reconstructed", "scale_distance_to_paper", "gap_distance_to_paper"],
        ascending=[False, True, True],
    )


def load_run_metrics() -> pd.DataFrame:
    path = TABLE_DIR / f"{PREFIX}_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_targets() -> pd.DataFrame:
    path = TABLE_DIR / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1" / "tau1_figure4_digitized_targets.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def curve_alignment(run_metrics: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "standard_mask_rmse",
        "pdf_literal_norm_rmse",
        "w_rmse",
        "mild_reward_w_rmse",
        "medium_reward_w_rmse",
        "sigmoid_reward_w_rmse",
        "reward_tail_proxy_rmse",
        "cumulative_reward",
    ]
    if run_metrics.empty or targets.empty:
        return pd.DataFrame(
            [
                {
                    "status": "not_testable_no_v2_metrics_or_targets",
                    "figure": "",
                    "metric": "",
                    "comparator": "",
                    "curve_rmse": np.nan,
                    "endpoint_abs_error_mean": np.nan,
                    "gap_abs_error": np.nan,
                }
            ]
        )

    target_map = {
        "Figure 4a": targets[targets["figure"] == "Figure 4a"],
        "Figure 4b": targets[targets["figure"] == "Figure 4b"],
    }
    local = run_metrics.groupby(["variant", "cycle"], as_index=False)[metric_cols].mean(numeric_only=True)
    rows: list[dict[str, object]] = []
    for comparator in COMPARATORS:
        variant_to_paper = {DU: "DU-Net", comparator: "Linear UCB"}
        for figure, target in target_map.items():
            if target.empty:
                continue
            local_metrics = ["cumulative_reward"] if figure == "Figure 4b" else [col for col in metric_cols if col != "cumulative_reward"]
            for metric in local_metrics:
                if metric not in local.columns:
                    continue
                diffs: list[float] = []
                endpoint_errors: list[float] = []
                local_endpoint: dict[str, float] = {}
                paper_endpoint: dict[str, float] = {}
                for variant, paper_method in variant_to_paper.items():
                    local_part = local[local["variant"] == variant][["cycle", metric]].rename(columns={metric: "local_value"})
                    paper_part = target[target["method"] == paper_method][["cycle", "paper_value"]]
                    merged = pd.merge(local_part, paper_part, on="cycle", how="inner")
                    if merged.empty:
                        continue
                    diffs.extend((merged["local_value"] - merged["paper_value"]).to_numpy(dtype=float).tolist())
                    end = merged.sort_values("cycle").tail(1).iloc[0]
                    endpoint_errors.append(float(abs(end["local_value"] - end["paper_value"])))
                    local_endpoint[paper_method] = float(end["local_value"])
                    paper_endpoint[paper_method] = float(end["paper_value"])
                if not diffs or "DU-Net" not in local_endpoint or "Linear UCB" not in local_endpoint:
                    continue
                local_gap = local_endpoint["Linear UCB"] - local_endpoint["DU-Net"]
                paper_gap = paper_endpoint["Linear UCB"] - paper_endpoint["DU-Net"]
                rows.append(
                    {
                        "status": "computed",
                        "figure": figure,
                        "metric": metric,
                        "comparator": comparator,
                        "curve_rmse": float(np.sqrt(np.mean(np.asarray(diffs, dtype=float) ** 2))),
                        "endpoint_abs_error_mean": float(np.mean(endpoint_errors)),
                        "local_gap": local_gap,
                        "paper_gap": paper_gap,
                        "gap_abs_error": float(abs(local_gap - paper_gap)),
                        "points": len(diffs),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            [
                {
                    "status": "not_testable_no_overlapping_curves",
                    "figure": "",
                    "metric": "",
                    "comparator": "",
                    "curve_rmse": np.nan,
                    "endpoint_abs_error_mean": np.nan,
                    "gap_abs_error": np.nan,
                }
            ]
        )
    return out.sort_values(["figure", "curve_rmse", "gap_abs_error"])


def write_plot(curves: pd.DataFrame) -> str:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return f"plot_not_written: {exc}"

    fig, ax = plt.subplots(figsize=(9, 4.8))
    computed = curves[curves.get("status", "") == "computed"].copy() if "status" in curves else pd.DataFrame()
    if computed.empty:
        ax.text(0.5, 0.5, "No v2 curve metrics yet", ha="center", va="center", fontsize=14)
        ax.set_axis_off()
    else:
        view = computed.head(12).copy()
        labels = [f"{row.figure}\n{row.metric}\n{row.comparator}" for row in view.itertuples(index=False)]
        ax.bar(np.arange(len(view)), view["curve_rmse"].to_numpy(dtype=float), color="#4C78A8")
        ax.set_xticks(np.arange(len(view)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("curve RMSE to visual target")
        ax.set_title("Metric curve alignment v4")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)
    return str(FIGURE_PATH.relative_to(ROOT)).replace("\\", "/")


def write_report(
    completion: pd.DataFrame,
    metric_grid: pd.DataFrame,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    curves: pd.DataFrame,
    plot_status: str,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    required_completion = (
        completion[completion["required_for_metric_gate"].astype(bool)]
        if "required_for_metric_gate" in completion.columns
        else completion
    )
    complete = bool((required_completion["status"] == "complete_20seed").all()) if not required_completion.empty else False
    metric_pass = bool((comparison["paper_wrmse_reconstructed"] == "yes").any()) if not comparison.empty else False
    gate = "pass" if complete and metric_pass else "fail"
    missing = required_completion[required_completion["status"] != "complete_20seed"] if not required_completion.empty else required_completion
    diagnostic = (
        completion[~completion["required_for_metric_gate"].astype(bool)]
        if "required_for_metric_gate" in completion.columns
        else pd.DataFrame()
    )

    lines = [
        "# Metric Reconstruction Audit v4",
        "",
        f"- Prefix: `{PREFIX}`",
        f"- Row-level source: `{ROW_LEVEL_DIR.relative_to(ROOT).as_posix()}`",
        f"- Figure 4a visual endpoints: DU-Net `{PAPER_FIG4A_DU}`, Linear UCB `{PAPER_FIG4A_LINEAR}`",
        f"- Gate status: **{gate}**",
        f"- Plot: `{plot_status}`",
        "",
        "## Row-Level Completeness",
        "",
        md_table(completion),
        "",
        "## Missing Work Queue",
        "",
        md_table(missing[["method", "available_seed_count", "missing_seeds", "status"]] if not missing.empty else missing),
        "",
        "## Diagnostic Branch Tracking",
        "",
        md_table(diagnostic[["method", "gate_role", "available_seed_count", "missing_seeds", "status"]] if not diagnostic.empty else diagnostic),
        "",
        "## Metric Candidates",
        "",
        f"- Row-level metric rows: `{len(metric_grid)}`",
        f"- Summary rows: `{len(summary)}`",
        "",
        md_table(comparison.head(25)),
        "",
        "## Curve Alignment",
        "",
        md_table(curves.head(20)),
        "",
        "## Decision",
        "",
        "No candidate is promoted to `paper_wrmse_reconstructed` unless row-level completeness is 20 seeds for the metric-gate-required methods and endpoint scale, gap, direction, and text-defensible formula all pass. Diagnostic branches are tracked but cannot promote or block the Figure 4a metric gate.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    completion = completion_status(predictions)
    metric_grid = compute_metric_grid(predictions)
    summary = summarize(metric_grid)
    comparison = compare_to_paper(summary)
    curves = curve_alignment(load_run_metrics(), load_targets())
    plot_status = write_plot(curves)

    completion.to_csv(COMPLETENESS_PATH, index=False)
    metric_grid.to_csv(METRIC_LONG_PATH, index=False)
    summary.to_csv(METRIC_SUMMARY_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    curves.to_csv(CURVE_PATH, index=False)
    write_report(completion, metric_grid, summary, comparison, curves, plot_status)

    print(REPORT_PATH)
    print(COMPLETENESS_PATH)
    print(METRIC_LONG_PATH)
    print(CURVE_PATH)
    print(plot_status)


if __name__ == "__main__":
    main()
