from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PREFIX = "inverse_reconstruction_v2"
ROW_LEVEL_DIR = ROOT / "outputs" / "row_level_predictions" / "paper_exact_attempt_movielens_v2_allcycles"
TARGET_PATH = (
    ROOT
    / "outputs"
    / "tables"
    / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
    / "tau1_figure4_digitized_targets.csv"
)
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

MAINLINE_DU = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
DU_METHODS = [MAINLINE_DU, "lambda0p3_diagnostic"]
LINEAR_METHODS = ["linucb_concat", "linucb_mf_interaction"]
REQUIRED_METHODS = [MAINLINE_DU, "linucb_concat", "linucb_mf_interaction", "lambda0p3_diagnostic"]
REQUIRED_SEEDS = list(range(20))
REQUIRED_CYCLES = list(range(21))

TOP50_PATH = TABLE_DIR / f"{PREFIX}_metric_curve_family_top50.csv"
ALIGNMENT_PATH = TABLE_DIR / f"{PREFIX}_metric_curve_alignment.csv"
COMPLETENESS_PATH = TABLE_DIR / f"{PREFIX}_allcycle_row_level_completeness.csv"
CURVES_PATH = TABLE_DIR / f"{PREFIX}_metric_curves_by_method.csv"
REPORT_PATH = REPORT_DIR / f"{PREFIX}_metric_curve_audit.md"


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


def ci95(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def rmse(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr * arr)))


def safe_weight(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    if float(arr.sum()) <= 0.0:
        return np.ones_like(arr, dtype=float)
    return arr


def weighted_rmse(err: np.ndarray, weight: pd.Series | np.ndarray) -> float:
    weight_arr = safe_weight(weight)
    return float(np.sqrt(np.sum(weight_arr * err * err) / np.sum(weight_arr)))


def weighted_norm_rmse(err: np.ndarray, weight: pd.Series | np.ndarray) -> float:
    weight_arr = safe_weight(weight)
    return float(np.sqrt(np.mean((weight_arr * err) ** 2)))


def metric_values(group: pd.DataFrame) -> dict[str, float]:
    truth = group["y_true"].to_numpy(dtype=float)
    err = group["err_clipped"].to_numpy(dtype=float)
    err_raw = group["err_raw"].to_numpy(dtype=float)
    pred = group["y_pred_clipped"].to_numpy(dtype=float)
    pos = np.maximum(truth - 3.0, 0.0)
    pos_eps = pos + 0.1
    reward = group["reward"].to_numpy(dtype=float)
    item_popularity = group["item_popularity"].to_numpy(dtype=float)
    n = max(len(group), 1)
    return {
        "standard_rmse": float(np.sqrt(np.mean(err * err))),
        "pdf_literal_norm_rmse": float(np.sqrt(np.sum(err * err)) / n),
        "current_w_rmse": weighted_rmse(err, pos_eps),
        "mild_reward_w_rmse": weighted_rmse(err, 1.0 + 0.5 * pos),
        "medium_reward_w_rmse": weighted_rmse(err, 1.0 + pos),
        "sigmoid_w_rmse": weighted_rmse(err, reward),
        "reward_tail_proxy": weighted_norm_rmse(err, pos_eps),
        "rating_weighted_rmse": weighted_rmse(err, truth),
        "inverse_popularity_weighted_rmse": weighted_rmse(err, 1.0 / np.maximum(item_popularity, 1.0)),
        "unclipped_standard_rmse": float(np.sqrt(np.mean(err_raw * err_raw))),
        "centered_rating_rmse": float(np.sqrt(np.mean(((truth - 3.0) - (pred - 3.0)) ** 2))),
        "0_1_normalized_rmse": float(np.sqrt(np.mean((err / 4.0) ** 2))),
    }


TEXT_DEFENSIBILITY = {
    "standard_rmse": True,
    "pdf_literal_norm_rmse": True,
    "current_w_rmse": True,
    "mild_reward_w_rmse": True,
    "medium_reward_w_rmse": True,
    "sigmoid_w_rmse": True,
    "reward_tail_proxy": True,
    "rating_weighted_rmse": True,
    "inverse_popularity_weighted_rmse": False,
    "unclipped_standard_rmse": True,
    "centered_rating_rmse": True,
    "0_1_normalized_rmse": False,
}


def row_level_files() -> list[Path]:
    if not ROW_LEVEL_DIR.exists():
        return []
    return sorted([*ROW_LEVEL_DIR.glob("*.csv"), *ROW_LEVEL_DIR.glob("*.csv.gz")])


def build_metric_long() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in row_level_files():
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame = frame[frame["selected_or_test"].astype(str).str.lower() == "test"].copy()
        if frame.empty:
            continue
        method = str(frame["method"].iloc[0])
        seed = int(frame["seed"].iloc[0]) if "seed" in frame.columns else int(path.stem.split("_")[-1])
        for cycle, group in frame.groupby("cycle", sort=False):
            values = metric_values(group)
            for metric, value in values.items():
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "cycle": int(cycle),
                        "metric": metric,
                        "value": float(value),
                        "row_level_file": path.name,
                        "rows": int(len(group)),
                    }
                )
    return pd.DataFrame(rows)


def completeness(metric_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in REQUIRED_METHODS:
        subset = metric_long[metric_long["method"] == method] if not metric_long.empty else pd.DataFrame()
        seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique()) if not subset.empty else []
        cycles = sorted(int(cycle) for cycle in subset["cycle"].dropna().unique()) if not subset.empty else []
        missing_seeds = [seed for seed in REQUIRED_SEEDS if seed not in seeds]
        missing_cycles = [cycle for cycle in REQUIRED_CYCLES if cycle not in cycles]
        complete = len(missing_seeds) == 0 and len(missing_cycles) == 0
        rows.append(
            {
                "method": method,
                "seed_count": len(seeds),
                "available_seeds": " ".join(map(str, seeds)),
                "missing_seeds": " ".join(map(str, missing_seeds)),
                "cycle_count": len(cycles),
                "available_cycles": " ".join(map(str, cycles)),
                "missing_cycles": " ".join(map(str, missing_cycles)),
                "status": "complete_20seed_21cycle" if complete else ("partial" if len(seeds) else "missing"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_curves(metric_long: pd.DataFrame) -> pd.DataFrame:
    if metric_long.empty:
        return pd.DataFrame()
    return (
        metric_long.groupby(["method", "metric", "cycle"], as_index=False)
        .agg(
            local_value=("value", "mean"),
            local_value_ci95=("value", ci95),
            seed_count=("seed", "nunique"),
            row_count=("rows", "sum"),
        )
        .sort_values(["method", "metric", "cycle"])
    )


def load_targets() -> pd.DataFrame:
    if not TARGET_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TARGET_PATH)


def align_metric(curves: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if curves.empty or targets.empty:
        return pd.DataFrame()
    paper_a = targets[(targets["figure"] == "Figure 4a") & (targets["method"].isin(["DU-Net", "Linear UCB"]))].copy()
    if paper_a.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for du_method in DU_METHODS:
        for linear_method in LINEAR_METHODS:
            for metric in sorted(curves["metric"].dropna().unique()):
                du_curve = curves[(curves["method"] == du_method) & (curves["metric"] == metric)][
                    ["cycle", "local_value", "seed_count"]
                ].rename(columns={"local_value": "du_local", "seed_count": "du_seed_count"})
                linear_curve = curves[(curves["method"] == linear_method) & (curves["metric"] == metric)][
                    ["cycle", "local_value", "seed_count"]
                ].rename(columns={"local_value": "linear_local", "seed_count": "linear_seed_count"})
                paper_du = paper_a[paper_a["method"] == "DU-Net"][["cycle", "paper_value"]].rename(
                    columns={"paper_value": "du_paper"}
                )
                paper_linear = paper_a[paper_a["method"] == "Linear UCB"][["cycle", "paper_value"]].rename(
                    columns={"paper_value": "linear_paper"}
                )
                merged = du_curve.merge(linear_curve, on="cycle", how="inner")
                merged = merged.merge(paper_du, on="cycle", how="inner").merge(paper_linear, on="cycle", how="inner")
                if merged.empty:
                    continue
                merged = merged.sort_values("cycle")
                final = merged.tail(1).iloc[0]
                local_gap = float(final["linear_local"] - final["du_local"])
                paper_gap = float(final["linear_paper"] - final["du_paper"])
                fig4a_du_curve_error = rmse(merged["du_local"] - merged["du_paper"])
                fig4a_linear_curve_error = rmse(merged["linear_local"] - merged["linear_paper"])
                final_gap_error = abs(local_gap - paper_gap)
                endpoint_scale_error = float(
                    np.mean(
                        [
                            abs(float(final["du_local"]) - float(final["du_paper"])),
                            abs(float(final["linear_local"]) - float(final["linear_paper"])),
                        ]
                    )
                )
                direction_match = local_gap > 0.0
                text_defensible = bool(TEXT_DEFENSIBILITY.get(str(metric), False))
                complete_seed_count = int(min(merged["du_seed_count"].min(), merged["linear_seed_count"].min()))
                score = (
                    fig4a_du_curve_error
                    + fig4a_linear_curve_error
                    + 5.0 * final_gap_error
                    + endpoint_scale_error
                    + (0.0 if direction_match else 1.0)
                    + (0.0 if text_defensible else 0.25)
                )
                gate_pass = (
                    fig4a_du_curve_error <= 0.05
                    and fig4a_linear_curve_error <= 0.05
                    and final_gap_error <= 0.01
                    and endpoint_scale_error <= 0.05
                    and direction_match
                    and text_defensible
                    and complete_seed_count >= 20
                    and len(merged) >= 21
                )
                rows.append(
                    {
                        "du_method": du_method,
                        "linear_method": linear_method,
                        "metric": metric,
                        "figure4a_du_curve_error": fig4a_du_curve_error,
                        "figure4a_linear_curve_error": fig4a_linear_curve_error,
                        "final_gap_error": final_gap_error,
                        "endpoint_scale_error": endpoint_scale_error,
                        "local_final_gap": local_gap,
                        "paper_final_gap": paper_gap,
                        "direction_match": direction_match,
                        "text_defensibility": "defensible" if text_defensible else "weak",
                        "seed_count": complete_seed_count,
                        "cycle_count": int(len(merged)),
                        "curve_score": score,
                        "paper_wrmse_reconstructed_candidate": "yes" if gate_pass else "no",
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values(["curve_score", "metric"]).reset_index(drop=True) if not out.empty else out


def write_report(completion: pd.DataFrame, alignment: pd.DataFrame, top50: pd.DataFrame, curves: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    complete = bool((completion["status"] == "complete_20seed_21cycle").all()) if not completion.empty else False
    promoted = bool((top50.get("paper_wrmse_reconstructed_candidate", pd.Series(dtype=str)) == "yes").any())
    lines = [
        "# inverse_reconstruction_v2 metric curve audit",
        "",
        "Scope: all-cycle row-level metric reconstruction for Figure 4a. This upgrades the previous endpoint-only inverse reconstruction once the all-cycle shards are present.",
        "",
        f"- Row-level directory: `{ROW_LEVEL_DIR.relative_to(ROOT).as_posix()}`",
        f"- Target curves: `{TARGET_PATH.relative_to(ROOT).as_posix()}`",
        f"- All-cycle completeness: `{'complete' if complete else 'incomplete'}`",
        f"- Metric gate promotion: `{'yes' if promoted else 'no'}`",
        "",
        "## All-Cycle Row-Level Completeness",
        "",
        md_table(completion),
        "",
        "## Top Metric Curve Candidates",
        "",
        md_table(top50, max_rows=50),
        "",
        "## Alignment Rows",
        "",
        md_table(alignment.head(30)),
        "",
        "## Decision",
        "",
        "A metric is promoted only if curve shape, endpoint scale, final gap, direction, text defensibility, 20 seeds, and 21 cycles all pass. If no metric is promoted after completeness is achieved, the metric gate remains failed with all-cycle evidence.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all() -> dict[str, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    metric_long = build_metric_long()
    completion = completeness(metric_long)
    curves = aggregate_curves(metric_long)
    alignment = align_metric(curves, load_targets())
    top50 = alignment.head(50).copy() if not alignment.empty else alignment
    completion.to_csv(COMPLETENESS_PATH, index=False)
    curves.to_csv(CURVES_PATH, index=False)
    alignment.to_csv(ALIGNMENT_PATH, index=False)
    top50.to_csv(TOP50_PATH, index=False)
    write_report(completion, alignment, top50, curves)
    return {
        "completion": completion,
        "curves": curves,
        "alignment": alignment,
        "top50": top50,
    }


def main() -> None:
    outputs = build_all()
    for path in [COMPLETENESS_PATH, CURVES_PATH, ALIGNMENT_PATH, TOP50_PATH, REPORT_PATH]:
        print(path)
    print(outputs["completion"].to_string(index=False))
    print(outputs["top50"].head(10).to_string(index=False) if not outputs["top50"].empty else "No metric rows yet.")


if __name__ == "__main__":
    main()
