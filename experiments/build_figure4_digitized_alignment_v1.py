from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
TABLE_ROOT = ROOT / "outputs" / "tables" / PREFIX
FIGURE_ROOT = ROOT / "outputs" / "figures" / PREFIX
REPORT_PATH = ROOT / "reports" / f"{PREFIX}_figure4_digitized_alignment_v1.md"

SOURCE_TAU1 = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_20seeds"
SOURCE_COMPARATORS = "paper_faithful_movielens_comparator_sensitivity_v1_20seeds"

TAU1 = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
CONCAT = "linucb_concat"


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
            formatted[col] = formatted[col].astype(str)
    lines = [
        "| " + " | ".join(str(col) for col in formatted.columns) + " |",
        "| " + " | ".join(["---"] * len(formatted.columns)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in formatted.columns) + " |")
    return "\n".join(lines)


def paper_targets() -> pd.DataFrame:
    cycles = list(range(21))
    fig4a_du = [
        1.925,
        1.885,
        1.862,
        1.846,
        1.836,
        1.828,
        1.820,
        1.814,
        1.809,
        1.806,
        1.804,
        1.802,
        1.800,
        1.798,
        1.796,
        1.794,
        1.792,
        1.791,
        1.790,
        1.789,
        1.788,
    ]
    fig4a_linear = [
        1.925,
        1.910,
        1.895,
        1.880,
        1.866,
        1.855,
        1.847,
        1.841,
        1.836,
        1.832,
        1.828,
        1.825,
        1.822,
        1.820,
        1.818,
        1.817,
        1.816,
        1.815,
        1.8145,
        1.814,
        1.814,
    ]
    fig4b_du = [
        0.0,
        62.0,
        128.0,
        198.0,
        270.0,
        345.0,
        420.0,
        495.0,
        570.0,
        645.0,
        715.0,
        785.0,
        855.0,
        925.0,
        990.0,
        1055.0,
        1120.0,
        1175.0,
        1225.0,
        1275.0,
        1325.0,
    ]
    fig4b_linear = [
        0.0,
        70.0,
        142.0,
        215.0,
        290.0,
        365.0,
        440.0,
        515.0,
        590.0,
        668.0,
        746.0,
        824.0,
        902.0,
        980.0,
        1058.0,
        1136.0,
        1214.0,
        1290.0,
        1362.0,
        1378.0,
        1390.0,
    ]
    specs = [
        ("Figure 4a", "DU-Net", "w_rmse", fig4a_du, "tmp/pdfs/figure4a_movielens_wrmse.png"),
        ("Figure 4a", "Linear UCB", "w_rmse", fig4a_linear, "tmp/pdfs/figure4a_movielens_wrmse.png"),
        ("Figure 4b", "DU-Net", "cumulative_reward", fig4b_du, "tmp/pdfs/figure4b_movielens_cumulative_crop.png"),
        ("Figure 4b", "Linear UCB", "cumulative_reward", fig4b_linear, "tmp/pdfs/figure4b_movielens_cumulative_crop.png"),
    ]
    rows = []
    for figure, method, paper_metric, values, source in specs:
        for cycle, value in zip(cycles, values):
            rows.append(
                {
                    "figure": figure,
                    "method": method,
                    "cycle": cycle,
                    "paper_metric": paper_metric,
                    "paper_value": float(value),
                    "digitization_source": source,
                    "digitization_status": "visual_approximation_not_author_data",
                }
            )
    return pd.DataFrame(rows)


def source_metrics() -> pd.DataFrame:
    tau1 = pd.read_csv(ROOT / "outputs" / "tables" / f"{SOURCE_TAU1}_metrics.csv")
    comparators = pd.read_csv(ROOT / "outputs" / "tables" / f"{SOURCE_COMPARATORS}_metrics.csv")
    comparators = comparators[comparators["variant"] == CONCAT].copy()
    frame = pd.concat([tau1, comparators], ignore_index=True, sort=False)
    frame = frame[frame["variant"].isin([TAU1, CONCAT])].copy()
    frame["method"] = frame["variant"].map({TAU1: "DU-Net", CONCAT: "Linear UCB"})
    return frame


def local_curves(metrics: pd.DataFrame) -> pd.DataFrame:
    available_metrics = ["w_rmse", "standard_mask_rmse", "cumulative_reward"]
    rows = []
    for metric_name in available_metrics:
        if metric_name not in metrics.columns:
            continue
        grouped = (
            metrics.groupby(["method", "cycle"], as_index=False)
            .agg(
                local_value=(metric_name, "mean"),
                local_ci95=(metric_name, ci95),
                seed_count=("seed", "nunique"),
            )
            .sort_values(["method", "cycle"])
        )
        grouped["local_metric"] = metric_name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True, sort=False)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1 or float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def affine_fit(local: np.ndarray, paper: np.ndarray) -> tuple[float, float, np.ndarray, float]:
    design = np.column_stack([np.ones(len(local)), local])
    intercept, slope = np.linalg.lstsq(design, paper, rcond=None)[0]
    fitted = intercept + slope * local
    rmse = float(np.sqrt(np.mean((fitted - paper) ** 2)))
    return float(intercept), float(slope), fitted, rmse


def alignment_rows(targets: pd.DataFrame, local: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [
        ("Figure 4a", "w_rmse"),
        ("Figure 4a", "standard_mask_rmse"),
        ("Figure 4b", "cumulative_reward"),
    ]
    merged_rows = []
    summary_rows = []
    endpoint_rows = []
    for figure, local_metric in specs:
        paper_subset = targets[targets["figure"] == figure].copy()
        local_subset = local[local["local_metric"] == local_metric].copy()
        merged = paper_subset.merge(local_subset, on=["method", "cycle"], how="inner")
        merged["local_metric"] = local_metric
        merged["paper_minus_local"] = merged["paper_value"] - merged["local_value"]
        merged_rows.append(merged)
        for method, group in merged.groupby("method"):
            group = group.sort_values("cycle")
            local_values = group["local_value"].to_numpy(dtype=float)
            paper_values = group["paper_value"].to_numpy(dtype=float)
            intercept, slope, _, affine_rmse = affine_fit(local_values, paper_values)
            summary_rows.append(
                {
                    "figure": figure,
                    "method": method,
                    "local_metric": local_metric,
                    "cycles": int(len(group)),
                    "endpoint_local": float(group[group["cycle"] == group["cycle"].max()]["local_value"].iloc[0]),
                    "endpoint_paper_visual": float(group[group["cycle"] == group["cycle"].max()]["paper_value"].iloc[0]),
                    "endpoint_abs_error": float(
                        abs(
                            group[group["cycle"] == group["cycle"].max()]["local_value"].iloc[0]
                            - group[group["cycle"] == group["cycle"].max()]["paper_value"].iloc[0]
                        )
                    ),
                    "raw_curve_rmse": float(np.sqrt(np.mean((local_values - paper_values) ** 2))),
                    "curve_correlation": correlation(local_values, paper_values),
                    "affine_intercept_to_paper": intercept,
                    "affine_slope_to_paper": slope,
                    "affine_curve_rmse": affine_rmse,
                }
            )
        endpoint = merged[merged["cycle"] == merged["cycle"].max()].copy()
        du = endpoint[endpoint["method"] == "DU-Net"]
        linear = endpoint[endpoint["method"] == "Linear UCB"]
        if not du.empty and not linear.empty:
            local_gap = float(linear["local_value"].iloc[0] - du["local_value"].iloc[0])
            paper_gap = float(linear["paper_value"].iloc[0] - du["paper_value"].iloc[0])
            endpoint_rows.append(
                {
                    "figure": figure,
                    "local_metric": local_metric,
                    "du_endpoint_local": float(du["local_value"].iloc[0]),
                    "linear_endpoint_local": float(linear["local_value"].iloc[0]),
                    "du_endpoint_paper_visual": float(du["paper_value"].iloc[0]),
                    "linear_endpoint_paper_visual": float(linear["paper_value"].iloc[0]),
                    "local_gap_linear_minus_du": local_gap,
                    "paper_gap_linear_minus_du": paper_gap,
                    "gap_abs_error": abs(local_gap - paper_gap),
                    "direction_match": bool(local_gap > 0 and paper_gap > 0),
                }
            )
    return (
        pd.concat(merged_rows, ignore_index=True, sort=False),
        pd.DataFrame(summary_rows).sort_values(["figure", "local_metric", "method"]),
        pd.DataFrame(endpoint_rows).sort_values(["figure", "local_metric"]),
    )


def plot_alignment(merged: pd.DataFrame) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for (figure, local_metric), group in merged.groupby(["figure", "local_metric"]):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method, method_group in group.groupby("method"):
            method_group = method_group.sort_values("cycle")
            ax.plot(method_group["cycle"], method_group["paper_value"], linestyle="--", linewidth=1.7, label=f"paper visual {method}")
            ax.plot(method_group["cycle"], method_group["local_value"], marker="o", markersize=2.8, linewidth=1.3, label=f"local {method}")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(local_metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        filename = f"tau1_{figure.lower().replace(' ', '')}_{local_metric}_digitized_alignment.png"
        path = FIGURE_ROOT / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(
    targets: pd.DataFrame,
    summary: pd.DataFrame,
    endpoints: pd.DataFrame,
    plot_paths: list[Path],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Figure 4 Digitized Alignment v1",
        "",
        "This audit uses visually digitized points from the local PDF crops. These values are not author data and are used only to test endpoint direction, curve shape, and affine scale diagnostics.",
        "",
        "## Endpoint Gate",
        "",
        md_table(endpoints),
        "",
        "## Curve Diagnostics",
        "",
        md_table(summary),
        "",
        "## Target Provenance",
        "",
        md_table(targets.head(8)),
        "",
        "## Interpretation",
        "",
        "- Figure 4a is evaluated under both local `w_rmse` and `standard_mask_rmse` because the paper text/plot ambiguity remains unresolved.",
        "- Figure 4b uses cumulative reward. The tau1 mainline matches the paper ordering against `linucb_concat`, but the local reward gap is larger than the visual paper gap.",
        "- Affine calibration is diagnostic only. A good affine RMSE cannot close an exact-replication gate unless the metric, split/mask, and Linear UCB protocol are also source-grounded.",
        "",
        "## Figures",
        "",
    ]
    for path in plot_paths:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    targets = paper_targets()
    metrics = source_metrics()
    local = local_curves(metrics)
    merged, summary, endpoints = alignment_rows(targets, local)
    targets_path = TABLE_ROOT / "tau1_figure4_digitized_targets.csv"
    merged_path = TABLE_ROOT / "tau1_figure4_digitized_local_vs_paper_curve.csv"
    summary_path = TABLE_ROOT / "tau1_figure4_curve_alignment_summary.csv"
    endpoints_path = TABLE_ROOT / "tau1_figure4_endpoint_alignment.csv"
    targets.to_csv(targets_path, index=False)
    merged.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)
    endpoints.to_csv(endpoints_path, index=False)
    plot_paths = plot_alignment(merged)
    write_report(targets, summary, endpoints, plot_paths)
    print(REPORT_PATH)
    print(summary_path)
    print(endpoints_path)


if __name__ == "__main__":
    main()
