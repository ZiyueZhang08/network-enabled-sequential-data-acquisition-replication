from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir

TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"


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


def load_predictions(row_level_dir: Path) -> pd.DataFrame:
    files = sorted(row_level_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No row-level prediction CSVs found under {row_level_dir}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    required = {"method", "seed", "cycle", "selected_or_test", "y_true", "y_pred_clipped", "y_pred_raw"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Row-level predictions missing required columns: {missing}")
    return predictions


def final_test_rows(predictions: pd.DataFrame, cycle: str) -> pd.DataFrame:
    frame = predictions[predictions["selected_or_test"].astype(str).str.lower() == "test"].copy()
    if frame.empty:
        raise ValueError("No test rows found in row-level predictions")
    if cycle.lower() in {"final", "last"}:
        final_cycles = frame.groupby(["method", "seed"])["cycle"].transform("max")
        return frame[frame["cycle"] == final_cycles].copy()
    return frame[frame["cycle"] == int(cycle)].copy()


def safe_weight(values: pd.Series | np.ndarray) -> np.ndarray:
    weight = np.asarray(values, dtype=float)
    weight = np.where(np.isfinite(weight), weight, 0.0)
    weight = np.maximum(weight, 0.0)
    if float(np.sum(weight)) <= 0.0:
        weight = np.ones_like(weight, dtype=float)
    return weight


def weight_candidates(group: pd.DataFrame, midpoint: float) -> dict[str, np.ndarray]:
    truth = group["y_true"].to_numpy(dtype=float)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(truth - midpoint, -40.0, 40.0)))
    candidates: dict[str, np.ndarray] = {
        "w=1": np.ones(len(group), dtype=float),
        "w=rating": truth,
        "w=rating/mean": truth / max(float(np.mean(truth)), 1e-12),
        "w=max(rating-midpoint,0)": np.maximum(truth - midpoint, 0.0),
        "w=max(rating-midpoint,0)+eps": np.maximum(truth - midpoint, 0.0) + 0.1,
        "w=sigmoid(rating-midpoint)": sigmoid,
        "w=sigmoid/mean": sigmoid / max(float(np.mean(sigmoid)), 1e-12),
    }
    optional_columns = {
        "w=selected_reward_contribution": "weight_selected_reward_contribution",
        "w=item_popularity": "weight_item_popularity",
        "w=item_popularity/mean": "weight_item_popularity_norm",
        "w=inverse_item_popularity": "weight_inverse_item_popularity",
        "w=test_weight_candidates": "test_weight_candidates",
    }
    for name, col in optional_columns.items():
        if col in group.columns:
            candidates[name] = group[col].to_numpy(dtype=float)
    return {name: safe_weight(weight) for name, weight in candidates.items()}


def metric_values(group: pd.DataFrame, midpoint: float) -> list[dict[str, float | int | str]]:
    truth = group["y_true"].to_numpy(dtype=float)
    pred_clipped = group["y_pred_clipped"].to_numpy(dtype=float)
    pred_raw = group["y_pred_raw"].to_numpy(dtype=float)
    err_clipped = truth - pred_clipped
    err_raw = truth - pred_raw
    n = max(len(group), 1)
    rows: list[dict[str, float | int | str]] = []

    def add(metric: str, weight: str, value: float) -> None:
        rows.append(
            {
                "method": str(group["method"].iloc[0]),
                "seed": int(group["seed"].iloc[0]),
                "cycle": int(group["cycle"].max()),
                "metric": metric,
                "weight": weight,
                "value": float(value),
                "rows": int(len(group)),
            }
        )

    add("M1_standard_rmse", "w=1", np.sqrt(np.mean(err_clipped * err_clipped)))
    add("M2_pdf_literal", "w=1", np.sqrt(np.sum(err_clipped * err_clipped)) / n)
    add("M3_normal_l2", "w=1", np.sqrt(np.sum(err_clipped * err_clipped)))
    add("M6_centered_rmse", "w=1", np.sqrt(np.mean(((truth - midpoint) - (pred_clipped - midpoint)) ** 2)))
    add("M7_normalized_0_1_rmse", "w=1", np.sqrt(np.mean(((truth - pred_clipped) / 4.0) ** 2)))
    add("M8_raw_unclipped_rmse", "w=1", np.sqrt(np.mean(err_raw * err_raw)))
    add("M9_clipped_rmse", "w=1", np.sqrt(np.mean(err_clipped * err_clipped)))

    for weight_name, weight in weight_candidates(group, midpoint).items():
        add("M4_weighted_rmse", weight_name, np.sqrt(np.sum(weight * err_clipped * err_clipped) / np.sum(weight)))
        add("M5_squared_weighted", weight_name, np.sqrt(np.mean((weight * err_clipped) ** 2)))
    return rows


def summarize(long_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        long_metrics.groupby(["method", "metric", "weight"], as_index=False)
        .agg(
            value=("value", "mean"),
            value_ci95=("value", ci95),
            seed_count=("seed", "nunique"),
            rows=("rows", "mean"),
        )
        .sort_values(["metric", "weight", "method"])
    )


def compare_to_paper(
    summary: pd.DataFrame,
    du_method: str,
    comparators: list[str],
    paper_du: float,
    paper_linear: float,
) -> pd.DataFrame:
    rows = []
    paper_gap = float(paper_linear - paper_du)
    for comparator in comparators:
        for (metric, weight), group in summary.groupby(["metric", "weight"], sort=False):
            du = group[group["method"] == du_method]
            base = group[group["method"] == comparator]
            if du.empty or base.empty:
                continue
            du_value = float(du["value"].iloc[0])
            base_value = float(base["value"].iloc[0])
            local_gap = base_value - du_value
            scale_distance = float(
                np.mean(
                    [
                        abs(du_value - paper_du) / max(abs(paper_du), 1e-12),
                        abs(base_value - paper_linear) / max(abs(paper_linear), 1e-12),
                    ]
                )
            )
            if abs(paper_gap) > 1e-12:
                gap_distance = float(abs(local_gap - paper_gap) / abs(paper_gap))
            else:
                gap_distance = np.nan
            direction = "DU-Net < Linear UCB" if local_gap > 0 else "not_matched"
            gate = (
                scale_distance <= 0.15
                and (np.isnan(gap_distance) or gap_distance <= 0.20)
                and local_gap > 0
            )
            rows.append(
                {
                    "metric": metric,
                    "weight": weight,
                    "comparator": comparator,
                    "DU-Net": du_value,
                    "Linear UCB": base_value,
                    "gap": local_gap,
                    "scale_distance_to_paper": scale_distance,
                    "gap_distance_to_paper": gap_distance,
                    "direction": direction,
                    "metric_gate": "pass" if gate else "unresolved",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["metric_gate", "scale_distance_to_paper", "gap_distance_to_paper"],
        ascending=[True, True, True],
    )


def write_report(
    prefix: str,
    row_level_dir: Path,
    comparison: pd.DataFrame,
    paper_du: float,
    paper_linear: float,
    report_path: Path,
) -> None:
    passing = comparison[comparison["metric_gate"] == "pass"]
    best = comparison.head(12)
    status = "candidate_found" if not passing.empty else "unresolved"
    lines = [
        "# Metric Reconstruction Audit v2",
        "",
        f"- Row-level source: `{row_level_dir}`",
        f"- Paper endpoint targets: DU-Net `{paper_du:.6g}`, Linear UCB `{paper_linear:.6g}`",
        f"- Status: **{status}**",
        "",
        "## Gate Interpretation",
        "",
        "A metric passes only if endpoint scale is within 15%, DU-Net is lower than Linear UCB, and the local gap is within 20% of the paper visual gap. If no row passes, metric exactness remains a P0 caveat.",
        "",
        "## Best Candidates",
        "",
        md_table(best),
    ]
    if not passing.empty:
        lines.extend(["", "## Passing Candidates", "", md_table(passing)])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-level-dir", required=True)
    parser.add_argument("--output-prefix", default="paper_faithful_metric_reconstruction_audit_v2")
    parser.add_argument("--du-method", default="schur_loglambda_tuned_full_pool_softmax_tau1_literal")
    parser.add_argument("--comparators", nargs="+", default=["linucb_concat", "linucb_mf_interaction"])
    parser.add_argument("--cycle", default="final")
    parser.add_argument("--midpoint", type=float, default=3.0)
    parser.add_argument("--paper-du-endpoint", type=float, default=1.788)
    parser.add_argument("--paper-linear-endpoint", type=float, default=1.814)
    args = parser.parse_args()

    ensure_dir(TABLE_DIR)
    ensure_dir(REPORT_DIR)
    row_level_dir = Path(args.row_level_dir)
    predictions = final_test_rows(load_predictions(row_level_dir), args.cycle)
    metric_rows = []
    for _, group in predictions.groupby(["method", "seed"], sort=False):
        metric_rows.extend(metric_values(group, args.midpoint))
    long_metrics = pd.DataFrame(metric_rows)
    summary = summarize(long_metrics)
    comparison = compare_to_paper(
        summary,
        du_method=args.du_method,
        comparators=[str(c) for c in args.comparators],
        paper_du=float(args.paper_du_endpoint),
        paper_linear=float(args.paper_linear_endpoint),
    )

    long_path = TABLE_DIR / f"{args.output_prefix}_long.csv"
    summary_path = TABLE_DIR / f"{args.output_prefix}_summary.csv"
    comparison_path = TABLE_DIR / f"{args.output_prefix}_comparison.csv"
    report_path = REPORT_DIR / f"{args.output_prefix}.md"
    long_metrics.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    write_report(
        args.output_prefix,
        row_level_dir,
        comparison,
        float(args.paper_du_endpoint),
        float(args.paper_linear_endpoint),
        report_path,
    )
    print(f"Wrote {long_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
