"""Summarize the MovieLens Table-1 initial-density Figure 4a diagnostic."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_CSV = TABLE_DIR / "figure4a_table1_initial_diagnostic.csv"
OUT_REPORT = REPORT_DIR / "figure4a_table1_initial_diagnostic.md"

DEFAULT_METRICS = TABLE_DIR / "movielens_100k_svdpp_optimized_20seeds_metrics.csv"
DEFAULT_SUMMARY = TABLE_DIR / "movielens_100k_svdpp_optimized_20seeds_final_summary.csv"
LINEAR_METRICS = TABLE_DIR / "movielens_100k_svdpp_figure4a_linear_ucb_20seeds_metrics.csv"
LINEAR_SUMMARY = TABLE_DIR / "movielens_100k_svdpp_figure4a_linear_ucb_20seeds_final_summary.csv"
TABLE1_METRICS = TABLE_DIR / "movielens_100k_svdpp_figure4a_table1_initial_3seeds_metrics.csv"
TABLE1_SUMMARY = TABLE_DIR / "movielens_100k_svdpp_figure4a_table1_initial_3seeds_final_summary.csv"

PAPER_VISUAL = {
    "DU-Net": 1.788,
    "MF-centered Linear UCB": 1.814,
    "Myopic": 1.823,
}

METHODS = [
    {
        "method": "DU-Net",
        "default_metrics_filter": ("policy", "du_net"),
        "default_summary_filter": ("policy", "du_net"),
        "table1_metrics_filter": ("variant", "du_net"),
        "table1_summary_filter": ("variant", "du_net"),
    },
    {
        "method": "MF-centered Linear UCB",
        "default_metrics_filter": ("variant", "mf_linear_ucb_a002"),
        "default_summary_filter": ("variant", "mf_linear_ucb_a002"),
        "table1_metrics_filter": ("variant", "mf_linear_ucb_a002"),
        "table1_summary_filter": ("variant", "mf_linear_ucb_a002"),
    },
    {
        "method": "Myopic",
        "default_metrics_filter": ("policy", "myopic"),
        "default_summary_filter": ("policy", "myopic"),
        "table1_metrics_filter": ("variant", "myopic"),
        "table1_summary_filter": ("variant", "myopic"),
    },
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def f(value: str | float | int) -> float:
    return float(value)


def row_matches(row: dict[str, str], field: str, value: str) -> bool:
    return row.get(field) == value


def final_seed_rows(
    rows: list[dict[str, str]],
    field: str,
    value: str,
    seeds: set[int] | None,
) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row_matches(row, field, value)
        and int(row["cycle"]) == 20
        and (seeds is None or int(row["seed"]) in seeds)
    ]
    if not selected:
        raise ValueError(f"No final rows for {field}={value}")
    return selected


def summary_row(rows: list[dict[str, str]], field: str, value: str) -> dict[str, str]:
    for row in rows:
        if row_matches(row, field, value):
            return row
    raise ValueError(f"No summary row for {field}={value}")


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def summarize_seed_rows(
    rows: list[dict[str, str]],
    method: str,
    source: str,
    paper_w_rmse: float,
    default_reference: float | None,
) -> dict[str, object]:
    rewards = [f(row["cumulative_reward"]) for row in rows]
    rmses = [f(row["rmse"]) for row in rows]
    w_rmses = [f(row["w_rmse"]) for row in rows]
    observed = [f(row["observed_size"]) for row in rows]
    initial_count = [f(row["observed_size"]) - 2000 for row in rows]
    w_rmse = mean(w_rmses)
    return {
        "source": source,
        "method": method,
        "seeds": ",".join(str(int(row["seed"])) for row in sorted(rows, key=lambda r: int(r["seed"]))),
        "initial_count": mean(initial_count),
        "final_observed_size": mean(observed),
        "cumulative_reward": mean(rewards),
        "cumulative_reward_ci95": ci95(rewards),
        "rmse": mean(rmses),
        "rmse_ci95": ci95(rmses),
        "w_rmse": w_rmse,
        "w_rmse_ci95": ci95(w_rmses),
        "paper_visual_w_rmse": paper_w_rmse,
        "w_rmse_ratio_to_paper_visual": w_rmse / paper_w_rmse,
        "w_rmse_delta_vs_default_3seed": ""
        if default_reference is None
        else w_rmse - default_reference,
    }


def summarize_20seed_row(
    row: dict[str, str],
    method: str,
    paper_w_rmse: float,
) -> dict[str, object]:
    w_rmse = f(row["w_rmse"])
    return {
        "source": "current_20seed_reference",
        "method": method,
        "seeds": "0-19",
        "initial_count": f(row["observed_size"]) - 2000,
        "final_observed_size": f(row["observed_size"]),
        "cumulative_reward": f(row["cumulative_reward"]),
        "cumulative_reward_ci95": f(row["cumulative_reward_ci95"]),
        "rmse": f(row["rmse"]),
        "rmse_ci95": f(row["rmse_ci95"]),
        "w_rmse": w_rmse,
        "w_rmse_ci95": f(row["w_rmse_ci95"]),
        "paper_visual_w_rmse": paper_w_rmse,
        "w_rmse_ratio_to_paper_visual": w_rmse / paper_w_rmse,
        "w_rmse_delta_vs_default_3seed": "",
    }


def build_rows() -> list[dict[str, object]]:
    default_metrics_rows = read_rows(DEFAULT_METRICS)
    default_summary_rows = read_rows(DEFAULT_SUMMARY)
    linear_metrics_rows = read_rows(LINEAR_METRICS)
    linear_summary_rows = read_rows(LINEAR_SUMMARY)
    table1_metrics_rows = read_rows(TABLE1_METRICS)
    table1_summary_rows = read_rows(TABLE1_SUMMARY)
    seeds = {0, 1, 2}

    rows: list[dict[str, object]] = []
    for spec in METHODS:
        method = spec["method"]
        paper_w_rmse = PAPER_VISUAL[method]

        if method == "MF-centered Linear UCB":
            default_metric_source = linear_metrics_rows
            default_summary_source = linear_summary_rows
        else:
            default_metric_source = default_metrics_rows
            default_summary_source = default_summary_rows

        default_field, default_value = spec["default_metrics_filter"]
        default_rows = final_seed_rows(default_metric_source, default_field, default_value, seeds)
        default_summary_field, default_summary_value = spec["default_summary_filter"]
        default_summary = summary_row(default_summary_source, default_summary_field, default_summary_value)

        default_3seed = summarize_seed_rows(
            default_rows,
            method,
            "current_default_initial_3seed",
            paper_w_rmse,
            default_reference=None,
        )
        rows.append(default_3seed)

        table1_field, table1_value = spec["table1_metrics_filter"]
        table1_rows = final_seed_rows(table1_metrics_rows, table1_field, table1_value, seeds)
        rows.append(
            summarize_seed_rows(
                table1_rows,
                method,
                "table1_initial_density_3seed",
                paper_w_rmse,
                default_reference=f(default_3seed["w_rmse"]),
            )
        )

        rows.append(summarize_20seed_row(default_summary, method, paper_w_rmse))

        # Assert the final-summary artifact agrees with the table generated by the
        # experiment runner. This catches accidental partial metric reads.
        table1_summary_field, table1_summary_value = spec["table1_summary_filter"]
        table1_summary = summary_row(table1_summary_rows, table1_summary_field, table1_summary_value)
        if abs(f(table1_summary["w_rmse"]) - f(rows[-2]["w_rmse"])) > 1e-12:
            raise AssertionError(f"Table1 summary mismatch for {method}")

    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def method_rows(rows: list[dict[str, object]], method: str) -> list[dict[str, object]]:
    return [row for row in rows if row["method"] == method]


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "source",
        "method",
        "initial",
        "reward",
        "RMSE",
        "w-RMSE",
        "paper visual",
        "local/paper",
        "delta vs default",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source"]),
                    str(row["method"]),
                    fmt(row["initial_count"], 0),
                    fmt(row["cumulative_reward"], 2),
                    fmt(row["rmse"], 4),
                    fmt(row["w_rmse"], 4),
                    fmt(row["paper_visual_w_rmse"], 4),
                    fmt(row["w_rmse_ratio_to_paper_visual"], 4),
                    fmt(row["w_rmse_delta_vs_default_3seed"], 4),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def conclusion_bullets(rows: list[dict[str, object]]) -> str:
    bullets = []
    for method in [spec["method"] for spec in METHODS]:
        method_data = {row["source"]: row for row in method_rows(rows, method)}
        default = method_data["current_default_initial_3seed"]
        table1 = method_data["table1_initial_density_3seed"]
        bullets.append(
            "- "
            + f"{method}: increasing the initial set from "
            + f"{fmt(default['initial_count'], 0)} to {fmt(table1['initial_count'], 0)} "
            + f"changes w-RMSE from {fmt(default['w_rmse'])} to {fmt(table1['w_rmse'])} "
            + f"({fmt(table1['w_rmse_delta_vs_default_3seed'], 4)}). "
            + f"The Table-1 setting is still only {fmt(table1['w_rmse_ratio_to_paper_visual'], 3)}x "
            + "of the paper visual endpoint."
        )
    return "\n".join(bullets)


def write_report(rows: list[dict[str, object]], path: Path, csv_path: Path) -> None:
    report = f"""# MovieLens Figure 4a Table-1 Initial-Density Diagnostic

Generated by `experiments/build_figure4a_table1_initial_diagnostic.py`.

## Summary

This diagnostic tests whether the MovieLens absolute w-RMSE gap to the paper's
Figure 4a is mainly caused by the current initial set being smaller than Table
1 implies. The answer is no. Matching the Table-1 initial density changes local
w-RMSE, but it does not move the results close to the paper's visual endpoints.

{conclusion_bullets(rows)}

## Result Table

{markdown_table(rows)}

## Interpretation

The current reproduction remains directionally consistent with Figure 4a:
DU-Net is better than the strongest local Linear-UCB-style comparator on
w-RMSE. The absolute scale gap is therefore more likely due to metric
definition, rating preprocessing, and model-backend fidelity than to the
initial-set density alone.

The next step should be metric instrumentation: report the PDF-formula
test-set RMSE/w-RMSE alongside the current reward-weighted `weighted_rmse()`,
then decide whether a noisy matrix-completion backend or exact DU-Net entropy
and DLS implementation is the largest remaining source of scale mismatch.

CSV artifact: `{csv_path.relative_to(ROOT)}`.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows, OUT_CSV)
    write_report(rows, OUT_REPORT, OUT_CSV)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
