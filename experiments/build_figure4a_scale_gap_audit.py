"""Audit why local MovieLens Figure 4a scale differs from the paper figure."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_movielens_100k, split_dataset

TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_CSV = TABLE_DIR / "figure4a_scale_gap_audit.csv"
OUT_REPORT = REPORT_DIR / "figure4a_scale_gap_audit.md"


PAPER = {
    "users": 943,
    "items": 1681,
    "initial_density": 0.00086,
    "test_density": 0.0126,
    "figure4a_du_net_w_rmse": 1.788,
    "figure4a_linear_ucb_w_rmse": 1.814,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"{value} not found in {key}")


def f(value: str | float) -> float:
    return float(value)


def dataset_rows() -> list[dict[str, object]]:
    dataset = load_movielens_100k(ROOT / "data" / "raw" / "ml-100k")
    split = split_dataset(dataset, seed=0, test_ratio=0.2, initial_fraction_of_train=0.01)
    total_entries = dataset.num_users * dataset.num_items
    paper_total_entries = PAPER["users"] * PAPER["items"]
    paper_initial_count = PAPER["initial_density"] * paper_total_entries
    paper_test_count = PAPER["test_density"] * paper_total_entries
    return [
        {
            "category": "dataset",
            "item": "users",
            "paper_value": PAPER["users"],
            "current_value": dataset.num_users,
            "ratio_current_to_paper": dataset.num_users / PAPER["users"],
            "interpretation": "User count matches the paper.",
        },
        {
            "category": "dataset",
            "item": "items",
            "paper_value": PAPER["items"],
            "current_value": dataset.num_items,
            "ratio_current_to_paper": dataset.num_items / PAPER["items"],
            "interpretation": "Current MovieLens loader keeps one more item than the paper table.",
        },
        {
            "category": "split",
            "item": "initial_count",
            "paper_value": paper_initial_count,
            "current_value": len(split.initial),
            "ratio_current_to_paper": len(split.initial) / paper_initial_count,
            "interpretation": "Current initial set is materially smaller than the paper Table 1 density implies.",
        },
        {
            "category": "split",
            "item": "test_count",
            "paper_value": paper_test_count,
            "current_value": len(split.test),
            "ratio_current_to_paper": len(split.test) / paper_test_count,
            "interpretation": "Current random 20% test size is essentially the paper's test density.",
        },
        {
            "category": "split",
            "item": "initial_density",
            "paper_value": PAPER["initial_density"],
            "current_value": len(split.initial) / total_entries,
            "ratio_current_to_paper": (len(split.initial) / total_entries) / PAPER["initial_density"],
            "interpretation": "Current config uses 1% of observed train ratings, not Table-1-matching matrix density.",
        },
        {
            "category": "split",
            "item": "test_density",
            "paper_value": PAPER["test_density"],
            "current_value": len(split.test) / total_entries,
            "ratio_current_to_paper": (len(split.test) / total_entries) / PAPER["test_density"],
            "interpretation": "Test density aligns closely, so test-size scale is not the main issue.",
        },
    ]


def metric_rows() -> list[dict[str, object]]:
    current = row_by(
        read_rows(TABLE_DIR / "movielens_100k_svdpp_optimized_20seeds_final_summary.csv"),
        "policy",
        "du_net",
    )
    current_linear = row_by(
        read_rows(TABLE_DIR / "movielens_100k_svdpp_figure4a_linear_ucb_20seeds_final_summary.csv"),
        "variant",
        "mf_linear_ucb_a002",
    )
    return [
        {
            "category": "metric",
            "item": "DU-Net local weighted_rmse",
            "paper_value": PAPER["figure4a_du_net_w_rmse"],
            "current_value": f(current["w_rmse"]),
            "ratio_current_to_paper": f(current["w_rmse"]) / PAPER["figure4a_du_net_w_rmse"],
            "interpretation": "Current weighted_rmse uses positive-rating weights and is not directly comparable to the PDF Figure 4a scale.",
        },
        {
            "category": "metric",
            "item": "DU-Net local plain rmse",
            "paper_value": PAPER["figure4a_du_net_w_rmse"],
            "current_value": f(current["rmse"]),
            "ratio_current_to_paper": f(current["rmse"]) / PAPER["figure4a_du_net_w_rmse"],
            "interpretation": "Plain RMSE is closer than current weighted_rmse but still well below the PDF visual estimate.",
        },
        {
            "category": "metric",
            "item": "Linear UCB local weighted_rmse",
            "paper_value": PAPER["figure4a_linear_ucb_w_rmse"],
            "current_value": f(current_linear["w_rmse"]),
            "ratio_current_to_paper": f(current_linear["w_rmse"]) / PAPER["figure4a_linear_ucb_w_rmse"],
            "interpretation": "Best MF-centered Linear UCB preserves paper ordering but not paper absolute scale.",
        },
        {
            "category": "metric",
            "item": "Linear UCB local plain rmse",
            "paper_value": PAPER["figure4a_linear_ucb_w_rmse"],
            "current_value": f(current_linear["rmse"]),
            "ratio_current_to_paper": f(current_linear["rmse"]) / PAPER["figure4a_linear_ucb_w_rmse"],
            "interpretation": "Plain RMSE remains below the PDF visual estimate.",
        },
    ]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = ["category", "item", "paper", "current", "current/paper", "interpretation"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["category"]),
                    str(row["item"]),
                    fmt(row["paper_value"], 4),
                    fmt(row["current_value"], 4),
                    fmt(row["ratio_current_to_paper"], 4),
                    str(row["interpretation"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]], path: Path, csv_path: Path) -> None:
    report = f"""# Figure 4a Absolute-Scale Gap Audit

Generated by `experiments/build_figure4a_scale_gap_audit.py`.

## Summary

The local MovieLens Figure 4a reproduction is directionally consistent with the
paper but not numerically scale-aligned. The main reason is not the Linear UCB
comparison anymore; that comparison now matches the paper's ordering. The
remaining absolute-scale gap has three likely sources:

1. The current `weighted_rmse()` is a local reward-weighted metric, whereas the
   PDF formula for w-RMSE is closer to plain test-set RMSE despite the text
   describing reward weighting.
2. The current MovieLens initial set has 800 ratings, while the paper Table 1
   density implies roughly 1,365 initial ratings.
3. The current matrix-completion and DU-Net mechanism are still modular proxies,
   not the author's exact noisy matrix-completion, observational entropy, and
   DLS implementation.

## Audit Table

{markdown_table(rows)}

## Decision

The next experiment should not add more baselines. The separate Table-1
initial-density diagnostic has now been run and does not close the absolute
scale gap. The next useful step is metric instrumentation: report the
PDF-formula test-set RMSE/w-RMSE next to the current local `weighted_rmse()` on
the same trajectories. That will tell us whether the remaining gap is mostly
reporting scale or model/preprocessing fidelity.

CSV artifact: `{csv_path.relative_to(ROOT)}`.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = dataset_rows() + metric_rows()
    write_csv(rows, OUT_CSV)
    write_report(rows, OUT_REPORT, OUT_CSV)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
