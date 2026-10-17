"""Compare local Figure 4a metrics against plausible PDF formula readings."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset_from_config, split_dataset
from src.utils import load_json


TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_FINAL_CSV = TABLE_DIR / "figure4a_paper_metric_instrumentation.csv"
OUT_CYCLE_CSV = TABLE_DIR / "figure4a_paper_metric_instrumentation_cycles.csv"
OUT_REPORT = REPORT_DIR / "figure4a_paper_metric_instrumentation.md"


PAPER_VISUAL = {
    "DU-Net": 1.788,
    "MF-centered Linear UCB": 1.814,
    "Myopic": 1.823,
}


@dataclass(frozen=True)
class MethodSpec:
    source: str
    method: str
    metrics_path: Path
    config_path: Path
    row_filter: Callable[[pd.DataFrame], pd.Series]


SPECS = [
    MethodSpec(
        source="current_default_initial_20seed",
        method="DU-Net",
        metrics_path=TABLE_DIR / "movielens_100k_svdpp_optimized_20seeds_metrics.csv",
        config_path=ROOT / "configs" / "movielens_100k_svdpp_optimized_20seeds.json",
        row_filter=lambda frame: frame["policy"] == "du_net",
    ),
    MethodSpec(
        source="current_default_initial_20seed",
        method="MF-centered Linear UCB",
        metrics_path=TABLE_DIR / "movielens_100k_svdpp_figure4a_linear_ucb_20seeds_metrics.csv",
        config_path=ROOT / "configs" / "movielens_100k_svdpp_figure4a_linear_ucb_20seeds.json",
        row_filter=lambda frame: frame["variant"] == "mf_linear_ucb_a002",
    ),
    MethodSpec(
        source="current_default_initial_20seed",
        method="Myopic",
        metrics_path=TABLE_DIR / "movielens_100k_svdpp_optimized_20seeds_metrics.csv",
        config_path=ROOT / "configs" / "movielens_100k_svdpp_optimized_20seeds.json",
        row_filter=lambda frame: frame["policy"] == "myopic",
    ),
    MethodSpec(
        source="table1_initial_density_3seed",
        method="DU-Net",
        metrics_path=TABLE_DIR / "movielens_100k_svdpp_figure4a_table1_initial_3seeds_metrics.csv",
        config_path=ROOT / "configs" / "movielens_100k_svdpp_figure4a_table1_initial_3seeds.json",
        row_filter=lambda frame: frame["variant"] == "du_net",
    ),
    MethodSpec(
        source="table1_initial_density_3seed",
        method="MF-centered Linear UCB",
        metrics_path=TABLE_DIR / "movielens_100k_svdpp_figure4a_table1_initial_3seeds_metrics.csv",
        config_path=ROOT / "configs" / "movielens_100k_svdpp_figure4a_table1_initial_3seeds.json",
        row_filter=lambda frame: frame["variant"] == "mf_linear_ucb_a002",
    ),
    MethodSpec(
        source="table1_initial_density_3seed",
        method="Myopic",
        metrics_path=TABLE_DIR / "movielens_100k_svdpp_figure4a_table1_initial_3seeds_metrics.csv",
        config_path=ROOT / "configs" / "movielens_100k_svdpp_figure4a_table1_initial_3seeds.json",
        row_filter=lambda frame: frame["variant"] == "myopic",
    ),
]


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def test_counts(config_path: Path) -> dict[int, int]:
    config = load_json(config_path)
    dataset = load_dataset_from_config(config)
    return {
        int(seed): len(split_dataset(dataset, seed=int(seed), **config["split"]).test)
        for seed in config["seeds"]
    }


def load_filtered_metrics(spec: MethodSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.metrics_path)
    filtered = frame.loc[spec.row_filter(frame)].copy()
    if filtered.empty:
        raise ValueError(f"No rows for {spec.method} in {spec.metrics_path}")
    counts = test_counts(spec.config_path)
    filtered["test_count"] = filtered["seed"].map(lambda seed: counts[int(seed)])
    filtered["paper_formula_rmse_conventional"] = filtered["rmse"].astype(float)
    filtered["pdf_literal_norm_per_test_sample"] = filtered["rmse"].astype(float) / filtered[
        "test_count"
    ].astype(float).pow(0.5)
    filtered["local_reward_weighted_w_rmse"] = filtered["w_rmse"].astype(float)
    filtered["method"] = spec.method
    filtered["source"] = spec.source
    return filtered


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    local_weighted = group["local_reward_weighted_w_rmse"].astype(float).tolist()
    paper_rmse = group["paper_formula_rmse_conventional"].astype(float).tolist()
    literal_norm = group["pdf_literal_norm_per_test_sample"].astype(float).tolist()
    rewards = group["cumulative_reward"].astype(float).tolist()
    observed = group["observed_size"].astype(float).tolist()
    test_count = group["test_count"].astype(float).tolist()
    method = str(group["method"].iloc[0])
    visual = PAPER_VISUAL[method]
    local_mean = mean(local_weighted)
    paper_mean = mean(paper_rmse)
    literal_mean = mean(literal_norm)
    return {
        "source": str(group["source"].iloc[0]),
        "method": method,
        "cycle": int(group["cycle"].iloc[0]),
        "seed_count": int(group["seed"].nunique()),
        "test_count": mean(test_count),
        "final_observed_size": mean(observed),
        "cumulative_reward": mean(rewards),
        "cumulative_reward_ci95": ci95(rewards),
        "local_reward_weighted_w_rmse": local_mean,
        "local_reward_weighted_w_rmse_ci95": ci95(local_weighted),
        "paper_formula_rmse_conventional": paper_mean,
        "paper_formula_rmse_conventional_ci95": ci95(paper_rmse),
        "pdf_literal_norm_per_test_sample": literal_mean,
        "pdf_literal_norm_per_test_sample_ci95": ci95(literal_norm),
        "paper_visual_w_rmse": visual,
        "local_weighted_ratio_to_visual": local_mean / visual,
        "paper_formula_ratio_to_visual": paper_mean / visual,
        "literal_norm_ratio_to_visual": literal_mean / visual,
    }


def build_cycle_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for spec in SPECS:
        metrics = load_filtered_metrics(spec)
        for _, group in metrics.groupby("cycle", sort=True):
            rows.append(summarize_group(group))
    return rows


def final_rows(cycle_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sorted({str(row["source"]) for row in cycle_rows}):
        source_rows = [row for row in cycle_rows if row["source"] == source]
        max_cycle = max(int(row["cycle"]) for row in source_rows)
        rows.extend(row for row in source_rows if int(row["cycle"]) == max_cycle)
    return sorted(rows, key=lambda row: (str(row["source"]), str(row["method"])))


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
    headers = [
        "source",
        "method",
        "reward",
        "local weighted",
        "paper RMSE",
        "literal norm",
        "paper visual",
        "weighted/visual",
        "paper RMSE/visual",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source"]),
                    str(row["method"]),
                    fmt(row["cumulative_reward"], 2),
                    fmt(row["local_reward_weighted_w_rmse"]),
                    fmt(row["paper_formula_rmse_conventional"]),
                    fmt(row["pdf_literal_norm_per_test_sample"], 6),
                    fmt(row["paper_visual_w_rmse"]),
                    fmt(row["local_weighted_ratio_to_visual"]),
                    fmt(row["paper_formula_ratio_to_visual"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def source_winner(rows: list[dict[str, object]], source: str, metric: str) -> dict[str, object]:
    scoped = [row for row in rows if row["source"] == source]
    return min(scoped, key=lambda row: float(row[metric]))


def gap_vs_du_net(rows: list[dict[str, object]], source: str, method: str, metric: str) -> float:
    scoped = [row for row in rows if row["source"] == source]
    du = next(row for row in scoped if row["method"] == "DU-Net")
    other = next(row for row in scoped if row["method"] == method)
    return float(other[metric]) - float(du[metric])


def write_report(rows: list[dict[str, object]], path: Path) -> None:
    default_local_winner = source_winner(rows, "current_default_initial_20seed", "local_reward_weighted_w_rmse")
    default_paper_winner = source_winner(
        rows,
        "current_default_initial_20seed",
        "paper_formula_rmse_conventional",
    )
    default_linear_gap_local = gap_vs_du_net(
        rows,
        "current_default_initial_20seed",
        "MF-centered Linear UCB",
        "local_reward_weighted_w_rmse",
    )
    default_linear_gap_paper = gap_vs_du_net(
        rows,
        "current_default_initial_20seed",
        "MF-centered Linear UCB",
        "paper_formula_rmse_conventional",
    )
    table1_linear_gap_local = gap_vs_du_net(
        rows,
        "table1_initial_density_3seed",
        "MF-centered Linear UCB",
        "local_reward_weighted_w_rmse",
    )
    table1_linear_gap_paper = gap_vs_du_net(
        rows,
        "table1_initial_density_3seed",
        "MF-centered Linear UCB",
        "paper_formula_rmse_conventional",
    )
    report = f"""# Figure 4a Paper-Metric Instrumentation

Generated by `experiments/build_figure4a_paper_metric_instrumentation.py`.

## What Was Instrumented

The local result files already contain two relevant metrics on the same
trajectories:

- `rmse`: conventional test-set RMSE, equivalent to reading the PDF formula as
  `sqrt(sum_squared_error / |Omega_test|)`.
- `w_rmse`: the local reward-weighted metric implemented in `src/metrics.py`,
  with weights `max(rating - 3, 0) + 0.1`.

The PDF text calls the metric weighted w-RMSE, but the extracted formula uses an
Omega test-set mask and is closer to conventional RMSE. For completeness, this
report also includes the literal extracted normalization
`sqrt(sum_squared_error) / |Omega_test|`, reported as `literal norm`.

## Final-Cycle Comparison

{markdown_table(rows)}

## Direction Check

- Under the current local reward-weighted metric, the 20-seed default-initial
  winner is **{default_local_winner["method"]}**. DU-Net beats MF-centered
  Linear UCB by {fmt(default_linear_gap_local, 4)} w-RMSE points.
- Under the conventional PDF-formula RMSE reading, the 20-seed default-initial
  winner is **{default_paper_winner["method"]}**. MF-centered Linear UCB is
  {fmt(abs(default_linear_gap_paper), 4)} RMSE points {'lower' if default_linear_gap_paper < 0 else 'higher'} than DU-Net.
- With the Table-1 initial density, DU-Net still beats MF-centered Linear UCB by
  {fmt(table1_linear_gap_local, 4)} under local weighted w-RMSE, but MF-centered
  Linear UCB is {fmt(abs(table1_linear_gap_paper), 4)} RMSE points {'lower' if table1_linear_gap_paper < 0 else 'higher'} under conventional RMSE.

## Interpretation

The previous Figure 4a alignment claim is valid only under the local
reward-weighted metric. If the PDF formula is interpreted as conventional
test-set RMSE, the local DU-Net proxy does not yet reproduce the paper's
MovieLens DU-Net-over-Linear-UCB result. The absolute scale also remains low:
conventional RMSE reaches roughly 0.64x of the paper visual endpoint, while the
local weighted metric is roughly 0.46x.

The literal extracted normalization is orders of magnitude too small to match
Figure 4a, so it is best treated as a PDF text-extraction/display ambiguity, not
as the plotted metric.

## Decision

Queue item 24 is complete. The next useful experiment is not another baseline
sweep. It should add a paper-faithful reward-weighted RMSE diagnostic, likely
using the paper's reward transform `sigmoid(rating - 3)` as candidate weights,
and rerun the smallest 3-seed MovieLens diagnostic. That will determine whether
the Figure 4a direction can be recovered under a more paper-faithful weighted
metric, or whether the remaining mismatch is mainly the matrix-completion and
DU-Net proxy implementation.

CSV artifacts:

- `{OUT_FINAL_CSV.relative_to(ROOT)}`
- `{OUT_CYCLE_CSV.relative_to(ROOT)}`
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cycle_rows = build_cycle_rows()
    rows = final_rows(cycle_rows)
    write_csv(cycle_rows, OUT_CYCLE_CSV)
    write_csv(rows, OUT_FINAL_CSV)
    write_report(rows, OUT_REPORT)
    print(f"Wrote {OUT_FINAL_CSV}")
    print(f"Wrote {OUT_CYCLE_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
