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


DIAGNOSTIC_COLUMNS = [
    "selected_avg_reward_percentile",
    "selected_avg_entropy_percentile",
    "selected_avg_user_degree",
    "selected_cold_user_share",
]


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    mean = float(np.mean(arr))
    ci = float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, ci


def summarize(prefix: str) -> None:
    table_dir = ROOT / "outputs" / "tables"
    report_dir = ensure_dir(ROOT / "reports")
    path = table_dir / f"{prefix}_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    metrics = pd.read_csv(path)
    present = [col for col in DIAGNOSTIC_COLUMNS if col in metrics.columns]
    if not present:
        raise ValueError(f"No diagnostic columns found in {path}")
    group_col = "variant" if "variant" in metrics.columns else "policy"

    rows: list[dict[str, float | str]] = []
    for group_name, group in metrics.groupby(group_col, sort=False):
        row: dict[str, float | str] = {group_col: group_name}
        for col in present:
            mean, ci = _mean_ci(group[col])
            row[col] = mean
            row[f"{col}_ci95"] = ci
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = table_dir / f"{prefix}_diagnostics_summary.csv"
    summary.to_csv(summary_path, index=False)

    by_cycle = (
        metrics.groupby([group_col, "cycle"], as_index=False)[present]
        .mean(numeric_only=True)
        .sort_values([group_col, "cycle"])
    )
    by_cycle_path = table_dir / f"{prefix}_diagnostics_by_cycle.csv"
    by_cycle.to_csv(by_cycle_path, index=False)

    lines = [
        f"# Diagnostics: {prefix}",
        "",
        "## Policy-Level Selection Diagnostics",
        "",
        f"| {group_col} | reward pct | entropy pct | user degree | cold-user share |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {getattr(row, group_col)} | "
            f"{getattr(row, 'selected_avg_reward_percentile', np.nan):.4f} | "
            f"{getattr(row, 'selected_avg_entropy_percentile', np.nan):.4f} | "
            f"{getattr(row, 'selected_avg_user_degree', np.nan):.4f} | "
            f"{getattr(row, 'selected_cold_user_share', np.nan):.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- Higher reward percentile means the policy behaves more like myopic exploitation.",
            "- Higher entropy percentile means stronger active-learning behavior.",
            "- Higher user degree indicates centrality bias from network diffusion or DLS.",
            "- Higher cold-user share indicates more cold-start exploration.",
            "",
            f"CSV artifacts: `{summary_path.relative_to(ROOT)}`, `{by_cycle_path.relative_to(ROOT)}`.",
        ]
    )
    (report_dir / f"{prefix}_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    summarize(args.prefix)


if __name__ == "__main__":
    main()
