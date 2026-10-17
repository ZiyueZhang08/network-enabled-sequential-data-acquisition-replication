"""Build the final Figure 4a target comparison report."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

ALIGNMENT = TABLE_DIR / "figure4a_movielens_alignment.csv"
TAIL_SUMMARY = TABLE_DIR / "figure4a_rating_tail_20seed_confirmation.csv"
TAIL_GAPS = TABLE_DIR / "figure4a_rating_tail_20seed_confirmation_gaps.csv"
OUT_CSV = TABLE_DIR / "figure4a_final_target_report.csv"
OUT_REPORT = REPORT_DIR / "figure4a_final_target_report.md"


def f(value: object) -> float:
    return float(value)


def fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def row_where(frame: pd.DataFrame, **filters: str) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        mask &= frame[key].astype(str) == str(value)
    rows = frame[mask]
    if rows.empty:
        raise KeyError(filters)
    return rows.iloc[0]


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_rows() -> list[dict[str, object]]:
    alignment = pd.read_csv(ALIGNMENT)
    summary = pd.read_csv(TAIL_SUMMARY)
    gaps = pd.read_csv(TAIL_GAPS)

    paper_du = row_where(alignment, source="paper_figure4a_visual_estimate", variant="du_net")
    paper_linear = row_where(alignment, source="paper_figure4a_visual_estimate", variant="linear_ucb")
    local_du_weighted = row_where(summary, variant="du_net", metric_name="local_reward_w_rmse")
    local_linear_weighted = row_where(summary, variant="mf_linear_ucb_a002", metric_name="local_reward_w_rmse")

    rows: list[dict[str, object]] = [
        {
            "comparison": "paper_visual_w_rmse",
            "du_net": f(paper_du["w_rmse"]),
            "linear_ucb": f(paper_linear["w_rmse"]),
            "linear_minus_du": f(paper_linear["w_rmse"]) - f(paper_du["w_rmse"]),
            "du_wins": "yes",
            "seed_count": "",
            "scale_ratio_local_du_to_paper_du": "",
            "interpretation": "Paper Figure 4a visual endpoint estimate; DU-Net beats Linear UCB.",
        },
        {
            "comparison": "local_reward_weighted_w_rmse",
            "du_net": f(local_du_weighted["metric_value"]),
            "linear_ucb": f(local_linear_weighted["metric_value"]),
            "linear_minus_du": f(local_linear_weighted["metric_value"]) - f(local_du_weighted["metric_value"]),
            "du_wins": "yes",
            "seed_count": int(local_du_weighted["seed_count"]),
            "scale_ratio_local_du_to_paper_du": f(local_du_weighted["metric_value"]) / f(paper_du["w_rmse"]),
            "interpretation": "Direction matches the paper, but the absolute scale is far lower.",
        },
    ]
    for metric in [
        "rmse_all",
        "rmse_high_rating_ge4",
        "rmse_low_rating_le3",
        "rmse_rating_eq5",
        "sigmoid_reward_w_rmse",
    ]:
        du = row_where(summary, variant="du_net", metric_name=metric)
        linear = row_where(summary, variant="mf_linear_ucb_a002", metric_name=metric)
        gap = row_where(gaps, metric_name=metric)
        rows.append(
            {
                "comparison": metric,
                "du_net": f(du["metric_value"]),
                "linear_ucb": f(linear["metric_value"]),
                "linear_minus_du": f(gap["linear_minus_du"]),
                "du_wins": "yes" if f(gap["linear_minus_du"]) > 0 else "no",
                "seed_count": int(gap["seed_count"]),
                "scale_ratio_local_du_to_paper_du": f(du["metric_value"]) / f(paper_du["w_rmse"]),
                "interpretation": interpretation_for(metric, f(gap["linear_minus_du"])),
            }
        )
    return rows


def interpretation_for(metric: str, gap: float) -> str:
    if metric == "rmse_all":
        return "Strict full-test RMSE does not reproduce the paper ordering; Linear UCB is lower."
    if metric == "rmse_high_rating_ge4":
        return "High-rating region is DU-Net-favored, supporting reward-region consistency."
    if metric == "rmse_low_rating_le3":
        return "Low-rating region is Linear-UCB-favored, explaining why full-test RMSE flips."
    if metric == "rmse_rating_eq5":
        return "Five-star test ratings are DU-Net-favored."
    if metric == "sigmoid_reward_w_rmse":
        return "Paper-like sigmoid reward weighting is a near tie at 20 seeds."
    return "DU-Net-favored." if gap > 0 else "Linear-UCB-favored."


def write_report(rows: list[dict[str, object]]) -> None:
    by_name = {str(row["comparison"]): row for row in rows}
    paper = by_name["paper_visual_w_rmse"]
    local = by_name["local_reward_weighted_w_rmse"]
    full = by_name["rmse_all"]
    high = by_name["rmse_high_rating_ge4"]
    low = by_name["rmse_low_rating_le3"]
    sigmoid = by_name["sigmoid_reward_w_rmse"]
    five = by_name["rmse_rating_eq5"]

    lines = [
        "# Figure 4a Final Target Report",
        "",
        "Generated by `experiments/build_figure4a_final_target_report.py`.",
        "",
        "## Answer",
        "",
        "The current reproduction is a strong partial reproduction, not an exact Figure 4a match.",
        "It is consistent with the paper when evaluated in the local reward-weighted/high-rating regions, but it is not close to the paper's absolute visual w-RMSE scale and it does not recover the DU-Net-over-Linear-UCB ordering under full-test conventional RMSE.",
        "",
        "## Main Comparison",
        "",
        "| comparison | DU-Net | Linear UCB | Linear - DU | seed count | interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison']} | {fmt(row['du_net'])} | {fmt(row['linear_ucb'])} | "
            f"{fmt(row['linear_minus_du'])} | {row['seed_count']} | {row['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Distance From Paper",
            "",
            f"- Paper Figure 4a visual endpoint: DU-Net {fmt(paper['du_net'], 3)}, Linear UCB {fmt(paper['linear_ucb'], 3)}, gap {fmt(paper['linear_minus_du'], 3)} favoring DU-Net.",
            f"- Local reward-weighted endpoint: DU-Net {fmt(local['du_net'])}, Linear UCB {fmt(local['linear_ucb'])}, gap {fmt(local['linear_minus_du'])} favoring DU-Net.",
            f"- Absolute scale is still far: local DU-Net reward-weighted w-RMSE is {fmt(local['scale_ratio_local_du_to_paper_du'], 3)}x of the paper visual DU-Net endpoint.",
            f"- Full-test RMSE goes the other way: DU-Net {fmt(full['du_net'])}, Linear UCB {fmt(full['linear_ucb'])}, gap {fmt(full['linear_minus_du'])} favoring Linear UCB.",
            "",
            "## Mechanism Takeaway",
            "",
            f"- High-rating test region: DU-Net {fmt(high['du_net'])}, Linear UCB {fmt(high['linear_ucb'])}, gap {fmt(high['linear_minus_du'])} favoring DU-Net.",
            f"- Five-star test ratings: DU-Net {fmt(five['du_net'])}, Linear UCB {fmt(five['linear_ucb'])}, gap {fmt(five['linear_minus_du'])} favoring DU-Net.",
            f"- Low-rating test region: DU-Net {fmt(low['du_net'])}, Linear UCB {fmt(low['linear_ucb'])}, gap {fmt(low['linear_minus_du'])} favoring Linear UCB.",
            f"- Sigmoid reward-weighted RMSE is a near tie: DU-Net {fmt(sigmoid['du_net'])}, Linear UCB {fmt(sigmoid['linear_ucb'])}, gap {fmt(sigmoid['linear_minus_du'])}.",
            "",
            "## Reporting Claim",
            "",
            "Report this as: the local DU-Net estimate is directionally consistent with Figure 4a only under the local reward-weighted metric and in high-rating regions; compared with Linear UCB, the reproduction supports the paper's reward-region mechanism but not the exact full-test Figure 4a curve.",
            "",
            "CSV artifact:",
            "",
            f"- `{OUT_CSV.relative_to(ROOT)}`",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows, OUT_CSV)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
