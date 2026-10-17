"""Build the current Figure 4a replication execution matrix.

The script intentionally reads the raw experiment CSVs instead of copying
numbers from a written report. It produces a compact comparison table for the
current frozen MovieLens protocol and a Markdown reflection note.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

METRICS_PREFIX = "movielens_100k_official_u1_head_late_half_start16_20seeds"
RATING_PREFIX = "movielens_100k_official_u1_best_candidate_rating_bins_20seeds"

BEST_VARIANT = "graph_late_lambda_half_start16"
COMPARATORS = ("mf_linear_ucb_a002", "myopic", "graph_base")
VARIANT_ORDER = (BEST_VARIANT, "myopic", "mf_linear_ucb_a002", "graph_base")

PAPER_VISUAL_DU = 1.788
PAPER_VISUAL_LINEAR_UCB = 1.814
PAPER_VISUAL_GAP = PAPER_VISUAL_LINEAR_UCB - PAPER_VISUAL_DU


def ci95(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) <= 1:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / np.sqrt(len(clean)))


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def variant_rank(variant: str) -> int:
    try:
        return VARIANT_ORDER.index(variant)
    except ValueError:
        return len(VARIANT_ORDER)


def markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    headers = list(frame.columns)
    rows = [[fmt(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_matrix(metrics: pd.DataFrame, summary: pd.DataFrame, rating_summary: pd.DataFrame) -> pd.DataFrame:
    selection = (
        metrics.groupby("variant", as_index=False)
        .agg(
            selected_reward_percentile_mean=("selected_avg_reward_percentile", "mean"),
            selected_entropy_percentile_mean=("selected_avg_entropy_percentile", "mean"),
            selected_reward_percentile_late16_19=(
                "selected_avg_reward_percentile",
                lambda s: s.loc[metrics.loc[s.index, "cycle"].between(16, 19)].mean(),
            ),
            selected_entropy_percentile_late16_19=(
                "selected_avg_entropy_percentile",
                lambda s: s.loc[metrics.loc[s.index, "cycle"].between(16, 19)].mean(),
            ),
            selected_high_degree_share_late16_19=(
                "selected_high_degree_share",
                lambda s: s.loc[metrics.loc[s.index, "cycle"].between(16, 19)].mean(),
            ),
        )
    )
    rating_cols = [
        "variant",
        "high_rating_rmse_ge4",
        "high_rating_rmse_ge4_ci95",
        "low_rating_rmse_le3",
        "low_rating_rmse_le3_ci95",
        "rating5_rmse",
        "rating5_rmse_ci95",
        "rating4_rmse",
        "rating4_rmse_ci95",
        "pred_mean",
        "bias",
        "high_bias_ge4",
        "low_bias_le3",
    ]
    matrix = summary.merge(rating_summary[rating_cols], on="variant", how="left")
    matrix = matrix.merge(selection, on="variant", how="left")
    matrix = matrix.sort_values("variant", key=lambda col: col.map(variant_rank)).reset_index(drop=True)
    return matrix


def build_comparisons(metrics: pd.DataFrame, matrix: pd.DataFrame, rating_summary: pd.DataFrame) -> pd.DataFrame:
    final = metrics.loc[metrics["cycle"].eq(20)].copy()
    best = final.loc[final["variant"].eq(BEST_VARIANT)].set_index("seed")
    matrix_by_variant = matrix.set_index("variant")
    rating_by_variant = rating_summary.set_index("variant")
    du_row = matrix_by_variant.loc[BEST_VARIANT]
    du_rating = rating_by_variant.loc[BEST_VARIANT]

    rows: list[dict[str, object]] = []
    for comparator in COMPARATORS:
        comp = final.loc[final["variant"].eq(comparator)].set_index("seed")
        comp_row = matrix_by_variant.loc[comparator]
        comp_rating = rating_by_variant.loc[comparator]
        primary_gaps = comp["linear_reward_norm_w_rmse"] - best["linear_reward_norm_w_rmse"]
        reward_gaps = best["cumulative_reward"] - comp["cumulative_reward"]
        rmse_gaps = comp["rmse"] - best["rmse"]
        w_gaps = comp["w_rmse"] - best["w_rmse"]
        positive_gaps = comp["positive_reward_norm_w_rmse"] - best["positive_reward_norm_w_rmse"]
        rows.append(
            {
                "du_variant": BEST_VARIANT,
                "comparator": comparator,
                "seed_count": int(primary_gaps.notna().sum()),
                "primary_linear_reward_norm_gap_favoring_du": float(primary_gaps.mean()),
                "primary_gap_ci95": ci95(primary_gaps),
                "primary_seed_wins": int((primary_gaps > 0).sum()),
                "cumulative_reward_gap_favoring_du": float(reward_gaps.mean()),
                "reward_gap_ci95": ci95(reward_gaps),
                "reward_seed_wins": int((reward_gaps > 0).sum()),
                "w_rmse_gap_favoring_du": float(w_gaps.mean()),
                "positive_reward_norm_gap_favoring_du": float(positive_gaps.mean()),
                "conventional_rmse_gap_favoring_du": float(rmse_gaps.mean()),
                "high_rating_rmse_gap_favoring_du": float(
                    comp_rating["high_rating_rmse_ge4"] - du_rating["high_rating_rmse_ge4"]
                ),
                "low_rating_rmse_gap_favoring_du": float(
                    comp_rating["low_rating_rmse_le3"] - du_rating["low_rating_rmse_le3"]
                ),
                "rating5_rmse_gap_favoring_du": float(comp_rating["rating5_rmse"] - du_rating["rating5_rmse"]),
                "seed15_primary_gap_favoring_du": float(primary_gaps.loc[15]),
                "seed15_reward_gap_favoring_du": float(reward_gaps.loc[15]),
                "seed15_rmse_gap_favoring_du": float(rmse_gaps.loc[15]),
                "du_value": float(du_row["linear_reward_norm_w_rmse"]),
                "comparator_value": float(comp_row["linear_reward_norm_w_rmse"]),
            }
        )
    return pd.DataFrame(rows)


def build_seed15_table(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variant",
        "cycle",
        "cumulative_reward",
        "linear_reward_norm_w_rmse",
        "rmse",
        "selected_avg_reward_percentile",
        "selected_avg_entropy_percentile",
        "selected_high_degree_share",
    ]
    keep_variants = [BEST_VARIANT, "graph_base", "mf_linear_ucb_a002", "myopic"]
    return metrics.loc[
        metrics["seed"].eq(15) & metrics["variant"].isin(keep_variants) & metrics["cycle"].between(15, 20),
        columns,
    ].sort_values(["variant", "cycle"], key=lambda col: col.map(variant_rank) if col.name == "variant" else col)


def write_report(matrix: pd.DataFrame, comparisons: pd.DataFrame, seed15: pd.DataFrame) -> Path:
    linear = comparisons.loc[comparisons["comparator"].eq("mf_linear_ucb_a002")].iloc[0]
    myopic = comparisons.loc[comparisons["comparator"].eq("myopic")].iloc[0]
    completion = 100.0 * linear["primary_linear_reward_norm_gap_favoring_du"] / PAPER_VISUAL_GAP
    remaining = PAPER_VISUAL_GAP - linear["primary_linear_reward_norm_gap_favoring_du"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "figure4a_latest_execution_reflection.md"
    lines = [
        "# Figure 4a Latest Execution Reflection",
        "",
        "## Source Tables",
        "",
        f"- Full-cycle metrics: `outputs/tables/{METRICS_PREFIX}_metrics.csv`",
        f"- Full-cycle summary: `outputs/tables/{METRICS_PREFIX}_summary.csv`",
        f"- Rating-bin diagnostics: `outputs/tables/{RATING_PREFIX}_summary.csv`",
        "",
        "## Current Paper Comparison",
        "",
        f"- Paper visual DU/Linear gap: `{PAPER_VISUAL_GAP:.6f}` "
        f"({PAPER_VISUAL_LINEAR_UCB:.3f} - {PAPER_VISUAL_DU:.3f}).",
        "- Local DU vs MF-centered Linear UCB proxy gap: "
        f"`{linear['primary_linear_reward_norm_gap_favoring_du']:.6f}`.",
        f"- Local completion of paper visual gap: `{completion:.1f}%`; remaining gap: `{remaining:.6f}`.",
        "- Local DU vs Myopic gap: "
        f"`{myopic['primary_linear_reward_norm_gap_favoring_du']:.6f}`.",
        "",
        "## Execution Matrix",
        "",
        markdown_table(
            matrix[
                [
                    "variant",
                    "policy",
                    "seed_count",
                    "cumulative_reward",
                    "rmse",
                    "w_rmse",
                    "linear_reward_norm_w_rmse",
                    "positive_reward_norm_w_rmse",
                    "high_rating_rmse_ge4",
                    "low_rating_rmse_le3",
                    "rating5_rmse",
                ]
            ]
        ),
        "",
        "## DU-Net Comparisons",
        "",
        markdown_table(
            comparisons[
                [
                    "comparator",
                    "primary_linear_reward_norm_gap_favoring_du",
                    "primary_seed_wins",
                    "cumulative_reward_gap_favoring_du",
                    "reward_seed_wins",
                    "conventional_rmse_gap_favoring_du",
                    "high_rating_rmse_gap_favoring_du",
                    "low_rating_rmse_gap_favoring_du",
                    "rating5_rmse_gap_favoring_du",
                ]
            ]
        ),
        "",
        "## Reflection",
        "",
        "- Phase 0/2 is now complete: Myopic is present in the same official-u1/head-init protocol.",
        "- Myopic does not change the main conclusion. It is almost tied with the MF-centered Linear UCB proxy, "
        "and DU-Net beats both under the local paper-like reward-weighted metric.",
        "- The remaining gap to the paper visual target did not shrink; the result is still a conditional, "
        "directionally consistent reproduction rather than an exact Figure 4a reproduction.",
        "- Conventional RMSE still favors the comparators, so the final claim should stay scoped to reward-weighted "
        "or high-reward-tail performance.",
        "- Seed 15 remains the useful next diagnostic: the frozen late-lambda schedule loses there to both Linear "
        "UCB proxy and Myopic on the primary metric and reward.",
        "",
        "## Seed 15 Late-Cycle Snapshot",
        "",
        markdown_table(seed15),
        "",
        "## Next Step",
        "",
        "Run the Phase 1 micro-screen on seeds `0,1,2,15`: add/measure `late lambda half start15`, "
        "reuse `late lambda half start17`, and test the existing `frontier95_floor25` guard. Promote nothing unless it "
        "improves seed 15 without lowering the four-seed mean against the frozen baseline.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    metrics = read_csv(f"{METRICS_PREFIX}_metrics.csv")
    summary = read_csv(f"{METRICS_PREFIX}_summary.csv")
    rating_summary = read_csv(f"{RATING_PREFIX}_summary.csv")

    matrix = build_matrix(metrics, summary, rating_summary)
    comparisons = build_comparisons(metrics, matrix, rating_summary)
    seed15 = build_seed15_table(metrics)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    matrix_path = TABLE_DIR / "figure4a_latest_execution_matrix.csv"
    comparisons_path = TABLE_DIR / "figure4a_latest_execution_comparisons.csv"
    seed15_path = TABLE_DIR / "figure4a_latest_seed15_snapshot.csv"
    matrix.to_csv(matrix_path, index=False)
    comparisons.to_csv(comparisons_path, index=False)
    seed15.to_csv(seed15_path, index=False)
    report_path = write_report(matrix, comparisons, seed15)

    print(f"Wrote {matrix_path}")
    print(f"Wrote {comparisons_path}")
    print(f"Wrote {seed15_path}")
    print(f"Wrote {report_path}")
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
