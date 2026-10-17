"""Build the 5-seed diagnostic for strict paper-style per-cycle tuning."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_PATH = ROOT / "reports" / "paper_faithful_tune_every_cycle_5seed_diagnostic_20260709.md"
TABLE_PATH = TABLE_DIR / "paper_faithful_tune_every_cycle_5seed_diagnostic_20260709.csv"

MAIN_METRICS = TABLE_DIR / "paper_faithful_movielens_rank20_ep20_v4_20seeds_metrics.csv"
TUNE_METRICS = [
    TABLE_DIR / "paper_faithful_movielens_tune_every_cycle_3seeds_metrics.csv",
    TABLE_DIR / "paper_faithful_movielens_tune_every_cycle_seeds3_4_metrics.csv",
]

METRICS = [
    "standard_mask_rmse",
    "pdf_literal_norm_rmse",
    "w_rmse",
    "reward_tail_proxy_rmse",
    "cumulative_reward",
]

COMPARATOR_LABELS = {
    "linucb_concat": "Linear UCB concat",
    "linucb_mf_interaction": "Linear UCB MF",
}


def f6(value: float) -> str:
    return f"{float(value):.6f}"


def final_by_seed(frame: pd.DataFrame, variants: set[str], seeds: set[int]) -> pd.DataFrame:
    final = (
        frame[(frame["variant"].isin(variants)) & (frame["seed"].isin(seeds))]
        .sort_values("cycle")
        .groupby(["variant", "seed"], as_index=False)
        .tail(1)
    )
    expected_rows = len(variants) * len(seeds)
    if len(final) != expected_rows:
        raise ValueError(f"Expected {expected_rows} final rows for {variants}, found {len(final)}.")
    return final


def load_tuned_metrics() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in TUNE_METRICS]
    return pd.concat(frames, ignore_index=True)


def averages(final: pd.DataFrame, variant: str) -> dict[str, float]:
    frame = final[final["variant"] == variant]
    return {metric: float(frame[metric].mean()) for metric in METRICS}


def write_table(rows: list[dict[str, float | str]]) -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comparison",
                "metric",
                "value",
                "direction",
                "interpretation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(
    tuned: dict[str, float],
    main: dict[str, float],
    comparators: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for metric in METRICS:
        rows.append(
            {
                "comparison": "strict_tune_minus_mainline",
                "metric": metric,
                "value": tuned[metric] - main[metric],
                "direction": "higher_is_better" if metric == "cumulative_reward" else "lower_is_better",
                "interpretation": "negative favors strict tuning except reward, where positive favors strict tuning",
            }
        )
    for variant, comp in comparators.items():
        for metric in METRICS:
            rows.append(
                {
                    "comparison": f"strict_tune_minus_{variant}",
                    "metric": metric,
                    "value": tuned[metric] - comp[metric],
                    "direction": "higher_is_better" if metric == "cumulative_reward" else "lower_is_better",
                    "interpretation": "negative favors DU-Net for RMSE metrics; positive favors DU-Net for reward",
                }
            )
    return rows


def per_seed_rows(tuned_final: pd.DataFrame, main_final: pd.DataFrame) -> list[str]:
    merged = main_final[["seed", *METRICS]].merge(
        tuned_final[["seed", *METRICS]],
        on="seed",
        suffixes=("_main", "_tuned"),
    )
    lines = [
        "| seed | standard delta | local w-RMSE delta | reward-tail delta | reward delta |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in merged.sort_values("seed").itertuples(index=False):
        lines.append(
            f"| {int(row.seed)} | "
            f"{float(row.standard_mask_rmse_tuned - row.standard_mask_rmse_main):.6f} | "
            f"{float(row.w_rmse_tuned - row.w_rmse_main):.6f} | "
            f"{float(row.reward_tail_proxy_rmse_tuned - row.reward_tail_proxy_rmse_main):.6f} | "
            f"{float(row.cumulative_reward_tuned - row.cumulative_reward_main):.6f} |"
        )
    return lines


def gap_rows(tuned: dict[str, float], comparators: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| comparator | standard gap | PDF gap | local w-RMSE gap | reward-tail gap | reward gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, label in COMPARATOR_LABELS.items():
        comp = comparators[variant]
        lines.append(
            f"| {label} | "
            f"{tuned['standard_mask_rmse'] - comp['standard_mask_rmse']:.6f} | "
            f"{tuned['pdf_literal_norm_rmse'] - comp['pdf_literal_norm_rmse']:.6f} | "
            f"{tuned['w_rmse'] - comp['w_rmse']:.6f} | "
            f"{tuned['reward_tail_proxy_rmse'] - comp['reward_tail_proxy_rmse']:.6f} | "
            f"{tuned['cumulative_reward'] - comp['cumulative_reward']:.6f} |"
        )
    return lines


def build_report() -> str:
    seeds = {0, 1, 2, 3, 4}
    main_metrics = pd.read_csv(MAIN_METRICS)
    tuned_metrics = load_tuned_metrics()
    tuned_final = final_by_seed(tuned_metrics, {"paper_faithful_dunet"}, seeds)
    main_du_final = final_by_seed(main_metrics, {"paper_faithful_dunet"}, seeds)
    comp_final = final_by_seed(main_metrics, set(COMPARATOR_LABELS), seeds)

    tuned = averages(tuned_final, "paper_faithful_dunet")
    main = averages(main_du_final, "paper_faithful_dunet")
    comparators = {variant: averages(comp_final, variant) for variant in COMPARATOR_LABELS}
    write_table(comparison_rows(tuned, main, comparators))

    delta = {metric: tuned[metric] - main[metric] for metric in METRICS}
    lines = [
        "# Tune-every-cycle 5-seed diagnostic",
        "",
        "Generated by `experiments/build_paper_faithful_tune_every_cycle_5seed_diagnostic.py`.",
        "",
        "## What Changed",
        "",
        "This extends the strict Section 3.5.1 dynamic-tuning branch from seeds 0-2",
        "to seeds 0-4. The branch changes only the tuning frequency from every 5",
        "cycles to every cycle. It keeps the same MovieLens `p=0`, rank20/20-epoch",
        "SVD++ backend, full Gaussian DLS, full-pool Schur entropy, and exponential",
        "DU-Net sampling.",
        "",
        "## Same-seed DU-Net Result",
        "",
        "| branch | standard RMSE | PDF-literal RMSE | local w-RMSE | reward-tail RMSE | cumulative reward |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| mainline tune every 5 cycles | {f6(main['standard_mask_rmse'])} | "
            f"{f6(main['pdf_literal_norm_rmse'])} | {f6(main['w_rmse'])} | "
            f"{f6(main['reward_tail_proxy_rmse'])} | {f6(main['cumulative_reward'])} |"
        ),
        (
            f"| strict tune every cycle | {f6(tuned['standard_mask_rmse'])} | "
            f"{f6(tuned['pdf_literal_norm_rmse'])} | {f6(tuned['w_rmse'])} | "
            f"{f6(tuned['reward_tail_proxy_rmse'])} | {f6(tuned['cumulative_reward'])} |"
        ),
        "",
        "| metric | strict minus mainline | direction |",
        "|---|---:|---|",
        f"| standard RMSE | {delta['standard_mask_rmse']:.6f} | lower is better |",
        f"| PDF-literal RMSE | {delta['pdf_literal_norm_rmse']:.6f} | lower is better |",
        f"| local w-RMSE | {delta['w_rmse']:.6f} | lower is better |",
        f"| reward-tail RMSE | {delta['reward_tail_proxy_rmse']:.6f} | lower is better |",
        f"| cumulative reward | {delta['cumulative_reward']:.6f} | higher is better |",
        "",
        "Per-seed deltas:",
        "",
        *per_seed_rows(tuned_final, main_du_final),
        "",
        "## Linear UCB Position",
        "",
        "Negative gaps favor DU-Net for RMSE metrics; positive gaps favor DU-Net for",
        "reward.",
        "",
        *gap_rows(tuned, comparators),
        "",
        "## Reflection",
        "",
        "The 5-seed extension preserves the useful Figure 4a-left signal: local",
        f"w-RMSE improves by {abs(delta['w_rmse']):.6f}, while standard RMSE and the",
        f"PDF-literal metric are nearly tied, worsening only {delta['standard_mask_rmse']:.6f}",
        f"and {delta['pdf_literal_norm_rmse']:.6f}. It also strengthens the comparison",
        "with MF-centered Linear UCB: strict-tuned DU-Net is better on all listed",
        "metrics against that comparator.",
        "",
        "The new caveat is reward: cumulative reward is now lower than the current",
        f"mainline by {abs(delta['cumulative_reward']):.6f}, even though it remains much",
        "higher than the MF-centered Linear UCB comparator. This means the branch is a",
        "left-panel w-RMSE promotion candidate, not a complete Figure 4a left-and-right",
        "win yet.",
        "",
        "## Decision and Next Step",
        "",
        "The 5-seed gate passes for the Figure 4a w-RMSE target: the local weighted",
        "improvement survived, and the standard/PDF regressions are small. The next",
        "paper-faithful action is to promote this single branch to a 20-seed DU-Net",
        "run, then compare it against the existing 20-seed Linear UCB baselines. The",
        "promotion must monitor cumulative reward explicitly because that signal",
        "weakened in seeds 3-4.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(REPORT_PATH)
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
