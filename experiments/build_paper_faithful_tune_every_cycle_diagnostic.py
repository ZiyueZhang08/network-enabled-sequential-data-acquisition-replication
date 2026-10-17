"""Compare paper-style every-cycle tuning against the current 5-cycle fallback."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_PATH = ROOT / "reports" / "paper_faithful_tune_every_cycle_diagnostic_20260709.md"
TABLE_PATH = TABLE_DIR / "paper_faithful_tune_every_cycle_diagnostic_20260709.csv"

MAIN_METRICS = TABLE_DIR / "paper_faithful_movielens_rank20_ep20_v4_20seeds_metrics.csv"
TUNE_METRICS = TABLE_DIR / "paper_faithful_movielens_tune_every_cycle_3seeds_metrics.csv"
TUNE_SUMMARY = TABLE_DIR / "paper_faithful_movielens_tune_every_cycle_3seeds_final_summary.csv"

METRICS = [
    "standard_mask_rmse",
    "pdf_literal_norm_rmse",
    "w_rmse",
    "reward_tail_proxy_rmse",
    "cumulative_reward",
]

METRIC_LABELS = {
    "standard_mask_rmse": "standard RMSE",
    "pdf_literal_norm_rmse": "PDF-literal RMSE",
    "w_rmse": "local weighted RMSE",
    "reward_tail_proxy_rmse": "reward-tail RMSE",
    "cumulative_reward": "cumulative reward",
}

COMPARATOR_LABELS = {
    "linucb_concat": "Linear UCB concat",
    "linucb_mf_interaction": "Linear UCB MF",
}


def f6(value: float) -> str:
    return f"{float(value):.6f}"


def mainline_seed_subset() -> dict[str, float]:
    final = mainline_final_by_seed()
    return {metric: float(final[metric].mean()) for metric in METRICS}


def mainline_final_by_seed() -> pd.DataFrame:
    metrics = pd.read_csv(MAIN_METRICS)
    final = (
        metrics[(metrics["variant"] == "paper_faithful_dunet") & (metrics["seed"].isin([0, 1, 2]))]
        .sort_values("cycle")
        .groupby(["variant", "seed"], as_index=False)
        .tail(1)
    )
    if final["seed"].nunique() != 3:
        raise ValueError("Expected mainline seeds 0, 1, and 2 in the 20-seed metrics file.")
    return final


def tuned_final_by_seed() -> pd.DataFrame:
    metrics = pd.read_csv(TUNE_METRICS)
    final = (
        metrics[(metrics["variant"] == "paper_faithful_dunet") & (metrics["seed"].isin([0, 1, 2]))]
        .sort_values("cycle")
        .groupby(["variant", "seed"], as_index=False)
        .tail(1)
    )
    if final["seed"].nunique() != 3:
        raise ValueError("Expected tune-every-cycle seeds 0, 1, and 2 in the metrics file.")
    return final


def comparator_final_by_seed() -> pd.DataFrame:
    metrics = pd.read_csv(MAIN_METRICS)
    final = (
        metrics[(metrics["variant"].isin(COMPARATOR_LABELS)) & (metrics["seed"].isin([0, 1, 2]))]
        .sort_values("cycle")
        .groupby(["variant", "seed"], as_index=False)
        .tail(1)
    )
    expected_rows = 3 * len(COMPARATOR_LABELS)
    if len(final) != expected_rows:
        raise ValueError(f"Expected {expected_rows} comparator seed rows, found {len(final)}.")
    return final


def summary_row(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    if int(float(row["seed_count"])) != 3:
        raise ValueError(f"Expected a 3-seed tune-every-cycle summary, found seed_count={row['seed_count']}.")
    return {metric: float(row[metric]) for metric in METRICS}


def parameter_counts() -> pd.DataFrame:
    metrics = pd.read_csv(TUNE_METRICS)
    changed = metrics[metrics["cycle"] > 0].copy()
    counts = (
        changed.groupby(["adaptive_lambda_entropy", "sigma"], as_index=False)
        .size()
        .sort_values(["size", "adaptive_lambda_entropy", "sigma"], ascending=[False, True, True])
    )
    counts.rename(columns={"size": "cycle_seed_count"}, inplace=True)
    return counts


def write_table(main: dict[str, float], tuned: dict[str, float]) -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "mainline_tune_every_5_seed0_2": main[metric],
                "tune_every_cycle_seed0_2": tuned[metric],
                "tune_every_cycle_minus_mainline": tuned[metric] - main[metric],
                "direction": "higher_is_better" if metric == "cumulative_reward" else "lower_is_better",
            }
        )
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "metric_label",
                "mainline_tune_every_5_seed0_2",
                "tune_every_cycle_seed0_2",
                "tune_every_cycle_minus_mainline",
                "direction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parameter_table(counts: pd.DataFrame) -> list[str]:
    lines = [
        "| lambda_H | sigma | cycle-seed count |",
        "|---:|---:|---:|",
    ]
    for row in counts.head(8).itertuples(index=False):
        lines.append(
            f"| {float(row.adaptive_lambda_entropy):.6f} | {float(row.sigma):.6f} | "
            f"{int(row.cycle_seed_count)} |"
        )
    return lines


def comparator_gap_table(tuned_final: pd.DataFrame, comparator_final: pd.DataFrame) -> list[str]:
    tuned_avg = {metric: float(tuned_final[metric].mean()) for metric in METRICS}
    lines = [
        "| comparator | standard gap | PDF gap | local w-RMSE gap | reward-tail gap | reward gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, label in COMPARATOR_LABELS.items():
        comp = comparator_final[comparator_final["variant"] == variant]
        comp_avg = {metric: float(comp[metric].mean()) for metric in METRICS}
        lines.append(
            f"| {label} | "
            f"{tuned_avg['standard_mask_rmse'] - comp_avg['standard_mask_rmse']:.6f} | "
            f"{tuned_avg['pdf_literal_norm_rmse'] - comp_avg['pdf_literal_norm_rmse']:.6f} | "
            f"{tuned_avg['w_rmse'] - comp_avg['w_rmse']:.6f} | "
            f"{tuned_avg['reward_tail_proxy_rmse'] - comp_avg['reward_tail_proxy_rmse']:.6f} | "
            f"{tuned_avg['cumulative_reward'] - comp_avg['cumulative_reward']:.6f} |"
        )
    return lines


def per_seed_table(main_final: pd.DataFrame, tuned_final: pd.DataFrame) -> list[str]:
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


def build_report() -> str:
    main_final = mainline_final_by_seed()
    tuned_final = tuned_final_by_seed()
    comparator_final = comparator_final_by_seed()
    main = mainline_seed_subset()
    tuned = summary_row(TUNE_SUMMARY)
    counts = parameter_counts()
    write_table(main, tuned)

    standard_delta = tuned["standard_mask_rmse"] - main["standard_mask_rmse"]
    pdf_delta = tuned["pdf_literal_norm_rmse"] - main["pdf_literal_norm_rmse"]
    w_delta = tuned["w_rmse"] - main["w_rmse"]
    tail_delta = tuned["reward_tail_proxy_rmse"] - main["reward_tail_proxy_rmse"]
    reward_delta = tuned["cumulative_reward"] - main["cumulative_reward"]

    lines = [
        "# Tune-every-cycle diagnostic",
        "",
        "Generated by `experiments/build_paper_faithful_tune_every_cycle_diagnostic.py`.",
        "",
        "## Why This Branch Is Paper-faithful",
        "",
        "Section 3.5.1 says the algorithm tunes the hyperparameters",
        "`lambda_H`, `sigma_t`, and `p_t` using a train/acquisition/test split of",
        "the currently observed data, and repeats this process across acquisition",
        "steps. The current mainline already uses this paper-style inner split, but",
        "updates every 5 cycles as a computational fallback. This branch changes only",
        "that frequency to every cycle, while keeping MovieLens `p=0`, the same",
        "rank20/20-epoch SVD++ backend, full Gaussian DLS, full-pool Schur entropy,",
        "and exponential DU-Net sampling.",
        "",
        "## Same-seed Result",
        "",
        "Both rows use seeds 0-2. Lower is better for RMSE metrics; higher is better",
        "for cumulative reward.",
        "",
        "| branch | standard RMSE | PDF-literal RMSE | local w-RMSE | reward-tail RMSE | cumulative reward |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| current mainline, tune every 5 cycles | {f6(main['standard_mask_rmse'])} | "
            f"{f6(main['pdf_literal_norm_rmse'])} | {f6(main['w_rmse'])} | "
            f"{f6(main['reward_tail_proxy_rmse'])} | {f6(main['cumulative_reward'])} |"
        ),
        (
            f"| strict tune every cycle | {f6(tuned['standard_mask_rmse'])} | "
            f"{f6(tuned['pdf_literal_norm_rmse'])} | {f6(tuned['w_rmse'])} | "
            f"{f6(tuned['reward_tail_proxy_rmse'])} | {f6(tuned['cumulative_reward'])} |"
        ),
        "",
        "Strict tuning minus mainline:",
        "",
        "| metric | delta | direction |",
        "|---|---:|---|",
        f"| standard RMSE | {standard_delta:.6f} | lower is better |",
        f"| PDF-literal RMSE | {pdf_delta:.6f} | lower is better |",
        f"| local w-RMSE | {w_delta:.6f} | lower is better |",
        f"| reward-tail RMSE | {tail_delta:.6f} | lower is better |",
        f"| cumulative reward | {reward_delta:.6f} | higher is better |",
        "",
        "Per-seed deltas show the local weighted signal is helpful but not unanimous:",
        "",
        *per_seed_table(main_final, tuned_final),
        "",
        "## Linear UCB Position",
        "",
        "The fallback criterion is to check consistency against Linear UCB when the",
        "absolute DU-Net endpoint is not yet paper-close. Using the same seeds 0-2",
        "and the existing mainline Linear UCB comparators, the strict-tuned DU-Net",
        "gaps are:",
        "",
        *comparator_gap_table(tuned_final, comparator_final),
        "",
        "Negative gaps favor DU-Net for RMSE metrics; positive gaps favor DU-Net for",
        "reward. The strict-tuned branch is cleanly better than the MF-centered",
        "Linear UCB comparator, but it still does not beat the concat Linear UCB",
        "comparator on local weighted RMSE or cumulative reward.",
        "",
        "## Parameter Movement",
        "",
        "The every-cycle branch did not collapse to a static schedule. The most common",
        "`(lambda_H, sigma_t)` choices across post-initial cycle-seed rows were:",
        "",
        *parameter_table(counts),
        "",
        "## Reflection",
        "",
        "This branch is closer to Section 3.5.1 than the 5-cycle fallback and it moves",
        "two Figure-4a-relevant local metrics in the desired direction: local w-RMSE",
        f"improves by {abs(w_delta):.6f}, reward-tail RMSE improves by {abs(tail_delta):.6f},",
        f"and cumulative reward improves by {reward_delta:.6f}. That is a real",
        "paper-faithful signal, not an engineering-only enhancement.",
        "",
        "The caveat is equally important: conventional standard RMSE and the",
        f"PDF-literal norm RMSE worsen by {standard_delta:.6f} and {pdf_delta:.6f}.",
        "Because the paper's visual Figure 4a endpoint is w-RMSE but the exact",
        "weighting/normalization remains unrecovered, this is not enough to replace",
        "the 20-seed mainline yet. It is a promotion candidate only if we explicitly",
        "treat local reward-weighted w-RMSE as the Figure 4a proxy.",
        "",
        "## Next Step",
        "",
        "Do not start another parameter sweep. The faithful next move is to extend this",
        "single branch to a slightly wider seed check, then promote to 20 seeds only if",
        "the local w-RMSE improvement survives and the standard/PDF regression stays",
        "small. In parallel, exact author evidence for the Figure 4a w-RMSE formula or",
        "MovieLens split remains the highest-value way to decide which metric should",
        "govern promotion.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(REPORT_PATH)
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
