"""Build a user-goal aligned Figure 4a comparison brief.

The user's target is: get close to the DU-Net estimate in Figure 4a; if worse,
compare with Linear UCB to show whether the result is consistent with the
paper's second-best baseline. This report turns the current evidence into that
exact decision table without launching a new proxy experiment.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FIGURE4A_CSV = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
DISTANCE_CSV = TABLE_DIR / "figure4a_distance_scorecard.csv"
TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"

OUT_CSV = TABLE_DIR / "figure4a_user_goal_comparison_brief.csv"
OUT_REPORT = REPORT_DIR / "figure4a_user_goal_comparison_brief.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def pct(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


def build_rows() -> list[dict[str, str]]:
    figure = read_rows(FIGURE4A_CSV)
    distance = read_rows(DISTANCE_CSV)
    trigger = read_rows(TRIGGER_CSV)

    paper = row_by(figure, "claim_area", "paper_target")
    reward = row_by(figure, "claim_area", "local_reward_weighted")
    full = row_by(figure, "claim_area", "local_full_rmse")
    high = row_by(figure, "claim_area", "local_high_rating_region")
    sigmoid = row_by(figure, "claim_area", "sigmoid_reward_weighted")

    reward_distance = row_by(distance, "claim_area", "local_reward_weighted")
    full_distance = row_by(distance, "claim_area", "local_full_rmse")
    high_distance = row_by(distance, "claim_area", "local_high_rating_region")
    rerun = row_by(trigger, "trigger", "rerun_decision")

    return [
        {
            "question": "paper_target",
            "status": "reference",
            "du_net": paper["du_net"],
            "linear_ucb": paper["linear_ucb"],
            "linear_minus_du": paper["linear_minus_du"],
            "distance_to_goal": "visual target from paper Figure 4a",
            "decision": "The target favors DU-Net over Linear UCB.",
        },
        {
            "question": "is_local_du_close_to_paper_du",
            "status": "not_close_in_absolute_scale",
            "du_net": reward["du_net"],
            "linear_ucb": reward["linear_ucb"],
            "linear_minus_du": reward["linear_minus_du"],
            "distance_to_goal": (
                f"DU endpoint scale={pct(reward_distance['du_scale_ratio_to_paper'])}; "
                f"absolute DU gap={reward_distance['du_absolute_gap_to_paper']}"
            ),
            "decision": "The best local DU-Net-style endpoint is directionally useful but not close to the paper's absolute w-RMSE scale.",
        },
        {
            "question": "does_linear_ucb_comparison_match_paper_direction",
            "status": reward["status"],
            "du_net": reward["du_net"],
            "linear_ucb": reward["linear_ucb"],
            "linear_minus_du": reward["linear_minus_du"],
            "distance_to_goal": (
                f"gap_ratio_to_paper={pct(reward_distance['paper_gap_ratio'])}; "
                f"relative_improvement_vs_paper={pct(reward_distance['relative_improvement_vs_paper'])}"
            ),
            "decision": "Yes under the best reportable reward-weighted metric: DU-Net-style remains below Linear UCB.",
        },
        {
            "question": "does_full_rmse_match_paper_direction",
            "status": full["status"],
            "du_net": full["du_net"],
            "linear_ucb": full["linear_ucb"],
            "linear_minus_du": full["linear_minus_du"],
            "distance_to_goal": (
                f"gap_ratio_to_paper={pct(full_distance['paper_gap_ratio'])}; "
                f"direction_matches={full_distance['direction_matches_paper']}"
            ),
            "decision": "No. Full-test conventional RMSE favors Linear UCB, so it blocks exact Figure 4a claims.",
        },
        {
            "question": "mechanism_support",
            "status": high["status"],
            "du_net": high["du_net"],
            "linear_ucb": high["linear_ucb"],
            "linear_minus_du": high["linear_minus_du"],
            "distance_to_goal": f"high-rating gap ratio={pct(high_distance['paper_gap_ratio'])}",
            "decision": "High-rating regions support the reward-region mechanism behind the local DU-Net-style advantage.",
        },
        {
            "question": "paper_like_sigmoid_weight_check",
            "status": sigmoid["status"],
            "du_net": sigmoid["du_net"],
            "linear_ucb": sigmoid["linear_ucb"],
            "linear_minus_du": sigmoid["linear_minus_du"],
            "distance_to_goal": "near tie with slight reversal",
            "decision": "The paper-like sigmoid-weighted check is not a stable DU-Net win at 20 seeds.",
        },
        {
            "question": "user_goal_verdict",
            "status": "partially_met",
            "du_net": reward["du_net"],
            "linear_ucb": reward["linear_ucb"],
            "linear_minus_du": reward["linear_minus_du"],
            "distance_to_goal": (
                f"absolute scale not close; reward-weighted Linear-minus-DU gap recovers {pct(reward_distance['paper_gap_ratio'])} of the paper visual gap"
            ),
            "decision": "The reproduction meets the Linear-UCB consistency fallback only as a strong partial result, not as an exact DU-Net Figure 4a estimate.",
        },
        {
            "question": "execution_decision",
            "status": rerun["status"],
            "du_net": "",
            "linear_ucb": "",
            "linear_minus_du": "",
            "distance_to_goal": rerun["evidence"],
            "decision": "Do not run another local proxy sweep; wait for source-backed metric/split/backend evidence or approved dependency trigger.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| question | status | DU-Net | Linear UCB | gap | decision |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question']} | {row['status']} | {row['du_net']} | "
            f"{row['linear_ucb']} | {row['linear_minus_du']} | {row['decision']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    verdict = row_by(rows, "question", "user_goal_verdict")
    report = f"""# Figure 4a User-Goal Comparison Brief

Generated by `experiments/build_figure4a_user_goal_comparison_brief.py`.

## Current Progress

The current result is a strong partial reproduction. It does not get a DU-Net
estimate close to the paper's Figure 4a absolute endpoint, but it does satisfy
the fallback comparison against Linear UCB under the best reportable local
reward-weighted evaluation.

## Goal-Aligned Decision Table

{markdown_table(rows)}

## Analysis

The local DU-Net-style endpoint is not close to the paper's absolute w-RMSE
scale: the best local reward-weighted DU endpoint is only about 46.1% of the
paper visual DU endpoint. However, the local reward-weighted Linear-minus-DU
gap is positive and recovers about 31.9% of the paper's visual Linear-UCB gap,
so the direction against Linear UCB is consistent in the best reportable
metric. High-rating regions support the same mechanism.

The caveat is important: conventional full-test RMSE reverses the ordering, and
the paper-like sigmoid-weighted check is essentially tied. Therefore the result
should be reported as Linear-UCB-consistent partial reproduction, not exact
DU-Net/Figure 4a replication.

## Verdict

Status: `{verdict['status']}`.

{verdict['decision']}

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
