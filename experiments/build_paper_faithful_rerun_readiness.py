"""Build a readiness checklist for a future paper-faithful rerun.

This is a planning artifact, not a new experiment. It records the current gate,
minimum prerequisites, and the smallest diagnostic that should run if a real
paper-faithful trigger appears.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
DISTANCE_CSV = TABLE_DIR / "figure4a_distance_scorecard.csv"
SNAPSHOT_CSV = TABLE_DIR / "reproduction_iteration_status_snapshot.csv"
INTEGRITY_CSV = TABLE_DIR / "reproduction_artifact_integrity_check.csv"

OUT_CSV = TABLE_DIR / "paper_faithful_rerun_readiness.csv"
OUT_REPORT = REPORT_DIR / "paper_faithful_rerun_readiness.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def count_status(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_rows() -> list[dict[str, str]]:
    trigger_rows = read_rows(TRIGGER_CSV)
    distance_rows = read_rows(DISTANCE_CSV)
    snapshot_rows = read_rows(SNAPSHOT_CSV)
    integrity_counts = count_status(read_rows(INTEGRITY_CSV))

    rerun = row_by(trigger_rows, "trigger", "rerun_decision")
    overall = row_by(snapshot_rows, "area", "overall_progress")
    reward = row_by(distance_rows, "claim_area", "local_reward_weighted")
    full = row_by(distance_rows, "claim_area", "local_full_rmse")
    high = row_by(distance_rows, "claim_area", "local_high_rating_region")

    ready = rerun["status"] != "blocked_by_missing_source"
    return [
        {
            "item": "current_reproduction_state",
            "status": overall["status"],
            "evidence": overall["evidence"],
            "action": "Use the current result as a strong partial reproduction, not an exact Figure 4a reproduction.",
        },
        {
            "item": "artifact_integrity_gate",
            "status": "pass" if integrity_counts.get("fail", 0) == 0 else "fail",
            "evidence": f"{integrity_counts.get('pass', 0)} pass; {integrity_counts.get('fail', 0)} fail",
            "action": "Keep the handoff package stable before any future rerun.",
        },
        {
            "item": "paper_faithful_trigger_gate",
            "status": "ready" if ready else "not_ready",
            "evidence": rerun["evidence"],
            "action": rerun["decision"],
        },
        {
            "item": "current_reward_weighted_threshold",
            "status": reward["status"],
            "evidence": (
                f"gap={reward['linear_minus_du']}; "
                f"paper_gap_ratio={float(reward['paper_gap_ratio']) * 100:.1f}%; "
                f"scale_ratio={float(reward['du_scale_ratio_to_paper']) * 100:.1f}%"
            ),
            "action": "A faithful diagnostic should improve this gap or explain the scale mismatch.",
        },
        {
            "item": "current_full_rmse_blocker",
            "status": full["status"],
            "evidence": f"gap={full['linear_minus_du']}; direction_matches={full['direction_matches_paper']}",
            "action": "A faithful diagnostic should not ignore the full-RMSE reversal.",
        },
        {
            "item": "current_high_rating_support",
            "status": high["status"],
            "evidence": (
                f"gap={high['linear_minus_du']}; "
                f"paper_gap_ratio={float(high['paper_gap_ratio']) * 100:.1f}%"
            ),
            "action": "Preserve this mechanism signal when testing any recovered mechanism.",
        },
        {
            "item": "minimum_next_diagnostic",
            "status": "blocked_until_trigger",
            "evidence": "Run only after source, exact metric/split/config, or approved validated mechanism dependency appears.",
            "action": "Run the smallest MovieLens Figure 4a 3-seed diagnostic against DU-Net, MF-centered Linear UCB, and Myopic.",
        },
        {
            "item": "promotion_rule",
            "status": "defined",
            "evidence": "Do not promote directly from source recovery to a 20-seed run.",
            "action": "Promote to 20 seeds only if the 3-seed diagnostic improves the current reward-weighted gap or resolves the full-RMSE ordering/scale mismatch.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["| item | status | evidence | action |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['item']} | {row['status']} | {row['evidence']} | {row['action']} |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Paper-Faithful Rerun Readiness

Generated by `experiments/build_paper_faithful_rerun_readiness.py`.

## Current Progress

The reproduction is ready for reporting as a strong partial reproduction, but it
is not ready for a paper-faithful rerun. The missing item is not another local
proxy row; it is a source-backed trigger such as author code, supplement,
split/config files, exact Figure 4a w-RMSE definition, or an approved validated
mechanism dependency.

## Readiness Table

{markdown_table(rows)}

## Next Step

Do not start a new 20-seed run from the current state. If a real trigger appears,
first run the smallest relevant MovieLens Figure 4a 3-seed diagnostic and compare
DU-Net-style behavior against MF-centered Linear UCB and Myopic. Promote to a
20-seed rerun only if that diagnostic improves the current reward-weighted gap
or resolves the full-RMSE ordering/absolute-scale mismatch.

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
