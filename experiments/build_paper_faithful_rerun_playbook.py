"""Build a playbook for the next paper-faithful rerun after a trigger.

The current gate is closed, so this script does not run a model. It converts
the source request packet, minimal diagnostic protocol, and current Figure 4a
goal verdict into a concrete sequence to execute once source-backed evidence or
an approved dependency trigger appears.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

SOURCE_PACKET = TABLE_DIR / "paper_source_request_packet.csv"
PROTOCOL = TABLE_DIR / "minimal_faithful_diagnostic_protocol.csv"
USER_GOAL = TABLE_DIR / "figure4a_user_goal_comparison_brief.csv"
TRIGGER = TABLE_DIR / "paper_faithful_trigger_status.csv"

OUT_CSV = TABLE_DIR / "paper_faithful_rerun_playbook.csv"
OUT_REPORT = REPORT_DIR / "paper_faithful_rerun_playbook.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def build_rows() -> list[dict[str, str]]:
    source = read_rows(SOURCE_PACKET)
    protocol = read_rows(PROTOCOL)
    goal = read_rows(USER_GOAL)
    trigger = read_rows(TRIGGER)

    metric = row_by(source, "request_id", "exact_figure4a_metric")
    split = row_by(source, "request_id", "exact_split_and_preprocessing")
    backend = row_by(source, "request_id", "du_net_backend_config")
    linear = row_by(source, "request_id", "linear_ucb_adapter")
    dependency = row_by(source, "request_id", "validated_external_dependency_approval")
    precondition = row_by(protocol, "stage", "precondition")
    diagnostic = row_by(protocol, "stage", "diagnostic_scope")
    primary = row_by(protocol, "stage", "primary_metric_threshold")
    blocker = row_by(protocol, "stage", "blocker_metric_threshold")
    preservation = row_by(protocol, "stage", "mechanism_preservation")
    promotion = row_by(protocol, "stage", "promotion_decision")
    verdict = row_by(goal, "question", "user_goal_verdict")
    rerun = row_by(trigger, "trigger", "rerun_decision")

    return [
        {
            "step": "0_gate_check",
            "status": precondition["status"],
            "input_required": precondition["requirement"],
            "action": "Start only if at least one P0 trigger is available.",
            "pass_rule": precondition["pass_rule"],
            "current_evidence": rerun["evidence"],
        },
        {
            "step": "1_source_intake",
            "status": "pending_until_trigger",
            "input_required": "Author code/supplement/source package or approved dependency trigger.",
            "action": "Classify the new material as metric, split/preprocessing, backend, Linear UCB adapter, or dependency evidence.",
            "pass_rule": "The artifact maps to at least one unresolved P0/P1 request without guessed implementation details.",
            "current_evidence": verdict["decision"],
        },
        {
            "step": "2_metric_mapping",
            "status": metric["current_status"],
            "input_required": metric["needed_artifact"],
            "action": "Implement or configure the exact Figure 4a w-RMSE mask, weights, normalization, and aggregation.",
            "pass_rule": metric["acceptance_test"],
            "current_evidence": metric["why_needed"],
        },
        {
            "step": "3_split_preprocessing_mapping",
            "status": split["current_status"],
            "input_required": split["needed_artifact"],
            "action": "Replace guessed MovieLens split/masks with recovered split files, initial set, candidate set, filtering, and seed mapping.",
            "pass_rule": split["acceptance_test"],
            "current_evidence": split["why_needed"],
        },
        {
            "step": "4_du_net_backend_mapping",
            "status": backend["current_status"],
            "input_required": backend["needed_artifact"],
            "action": "Map recovered DU-Net architecture, entropy, DLS, diffusion, and SVD++ settings to local code/config.",
            "pass_rule": backend["acceptance_test"],
            "current_evidence": backend["why_needed"],
        },
        {
            "step": "5_linear_ucb_adapter_mapping",
            "status": linear["current_status"],
            "input_required": linear["needed_artifact"],
            "action": "Verify or replace MF-centered Linear UCB features, reward transform, regularization, alpha, and update schedule.",
            "pass_rule": linear["acceptance_test"],
            "current_evidence": linear["why_needed"],
        },
        {
            "step": "6_dependency_path_if_approved",
            "status": dependency["current_status"],
            "input_required": dependency["needed_artifact"],
            "action": "If author source remains absent and approval exists, introduce only the validated dependency needed for a bounded mechanism diagnostic.",
            "pass_rule": dependency["acceptance_test"],
            "current_evidence": dependency["why_needed"],
        },
        {
            "step": "7_minimal_3seed_diagnostic",
            "status": diagnostic["status"],
            "input_required": diagnostic["requirement"],
            "action": "Run MovieLens Figure 4a seeds 0-2 only, comparing DU-Net-style, MF-centered Linear UCB, and Myopic under the same recovered setup.",
            "pass_rule": diagnostic["pass_rule"],
            "current_evidence": diagnostic["evidence"],
        },
        {
            "step": "8_primary_threshold",
            "status": primary["status"],
            "input_required": primary["requirement"],
            "action": "Compare against frozen baseline reward-weighted gap and paper scale ratio.",
            "pass_rule": primary["pass_rule"],
            "current_evidence": primary["evidence"],
        },
        {
            "step": "9_blocker_threshold",
            "status": blocker["status"],
            "input_required": blocker["requirement"],
            "action": "Check whether full-RMSE reversal is resolved, explained, or proven irrelevant to the paper metric.",
            "pass_rule": blocker["pass_rule"],
            "current_evidence": blocker["evidence"],
        },
        {
            "step": "10_mechanism_preservation",
            "status": preservation["status"],
            "input_required": preservation["requirement"],
            "action": "Confirm recovered mechanism does not erase high-rating/reward-region support unless it replaces the metric interpretation.",
            "pass_rule": preservation["pass_rule"],
            "current_evidence": preservation["evidence"],
        },
        {
            "step": "11_promote_or_freeze",
            "status": promotion["status"],
            "input_required": promotion["requirement"],
            "action": "Promote to 20 seeds only after the 3-seed diagnostic improves a primary blocker; otherwise freeze and report the diagnostic.",
            "pass_rule": promotion["pass_rule"],
            "current_evidence": promotion["evidence"],
        },
        {
            "step": "execution_decision",
            "status": "ready_but_blocked",
            "input_required": "Any source-backed P0 trigger or explicit approved dependency trigger.",
            "action": "Keep this playbook as the next executable path; do not run it while rerun_decision is blocked_by_missing_source.",
            "pass_rule": "Run only after the gate check passes.",
            "current_evidence": rerun["status"],
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| step | status | action | pass rule |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['step']} | {row['status']} | {row['action']} | {row['pass_rule']} |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    decision = row_by(rows, "step", "execution_decision")
    report = f"""# Paper-Faithful Rerun Playbook

Generated by `experiments/build_paper_faithful_rerun_playbook.py`.

## Current Progress

The current reproduction is a strong partial result. The user-goal verdict is
partially met: the local reward-weighted comparison is directionally consistent
with Linear UCB, but the local DU-Net-style endpoint is not close to the paper's
absolute Figure 4a scale and full RMSE still reverses the ordering.

This playbook is the next executable path after a source-backed trigger. It
does not authorize another local proxy sweep while the gate is closed.

## Playbook

{markdown_table(rows)}

## Decision

Status: `{decision['status']}`.

{decision['action']}

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
