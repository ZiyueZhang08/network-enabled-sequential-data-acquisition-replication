"""Build a claim-to-evidence index for the reproduction handoff.

The index is a reporting aid. It maps each allowed/non-allowed claim to the
current evidence artifact and keeps the final story auditable.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FIGURE4A_CSV = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
USER_GOAL_CSV = TABLE_DIR / "figure4a_user_goal_comparison_brief.csv"
SNAPSHOT_CSV = TABLE_DIR / "reproduction_iteration_status_snapshot.csv"
INTEGRITY_CSV = TABLE_DIR / "reproduction_artifact_integrity_check.csv"
TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
SOURCE_CSV = TABLE_DIR / "paper_source_request_packet.csv"
AUTHOR_REQUEST_CSV = TABLE_DIR / "author_source_request_email.csv"
PROTOCOL_CSV = TABLE_DIR / "minimal_faithful_diagnostic_protocol.csv"
PLAYBOOK_CSV = TABLE_DIR / "paper_faithful_rerun_playbook.csv"
INSPECTION_PROTOCOL_CSV = TABLE_DIR / "source_artifact_inspection_protocol.csv"
SOURCE_TRIGGER_MONITOR_CSV = TABLE_DIR / "source_trigger_monitor_snapshot.csv"
SOURCE_TRIGGER_INTAKE_CSV = TABLE_DIR / "source_trigger_intake_check.csv"
CONTINUATION_CSV = TABLE_DIR / "reproduction_continuation_audit.csv"
OFFICIAL_U1_GAPS_CSV = TABLE_DIR / "movielens_100k_svdpp_official_u1_sensitivity_3seeds_gaps.csv"
OFFICIAL_FOLDS_GAPS_CSV = TABLE_DIR / "movielens_100k_svdpp_official_folds_sensitivity_fold_gaps.csv"
METHOD_DETAIL_CSV = TABLE_DIR / "paper_method_detail_keyword_audit.csv"
PAPER_URL_CSV = TABLE_DIR / "paper_url_dependency_audit.csv"
LOCAL_SOURCE_INVENTORY_CSV = TABLE_DIR / "local_source_artifact_inventory_audit.csv"
METRIC_FORMULA_CSV = TABLE_DIR / "figure4a_metric_formula_visual_audit.csv"
PUBLIC_RESCAN_CSV = TABLE_DIR / "public_source_rescan_audit.csv"
REMAINING_GAP_CSV = TABLE_DIR / "remaining_gap_priority_audit.csv"
GCMC_SPLIT_AUDIT_CSV = TABLE_DIR / "gcmc_monti_split_provenance_audit.csv"
EXECUTION_GATE_CSV = TABLE_DIR / "execution_gate_closure_audit.csv"

OUT_CSV = TABLE_DIR / "reproduction_claim_evidence_index.csv"
OUT_REPORT = REPORT_DIR / "reproduction_claim_evidence_index.md"


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


def linear_gaps_for_metric(rows: list[dict[str, str]], metric: str) -> list[float]:
    return [float(row["linear_gap_favoring_du"]) for row in rows if row["metric"] == metric]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_rows() -> list[dict[str, str]]:
    figure = read_rows(FIGURE4A_CSV)
    user_goal = read_rows(USER_GOAL_CSV)
    snapshot = read_rows(SNAPSHOT_CSV)
    integrity = count_status(read_rows(INTEGRITY_CSV))
    trigger = read_rows(TRIGGER_CSV)
    source = read_rows(SOURCE_CSV)
    author_request = read_rows(AUTHOR_REQUEST_CSV)
    protocol = read_rows(PROTOCOL_CSV)
    playbook = read_rows(PLAYBOOK_CSV)
    inspection_protocol = read_rows(INSPECTION_PROTOCOL_CSV)
    source_trigger_monitor = read_rows(SOURCE_TRIGGER_MONITOR_CSV)
    source_trigger_intake = read_rows(SOURCE_TRIGGER_INTAKE_CSV)
    continuation = read_rows(CONTINUATION_CSV)
    official_u1_gaps = read_rows(OFFICIAL_U1_GAPS_CSV)
    official_folds_gaps = read_rows(OFFICIAL_FOLDS_GAPS_CSV)
    method_detail = read_rows(METHOD_DETAIL_CSV)
    paper_urls = read_rows(PAPER_URL_CSV)
    local_inventory = read_rows(LOCAL_SOURCE_INVENTORY_CSV)
    metric_formula = read_rows(METRIC_FORMULA_CSV)
    public_rescan = read_rows(PUBLIC_RESCAN_CSV)
    remaining_gap = read_rows(REMAINING_GAP_CSV)
    gcmc_split = read_rows(GCMC_SPLIT_AUDIT_CSV)
    execution_gate = read_rows(EXECUTION_GATE_CSV)

    overall = row_by(snapshot, "area", "overall_progress")
    paper = row_by(figure, "claim_area", "paper_target")
    user_goal_verdict = row_by(user_goal, "question", "user_goal_verdict")
    weighted = row_by(figure, "claim_area", "local_reward_weighted")
    full = row_by(figure, "claim_area", "local_full_rmse")
    high = row_by(figure, "claim_area", "local_high_rating_region")
    sigmoid = row_by(figure, "claim_area", "sigmoid_reward_weighted")
    rerun = row_by(trigger, "trigger", "rerun_decision")
    author_request_decision = row_by(author_request, "item", "execution_decision")
    continuation_decision = row_by(continuation, "check", "continuation_decision")
    source_p0 = [
        row["request_id"]
        for row in source
        if row["priority"] == "P0"
        and row["request_id"] not in {"minimum_rerun_protocol", "promotion_rule"}
        and row["current_status"] not in {"available", "defined"}
    ]
    protocol_precondition = row_by(protocol, "stage", "precondition")
    playbook_decision = row_by(playbook, "step", "execution_decision")
    inspection_decision = row_by(inspection_protocol, "candidate_class", "execution_decision")
    source_trigger_gate = row_by(source_trigger_monitor, "monitor_item", "rerun_gate")
    source_trigger_intake_gate = row_by(source_trigger_intake, "item", "rerun_gate")
    official_u1_weighted = row_by(official_u1_gaps, "metric", "w_rmse")
    official_u1_rmse = row_by(official_u1_gaps, "metric", "rmse")
    official_folds_weighted = linear_gaps_for_metric(official_folds_gaps, "w_rmse")
    official_folds_rmse = linear_gaps_for_metric(official_folds_gaps, "rmse")
    method_execution = row_by(method_detail, "keyword_group", "execution_decision")
    paper_url_decision = row_by(paper_urls, "url", "execution_decision")
    inventory_decision = row_by(local_inventory, "artifact", "execution_decision")
    metric_standard = row_by(metric_formula, "metric_interpretation", "standard_rmse_indicator_mask")
    metric_literal = row_by(metric_formula, "metric_interpretation", "page28_literal_cardinality_outside_sqrt")
    public_rescan_decision = row_by(public_rescan, "check_id", "execution_decision")
    remaining_gap_decision = row_by(remaining_gap, "gap_id", "execution_decision")
    gcmc_exec = row_by(gcmc_split, "check_id", "execution_decision")
    gate_decision = row_by(execution_gate, "gate_id", "execution_decision")

    return [
        {
            "claim_id": "overall_status",
            "claim_type": "allowed",
            "claim": "The project is a strong partial reproduction, not an exact DU-Net replication.",
            "evidence": overall["evidence"],
            "primary_artifact": "reports/reproduction_iteration_status_snapshot.md",
            "reporting_note": overall["decision"],
        },
        {
            "claim_id": "paper_target",
            "claim_type": "allowed",
            "claim": "The Figure 4a visual target favors DU-Net over Linear UCB.",
            "evidence": f"DU-Net {paper['du_net']} vs Linear UCB {paper['linear_ucb']}; gap {paper['linear_minus_du']}",
            "primary_artifact": "reports/final_figure4a_caveated_delivery.md",
            "reporting_note": "Use as visual target, not exact extracted numeric ground truth.",
        },
        {
            "claim_id": "reward_weighted_direction",
            "claim_type": "allowed",
            "claim": "Local reward-weighted Figure 4a direction is consistent with DU-Net outperforming Linear UCB.",
            "evidence": f"DU-Net {weighted['du_net']} vs Linear UCB {weighted['linear_ucb']}; gap {weighted['linear_minus_du']}",
            "primary_artifact": "reports/figure4a_distance_scorecard.md",
            "reporting_note": "This is the strongest local Figure 4a comparison.",
        },
        {
            "claim_id": "user_goal_verdict",
            "claim_type": "allowed",
            "claim": "The user goal is partially met: not close to the paper DU-Net endpoint, but the Linear-UCB fallback is directionally consistent under the best local metric.",
            "evidence": user_goal_verdict["distance_to_goal"],
            "primary_artifact": "reports/figure4a_user_goal_comparison_brief.md",
            "reporting_note": user_goal_verdict["decision"],
        },
        {
            "claim_id": "high_rating_mechanism",
            "claim_type": "allowed",
            "claim": "High-rating regions support the DU-Net reward-region mechanism.",
            "evidence": f"DU-Net {high['du_net']} vs Linear UCB {high['linear_ucb']}; gap {high['linear_minus_du']}",
            "primary_artifact": "reports/figure4a_distance_scorecard.md",
            "reporting_note": "Use as mechanism support, not as full-test proof.",
        },
        {
            "claim_id": "exact_figure4a",
            "claim_type": "not_allowed",
            "claim": "Do not claim exact Figure 4a reproduction.",
            "evidence": f"Full RMSE gap {full['linear_minus_du']}; sigmoid gap {sigmoid['linear_minus_du']}; rerun gate {rerun['status']}",
            "primary_artifact": "reports/reproduction_continuation_audit.md",
            "reporting_note": "Absolute scale and full-test ordering remain non-matching.",
        },
        {
            "claim_id": "paper_faithful_rerun",
            "claim_type": "not_ready",
            "claim": "Do not start a paper-faithful rerun from the current state.",
            "evidence": rerun["evidence"],
            "primary_artifact": "reports/paper_faithful_rerun_readiness.md",
            "reporting_note": rerun["decision"],
        },
        {
            "claim_id": "source_gap",
            "claim_type": "actionable_gap",
            "claim": "The remaining blocker is missing source-backed paper details.",
            "evidence": "; ".join(source_p0),
            "primary_artifact": "reports/paper_pdf_evidence_audit.md; reports/paper_method_detail_keyword_audit.md; reports/paper_url_dependency_audit.md; reports/local_source_artifact_inventory_audit.md; reports/figure4a_metric_formula_visual_audit.md; reports/public_source_rescan_audit.md; reports/remaining_gap_priority_audit.md; reports/movielens_protocol_alignment_audit.md; reports/movielens_split_overlap_audit.md; reports/gcmc_monti_split_provenance_audit.md; reports/movielens_100k_svdpp_official_u1_sensitivity_3seeds_report.md; reports/movielens_100k_svdpp_official_folds_sensitivity_report.md; reports/paper_source_request_packet.md",
            "reporting_note": "The local PDF and MovieLens split audits narrow the protocol, but do not replace P0 source/metric/split/backend artifacts.",
        },
        {
            "claim_id": "author_source_request_email",
            "claim_type": "defined_next_step",
            "claim": "A copy-ready author/source request email is prepared.",
            "evidence": author_request_decision["content"],
            "primary_artifact": "reports/author_source_request_email.md",
            "reporting_note": "Use this to request the source-backed trigger needed for the next diagnostic.",
        },
        {
            "claim_id": "local_source_inventory",
            "claim_type": "allowed",
            "claim": "The local workspace has no new actionable source/supplement/PDF/archive trigger.",
            "evidence": inventory_decision["interpretation"],
            "primary_artifact": "reports/local_source_artifact_inventory_audit.md",
            "reporting_note": "Use as the local-file counterpart to the public source rescan.",
        },
        {
            "claim_id": "gcmc_monti_split_provenance",
            "claim_type": "allowed",
            "claim": "Local GCMC/Monti files do not recover the exact MovieLens Figure 4a split/preprocessing protocol.",
            "evidence": gcmc_exec["evidence"],
            "primary_artifact": "reports/gcmc_monti_split_provenance_audit.md",
            "reporting_note": "Use as source-backed support that split/preprocessing remains a P0 gap.",
        },
        {
            "claim_id": "paper_method_detail_keyword_audit",
            "claim_type": "allowed",
            "claim": "Keyword-level PDF extraction finds protocol clues but no executable exact Figure 4a details.",
            "evidence": method_execution["decision"],
            "primary_artifact": "reports/paper_method_detail_keyword_audit.md",
            "reporting_note": "Use as source-backed support for the missing-source rerun gate.",
        },
        {
            "claim_id": "paper_url_dependency_audit",
            "claim_type": "allowed",
            "claim": "Embedded PDF URLs do not include an author source or supplement candidate.",
            "evidence": paper_url_decision["decision"],
            "primary_artifact": "reports/paper_url_dependency_audit.md",
            "reporting_note": "Paper-provided URLs confirm comparator dependencies only, not DU-Net source.",
        },
        {
            "claim_id": "figure4a_metric_formula_visual_audit",
            "claim_type": "allowed",
            "claim": "The rendered page-28 w-RMSE formula does not recover the Figure 4a target under local metrics.",
            "evidence": (
                f"literal formula DU {metric_literal['du_net']} scale {metric_literal['scale_vs_paper_du']}x; "
                f"standard RMSE gap {metric_standard['linear_minus_du']}"
            ),
            "primary_artifact": "reports/figure4a_metric_formula_visual_audit.md",
            "reporting_note": "The formula narrows the metric discussion but does not clear the exact reproduction blocker.",
        },
        {
            "claim_id": "public_source_rescan",
            "claim_type": "allowed",
            "claim": "The latest public source rescan did not recover a DU-Net source or supplement trigger.",
            "evidence": public_rescan_decision["evidence"],
            "primary_artifact": "reports/public_source_rescan_audit.md",
            "reporting_note": "Comparator dependency repositories are context only; they do not make the run paper-faithful.",
        },
        {
            "claim_id": "remaining_gap_priority",
            "claim_type": "defined_next_step",
            "claim": "The remaining exact-reproduction gaps are prioritized and all P0 gaps require source-backed input.",
            "evidence": remaining_gap_decision["latest_evidence"],
            "primary_artifact": "reports/remaining_gap_priority_audit.md",
            "reporting_note": remaining_gap_decision["next_action"],
        },
        {
            "claim_id": "official_u1_sensitivity",
            "claim_type": "allowed",
            "claim": "Official MovieLens u1 fold sensitivity preserves local reward-weighted DU-Net support but not exact Figure 4a support.",
            "evidence": (
                f"w_rmse Linear-DU gap {official_u1_weighted['linear_gap_favoring_du']}; "
                f"full-RMSE Linear-DU gap {official_u1_rmse['linear_gap_favoring_du']}"
            ),
            "primary_artifact": "reports/movielens_100k_svdpp_official_u1_sensitivity_3seeds_report.md",
            "reporting_note": "Use only as a local split-sensitivity proxy, not as paper-faithful evidence.",
        },
        {
            "claim_id": "official_folds_sensitivity",
            "claim_type": "allowed",
            "claim": "Official MovieLens u1-u5 fold sensitivity does not support a robust split-only fix.",
            "evidence": (
                f"w_rmse mean Linear-DU gap {mean(official_folds_weighted):.6f}; "
                f"DU-favored w_rmse folds {sum(value > 0 for value in official_folds_weighted)}/5; "
                f"full-RMSE mean Linear-DU gap {mean(official_folds_rmse):.6f}"
            ),
            "primary_artifact": "reports/movielens_100k_svdpp_official_folds_sensitivity_report.md",
            "reporting_note": "Use only as a local proxy; it weakens the split-only explanation.",
        },
        {
            "claim_id": "next_experiment_protocol",
            "claim_type": "defined_next_step",
            "claim": "The next executable experiment is the minimal MovieLens Figure 4a 3-seed diagnostic after a trigger.",
            "evidence": f"precondition={protocol_precondition['status']}; {protocol_precondition['evidence']}",
            "primary_artifact": "reports/minimal_faithful_diagnostic_protocol.md",
            "reporting_note": "Run only after a source-backed trigger appears.",
        },
        {
            "claim_id": "paper_faithful_rerun_playbook",
            "claim_type": "defined_next_step",
            "claim": "The paper-faithful rerun playbook is ready but blocked until a trigger appears.",
            "evidence": playbook_decision["current_evidence"],
            "primary_artifact": "reports/paper_faithful_rerun_playbook.md",
            "reporting_note": playbook_decision["action"],
        },
        {
            "claim_id": "source_artifact_inspection_protocol",
            "claim_type": "defined_next_step",
            "claim": "The source-artifact inspection protocol is ready for any future inspection_required candidate.",
            "evidence": inspection_decision["current_state"],
            "primary_artifact": "reports/source_artifact_inspection_protocol.md",
            "reporting_note": inspection_decision["inspection_action"],
        },
        {
            "claim_id": "source_trigger_monitor_snapshot",
            "claim_type": "defined_next_step",
            "claim": "The source-trigger monitor consolidates the current blocked gate and next authorized action.",
            "evidence": source_trigger_gate["evidence"],
            "primary_artifact": "reports/source_trigger_monitor_snapshot.md",
            "reporting_note": source_trigger_gate["next_action"],
        },
        {
            "claim_id": "source_trigger_intake_check",
            "claim_type": "defined_next_step",
            "claim": "The source-trigger intake runner is the one-command refresh entry point for future source materials.",
            "evidence": source_trigger_intake_gate["evidence"],
            "primary_artifact": "reports/source_trigger_intake_check.md",
            "reporting_note": source_trigger_intake_gate["decision"],
        },
        {
            "claim_id": "handoff_integrity",
            "claim_type": "allowed",
            "claim": "The current handoff artifact set is internally consistent.",
            "evidence": f"{integrity.get('pass', 0)} pass; {integrity.get('fail', 0)} fail",
            "primary_artifact": "reports/reproduction_artifact_integrity_check.md",
            "reporting_note": "Use this as the handoff validation status.",
        },
        {
            "claim_id": "continuation_decision",
            "claim_type": "allowed",
            "claim": "No further local proxy experiment is justified without a trigger.",
            "evidence": continuation_decision["status"],
            "primary_artifact": "reports/reproduction_continuation_audit.md",
            "reporting_note": continuation_decision["decision"],
        },
        {
            "claim_id": "execution_gate_closure",
            "claim_type": "defined_next_step",
            "claim": "The current execution gate is closed until a primary source or approved dependency trigger appears.",
            "evidence": gate_decision["evidence"],
            "primary_artifact": "reports/execution_gate_closure_audit.md",
            "reporting_note": gate_decision["decision"],
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| claim id | type | claim | evidence | primary artifact | reporting note |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['claim_id']} | {row['claim_type']} | {row['claim']} | "
            f"{row['evidence']} | `{row['primary_artifact']}` | {row['reporting_note']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    allowed = sum(1 for row in rows if row["claim_type"] == "allowed")
    not_allowed = sum(1 for row in rows if row["claim_type"] == "not_allowed")
    report = f"""# Reproduction Claim-Evidence Index

Generated by `experiments/build_reproduction_claim_evidence_index.py`.

## Current Progress

The reproduction remains a strong partial reproduction. This index maps the
current reportable claims, non-claims, source gaps, and next-step protocol to
their evidence artifacts.

## Summary

- Allowed reportable claims: {allowed}
- Explicit non-claims: {not_allowed}
- Current continuation decision: wait for source or approved dependency trigger.

## Claim-Evidence Table

{markdown_table(rows)}

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
