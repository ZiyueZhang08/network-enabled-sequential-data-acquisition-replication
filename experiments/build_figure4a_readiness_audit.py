"""Audit final Figure 4a replication readiness and rerun triggers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
STATUS_ZH = REPORT_DIR / "figure4a_current_status_zh.md"
SOURCE_PACKET = REPORT_DIR / "figure4a_source_recovery_packet.md"

OUT_CSV = TABLE_DIR / "figure4a_readiness_audit.csv"
OUT_REPORT = REPORT_DIR / "figure4a_readiness_audit.md"


REQUIRED_ARTIFACTS = [
    ("handoff_index", REPORT_DIR / "figure4a_handoff_index.md"),
    ("submission_packet", REPORT_DIR / "figure4a_submission_packet.md"),
    ("submission_bundle_latest", REPORT_DIR / "figure4a_submission_bundle_latest.md"),
    ("final_package_report", REPORT_DIR / "figure4a_final_package.md"),
    ("final_delivery_zh", REPORT_DIR / "figure4a_final_delivery_zh.md"),
    ("delivery_quality_gate", REPORT_DIR / "figure4a_delivery_quality_gate.md"),
    ("latest_execution_reflection", REPORT_DIR / "figure4a_latest_execution_reflection.md"),
    ("latest_metric_reconciliation", REPORT_DIR / "figure4a_latest_metric_reconciliation.md"),
    ("current_status_zh", STATUS_ZH),
    ("retrospective_zh", REPORT_DIR / "figure4a_retrospective_zh.md"),
    ("no_author_source_closure", REPORT_DIR / "figure4a_no_author_source_closure.md"),
    ("report_ready_section_zh", REPORT_DIR / "figure4a_report_ready_section_zh.md"),
    ("evidence_appendix_zh", REPORT_DIR / "figure4a_evidence_appendix_zh.md"),
    ("source_recovery_packet", SOURCE_PACKET),
    ("author_source_request_email", REPORT_DIR / "author_source_request_email.md"),
    ("artifact_checksums", REPORT_DIR / "figure4a_artifact_checksums.md"),
    ("metric_alignment_report", REPORT_DIR / "figure4a_metric_alignment_audit.md"),
    ("phase1_seed15_micro_report", REPORT_DIR / "figure4a_phase1_seed15_micro_reflection.md"),
    ("phase4_adaptive_report", REPORT_DIR / "figure4a_phase4_adaptive_reflection.md"),
    ("no_network_fidelity_probe", REPORT_DIR / "figure4a_no_network_fidelity_probe.md"),
    ("exp_fidelity_probe", REPORT_DIR / "figure4a_exp_fidelity_probe.md"),
    ("observational_entropy_report", REPORT_DIR / "figure4a_observational_entropy_diagnostic.md"),
    ("adaptive_grid_report", REPORT_DIR / "figure4a_adaptive_grid_diagnostic.md"),
    ("linear_ucb_comparator_report", REPORT_DIR / "figure4a_linear_ucb_comparator_diagnostic.md"),
    ("final_summary_csv", FINAL_SUMMARY),
    ("latest_execution_matrix_csv", TABLE_DIR / "figure4a_latest_execution_matrix.csv"),
    ("latest_execution_comparisons_csv", TABLE_DIR / "figure4a_latest_execution_comparisons.csv"),
    ("latest_metric_reconciliation_csv", TABLE_DIR / "figure4a_latest_metric_reconciliation.csv"),
    ("latest_metric_reconciliation_gaps_csv", TABLE_DIR / "figure4a_latest_metric_reconciliation_gaps.csv"),
    ("no_network_fidelity_gate_csv", TABLE_DIR / "figure4a_no_network_fidelity_gate.csv"),
    ("no_network_fidelity_seed_values_csv", TABLE_DIR / "figure4a_no_network_fidelity_seed_values.csv"),
    ("exp_fidelity_gate_csv", TABLE_DIR / "figure4a_exp_fidelity_gate.csv"),
    ("exp_fidelity_seed_values_csv", TABLE_DIR / "figure4a_exp_fidelity_seed_values.csv"),
    ("handoff_index_csv", TABLE_DIR / "figure4a_handoff_index.csv"),
    ("submission_packet_csv", TABLE_DIR / "figure4a_submission_packet.csv"),
    ("submission_bundle_latest_csv", TABLE_DIR / "figure4a_submission_bundle_latest.csv"),
    ("final_delivery_zh_csv", TABLE_DIR / "figure4a_final_delivery_zh.csv"),
    ("delivery_quality_gate_csv", TABLE_DIR / "figure4a_delivery_quality_gate.csv"),
    ("retrospective_zh_csv", TABLE_DIR / "figure4a_retrospective_zh.csv"),
    ("no_author_source_closure_csv", TABLE_DIR / "figure4a_no_author_source_closure.csv"),
    ("report_ready_section_zh_csv", TABLE_DIR / "figure4a_report_ready_section_zh.csv"),
    ("evidence_appendix_zh_csv", TABLE_DIR / "figure4a_evidence_appendix_zh.csv"),
    ("artifact_checksums_csv", TABLE_DIR / "figure4a_artifact_checksums.csv"),
    ("paired_seed_gaps_csv", TABLE_DIR / "figure4a_reward_weight_metric_20seeds_paired_seed_gaps.csv"),
    ("rating_tail_diagnostic_csv", TABLE_DIR / "figure4a_rating_tail_20seed_confirmation.csv"),
    ("source_recovery_blockers_csv", TABLE_DIR / "figure4a_source_recovery_blockers.csv"),
    ("author_source_request_email_csv", TABLE_DIR / "author_source_request_email.csv"),
    ("w_rmse_figure", FIGURE_DIR / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_w_rmse.png"),
    ("reward_figure", FIGURE_DIR / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_cumulative_reward.png"),
]


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def artifact_rows() -> pd.DataFrame:
    rows = []
    for name, path in REQUIRED_ARTIFACTS:
        rows.append(
            {
                "artifact": name,
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def markdown_artifacts(rows: pd.DataFrame) -> str:
    lines = [
        "| artifact | path | exists | bytes |",
        "|---|---|---|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(f"| {row.artifact} | `{row.path}` | {str(row.exists).lower()} | {int(row.bytes)} |")
    return "\n".join(lines)


def write_report(rows: pd.DataFrame) -> None:
    final = pd.read_csv(FINAL_SUMMARY)
    du = final[final["variant"] == "du_net"].iloc[0]
    linear = final[final["variant"] == "mf_linear_ucb_a002"].iloc[0]
    gap = float(linear.local_reward_weighted_w_rmse) - float(du.local_reward_weighted_w_rmse)
    ready = bool(rows["exists"].all())
    missing = rows[~rows["exists"]]["artifact"].tolist()
    missing_text = "none" if not missing else ", ".join(missing)
    report = f"""# Figure 4a Readiness Audit

Generated by `experiments/build_figure4a_readiness_audit.py`.

## Readiness

Ready for handoff: `{str(ready).lower()}`.

Missing artifacts: `{missing_text}`.

Current final result: DU-Net primary linear reward-norm w-RMSE
`{fmt(du.local_reward_weighted_w_rmse)}` versus MF-centered Linear UCB proxy
`{fmt(linear.local_reward_weighted_w_rmse)}`, gap `{fmt(gap)}`.

## Artifact Check

{markdown_artifacts(rows)}

## Next Execution Trigger

Do not run another local sweep without one of these new inputs:

- exact Figure 4a w-RMSE weighting/normalization;
- author MovieLens split, initial observed mask, or candidate mask;
- DU-Net source/supplement details for observational entropy, DLS, and
  per-cycle tuning;
- paper Linear UCB feature/update protocol.

When a trigger appears, run only a seeds `0,1,2` diagnostic first. Promote to
20 seeds only if the recovered setup improves the primary blocker and does not
reintroduce the conventional/sigmoid/local metric contradiction.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = artifact_rows()
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
