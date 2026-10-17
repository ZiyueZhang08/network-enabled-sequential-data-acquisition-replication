"""Build a one-page handoff index for the Figure 4a replication branch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
CLAIM_FREEZE = TABLE_DIR / "figure4a_claim_freeze.csv"
READINESS_AUDIT = TABLE_DIR / "figure4a_readiness_audit.csv"

OUT_REPORT = REPORT_DIR / "figure4a_handoff_index.md"
OUT_CSV = TABLE_DIR / "figure4a_handoff_index.csv"

PAPER_GAP = 1.814 - 1.788


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def index_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order": 1,
                "purpose": "Submission packet",
                "artifact": "reports/figure4a_submission_packet.md",
                "use_when": "Need the top-level submit/request packet with metrics, guardrails, and next action.",
            },
            {
                "order": 2,
                "purpose": "Submission bundle",
                "artifact": "reports/figure4a_submission_bundle_latest.md",
                "use_when": "Need the latest script-generated zip path, SHA256, and bundle manifest.",
            },
            {
                "order": 3,
                "purpose": "Final conclusion",
                "artifact": "reports/figure4a_final_package.md",
                "use_when": "Need the headline result, final table, and caveated conclusion.",
            },
            {
                "order": 4,
                "purpose": "Latest metric reconciliation",
                "artifact": "reports/figure4a_latest_metric_reconciliation.md",
                "use_when": "Need the cross-metric caveat for primary, sigmoid, and conventional RMSE.",
            },
            {
                "order": 5,
                "purpose": "No-network fidelity",
                "artifact": "reports/figure4a_no_network_fidelity_probe.md",
                "use_when": "Need to explain why MovieLens no-diffusion/no-network was not promoted.",
            },
            {
                "order": 6,
                "purpose": "Exp-sampling fidelity",
                "artifact": "reports/figure4a_exp_fidelity_probe.md",
                "use_when": "Need to explain why exponential batch sampling was not promoted.",
            },
            {
                "order": 7,
                "purpose": "Final Chinese delivery",
                "artifact": "reports/figure4a_final_delivery_zh.md",
                "use_when": "Need the final Chinese handoff wording, reflection, and next step.",
            },
            {
                "order": 8,
                "purpose": "Delivery quality gate",
                "artifact": "reports/figure4a_delivery_quality_gate.md",
                "use_when": "Need to verify no stale early-gap wording or mojibake remains in curated handoff reports.",
            },
            {
                "order": 9,
                "purpose": "Chinese status",
                "artifact": "reports/figure4a_current_status_zh.md",
                "use_when": "Need a concise Chinese progress summary.",
            },
            {
                "order": 10,
                "purpose": "Allowed/forbidden claims",
                "artifact": "reports/figure4a_claim_freeze.md",
                "use_when": "Need to avoid overstating the reproduction.",
            },
            {
                "order": 11,
                "purpose": "Artifact readiness",
                "artifact": "reports/figure4a_readiness_audit.md",
                "use_when": "Need to verify all final deliverables exist.",
            },
            {
                "order": 12,
                "purpose": "Source recovery",
                "artifact": "reports/figure4a_source_recovery_packet.md",
                "use_when": "Need author/source request details and unresolved blockers.",
            },
            {
                "order": 13,
                "purpose": "Author request email",
                "artifact": "reports/author_source_request_email.md",
                "use_when": "Need copy-ready wording to request source-backed Figure 4a details.",
            },
            {
                "order": 14,
                "purpose": "Reopen protocol",
                "artifact": "reports/figure4a_next_trigger_protocol.md",
                "use_when": "Need to decide whether a new experiment is allowed.",
            },
            {
                "order": 15,
                "purpose": "No-author-source closure",
                "artifact": "reports/figure4a_no_author_source_closure.md",
                "use_when": "Need the final decision after confirming author source code is unavailable.",
            },
            {
                "order": 16,
                "purpose": "Report-ready wording",
                "artifact": "reports/figure4a_report_ready_section_zh.md",
                "use_when": "Need a directly usable Chinese report paragraph with caveats.",
            },
            {
                "order": 17,
                "purpose": "Evidence appendix",
                "artifact": "reports/figure4a_evidence_appendix_zh.md",
                "use_when": "Need claim-to-evidence mapping for the final report.",
            },
            {
                "order": 18,
                "purpose": "Retrospective",
                "artifact": "reports/figure4a_retrospective_zh.md",
                "use_when": "Need to understand why local sweeps stopped and what was learned.",
            },
            {
                "order": 19,
                "purpose": "Artifact checksums",
                "artifact": "reports/figure4a_artifact_checksums.md",
                "use_when": "Need to verify final handoff files have not changed unexpectedly.",
            },
        ]
    )


def markdown_index(rows: pd.DataFrame) -> str:
    lines = [
        "| order | purpose | artifact | use when |",
        "|---:|---|---|---|",
    ]
    for row in rows.sort_values("order").itertuples(index=False):
        lines.append(f"| {row.order} | {row.purpose} | `{row.artifact}` | {row.use_when} |")
    return "\n".join(lines)


def write_report(rows: pd.DataFrame) -> None:
    final = pd.read_csv(FINAL_SUMMARY)
    claims = pd.read_csv(CLAIM_FREEZE)
    readiness = pd.read_csv(READINESS_AUDIT)
    du = final[final["variant"] == "du_net"].iloc[0]
    linear = final[final["variant"] == "mf_linear_ucb_a002"].iloc[0]
    gap = float(linear.local_reward_weighted_w_rmse) - float(du.local_reward_weighted_w_rmse)
    ready = bool(readiness["exists"].all())
    forbidden_count = int((claims["claim_type"] == "forbidden").sum())
    report = f"""# Figure 4a Handoff Index

Generated by `experiments/build_figure4a_handoff_index.py`.

## One-Line Status

Local Figure 4a replication is handoff ready: `{str(ready).lower()}`.
The primary linear reward-norm gap is `{fmt(gap)}`:
DU-Net `{fmt(du.local_reward_weighted_w_rmse)}` versus MF-centered Linear UCB
proxy `{fmt(linear.local_reward_weighted_w_rmse)}`. This is
`{fmt(100.0 * gap / PAPER_GAP, 1)}%` of the paper visual gap. It is a strong
partial, caveated reproduction, not an exact Figure 4a reproduction.

## Read Order

{markdown_index(rows)}

## Guardrails

- Use `MF-centered Linear UCB proxy`, not exact paper Linear UCB.
- Do not claim exact Figure 4a replication.
- Do not run more 20-seed experiments without a source-triggered 3-seed pass.
- There are `{forbidden_count}` frozen forbidden claim patterns in
  `reports/figure4a_claim_freeze.md`.

## Current Next Step

Author source code is unavailable. Do not wait on a code drop as an active
execution branch. Reopen only if non-code source evidence arrives: exact
w-RMSE, exact split/masks, DU-Net entropy/DLS/tuning details, or paper Linear
UCB protocol. If one arrives, follow `reports/figure4a_next_trigger_protocol.md`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = index_rows()
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
