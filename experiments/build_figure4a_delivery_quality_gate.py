"""Validate the final Figure 4a handoff surface for stale or garbled claims."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
READINESS_AUDIT = TABLE_DIR / "figure4a_readiness_audit.csv"
NO_NETWORK_GATE = TABLE_DIR / "figure4a_no_network_fidelity_gate.csv"
EXP_GATE = TABLE_DIR / "figure4a_exp_fidelity_gate.csv"

OUT_CSV = TABLE_DIR / "figure4a_delivery_quality_gate.csv"
OUT_REPORT = REPORT_DIR / "figure4a_delivery_quality_gate.md"

PAPER_GAP = 1.814 - 1.788

CURATED_REPORTS = [
    REPORT_DIR / "figure4a_submission_packet.md",
    REPORT_DIR / "figure4a_submission_bundle_latest.md",
    REPORT_DIR / "figure4a_final_package.md",
    REPORT_DIR / "figure4a_final_delivery_zh.md",
    REPORT_DIR / "figure4a_current_status_zh.md",
    REPORT_DIR / "figure4a_report_ready_section_zh.md",
    REPORT_DIR / "figure4a_retrospective_zh.md",
    REPORT_DIR / "figure4a_evidence_appendix_zh.md",
    REPORT_DIR / "figure4a_source_recovery_packet.md",
    REPORT_DIR / "author_source_request_email.md",
    REPORT_DIR / "figure4a_next_trigger_protocol.md",
    REPORT_DIR / "figure4a_handoff_index.md",
    REPORT_DIR / "figure4a_readiness_audit.md",
    REPORT_DIR / "figure4a_no_network_fidelity_probe.md",
    REPORT_DIR / "figure4a_exp_fidelity_probe.md",
]

MOJIBAKE_MARKERS = ["锛", "褰", "鎶", "璁", "瑙", "鐨", "宺", "孧", "�"]


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def status(ok: bool) -> str:
    return "pass" if ok else "fail"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def final_metrics() -> dict[str, float]:
    final = pd.read_csv(FINAL_SUMMARY)
    du = final[final["variant"] == "du_net"].iloc[0]
    linear = final[final["variant"] == "mf_linear_ucb_a002"].iloc[0]
    gap = float(linear.linear_reward_norm_w_rmse) - float(du.linear_reward_norm_w_rmse)
    return {
        "du_primary": float(du.linear_reward_norm_w_rmse),
        "linear_primary": float(linear.linear_reward_norm_w_rmse),
        "gap": gap,
        "paper_share": 100.0 * gap / PAPER_GAP,
        "du_rmse": float(du.rmse),
        "linear_rmse": float(linear.rmse),
    }


def build_rows() -> pd.DataFrame:
    metrics = final_metrics()
    no_network = pd.read_csv(NO_NETWORK_GATE).iloc[0]
    exp = pd.read_csv(EXP_GATE).iloc[0]
    readiness = pd.read_csv(READINESS_AUDIT)

    rows: list[dict[str, object]] = []

    def add(check: str, ok: bool, evidence: str, expected: str) -> None:
        rows.append(
            {
                "check": check,
                "status": status(ok),
                "evidence": evidence,
                "expected": expected,
            }
        )

    add(
        "primary_gap_current",
        metrics["gap"] >= 0.020,
        f"gap={fmt(metrics['gap'])}; share={fmt(metrics['paper_share'], 1)}%",
        "Final primary gap is the latest >= +0.020 result, not the stale early-gap result.",
    )
    add(
        "readiness_complete",
        bool(readiness["exists"].all()),
        f"missing={int((~readiness['exists']).sum())}",
        "All final handoff artifacts exist.",
    )
    add(
        "no_network_rejected",
        str(no_network.decision) == "reject_primary_and_reward_regression"
        and str(no_network.promote_to_20seed) == "False",
        f"decision={no_network.decision}; promote={no_network.promote_to_20seed}",
        "No-network fidelity branch remains rejected.",
    )
    add(
        "exp_sampling_rejected",
        str(exp.decision) == "reject_reward_tail_collapse"
        and str(exp.promote_to_20seed) == "False",
        f"decision={exp.decision}; promote={exp.promote_to_20seed}",
        "Exponential-sampling fidelity branch remains rejected.",
    )

    missing_reports = [str(path.relative_to(ROOT)) for path in CURATED_REPORTS if not path.exists()]
    add(
        "curated_reports_exist",
        not missing_reports,
        "missing=" + ("none" if not missing_reports else "; ".join(missing_reports)),
        "Every curated final report exists.",
    )

    stale_hits = []
    mojibake_hits = []
    missing_latest_gap = []
    for path in CURATED_REPORTS:
        if not path.exists():
            continue
        text = read_text(path)
        rel = str(path.relative_to(ROOT))
        if "0.0083" in text and "not `0.0083`" not in text:
            stale_hits.append(rel)
        markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
        if markers:
            mojibake_hits.append(f"{rel}:{','.join(markers)}")
        if path.name in {
            "figure4a_submission_packet.md",
            "figure4a_final_package.md",
            "figure4a_final_delivery_zh.md",
            "figure4a_current_status_zh.md",
            "figure4a_source_recovery_packet.md",
            "author_source_request_email.md",
            "figure4a_next_trigger_protocol.md",
        } and "0.0207" not in text:
            missing_latest_gap.append(rel)

    add(
        "no_stale_early_gap_in_curated_reports",
        not stale_hits,
        "hits=" + ("none" if not stale_hits else "; ".join(stale_hits)),
        "Curated reports do not present stale early-gap wording as the current result.",
    )
    add(
        "no_mojibake_in_curated_reports",
        not mojibake_hits,
        "hits=" + ("none" if not mojibake_hits else "; ".join(mojibake_hits)),
        "Curated reports are readable UTF-8 text without common mojibake markers.",
    )
    add(
        "latest_gap_visible_in_primary_reports",
        not missing_latest_gap,
        "missing=" + ("none" if not missing_latest_gap else "; ".join(missing_latest_gap)),
        "Primary reports explicitly show the latest +0.0207 gap.",
    )

    next_trigger = read_text(REPORT_DIR / "figure4a_next_trigger_protocol.md")
    add(
        "next_trigger_no_old_threshold",
        "gap >= 0.010" not in next_trigger,
        "old_threshold_present=" + str("gap >= 0.010" in next_trigger).lower(),
        "Next-trigger protocol is calibrated to the frozen +0.0207 candidate.",
    )

    source_packet = read_text(REPORT_DIR / "figure4a_source_recovery_packet.md")
    add(
        "source_packet_mentions_fidelity_probes",
        "No-network fidelity: rejected" in source_packet
        and "Exponential sampling fidelity: rejected" in source_packet,
        "no_network=" + str("No-network fidelity: rejected" in source_packet).lower()
        + "; exp="
        + str("Exponential sampling fidelity: rejected" in source_packet).lower(),
        "Source-recovery packet includes the latest rejected fidelity probes.",
    )

    return pd.DataFrame(rows)


def markdown_table(rows: pd.DataFrame) -> str:
    lines = [
        "| check | status | evidence | expected |",
        "|---|---|---|---|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(f"| {row.check} | {row.status} | `{row.evidence}` | {row.expected} |")
    return "\n".join(lines)


def write_report(rows: pd.DataFrame) -> None:
    complete = bool((rows["status"] == "pass").all())
    report = f"""# Figure 4a Delivery Quality Gate

Generated by `experiments/build_figure4a_delivery_quality_gate.py`.

Complete: `{str(complete).lower()}`.

This gate protects the final handoff surface from two regressions found late in
the reproduction branch: stale early-gap wording and garbled Chinese text.

{markdown_table(rows)}
"""
    OUT_REPORT.write_text(report, encoding="utf-8-sig")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")
    print(rows.to_string(index=False))
    if not bool((rows["status"] == "pass").all()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
