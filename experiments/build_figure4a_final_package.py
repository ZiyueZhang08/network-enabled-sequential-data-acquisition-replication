"""Build the final Figure 4a replication package summary."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

LATEST_MATRIX = TABLE_DIR / "figure4a_latest_execution_matrix.csv"
LATEST_COMPARISONS = TABLE_DIR / "figure4a_latest_execution_comparisons.csv"
LATEST_RECONCILIATION = TABLE_DIR / "figure4a_latest_metric_reconciliation.csv"
LATEST_RECONCILIATION_GAPS = TABLE_DIR / "figure4a_latest_metric_reconciliation_gaps.csv"
PHASE1_GATE = TABLE_DIR / "figure4a_phase1_seed15_micro_gates.csv"
PHASE4_GATE = TABLE_DIR / "figure4a_phase4_adaptive_gate.csv"
NO_NETWORK_GATE = TABLE_DIR / "figure4a_no_network_fidelity_gate.csv"
EXP_GATE = TABLE_DIR / "figure4a_exp_fidelity_gate.csv"

OUT_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
OUT_MANIFEST = TABLE_DIR / "figure4a_final_package_manifest.csv"
OUT_REPORT = REPORT_DIR / "figure4a_final_package.md"

DU_SOURCE_VARIANT = "graph_late_lambda_half_start16"
PAPER_DU = 1.788
PAPER_LINEAR = 1.814
PAPER_GAP = PAPER_LINEAR - PAPER_DU


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def final_rows() -> pd.DataFrame:
    matrix = pd.read_csv(LATEST_MATRIX)
    du = matrix[matrix["variant"] == DU_SOURCE_VARIANT].iloc[0]
    wanted = [DU_SOURCE_VARIANT, "mf_linear_ucb_a002", "myopic", "graph_base"]
    labels = {
        DU_SOURCE_VARIANT: ("du_net", "Frozen DU-Net best candidate"),
        "mf_linear_ucb_a002": ("mf_linear_ucb_a002", "MF-centered Linear UCB proxy"),
        "myopic": ("myopic", "Myopic"),
        "graph_base": ("graph_base", "DU-Net graph-base diagnostic"),
    }
    rows = []
    for source_variant in wanted:
        row = matrix[matrix["variant"] == source_variant].iloc[0]
        public_variant, label = labels[source_variant]
        primary = float(row.linear_reward_norm_w_rmse)
        rows.append(
            {
                "variant": public_variant,
                "source_variant": source_variant,
                "policy": row.policy,
                "seed_count": int(row.seed_count),
                "cumulative_reward": float(row.cumulative_reward),
                "rmse": float(row.rmse),
                "local_reward_weighted_w_rmse": primary,
                "linear_reward_norm_w_rmse": primary,
                "sum_norm_weighted_w_rmse": float(row.w_rmse),
                "sigmoid_reward_w_rmse": float(row.sigmoid_reward_w_rmse),
                "positive_reward_norm_w_rmse": float(row.positive_reward_norm_w_rmse),
                "local_linear_minus_du_gap": primary - float(du.linear_reward_norm_w_rmse),
                "sum_norm_gap_vs_du": float(row.w_rmse) - float(du.w_rmse),
                "sigmoid_gap_vs_du": float(row.sigmoid_reward_w_rmse) - float(du.sigmoid_reward_w_rmse),
                "rmse_gap_vs_du": float(row.rmse) - float(du.rmse),
                "high_rating_rmse_ge4": float(row.high_rating_rmse_ge4),
                "low_rating_rmse_le3": float(row.low_rating_rmse_le3),
                "rating5_rmse": float(row.rating5_rmse),
                "paper_visual_w_rmse_target": PAPER_DU
                if public_variant == "du_net"
                else PAPER_LINEAR
                if public_variant == "mf_linear_ucb_a002"
                else None,
                "comparator_label": label,
            }
        )
    return pd.DataFrame(rows)


def manifest_rows() -> pd.DataFrame:
    artifacts = [
        ("final_summary_csv", OUT_SUMMARY),
        ("submission_packet", REPORT_DIR / "figure4a_submission_packet.md"),
        ("submission_packet_csv", TABLE_DIR / "figure4a_submission_packet.csv"),
        ("submission_bundle_latest", REPORT_DIR / "figure4a_submission_bundle_latest.md"),
        ("submission_bundle_latest_csv", TABLE_DIR / "figure4a_submission_bundle_latest.csv"),
        ("final_delivery_zh", REPORT_DIR / "figure4a_final_delivery_zh.md"),
        ("final_delivery_zh_csv", TABLE_DIR / "figure4a_final_delivery_zh.csv"),
        ("delivery_quality_gate", REPORT_DIR / "figure4a_delivery_quality_gate.md"),
        ("delivery_quality_gate_csv", TABLE_DIR / "figure4a_delivery_quality_gate.csv"),
        ("latest_execution_matrix", LATEST_MATRIX),
        ("latest_execution_comparisons", LATEST_COMPARISONS),
        ("latest_execution_reflection", REPORT_DIR / "figure4a_latest_execution_reflection.md"),
        ("latest_metric_reconciliation", REPORT_DIR / "figure4a_latest_metric_reconciliation.md"),
        ("latest_metric_reconciliation_csv", LATEST_RECONCILIATION),
        ("latest_metric_reconciliation_gaps", LATEST_RECONCILIATION_GAPS),
        ("phase1_seed15_micro_report", REPORT_DIR / "figure4a_phase1_seed15_micro_reflection.md"),
        ("phase4_adaptive_report", REPORT_DIR / "figure4a_phase4_adaptive_reflection.md"),
        ("no_network_fidelity_probe", REPORT_DIR / "figure4a_no_network_fidelity_probe.md"),
        ("no_network_fidelity_gate", NO_NETWORK_GATE),
        ("exp_fidelity_probe", REPORT_DIR / "figure4a_exp_fidelity_probe.md"),
        ("exp_fidelity_gate", EXP_GATE),
        ("public_source_rescan_audit", REPORT_DIR / "public_source_rescan_audit.md"),
        ("paper_source_request_packet", REPORT_DIR / "paper_source_request_packet.md"),
        ("figure4a_source_recovery_packet", REPORT_DIR / "figure4a_source_recovery_packet.md"),
        ("author_source_request_email", REPORT_DIR / "author_source_request_email.md"),
        ("author_source_request_email_csv", TABLE_DIR / "author_source_request_email.csv"),
        ("paper_pdf_evidence_audit", REPORT_DIR / "paper_pdf_evidence_audit.md"),
        ("readiness_audit", REPORT_DIR / "figure4a_readiness_audit.md"),
        ("handoff_index", REPORT_DIR / "figure4a_handoff_index.md"),
        ("artifact_checksums", REPORT_DIR / "figure4a_artifact_checksums.md"),
        ("final_report", OUT_REPORT),
    ]
    return pd.DataFrame(
        {
            "artifact": name,
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
        }
        for name, path in artifacts
    )


def markdown_table(frame: pd.DataFrame) -> str:
    lines = [
        "| variant | source variant | label | reward | primary w-RMSE | sum-norm w-RMSE | sigmoid w-RMSE | RMSE | primary gap vs DU |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {row.source_variant} | {row.comparator_label} | "
            f"{fmt(row.cumulative_reward, 2)} | {fmt(row.linear_reward_norm_w_rmse)} | "
            f"{fmt(row.sum_norm_weighted_w_rmse)} | {fmt(row.sigmoid_reward_w_rmse)} | "
            f"{fmt(row.rmse)} | {fmt(row.local_linear_minus_du_gap)} |"
        )
    return "\n".join(lines)


def high_low_table(frame: pd.DataFrame) -> str:
    lines = [
        "| variant | high>=4 RMSE | low<=3 RMSE | rating-5 RMSE | RMSE gap vs DU |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.variant} | {fmt(row.high_rating_rmse_ge4)} | "
            f"{fmt(row.low_rating_rmse_le3)} | {fmt(row.rating5_rmse)} | {fmt(row.rmse_gap_vs_du)} |"
        )
    return "\n".join(lines)


def gap_table(comparisons: pd.DataFrame) -> str:
    lines = [
        "| comparator | primary gap favoring DU | DU seed wins | reward gap favoring DU | RMSE gap favoring DU | sigmoid caveat |",
        "|---|---:|---:|---:|---:|---|",
    ]
    recon_gaps = pd.read_csv(LATEST_RECONCILIATION_GAPS)
    for row in comparisons.itertuples(index=False):
        if row.comparator == "graph_base":
            continue
        sigmoid = recon_gaps[
            (recon_gaps["comparator"] == row.comparator) & (recon_gaps["metric"] == "sigmoid_reward_w_rmse")
        ].iloc[0]
        sigmoid_text = (
            f"{fmt(sigmoid.comparator_minus_du)} "
            f"({'favors DU' if bool(sigmoid.positive_favors_du) else 'favors comparator/tie'})"
        )
        lines.append(
            f"| {row.comparator} | {fmt(row.primary_linear_reward_norm_gap_favoring_du)} | "
            f"{int(row.primary_seed_wins)}/{int(row.seed_count)} | "
            f"{fmt(row.cumulative_reward_gap_favoring_du, 2)} | "
            f"{fmt(row.conventional_rmse_gap_favoring_du)} | {sigmoid_text} |"
        )
    return "\n".join(lines)


def manifest_table(frame: pd.DataFrame) -> str:
    lines = [
        "| artifact | path | exists |",
        "|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.artifact} | `{row.path}` | {str(row.exists).lower()} |")
    return "\n".join(lines)


def phase_decisions() -> str:
    phase1 = pd.read_csv(PHASE1_GATE)
    phase4 = pd.read_csv(PHASE4_GATE).iloc[0]
    no_network = pd.read_csv(NO_NETWORK_GATE).iloc[0] if NO_NETWORK_GATE.exists() else None
    exp = pd.read_csv(EXP_GATE).iloc[0] if EXP_GATE.exists() else None
    start17 = phase1[phase1["variant"] == "graph_late_lambda_half_start17"].iloc[0]
    start15 = phase1[phase1["variant"] == "graph_late_lambda_half_start15"].iloc[0]
    lines = (
        "- Phase 1 seed-15 micro screen: no candidate promoted. "
        f"`graph_late_lambda_half_start17` improved primary gap by `{fmt(start17.primary_delta_vs_frozen, 6)}` "
        f"but harmed `{int(start17.stable_seed_harm_count)}` stable seeds; "
        f"`graph_late_lambda_half_start15` improved by `{fmt(start15.primary_delta_vs_frozen, 6)}` "
        f"but harmed `{int(start15.stable_seed_harm_count)}` stable seeds.\n"
        "- Phase 4 adaptive probe: no candidate promoted. "
        f"`{phase4.variant}` had primary delta `{fmt(phase4.primary_delta_vs_frozen, 6)}` and "
        f"reward delta `{fmt(phase4.reward_delta_vs_frozen, 2)}` versus the frozen candidate."
    )
    if no_network is not None:
        lines += (
            "\n- MovieLens no-network fidelity probe: no candidate promoted. "
            f"`{no_network.variant}` had primary delta `{fmt(no_network.primary_delta_vs_frozen, 6)}` "
            f"and reward delta `{fmt(no_network.reward_delta_vs_frozen, 2)}` versus the frozen candidate."
        )
    if exp is not None:
        lines += (
            "\n- Exponential-sampling fidelity probe: no candidate promoted. "
            f"`{exp.variant}` had primary delta `{fmt(exp.primary_delta_vs_frozen, 6)}` "
            f"and reward delta `{fmt(exp.reward_delta_vs_frozen, 2)}` versus the frozen candidate."
        )
    return lines


def write_report(summary: pd.DataFrame, manifest: pd.DataFrame) -> None:
    comparisons = pd.read_csv(LATEST_COMPARISONS)
    du = summary[summary["variant"] == "du_net"].iloc[0]
    linear = summary[summary["variant"] == "mf_linear_ucb_a002"].iloc[0]
    myopic = summary[summary["variant"] == "myopic"].iloc[0]
    primary_gap = float(linear.linear_reward_norm_w_rmse - du.linear_reward_norm_w_rmse)
    myopic_gap = float(myopic.linear_reward_norm_w_rmse - du.linear_reward_norm_w_rmse)
    gap_share = primary_gap / PAPER_GAP
    linear_comparison = comparisons[comparisons["comparator"] == "mf_linear_ucb_a002"].iloc[0]
    myopic_comparison = comparisons[comparisons["comparator"] == "myopic"].iloc[0]

    report = f"""# Figure 4a Final Replication Package

Generated by `experiments/build_figure4a_final_package.py`.

## Status

Final status: **strong partial reproduction; not exact Figure 4a replication**.

The frozen best local candidate is `{DU_SOURCE_VARIANT}`. Under the current
transparent primary proxy (`linear_reward_norm_w_rmse`), DU-Net is
`{fmt(du.linear_reward_norm_w_rmse)}` and the MF-centered Linear UCB proxy is
`{fmt(linear.linear_reward_norm_w_rmse)}`, giving a Linear-minus-DU gap of
`{fmt(primary_gap)}`. The paper visual gap is approximately `{fmt(PAPER_GAP)}`,
so the local primary gap reaches `{fmt(gap_share * 100, 1)}%` of the visual
target. DU-Net wins `{int(linear_comparison.primary_seed_wins)}/{int(linear_comparison.seed_count)}`
seeds against Linear UCB and `{int(myopic_comparison.primary_seed_wins)}/{int(myopic_comparison.seed_count)}`
seeds against Myopic on the primary metric.

## Final 20-Seed Table

{markdown_table(summary)}

## Comparator Gaps

Positive gaps favor DU-Net.

{gap_table(comparisons)}

## High/Low Rating Diagnostic

{high_low_table(summary)}

## Interpretation

The updated result is materially stronger than the stale final package: the
primary Linear-minus-DU gap is now `{fmt(primary_gap)}`, not `0.0083`. It clears
the local `+0.020` success gate and reaches `{fmt(gap_share * 100, 1)}%` of the
paper visual target, but it still does not clear the paper-scale target itself.

The mechanism remains high-reward-tail-specific. DU-Net improves high-rating
and rating-5 errors versus the Linear UCB proxy, but conventional full RMSE
still favors the comparator. The latest metric reconciliation also shows that
sigmoid reward-weighted RMSE is effectively tied and slightly favors the
comparator. Therefore the reportable claim is a Linear-UCB-consistent partial
reproduction under the local reward-tail proxy, not an exact DU-Net/Figure 4a
reproduction.

## Phase Decisions

{phase_decisions()}

## Remaining Blockers

- Linear UCB remains an MF-centered proxy, not the paper author's exact adapter.
- The paper w-RMSE formula, weighting, normalization, mask, and aggregation are
  not fully executable from the public artifact.
- The exact MovieLens split, initial observed mask, candidate mask, and seed
  protocol remain unrecovered.
- The DU-Net backend details for observational entropy, DLS, diffusion handling,
  and per-cycle tuning remain proxy-level.

## Next Step

Keep the frozen best candidate as the reportable local result. The no-network
MovieLens fidelity probe and exponential-sampling fidelity probe have both been
run and rejected because they weaken the primary reward-tail metric and/or
reward despite improving conventional RMSE-style checks. The plan should now
shift back to source-gated work only. Do not open another broad tuning sweep.

## Manifest

{manifest_table(manifest)}
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = final_rows()
    manifest = manifest_rows()
    summary.to_csv(OUT_SUMMARY, index=False)
    manifest.to_csv(OUT_MANIFEST, index=False)
    write_report(summary, manifest)
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_MANIFEST}")
    print(f"Wrote {OUT_REPORT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
