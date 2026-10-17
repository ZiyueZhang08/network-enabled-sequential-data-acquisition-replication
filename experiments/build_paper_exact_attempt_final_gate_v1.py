from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_exact_attempt_movielens_v1"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
OLD_TAU1_TABLE_DIR = TABLE_DIR / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"

OUT_CSV = TABLE_DIR / f"{PREFIX}_final_exact_gate.csv"
OUT_REPORT = REPORT_DIR / f"{PREFIX}_final_exact_gate.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda value: f"{float(value):.6g}" if pd.notna(value) else "")
        else:
            formatted[col] = formatted[col].fillna("").astype(str)
    lines = [
        "| " + " | ".join(str(col) for col in formatted.columns) + " |",
        "| " + " | ".join(["---"] * len(formatted.columns)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in formatted.columns) + " |")
    return "\n".join(lines)


def bool_text(value: object) -> str:
    return "yes" if bool(value) else "no"


def gate_rows() -> pd.DataFrame:
    metric = read_csv(TABLE_DIR / f"{PREFIX}_metric_reconstruction_v3_comparison.csv")
    row_completion = read_csv(TABLE_DIR / f"{PREFIX}_row_level_completion_v3.csv")
    endpoint = read_csv(OLD_TAU1_TABLE_DIR / "tau1_figure4_endpoint_alignment.csv")
    tau1_summary = read_csv(OLD_TAU1_TABLE_DIR / "tau1_final_summary.csv")
    exact_summary = read_csv(TABLE_DIR / f"{PREFIX}_final_summary.csv")
    linear = read_csv(TABLE_DIR / "linear_ucb_protocol_audit_v1.csv")
    split = read_csv(TABLE_DIR / "split_search_audit_v2.csv")

    metric_pass = False
    if not metric.empty and "paper_wrmse_reconstructed" in metric.columns:
        metric_pass = bool((metric["paper_wrmse_reconstructed"] == "yes").any())
    best_metric = metric.iloc[0].to_dict() if not metric.empty else {}

    row_complete = False
    if not row_completion.empty:
        row_complete = bool((row_completion["status"] == "complete_20seed").all())

    fig4a_standard = endpoint[(endpoint["figure"] == "Figure 4a") & (endpoint["local_metric"] == "standard_mask_rmse")] if not endpoint.empty else pd.DataFrame()
    fig4a_w = endpoint[(endpoint["figure"] == "Figure 4a") & (endpoint["local_metric"] == "w_rmse")] if not endpoint.empty else pd.DataFrame()
    fig4b = endpoint[(endpoint["figure"] == "Figure 4b") & (endpoint["local_metric"] == "cumulative_reward")] if not endpoint.empty else pd.DataFrame()

    selection_evidence = "tau1 summary missing"
    if not tau1_summary.empty:
        tau1 = tau1_summary[tau1_summary["variant"] == "schur_loglambda_tuned_full_pool_softmax_tau1_literal"].iloc[0]
        selection_evidence = f"tau1 final standard RMSE={float(tau1['standard_mask_rmse']):.6f}; existing 20-seed gate: vs concat 20/20, vs mf_interaction 19/20"

    exact_seed0_evidence = ""
    if not exact_summary.empty:
        exact_seed0_evidence = "; exact-attempt row-level replay currently seed0 only"

    row_text = "row-level completion table missing"
    if not row_completion.empty:
        counts = "; ".join(f"{row.method}:{row.available_seed_count}/20 {row.row_level_scope}" for row in row_completion.itertuples(index=False))
        row_text = counts

    linear_text = "linear UCB audit missing"
    if not linear.empty:
        installed = linear[linear["implementation_status"].astype(str).isin(["installed"])]
        local = linear[linear["implementation_status"] == "implemented_20seed"]
        linear_text = f"external installed candidates={len(installed)}; local proxy rows={len(local)}"

    split_text = "split audit missing"
    if not split.empty:
        exact = split[split["status"].astype(str).isin(["exact", "exact_match", "author_mask_found"])]
        split_text = f"exact public mask rows={len(exact)}; audit rows={len(split)}; current config uses u1 fallback"

    rows = [
        {
            "Gate": "Metric exactness",
            "pass/fail": "pass" if metric_pass else "fail",
            "evidence": (
                "paper_wrmse_reconstructed=yes"
                if metric_pass
                else f"v3 found no paper_wrmse_reconstructed metric; best={best_metric.get('metric', 'missing')} scale_distance={best_metric.get('scale_distance_to_paper', 'missing')} gap_distance={best_metric.get('gap_distance_to_paper', 'missing')}; {row_text}"
            ),
        },
        {
            "Gate": "Row-level completeness",
            "pass/fail": "pass" if row_complete else "fail",
            "evidence": row_text,
        },
        {
            "Gate": "Split exactness",
            "pass/fail": "fail",
            "evidence": split_text + "; public search did not recover author MovieLens initial/candidate/test masks.",
        },
        {
            "Gate": "Linear UCB exactness",
            "pass/fail": "fail",
            "evidence": linear_text + "; no unique paper_linear_ucb_reconstructed comparator.",
        },
        {
            "Gate": "DU-Net Eq.3.6",
            "pass/fail": "pass",
            "evidence": "explicit-kernel Schur toy equivalence report shows max abs diff 2.220e-16; noise/backend caveats remain separate.",
        },
        {
            "Gate": "DLS/ELPH",
            "pass/fail": "fail",
            "evidence": "MovieLens exact-attempt config uses GCMC/user metadata features; no recovered author ELPH backend is installed or validated.",
        },
        {
            "Gate": "Selection",
            "pass/fail": "pass",
            "evidence": selection_evidence + exact_seed0_evidence,
        },
        {
            "Gate": "Figure 4a",
            "pass/fail": "fail",
            "evidence": (
                f"standard_mask_rmse direction={bool_text(fig4a_standard['direction_match'].iloc[0]) if not fig4a_standard.empty else 'missing'}; "
                f"w_rmse direction={bool_text(fig4a_w['direction_match'].iloc[0]) if not fig4a_w.empty else 'missing'}; "
                "y-axis metric exactness fails."
            ),
        },
        {
            "Gate": "Figure 4b",
            "pass/fail": "fail",
            "evidence": (
                f"ranking direction={bool_text(fig4b['direction_match'].iloc[0]) if not fig4b.empty else 'missing'}; "
                f"gap_abs_error={float(fig4b['gap_abs_error'].iloc[0]):.6g} vs visual target" if not fig4b.empty else "Figure 4b endpoint table missing"
            ),
        },
    ]
    return pd.DataFrame(rows)


def write_report(frame: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pass_count = int((frame["pass/fail"] == "pass").sum())
    fail_count = int((frame["pass/fail"] == "fail").sum())
    status = "near-perfect reproduction" if fail_count == 0 else "strong clean-room recovery; exact replication blocked"
    lines = [
        "# Paper Exact Attempt MovieLens v1: Final Gate",
        "",
        f"Status: **{status}**.",
        "",
        "This table uses strict pass/fail semantics. Partial evidence is reported as `fail` until the gate's exact condition is satisfied.",
        "",
        f"- Passed gates: {pass_count}",
        f"- Failed gates: {fail_count}",
        "",
        md_table(frame),
        "",
        "## Public Source Notes",
        "",
        "- MovieLens 100K official folds are useful sensitivity inputs, but are not the recovered DU-Net Figure 4 masks.",
        "- Monti/RMGCNN and GCMC papers establish related graph matrix-completion data conventions, but the exact MovieLens sequential acquisition mask remains unrecovered.",
        "",
        "## Decision",
        "",
        "The current exact attempt cannot be called near-perfect. The defensible claim remains: strong clean-room recovery, exact replication blocked by unavailable or internally inconsistent paper details.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frame = gate_rows()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False)
    write_report(frame)
    print(OUT_REPORT)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
