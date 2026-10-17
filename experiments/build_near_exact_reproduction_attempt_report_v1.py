from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
TABLE_ROOT = ROOT / "outputs" / "tables" / PREFIX
REPORT_PATH = ROOT / "reports" / "from_strong_clean_room_recovery_to_near_exact_reproduction_attempt.md"

TAU1 = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def md_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame[columns].copy() if columns else frame.copy()
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


def first_value(frame: pd.DataFrame, column: str, default: str = "missing") -> str:
    if frame.empty or column not in frame.columns:
        return default
    value = frame[column].iloc[0]
    if pd.isna(value):
        return default
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_report() -> None:
    summary = read_csv(TABLE_ROOT / "tau1_final_summary.csv")
    gaps = read_csv(TABLE_ROOT / "tau1_per_seed_gaps.csv")
    endpoints = read_csv(TABLE_ROOT / "tau1_figure4_endpoint_alignment.csv")
    curve_summary = read_csv(TABLE_ROOT / "tau1_figure4_curve_alignment_summary.csv")
    row_status = read_csv(TABLE_ROOT / "tau1_row_level_status.csv")
    gates = read_csv(ROOT / "outputs" / "tables" / "exact_replication_gate_audit_v1.csv")
    linear = read_csv(ROOT / "outputs" / "tables" / "linear_ucb_protocol_audit_v1.csv")
    split = read_csv(ROOT / "outputs" / "tables" / "split_search_audit_v2.csv")

    tau1 = summary[summary["variant"] == TAU1] if not summary.empty and "variant" in summary.columns else pd.DataFrame()
    concat = gaps[gaps["comparator"] == "linucb_concat"] if not gaps.empty and "comparator" in gaps.columns else pd.DataFrame()
    mf = gaps[gaps["comparator"] == "linucb_mf_interaction"] if not gaps.empty and "comparator" in gaps.columns else pd.DataFrame()
    concat_wins = f"{int(concat['standard_win'].sum())}/{concat['seed'].nunique()}" if not concat.empty else "missing"
    mf_wins = f"{int(mf['standard_win'].sum())}/{mf['seed'].nunique()}" if not mf.empty else "missing"

    lines = [
        "# From Strong Clean-Room Recovery to Near-Exact Reproduction Attempt: A Fidelity Audit of Figure 4 in Network-Enabled Sequential Data Acquisition",
        "",
        "## Bottom Line",
        "",
        "The tau1 literal-softmax DU-Net path is now the frozen clean-room mainline, but it is not an exact or perfect replication. The remaining blockers are source/protocol blockers, not simple tuning knobs.",
        "",
        "## Mainline Snapshot",
        "",
        f"- Config/report prefix: `{PREFIX}`",
        f"- tau1 final standard RMSE: `{first_value(tau1, 'standard_mask_rmse')}`",
        f"- tau1 final cumulative reward: `{first_value(tau1, 'cumulative_reward')}`",
        f"- RMSE wins vs `linucb_concat`: `{concat_wins}`",
        f"- RMSE wins vs `linucb_mf_interaction`: `{mf_wins}`",
        "",
        "## Figure 4 Endpoint Gate",
        "",
        md_table(endpoints),
        "",
        "## Curve Alignment",
        "",
        md_table(
            curve_summary,
            [
                "figure",
                "method",
                "local_metric",
                "endpoint_local",
                "endpoint_paper_visual",
                "raw_curve_rmse",
                "curve_correlation",
                "affine_curve_rmse",
            ],
        ),
        "",
        "## Row-Level Prediction Status",
        "",
        md_table(row_status),
        "",
        "## Exact Replication Gates",
        "",
        md_table(gates),
        "",
        "## Linear UCB Protocol Caveat",
        "",
        md_table(linear),
        "",
        "## Split/Mask Provenance",
        "",
        md_table(split),
        "",
        "## Claim Language",
        "",
        "Recommended claim: strong clean-room recovery with partial Figure 4 alignment. Do not use exact replication, perfect replication, or paper-faithful completion until the unresolved gates close with source-backed evidence.",
        "",
        "## Primary Artifacts",
        "",
        f"- `configs/{PREFIX}.json`",
        f"- `outputs/tables/{PREFIX}/tau1_final_summary.csv`",
        f"- `outputs/tables/{PREFIX}/tau1_per_seed_gaps.csv`",
        f"- `outputs/tables/{PREFIX}/tau1_per_cycle_curve.csv`",
        f"- `outputs/tables/{PREFIX}/tau1_figure4_endpoint_alignment.csv`",
        f"- `outputs/tables/{PREFIX}/tau1_figure4_curve_alignment_summary.csv`",
        f"- `outputs/row_level_predictions/{PREFIX}/`",
        "- `outputs/tables/split_search_audit_v2.csv`",
        "- `outputs/tables/linear_ucb_protocol_audit_v1.csv`",
        "- `outputs/tables/exact_replication_gate_audit_v1.csv`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    write_report()
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
