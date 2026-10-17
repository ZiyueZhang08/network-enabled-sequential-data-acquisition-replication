from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
MAINLINE_TABLE_DIR = TABLE_DIR / PREFIX

SPLIT_AUDIT_PATH = TABLE_DIR / "split_search_audit_v2.csv"
LINEAR_AUDIT_PATH = TABLE_DIR / "linear_ucb_protocol_audit_v1.csv"
GATE_AUDIT_PATH = TABLE_DIR / "exact_replication_gate_audit_v1.csv"
REPORT_PATH = REPORT_DIR / "exact_replication_gate_audit_v1.md"

TAU1 = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
CONCAT = "linucb_concat"
MF = "linucb_mf_interaction"
BIAS = "linucb_bias_interaction"


def exists(path: str) -> bool:
    return (ROOT / path).exists()


def optional_csv(path: Path) -> pd.DataFrame:
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
            formatted[col] = formatted[col].astype(str)
    lines = [
        "| " + " | ".join(str(col) for col in formatted.columns) + " |",
        "| " + " | ".join(["---"] * len(formatted.columns)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in formatted.columns) + " |")
    return "\n".join(lines)


def split_search_rows() -> pd.DataFrame:
    rows = [
        {
            "search_group": "exact_paper_public_code",
            "query_or_source": '"Network-Enabled Sequential Data Acquisition" GitHub',
            "status": "not_found_public_web_or_local",
            "evidence": "Web search returned no author-code result for the exact title; local repo has PDF/Markdown/crops but no author code bundle.",
            "local_artifact": "Network-Enabled Sequential Data Acquisition (1).pdf",
            "decision": "Do not claim exact paper masks from public code.",
        },
        {
            "search_group": "exact_paper_public_code",
            "query_or_source": '"Network-Enabled Sequential Data Acquisition" MovieLens split code',
            "status": "not_found_public_web_or_local",
            "evidence": "Exact-title and DU-Net/MovieLens searches did not recover split files or executable Figure 4 scripts.",
            "local_artifact": "reports/paper_url_dependency_audit.md",
            "decision": "Keep author split/mask provenance unresolved.",
        },
        {
            "search_group": "grouplens_official_folds",
            "query_or_source": "MovieLens 100K u1-u5 official folds",
            "status": "available_not_exact",
            "evidence": "u1-u5 and ua/ub folds exist locally, but the paper does not state that Figure 4 uses one of these masks as the sequential acquisition test/candidate split.",
            "local_artifact": "data/raw/ml-100k/u1.base; data/raw/ml-100k/u1.test",
            "decision": "Use only as sensitivity evidence, never as mainline by best-looking fold.",
        },
        {
            "search_group": "gcmc_loader",
            "query_or_source": "GCMC MovieLens official split loader",
            "status": "available_partial",
            "evidence": "Local GCMC provenance confirms a u1.base/u1.test loader pattern and a validation split, but not DU-Net's Figure 4 initial/candidate/test masks.",
            "local_artifact": "reports/gcmc_monti_split_provenance_audit.md",
            "decision": "Provenance support only; exact mask still blocked.",
        },
        {
            "search_group": "monti_mat_files",
            "query_or_source": "Monti training_test_dataset.mat",
            "status": "available_for_flixster_douban_not_movielens",
            "evidence": "Local Monti-style .mat masks are present for Flixster and Douban, but not MovieLens.",
            "local_artifact": "data/raw/gcmc/flixster/training_test_dataset.mat; data/raw/gcmc/douban/training_test_dataset.mat",
            "decision": "Do not infer MovieLens masks from other datasets.",
        },
        {
            "search_group": "local_split_distribution",
            "query_or_source": "official folds vs seeded random masks",
            "status": "distribution_similar_mask_not_exact",
            "evidence": "Prior overlap audit shows similar marginal distributions but different masks.",
            "local_artifact": "outputs/tables/movielens_split_overlap_audit.csv",
            "decision": "Split alone is not the dominant scale fix, but exactness remains blocked.",
        },
    ]
    return pd.DataFrame(rows)


def linear_ucb_rows() -> pd.DataFrame:
    summary = optional_csv(MAINLINE_TABLE_DIR / "tau1_final_summary.csv")
    rows = []
    package_specs = [
        ("contextualbandits", "contextualbandits.online.LinUCB/default-disjoint", "not_installed" if importlib.util.find_spec("contextualbandits") is None else "installed"),
        ("lrbandit", "lrbandit/OFUL paper-adjacent adapter", "not_installed" if importlib.util.find_spec("lrbandit") is None else "installed"),
    ]
    for package, candidate, status in package_specs:
        rows.append(
            {
                "candidate": candidate,
                "implementation_status": status,
                "paper_faithfulness": "candidate_external_protocol",
                "final_standard_rmse": "",
                "final_w_rmse": "",
                "final_cumulative_reward": "",
                "evidence": f"Python package `{package}` availability checked locally without adding dependencies.",
                "decision": "Do not add dependency or promote until explicit author-code/package link is recovered.",
            }
        )
    for variant, label in [
        (CONCAT, "local linucb_concat"),
        (MF, "local linucb_mf_interaction"),
        (BIAS, "local linucb_bias_interaction"),
    ]:
        if summary.empty or variant not in set(summary["variant"]):
            rows.append(
                {
                    "candidate": label,
                    "implementation_status": "missing_from_mainline_summary",
                    "paper_faithfulness": "local_proxy",
                    "final_standard_rmse": "",
                    "final_w_rmse": "",
                    "final_cumulative_reward": "",
                    "evidence": "No row in tau1 final summary.",
                    "decision": "Cannot use as comparator until regenerated.",
                }
            )
            continue
        row = summary[summary["variant"] == variant].iloc[0]
        rows.append(
            {
                "candidate": label,
                "implementation_status": "implemented_20seed",
                "paper_faithfulness": "local_proxy_not_author_protocol",
                "final_standard_rmse": float(row["standard_mask_rmse"]),
                "final_w_rmse": float(row["w_rmse"]),
                "final_cumulative_reward": float(row["cumulative_reward"]),
                "evidence": "Included in tau1 mainline bundle summary.",
                "decision": "Use as comparator range, not exact paper Linear UCB.",
            }
        )
    return pd.DataFrame(rows)


def gate_rows() -> pd.DataFrame:
    summary = optional_csv(MAINLINE_TABLE_DIR / "tau1_final_summary.csv")
    gaps = optional_csv(MAINLINE_TABLE_DIR / "tau1_per_seed_gaps.csv")
    endpoints = optional_csv(MAINLINE_TABLE_DIR / "tau1_figure4_endpoint_alignment.csv")
    row_status = optional_csv(MAINLINE_TABLE_DIR / "tau1_row_level_status.csv")
    metric_audit = optional_csv(MAINLINE_TABLE_DIR / "tau1_metric_audit.csv")

    tau1_row = summary[summary["variant"] == TAU1].iloc[0] if not summary.empty and TAU1 in set(summary["variant"]) else None
    concat_wins = ""
    mf_wins = ""
    if not gaps.empty:
        concat = gaps[gaps["comparator"] == CONCAT]
        mf = gaps[gaps["comparator"] == MF]
        concat_wins = f"{int(concat['standard_win'].sum())}/{concat['seed'].nunique()}"
        mf_wins = f"{int(mf['standard_win'].sum())}/{mf['seed'].nunique()}"

    fig4a_standard = endpoints[(endpoints["figure"] == "Figure 4a") & (endpoints["local_metric"] == "standard_mask_rmse")] if not endpoints.empty else pd.DataFrame()
    fig4a_w = endpoints[(endpoints["figure"] == "Figure 4a") & (endpoints["local_metric"] == "w_rmse")] if not endpoints.empty else pd.DataFrame()
    fig4b = endpoints[(endpoints["figure"] == "Figure 4b") & (endpoints["local_metric"] == "cumulative_reward")] if not endpoints.empty else pd.DataFrame()

    row_level_complete = False
    if not row_status.empty:
        row_level_complete = bool((row_status["status"] == "complete").all())

    metric_pass = False
    if not metric_audit.empty and "metric_gate" in metric_audit.columns:
        metric_pass = bool((metric_audit["metric_gate"] == "pass").any())

    rows = [
        {
            "gate": "tau1_literal_mainline_freeze",
            "status": "closed_cleanroom",
            "evidence": f"tau1 RMSE={float(tau1_row['standard_mask_rmse']):.6f}; wins vs concat={concat_wins}; wins vs mf_interaction={mf_wins}" if tau1_row is not None else "tau1 summary missing",
            "blocking_reason": "",
            "next_action": "Keep tau1 as mainline; keep tau0.3 robustness only.",
        },
        {
            "gate": "row_level_metric_reconstruction_v3",
            "status": "partial",
            "evidence": "Row-level schema is implemented, but completion table shows only seed0 tau1/concat currently available.",
            "blocking_reason": "Full 20-seed row-level CSVs for all required methods are not generated.",
            "next_action": "Generate row-level predictions in bounded batches only when compute budget is explicitly allocated.",
        },
        {
            "gate": "metric_definition_exactness",
            "status": "unresolved",
            "evidence": "Metric audit v2 finds direction under standard RMSE but no endpoint scale/gap candidate passes.",
            "blocking_reason": "Paper formula, plotted Figure 4a scale, and local row-level metric candidates do not agree.",
            "next_action": "Do not call exact until a source-backed metric definition or author table is recovered.",
        },
        {
            "gate": "figure4a_joint_endpoint_curve",
            "status": "partial",
            "evidence": f"standard direction={bool(fig4a_standard['direction_match'].iloc[0]) if not fig4a_standard.empty else 'missing'}; w_rmse direction={bool(fig4a_w['direction_match'].iloc[0]) if not fig4a_w.empty else 'missing'}",
            "blocking_reason": "Figure 4a standard RMSE proxy aligns in direction, but local w_rmse reverses the paper ordering.",
            "next_action": "Report both metric readings; do not select the convenient one as exact.",
        },
        {
            "gate": "figure4b_same_config",
            "status": "partial",
            "evidence": f"direction={bool(fig4b['direction_match'].iloc[0]) if not fig4b.empty else 'missing'}; gap_abs_error={float(fig4b['gap_abs_error'].iloc[0]):.6g}" if not fig4b.empty else "missing",
            "blocking_reason": "Ranking is paper-like, but reward gap magnitude is too large.",
            "next_action": "Treat lambda=0.3 and tau variants as diagnostics only; same-config gate remains partial.",
        },
        {
            "gate": "split_mask_public_provenance",
            "status": "blocked_by_missing_source",
            "evidence": "split_search_audit_v2 records no exact author MovieLens masks or code; official u1-u5 folds are sensitivity-only.",
            "blocking_reason": "Exact initial/candidate/test masks and seed mapping are not public in recovered sources.",
            "next_action": "Do not choose most paper-like fold as mainline.",
        },
        {
            "gate": "linear_ucb_protocol",
            "status": "proxy_set_only",
            "evidence": "Local concat/MF/bias variants form a comparator range; contextualbandits/lrbandit are not installed and not author-confirmed.",
            "blocking_reason": "Exact paper Linear UCB backend/protocol is not recovered.",
            "next_action": "Keep clean-room comparator caveat.",
        },
        {
            "gate": "du_net_entropy_eq36",
            "status": "closed_cleanroom",
            "evidence": "Toy equivalence report and tests show Schur conditional variance matches the explicit kernel Eq. 3.6 form to numerical precision.",
            "blocking_reason": "",
            "next_action": "Keep as closed unless paper reveals a different noise/backend convention.",
        },
        {
            "gate": "all_candidate_shortlist",
            "status": "unresolved",
            "evidence": "Mainline uses paper_entropy_shortlist_size=20000 while final candidate size is about 76635.",
            "blocking_reason": "All-candidate Schur is not run as a frozen diagnostic.",
            "next_action": "Run a small all-candidate diagnostic before any exact claim, not another tuning sweep.",
        },
        {
            "gate": "lambda_noise_backend",
            "status": "unresolved",
            "evidence": "lambda_noise is tied to entropy_logdet_regularization=1.0 in the current executable path.",
            "blocking_reason": "Paper does not expose enough backend/noise details to prove the literal convention.",
            "next_action": "Keep lambda_noise audit as source-blocked unless author code appears.",
        },
        {
            "gate": "elph_backend",
            "status": "unresolved",
            "evidence": "Local graph embeddings include dependency-free ELPH-style proxies, not a recovered paper ELPH backend.",
            "blocking_reason": "Exact ELPH implementation/parameters are not recovered.",
            "next_action": "Do not promote proxy ELPH into exact MovieLens Figure 4 claims.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(split_audit: pd.DataFrame, linear_audit: pd.DataFrame, gate_audit: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exact Replication Gate Audit v1",
        "",
        "This is a boundary-setting audit, not a tuning report. It freezes what is closed, separates clean-room evidence from exact replication, and names the remaining source blockers.",
        "",
        "## Gate Status",
        "",
        md_table(gate_audit),
        "",
        "## Linear UCB Protocol",
        "",
        md_table(linear_audit),
        "",
        "## Split Search",
        "",
        md_table(split_audit),
        "",
        "## Decision",
        "",
        "The current result should be described as a strong clean-room recovery attempt with partial Figure 4 alignment. It should not be called exact or perfect until metric definition, split/mask provenance, Linear UCB protocol, all-candidate/noise backend, and ELPH backend gates are closed.",
        "",
        f"CSV artifacts: `{SPLIT_AUDIT_PATH.relative_to(ROOT)}`, `{LINEAR_AUDIT_PATH.relative_to(ROOT)}`, `{GATE_AUDIT_PATH.relative_to(ROOT)}`.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    split_audit = split_search_rows()
    linear_audit = linear_ucb_rows()
    gate_audit = gate_rows()
    split_audit.to_csv(SPLIT_AUDIT_PATH, index=False)
    linear_audit.to_csv(LINEAR_AUDIT_PATH, index=False)
    gate_audit.to_csv(GATE_AUDIT_PATH, index=False)
    write_report(split_audit, linear_audit, gate_audit)
    print(REPORT_PATH)
    print(SPLIT_AUDIT_PATH)
    print(LINEAR_AUDIT_PATH)
    print(GATE_AUDIT_PATH)


if __name__ == "__main__":
    main()
