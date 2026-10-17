from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_exact_attempt_movielens_v2"
CONFIG_PATH = ROOT / "configs" / f"{PREFIX}.json"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
FINAL_SUMMARY_PATH = TABLE_DIR / f"{PREFIX}_final_summary.csv"
METRIC_COMPARISON_PATH = TABLE_DIR / "metric_reconstruction_v4_comparison.csv"
PRIOR_TUNE_5SEED_PATH = TABLE_DIR / "paper_faithful_tune_every_cycle_5seed_diagnostic_20260709.csv"
OUT_CSV = TABLE_DIR / f"{PREFIX}_tuning_frequency_objective_audit.csv"
OUT_REPORT = REPORT_DIR / f"{PREFIX}_tuning_frequency_objective_audit.md"


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.6g}" if pd.notna(value) else "")
        else:
            view[col] = view[col].fillna("").astype(str)

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(col) for col in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(clean(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def final_row(summary: pd.DataFrame, variant: str) -> pd.Series | None:
    rows = summary[summary["variant"] == variant]
    if rows.empty:
        return None
    return rows.iloc[0]


def metric_text(value: object, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def summarize_current_objective(summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    keep = [
        "variant",
        "seed_count",
        "standard_mask_rmse",
        "w_rmse",
        "reward_tail_proxy_rmse",
        "cumulative_reward",
        "centered_cumulative_reward",
    ]
    if summary.empty:
        return pd.DataFrame(columns=keep), "final summary missing"
    available = [col for col in keep if col in summary.columns]
    view = summary[available].copy()
    view = view.sort_values(["seed_count", "variant"], ascending=[False, True]).reset_index(drop=True)

    main = final_row(summary, "schur_loglambda_tuned_full_pool_softmax_tau1_literal")
    if main is None:
        return view, "mainline final-summary row missing"
    parts = [
        f"mainline seed_count={int(float(main['seed_count']))}",
        f"standard_mask_rmse={metric_text(main['standard_mask_rmse'])}",
        f"w_rmse={metric_text(main['w_rmse'])}",
        f"reward_tail_proxy_rmse={metric_text(main['reward_tail_proxy_rmse'])}",
        f"cumulative_reward={metric_text(main['cumulative_reward'])}",
    ]
    for comp in ["linucb_concat", "linucb_mf_interaction", "linucb_bias_interaction"]:
        row = final_row(summary, comp)
        if row is None:
            continue
        standard_gap = float(row["standard_mask_rmse"]) - float(main["standard_mask_rmse"])
        reward_gap = float(main["cumulative_reward"]) - float(row["cumulative_reward"])
        tail_gap = float(row["reward_tail_proxy_rmse"]) - float(main["reward_tail_proxy_rmse"])
        parts.append(
            f"vs {comp}: standard_gap_linear_minus_du={standard_gap:.6g}, "
            f"reward_gap_du_minus_linear={reward_gap:.6g}, "
            f"reward_tail_gap_linear_minus_du={tail_gap:.6g}"
        )
    return view, "; ".join(parts)


def summarize_metric_gate() -> str:
    comparison = read_csv(METRIC_COMPARISON_PATH)
    if comparison.empty:
        return "metric reconstruction v4 missing"
    best = comparison.iloc[0]
    return (
        f"best_metric={best['metric']}; comparator={best['comparator']}; "
        f"row_level_seed_count={int(best['row_level_seed_count'])}; "
        f"scale_distance={float(best['scale_distance_to_paper']):.6g}; "
        f"gap_distance={float(best['gap_distance_to_paper']):.6g}; "
        f"paper_wrmse_reconstructed={best['paper_wrmse_reconstructed']}"
    )


def prior_every_cycle_evidence() -> str:
    prior = read_csv(PRIOR_TUNE_5SEED_PATH)
    if prior.empty:
        return "prior every-cycle diagnostic missing"
    subset = prior[prior["comparison"] == "strict_tune_minus_mainline"]
    if subset.empty:
        return f"prior every-cycle diagnostic exists: {PRIOR_TUNE_5SEED_PATH.relative_to(ROOT).as_posix()}"
    pieces = []
    for metric in ["standard_mask_rmse", "w_rmse", "reward_tail_proxy_rmse", "cumulative_reward"]:
        row = subset[subset["metric"] == metric]
        if not row.empty:
            pieces.append(f"{metric}_delta={float(row.iloc[0]['value']):.6g}")
    return (
        f"prior 5-seed every-cycle vs every-5 diagnostic: {'; '.join(pieces)}; "
        f"source={PRIOR_TUNE_5SEED_PATH.relative_to(ROOT).as_posix()}"
    )


def build_rows(config: dict, objective_text: str, metric_gate_text: str, prior_text: str) -> pd.DataFrame:
    acquisition = config["acquisition"]
    lambda_grid = " ".join(str(value) for value in acquisition["adaptive_lambda_grid"])
    sigma_grid = " ".join(str(value) for value in acquisition["sigma_grid"])
    diffusion_grid = " ".join(str(value) for value in acquisition["adaptive_diffusion_prob_grid"])
    rows = [
        {
            "audit_item": "retuning_frequency",
            "current_protocol": f"every {acquisition['paper_tuning_frequency']} cycles",
            "paper_required_for_exact": "author-equivalent retuning cadence for lambda_H, sigma_t, and p_t",
            "status": "source_blocked_not_closed",
            "evidence": prior_text,
            "exactness_risk": "Paper text says hyperparameters are tuned during acquisition, but does not publish a cadence sufficient to choose every-cycle vs every-5-cycle exactly.",
        },
        {
            "audit_item": "inner_objective",
            "current_protocol": str(acquisition["paper_tuning_metric"]),
            "paper_required_for_exact": "author-equivalent inner train/acquisition/test objective and held-out split",
            "status": "source_blocked_not_closed",
            "evidence": objective_text,
            "exactness_risk": "The current objective optimizes Figure-4a-like RMSE behavior, while Figure 4b reward can move differently under the same branch.",
        },
        {
            "audit_item": "lambda_sigma_p_grid",
            "current_protocol": f"lambda={lambda_grid}; sigma={sigma_grid}; p={diffusion_grid}",
            "paper_required_for_exact": "author grid or continuous optimizer, tie-breaks, and any staged schedule",
            "status": "source_blocked_not_closed",
            "evidence": "v2 config grid is explicit and reproducible, but not source-verified.",
            "exactness_risk": "Grid endpoints and tie-breaks can change selected batches even when the acquisition formula is otherwise close.",
        },
        {
            "audit_item": "metric_gate_interaction",
            "current_protocol": "tuning evidence must be judged under metric v4 reconstruction",
            "paper_required_for_exact": "same metric definition used in paper Figure 4a",
            "status": "blocked_by_metric_gate",
            "evidence": metric_gate_text,
            "exactness_risk": "No tuning protocol can be called exact while the target w-RMSE formula and y-axis scale remain unreconstructed.",
        },
        {
            "audit_item": "joint_figure4_objective",
            "current_protocol": "standard_mask_rmse inner objective with reward tracked as an outcome",
            "paper_required_for_exact": "single same-config protocol matching Figure 4a and Figure 4b",
            "status": "not_jointly_closed",
            "evidence": objective_text,
            "exactness_risk": "A branch that improves RMSE can reduce immediate reward, so Figure 4b cannot be inferred from Figure 4a tuning success.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(config: dict, audit: pd.DataFrame, summary_view: pd.DataFrame) -> None:
    acquisition = config["acquisition"]
    source_blocked = int((audit["status"] == "source_blocked_not_closed").sum())
    lines = [
        "# paper_exact_attempt_movielens_v2 Tuning Frequency/Objective Audit",
        "",
        "This Phase 5 artifact audits the tuning protocol as an exact-replication gate. It does not promote a new branch by performance.",
        "",
        "## Current v2 Protocol",
        "",
        f"- Retuning frequency: every `{acquisition['paper_tuning_frequency']}` cycles",
        f"- Inner objective: `{acquisition['paper_tuning_metric']}`",
        f"- Lambda grid: `{acquisition['adaptive_lambda_grid']}`",
        f"- Sigma grid: `{acquisition['sigma_grid']}`",
        f"- Diffusion p grid: `{acquisition['adaptive_diffusion_prob_grid']}`",
        "",
        "## Audit Table",
        "",
        md_table(audit),
        "",
        "## Current Same-Config Outcome Snapshot",
        "",
        md_table(summary_view),
        "",
        "## Decision",
        "",
        f"- Source-blocked tuning items: `{source_blocked}`",
        "- The v2 tuning protocol is reproducible and fixed, but it is not author-source exact.",
        "- Phase 5 remains open until the author-equivalent retuning cadence, inner objective, held-out construction, grid/optimizer, and tie-break rules are recovered.",
        "- Because metric v4 still fails, running broader local tuning would not close exact replication; it would only create additional proxy branches.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    summary = read_csv(FINAL_SUMMARY_PATH)
    summary_view, objective_text = summarize_current_objective(summary)
    metric_gate_text = summarize_metric_gate()
    prior_text = prior_every_cycle_evidence()
    audit = build_rows(config, objective_text, metric_gate_text, prior_text)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(OUT_CSV, index=False)
    write_report(config, audit, summary_view)
    print(OUT_REPORT)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
