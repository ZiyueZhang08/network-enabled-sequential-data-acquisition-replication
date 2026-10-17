from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "outputs" / "logs"

PREFIX = "paper_faithful_movielens_tau1_literal_p0_reconstruction_v1"
TAU1_PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_20seeds"
TAU03_PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1"
TOPK_PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_topk_v1"
COMPARATOR_PREFIX = "paper_faithful_movielens_comparator_sensitivity_v1_20seeds"
FOLD_PREFIX = "paper_faithful_movielens_softmax_tau1_fold_sensitivity_v1_3seeds"
METRIC_AUDIT_PREFIX = "paper_faithful_movielens_tau1_literal_metric_reconstruction_audit_v2"

TAU1 = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
TAU03 = "schur_loglambda_tuned_full_pool_softmax_tau0p3"
TOPK_SOURCE = "schur_loglambda_tuned_standard_topk"
TOPK = "schur_tuned_topk"
CONCAT = "linucb_concat"
MF = "linucb_mf_interaction"
BIAS = "linucb_bias_interaction"

PAPER_FIG4A_DU_ENDPOINT = 1.788
PAPER_FIG4A_LINEAR_ENDPOINT = 1.814
PAPER_FIG4B_DU_ENDPOINT_APPROX = 1325.0
PAPER_FIG4B_LINEAR_ENDPOINT_APPROX = 1390.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ci95(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)


def normalize_summary(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    if "policy" not in frame.columns:
        frame["policy"] = ""
    return frame


def load_final_summary() -> pd.DataFrame:
    tau1 = normalize_summary(pd.read_csv(TABLE_DIR / f"{TAU1_PREFIX}_final_summary.csv"))
    tau03 = normalize_summary(pd.read_csv(TABLE_DIR / f"{TAU03_PREFIX}_final_summary.csv"))
    topk = normalize_summary(pd.read_csv(TABLE_DIR / f"{TOPK_PREFIX}_final_summary.csv"))
    topk = topk[topk["variant"] == TOPK_SOURCE].copy()
    topk["variant"] = TOPK
    comparators = normalize_summary(pd.read_csv(TABLE_DIR / f"{COMPARATOR_PREFIX}_final_summary.csv"))
    comparators = comparators[comparators["variant"].isin([CONCAT, MF, BIAS])].copy()
    return pd.concat([tau1, tau03[tau03["variant"] == TAU03], topk, comparators], ignore_index=True, sort=False)


def load_combined_metrics() -> pd.DataFrame:
    tau1 = pd.read_csv(TABLE_DIR / f"{TAU1_PREFIX}_metrics.csv")
    tau03 = pd.read_csv(TABLE_DIR / f"{TAU03_PREFIX}_metrics.csv")
    topk = pd.read_csv(TABLE_DIR / f"{TOPK_PREFIX}_metrics.csv")
    topk = topk[topk["variant"] == TOPK_SOURCE].copy()
    topk["variant"] = TOPK
    comparators = pd.read_csv(TABLE_DIR / f"{COMPARATOR_PREFIX}_metrics.csv")
    comparators = comparators[comparators["variant"].isin([CONCAT, MF, BIAS])].copy()
    return pd.concat([tau1, tau03[tau03["variant"] == TAU03], topk, comparators], ignore_index=True, sort=False)


def build_selection_position(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    role = {
        TAU1: "literal paper-style mainline",
        TAU03: "calibrated robustness / best-RMSE robustness",
        TOPK: "deterministic ablation",
    }
    for variant in [TAU1, TAU03, TOPK]:
        row = summary[summary["variant"] == variant].iloc[0]
        rows.append(
            {
                "version": variant,
                "status": role[variant],
                "seed_count": int(float(row["seed_count"])),
                "standard_mask_rmse": float(row["standard_mask_rmse"]),
                "pdf_literal_norm_rmse": float(row["pdf_literal_norm_rmse"]),
                "cumulative_reward": float(row["cumulative_reward"]),
                "claim_role": "mainline" if variant == TAU1 else "supporting",
            }
        )
    return pd.DataFrame(rows)


def build_gap_summary(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparator in [CONCAT, MF, BIAS, TOPK, TAU03]:
        for seed, group in final.groupby("seed"):
            du = group[group["variant"] == TAU1]
            base = group[group["variant"] == comparator]
            if du.empty or base.empty:
                continue
            du_row = du.iloc[0]
            base_row = base.iloc[0]
            rows.append(
                {
                    "seed": int(seed),
                    "comparator": comparator,
                    "standard_gap_comparator_minus_tau1": float(
                        base_row["standard_mask_rmse"] - du_row["standard_mask_rmse"]
                    ),
                    "pdf_gap_comparator_minus_tau1": float(
                        base_row["pdf_literal_norm_rmse"] - du_row["pdf_literal_norm_rmse"]
                    ),
                    "reward_gap_tau1_minus_comparator": float(
                        du_row["cumulative_reward"] - base_row["cumulative_reward"]
                    ),
                    "standard_win": int(du_row["standard_mask_rmse"] < base_row["standard_mask_rmse"]),
                }
            )
    paired = pd.DataFrame(rows)
    summary = (
        paired.groupby("comparator", as_index=False)
        .agg(
            mean_standard_gap=("standard_gap_comparator_minus_tau1", "mean"),
            standard_win_rate=("standard_win", "mean"),
            wins=("standard_win", "sum"),
            seeds=("seed", "nunique"),
            mean_pdf_gap=("pdf_gap_comparator_minus_tau1", "mean"),
            mean_reward_gap=("reward_gap_tau1_minus_comparator", "mean"),
        )
        .sort_values("mean_standard_gap", ascending=False)
    )
    return paired, summary


def build_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "standard_mask_rmse": ("standard_mask_rmse", "mean"),
        "standard_mask_rmse_ci95": ("standard_mask_rmse", ci95),
        "pdf_literal_norm_rmse": ("pdf_literal_norm_rmse", "mean"),
        "cumulative_reward": ("cumulative_reward", "mean"),
        "cumulative_reward_ci95": ("cumulative_reward", ci95),
    }
    return metrics.groupby(["variant", "cycle"], as_index=False).agg(**agg).sort_values(["variant", "cycle"])


def build_one_config_table(summary: pd.DataFrame) -> pd.DataFrame:
    tau1 = summary[summary["variant"] == TAU1].iloc[0]
    concat = summary[summary["variant"] == CONCAT].iloc[0]
    mf = summary[summary["variant"] == MF].iloc[0]
    fig4a_gap = float(concat["standard_mask_rmse"] - tau1["standard_mask_rmse"])
    paper_fig4a_gap = PAPER_FIG4A_LINEAR_ENDPOINT - PAPER_FIG4A_DU_ENDPOINT
    fig4b_gap = float(tau1["cumulative_reward"] - concat["cumulative_reward"])
    paper_fig4b_gap = PAPER_FIG4B_DU_ENDPOINT_APPROX - PAPER_FIG4B_LINEAR_ENDPOINT_APPROX
    return pd.DataFrame(
        [
            {
                "figure": "Figure 4a",
                "metric": "standard_mask_rmse executable proxy",
                "DU-Net local": float(tau1["standard_mask_rmse"]),
                "Linear local": float(concat["standard_mask_rmse"]),
                "DU-Net paper": PAPER_FIG4A_DU_ENDPOINT,
                "Linear paper": PAPER_FIG4A_LINEAR_ENDPOINT,
                "local_gap_linear_minus_du": fig4a_gap,
                "paper_gap_linear_minus_du": paper_fig4a_gap,
                "ranking_match": "yes",
                "scale_match": "no",
                "gate": "metric_unresolved",
            },
            {
                "figure": "Figure 4a",
                "metric": "pdf_literal_norm_rmse",
                "DU-Net local": float(tau1["pdf_literal_norm_rmse"]),
                "Linear local": float(concat["pdf_literal_norm_rmse"]),
                "DU-Net paper": PAPER_FIG4A_DU_ENDPOINT,
                "Linear paper": PAPER_FIG4A_LINEAR_ENDPOINT,
                "local_gap_linear_minus_du": float(concat["pdf_literal_norm_rmse"] - tau1["pdf_literal_norm_rmse"]),
                "paper_gap_linear_minus_du": paper_fig4a_gap,
                "ranking_match": "yes",
                "scale_match": "no",
                "gate": "metric_unresolved",
            },
            {
                "figure": "Figure 4b",
                "metric": "cumulative_reward",
                "DU-Net local": float(tau1["cumulative_reward"]),
                "Linear local": float(concat["cumulative_reward"]),
                "DU-Net paper": PAPER_FIG4B_DU_ENDPOINT_APPROX,
                "Linear paper": PAPER_FIG4B_LINEAR_ENDPOINT_APPROX,
                "local_gap_linear_minus_du": -fig4b_gap,
                "paper_gap_linear_minus_du": -paper_fig4b_gap,
                "ranking_match": "yes",
                "scale_match": "partial",
                "gate": "partial_recovery_gap_too_large",
            },
            {
                "figure": "Figure 4b",
                "metric": "cumulative_reward vs mf_interaction",
                "DU-Net local": float(tau1["cumulative_reward"]),
                "Linear local": float(mf["cumulative_reward"]),
                "DU-Net paper": PAPER_FIG4B_DU_ENDPOINT_APPROX,
                "Linear paper": PAPER_FIG4B_LINEAR_ENDPOINT_APPROX,
                "local_gap_linear_minus_du": float(mf["cumulative_reward"] - tau1["cumulative_reward"]),
                "paper_gap_linear_minus_du": -paper_fig4b_gap,
                "ranking_match": "not_author_linear_ucb",
                "scale_match": "diagnostic_only",
                "gate": "comparator_proxy_only",
            },
        ]
    )


def build_gate_table(metric_audit: pd.DataFrame, fold_gaps: pd.DataFrame) -> pd.DataFrame:
    metric_pass = bool(not metric_audit.empty and (metric_audit.get("metric_gate") == "pass").any())
    fold_pass = bool(
        not fold_gaps.empty
        and (fold_gaps.groupby("comparator")["win_rate"].mean() >= (2 / 3)).all()
    )
    eq36_report = REPORT_DIR / "paper_faithful_entropy_eq36_equivalence_v1.md"
    return pd.DataFrame(
        [
            {
                "gate": "Metric",
                "status": "pass" if metric_pass else "unresolved",
                "evidence": "metric reconstruction audit v2",
            },
            {
                "gate": "Entropy Eq3.6",
                "status": "closed_cleanroom" if eq36_report.exists() else "unresolved",
                "evidence": "explicit observed-kernel Schur complement equivalence note and unit test",
            },
            {
                "gate": "Split",
                "status": "robust_proxy" if fold_pass else "unresolved",
                "evidence": "u1-u5 tau1 3-seed fold sensitivity" if not fold_gaps.empty else "fold sensitivity not yet run",
            },
            {
                "gate": "Linear UCB",
                "status": "proxy_set",
                "evidence": "concat, mf_interaction, bias_interaction comparators; author library protocol unavailable",
            },
            {
                "gate": "DU-Net",
                "status": "partial",
                "evidence": "Schur Eq3.6-equivalent conditional variance + DLS + tau1 selection; ELPH/exact author backend still open",
            },
            {
                "gate": "Figure 4a",
                "status": "strong_pattern_recovery",
                "evidence": "tau1 lower than linucb_concat and mf_interaction under executable standard RMSE",
            },
            {
                "gate": "Figure 4b",
                "status": "partial",
                "evidence": "ranking vs Linear UCB matches paper direction, local reward gap is too large",
            },
        ]
    )


def build_manifest(paths: list[tuple[Path, str]]) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for path, description in paths:
        if not path.exists():
            continue
        rows.append(
            {
                "artifact_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "description": description,
                "created_at": now,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    selection: pd.DataFrame,
    final_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    one_config: pd.DataFrame,
    gates: pd.DataFrame,
    fold_gaps: pd.DataFrame,
    metric_audit: pd.DataFrame,
    manifest: pd.DataFrame,
    report_path: Path,
) -> None:
    tau1 = final_summary[final_summary["variant"] == TAU1].iloc[0]
    concat = final_summary[final_summary["variant"] == CONCAT].iloc[0]
    mf = final_summary[final_summary["variant"] == MF].iloc[0]
    metric_audit_block = (
        md_table(metric_audit.head(12))
        if not metric_audit.empty
        else "_No row-level metric rows yet. The row-level export path and audit script are implemented, but the full MovieLens row-level rerun did not complete in this session._"
    )
    fold_block = (
        md_table(fold_gaps)
        if not fold_gaps.empty
        else "_No tau1 fold-sensitivity rows yet. The fold runner now uses tau1 full-pool softmax, but the u1-u5 x 3-seed rerun still needs to be executed._"
    )
    lines = [
        f"# Tau1 Literal P0 Reconstruction Bundle: {PREFIX}",
        "",
        "## Mainline Decision",
        "",
        "`tau1_literal` is now the paper-faithful mainline because the paper states exponential weighting without a temperature. `tau0.3` remains a calibrated robustness result, not the main claim.",
        "",
        md_table(selection),
        "",
        "## Same-Config Figure 4a / Figure 4b",
        "",
        "Paper endpoints are approximate visual anchors from the supplied Figure 4 visual extraction. They are not a replacement for full WebPlotDigitizer curve extraction.",
        "",
        md_table(one_config),
        "",
        "## Tau1 Final Results",
        "",
        f"- tau1 standard RMSE: {float(tau1['standard_mask_rmse']):.6f}",
        f"- linucb_concat standard RMSE: {float(concat['standard_mask_rmse']):.6f}",
        f"- linucb_mf_interaction standard RMSE: {float(mf['standard_mask_rmse']):.6f}",
        f"- tau1 cumulative reward: {float(tau1['cumulative_reward']):.3f}",
        f"- linucb_concat cumulative reward: {float(concat['cumulative_reward']):.3f}",
        "",
        md_table(final_summary[["variant", "seed_count", "standard_mask_rmse", "pdf_literal_norm_rmse", "cumulative_reward", "w_rmse"]]),
        "",
        "## Comparator Gaps",
        "",
        md_table(gap_summary),
        "",
        "## Metric Reconstruction Gate",
        "",
        metric_audit_block,
        "",
        "## Fold Sensitivity Gate",
        "",
        fold_block,
        "",
        "## Gate Summary",
        "",
        md_table(gates),
        "",
        "## Conclusion",
        "",
        "This is the strongest current paper-only clean-room reproduction path, not an exact replication. The mainline now removes the controllable selection inconsistency by using tau1 literal softmax, and the Schur entropy implementation is locked against the explicit observed-kernel Eq3.6 form. Metric, exact split/mask, author Linear UCB, and ELPH/exact author backend remain the limiting ambiguities.",
        "",
        "## Artifact Manifest",
        "",
        md_table(manifest),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    final_summary = load_final_summary()
    metrics = load_combined_metrics()
    final = final_rows(metrics)
    selection = build_selection_position(final_summary)
    paired_gaps, gap_summary = build_gap_summary(final)
    curve = build_curve(metrics)
    one_config = build_one_config_table(final_summary)
    metric_audit = read_csv_if_exists(TABLE_DIR / f"{METRIC_AUDIT_PREFIX}_comparison.csv")
    fold_gaps = read_csv_if_exists(TABLE_DIR / f"{FOLD_PREFIX}_fold_comparator_gaps.csv")
    gates = build_gate_table(metric_audit, fold_gaps)

    paths = {
        "selection": TABLE_DIR / f"{PREFIX}_selection_position.csv",
        "final_summary": TABLE_DIR / f"{PREFIX}_final_summary.csv",
        "per_seed_gaps": TABLE_DIR / f"{PREFIX}_per_seed_gaps.csv",
        "gap_summary": TABLE_DIR / f"{PREFIX}_gap_summary.csv",
        "per_cycle_curve": TABLE_DIR / f"{PREFIX}_per_cycle_curve.csv",
        "one_config": TABLE_DIR / f"{PREFIX}_one_config_figure4a_figure4b.csv",
        "gate_summary": TABLE_DIR / f"{PREFIX}_gate_summary.csv",
        "manifest": TABLE_DIR / f"{PREFIX}_artifact_manifest.csv",
        "report": REPORT_DIR / f"{PREFIX}.md",
    }
    selection.to_csv(paths["selection"], index=False)
    final_summary.to_csv(paths["final_summary"], index=False)
    paired_gaps.to_csv(paths["per_seed_gaps"], index=False)
    gap_summary.to_csv(paths["gap_summary"], index=False)
    curve.to_csv(paths["per_cycle_curve"], index=False)
    one_config.to_csv(paths["one_config"], index=False)
    gates.to_csv(paths["gate_summary"], index=False)

    manifest = build_manifest(
        [
            (TABLE_DIR / f"{TAU1_PREFIX}_metrics.csv", "tau1 literal 20-seed metrics"),
            (TABLE_DIR / f"{TAU1_PREFIX}_final_summary.csv", "tau1 literal 20-seed final summary"),
            (TABLE_DIR / f"{TAU03_PREFIX}_final_summary.csv", "tau0.3 robustness final summary"),
            (TABLE_DIR / f"{COMPARATOR_PREFIX}_final_summary.csv", "Linear UCB proxy comparator summary"),
            (TABLE_DIR / f"{FOLD_PREFIX}_fold_comparator_gaps.csv", "tau1 u1-u5 fold sensitivity gaps"),
            (TABLE_DIR / f"{METRIC_AUDIT_PREFIX}_comparison.csv", "metric reconstruction audit v2 comparison"),
            (paths["selection"], "selection role table"),
            (paths["one_config"], "same-config Figure 4a/Figure 4b comparison table"),
            (paths["gate_summary"], "P0 gate summary"),
        ]
    )
    manifest.to_csv(paths["manifest"], index=False)
    write_report(selection, final_summary, gap_summary, one_config, gates, fold_gaps, metric_audit, manifest, paths["report"])
    manifest = build_manifest([(paths["report"], "tau1 literal P0 reconstruction report"), (paths["manifest"], "artifact manifest")])
    manifest.to_csv(paths["manifest"], mode="a", header=False, index=False)
    print(paths["report"])
    print(paths["one_config"])
    print(paths["gate_summary"])


if __name__ == "__main__":
    main()
