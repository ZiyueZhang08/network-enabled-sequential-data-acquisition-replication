from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
TABLE_ROOT = ROOT / "outputs" / "tables" / PREFIX
FIGURE_ROOT = ROOT / "outputs" / "figures" / PREFIX
LOG_ROOT = ROOT / "outputs" / "logs" / PREFIX
REPORT_PATH = ROOT / "reports" / f"{PREFIX}.md"
CONFIG_PATH = ROOT / "configs" / f"{PREFIX}.json"
ROW_LEVEL_ROOT = ROOT / "outputs" / "row_level_predictions" / PREFIX

SOURCE_TAU1 = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_20seeds"
SOURCE_TAU03 = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1"
SOURCE_TOPK = "paper_faithful_movielens_schur_loglambda_tuned_topk_v1"
SOURCE_COMPARATORS = "paper_faithful_movielens_comparator_sensitivity_v1_20seeds"
METRIC_AUDIT = "paper_faithful_movielens_tau1_literal_metric_reconstruction_audit_v2"

TAU1 = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
TAU03 = "schur_loglambda_tuned_full_pool_softmax_tau0p3"
TOPK_SOURCE = "schur_loglambda_tuned_standard_topk"
TOPK = "schur_tuned_topk"
CONCAT = "linucb_concat"
MF = "linucb_mf_interaction"
BIAS = "linucb_bias_interaction"

REQUIRED_ROW_LEVEL_METHODS = [
    TAU1,
    CONCAT,
    MF,
    TAU03,
    TOPK,
    "lambda0p3_diagnostic",
]


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


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "tables" / name)


def normalize_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "policy" not in out.columns:
        out["policy"] = ""
    return out


def source_final_summary() -> pd.DataFrame:
    tau1 = normalize_summary(read_table(f"{SOURCE_TAU1}_final_summary.csv"))
    tau03 = normalize_summary(read_table(f"{SOURCE_TAU03}_final_summary.csv"))
    topk = normalize_summary(read_table(f"{SOURCE_TOPK}_final_summary.csv"))
    topk = topk[topk["variant"] == TOPK_SOURCE].copy()
    topk["variant"] = TOPK
    comparators = normalize_summary(read_table(f"{SOURCE_COMPARATORS}_final_summary.csv"))
    comparators = comparators[comparators["variant"].isin([CONCAT, MF, BIAS])].copy()
    summary = pd.concat(
        [tau1, comparators, tau03[tau03["variant"] == TAU03], topk],
        ignore_index=True,
        sort=False,
    )
    roles = {
        TAU1: "mainline",
        CONCAT: "linear_ucb_proxy",
        MF: "linear_ucb_proxy",
        BIAS: "linear_ucb_proxy",
        TAU03: "robustness_diagnostic",
        TOPK: "deterministic_ablation",
    }
    summary.insert(0, "claim_role", summary["variant"].map(roles).fillna("supporting"))
    return summary


def source_metrics() -> pd.DataFrame:
    tau1 = read_table(f"{SOURCE_TAU1}_metrics.csv")
    tau03 = read_table(f"{SOURCE_TAU03}_metrics.csv")
    topk = read_table(f"{SOURCE_TOPK}_metrics.csv")
    topk = topk[topk["variant"] == TOPK_SOURCE].copy()
    topk["variant"] = TOPK
    comparators = read_table(f"{SOURCE_COMPARATORS}_metrics.csv")
    comparators = comparators[comparators["variant"].isin([CONCAT, MF, BIAS])].copy()
    return pd.concat(
        [tau1, comparators, tau03[tau03["variant"] == TAU03], topk],
        ignore_index=True,
        sort=False,
    )


def final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)


def per_seed_gaps(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparator in [CONCAT, MF, BIAS, TAU03, TOPK]:
        for seed, group in final.groupby("seed"):
            tau1 = group[group["variant"] == TAU1]
            base = group[group["variant"] == comparator]
            if tau1.empty or base.empty:
                continue
            t = tau1.iloc[0]
            b = base.iloc[0]
            rows.append(
                {
                    "seed": int(seed),
                    "comparator": comparator,
                    "standard_rmse_gap_comparator_minus_tau1": float(b["standard_mask_rmse"] - t["standard_mask_rmse"]),
                    "pdf_gap_comparator_minus_tau1": float(b["pdf_literal_norm_rmse"] - t["pdf_literal_norm_rmse"]),
                    "reward_gap_tau1_minus_comparator": float(t["cumulative_reward"] - b["cumulative_reward"]),
                    "tau1_standard_mask_rmse": float(t["standard_mask_rmse"]),
                    "comparator_standard_mask_rmse": float(b["standard_mask_rmse"]),
                    "tau1_cumulative_reward": float(t["cumulative_reward"]),
                    "comparator_cumulative_reward": float(b["cumulative_reward"]),
                    "standard_win": int(float(t["standard_mask_rmse"]) < float(b["standard_mask_rmse"])),
                }
            )
    return pd.DataFrame(rows)


def per_cycle_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["variant", "cycle"], as_index=False)
        .agg(
            standard_mask_rmse=("standard_mask_rmse", "mean"),
            standard_mask_rmse_ci95=("standard_mask_rmse", ci95),
            pdf_literal_norm_rmse=("pdf_literal_norm_rmse", "mean"),
            cumulative_reward=("cumulative_reward", "mean"),
            cumulative_reward_ci95=("cumulative_reward", ci95),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["variant", "cycle"])
    )


def one_config_figure_table(summary: pd.DataFrame) -> pd.DataFrame:
    tau1 = summary[summary["variant"] == TAU1].iloc[0]
    concat = summary[summary["variant"] == CONCAT].iloc[0]
    mf = summary[summary["variant"] == MF].iloc[0]
    tau03 = summary[summary["variant"] == TAU03].iloc[0]
    return pd.DataFrame(
        [
            {
                "figure": "Figure 4a",
                "metric": "standard_mask_rmse executable proxy",
                "tau1_du_net": float(tau1["standard_mask_rmse"]),
                "linear_ucb": float(concat["standard_mask_rmse"]),
                "paper_du_net_visual": 1.788,
                "paper_linear_ucb_visual": 1.814,
                "local_gap_linear_minus_tau1": float(concat["standard_mask_rmse"] - tau1["standard_mask_rmse"]),
                "paper_gap_linear_minus_du": 0.026,
                "match": "direction_yes_scale_no",
            },
            {
                "figure": "Figure 4b",
                "metric": "cumulative_reward",
                "tau1_du_net": float(tau1["cumulative_reward"]),
                "linear_ucb": float(concat["cumulative_reward"]),
                "paper_du_net_visual": 1325.0,
                "paper_linear_ucb_visual": 1390.0,
                "local_gap_linear_minus_tau1": float(concat["cumulative_reward"] - tau1["cumulative_reward"]),
                "paper_gap_linear_minus_du": 65.0,
                "match": "ranking_yes_magnitude_partial",
            },
            {
                "figure": "Figure 4b",
                "metric": "cumulative_reward_lambda0p3_diagnostic",
                "tau1_du_net": float(tau03["cumulative_reward"]),
                "linear_ucb": float(concat["cumulative_reward"]),
                "paper_du_net_visual": 1325.0,
                "paper_linear_ucb_visual": 1390.0,
                "local_gap_linear_minus_tau1": float(concat["cumulative_reward"] - tau03["cumulative_reward"]),
                "paper_gap_linear_minus_du": 65.0,
                "match": "diagnostic_only",
            },
            {
                "figure": "Figure 4a",
                "metric": "standard_mask_rmse_vs_mf_interaction",
                "tau1_du_net": float(tau1["standard_mask_rmse"]),
                "linear_ucb": float(mf["standard_mask_rmse"]),
                "paper_du_net_visual": 1.788,
                "paper_linear_ucb_visual": 1.814,
                "local_gap_linear_minus_tau1": float(mf["standard_mask_rmse"] - tau1["standard_mask_rmse"]),
                "paper_gap_linear_minus_du": 0.026,
                "match": "proxy_direction_yes",
            },
        ]
    )


def row_level_status() -> pd.DataFrame:
    ROW_LEVEL_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_dirs = [
        ROW_LEVEL_ROOT,
        ROOT / "outputs" / "row_level_predictions" / "paper_faithful_movielens_tau1_literal_p0_row_level_seed0",
    ]
    rows = []
    for method in REQUIRED_ROW_LEVEL_METHODS:
        files: list[Path] = []
        for directory in candidate_dirs:
            if directory.exists():
                files.extend(sorted(directory.glob(f"{method}__seed_*.csv")))
        seeds = sorted(
            {
                int(path.stem.rsplit("_", 1)[-1])
                for path in files
                if path.stem.rsplit("_", 1)[-1].isdigit()
            }
        )
        rows.append(
            {
                "method": method,
                "available_seed_count": len(seeds),
                "available_seeds": " ".join(str(seed) for seed in seeds),
                "required_seed_count": 20,
                "status": "complete" if len(seeds) >= 20 else "partial" if seeds else "missing",
                "source_dirs": ";".join(str(path.relative_to(ROOT)) for path in candidate_dirs if path.exists()),
            }
        )
    return pd.DataFrame(rows)


def stage_available_row_level_predictions() -> list[Path]:
    """Copy already generated row-level CSVs into the canonical tau1 bundle dir."""
    ROW_LEVEL_ROOT.mkdir(parents=True, exist_ok=True)
    source_dirs = [
        ROOT / "outputs" / "row_level_predictions" / "paper_faithful_movielens_tau1_literal_p0_row_level_seed0",
    ]
    staged: list[Path] = []
    for directory in source_dirs:
        if not directory.exists():
            continue
        for method in REQUIRED_ROW_LEVEL_METHODS:
            for source in sorted(directory.glob(f"{method}__seed_*.csv")):
                destination = ROW_LEVEL_ROOT / source.name
                if not destination.exists() or source.stat().st_mtime > destination.stat().st_mtime:
                    shutil.copy2(source, destination)
                staged.append(destination)
    return staged


def plot_curves(curve: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return paths
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    labels = {
        TAU1: "tau1 DU-Net",
        CONCAT: "Linear UCB concat",
        MF: "Linear UCB MF",
        TAU03: "tau0.3 diagnostic",
        TOPK: "top-k ablation",
    }
    for metric, ylabel, filename in [
        ("standard_mask_rmse", "Standard RMSE", "tau1_standard_rmse_curve.png"),
        ("cumulative_reward", "Cumulative reward", "tau1_cumulative_reward_curve.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for variant, group in curve.groupby("variant"):
            if variant not in labels:
                continue
            ax.plot(group["cycle"], group[metric], marker="o", linewidth=1.6, markersize=3.0, label=labels[variant])
        ax.set_xlabel("Cycle")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIGURE_ROOT / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def manifest(paths: list[tuple[Path, str]]) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for path, description in paths:
        artifact_path = str(path.relative_to(ROOT)).replace("\\", "/")
        if not path.exists():
            rows.append(
                {
                    "artifact_path": artifact_path,
                    "description": description,
                    "status": "missing",
                    "bytes": 0,
                    "sha256": "",
                    "created_at": now,
                }
            )
            continue
        rows.append(
            {
                "artifact_path": artifact_path,
                "description": description,
                "status": "available_directory" if path.is_dir() else "available",
                "bytes": 0 if path.is_dir() else path.stat().st_size,
                "sha256": "" if path.is_dir() else sha256(path),
                "created_at": now,
            }
        )
    return pd.DataFrame(rows)


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


def write_report(
    summary: pd.DataFrame,
    gaps: pd.DataFrame,
    one_config: pd.DataFrame,
    metric_audit: pd.DataFrame,
    row_status: pd.DataFrame,
    manifest_frame: pd.DataFrame,
) -> None:
    tau1 = summary[summary["variant"] == TAU1].iloc[0]
    concat_gap = gaps[gaps["comparator"] == CONCAT]
    mf_gap = gaps[gaps["comparator"] == MF]
    lines = [
        f"# Tau1 Literal Mainline Bundle: {PREFIX}",
        "",
        "## Decision",
        "",
        "`tau1 literal softmax` is the canonical mainline. `tau0.3` is retained only as a robustness/diagnostic branch.",
        "",
        "## Gate Snapshot",
        "",
        f"- tau1 final standard RMSE: {float(tau1['standard_mask_rmse']):.6f}",
        f"- vs linucb_concat wins: {int(concat_gap['standard_win'].sum())}/{concat_gap['seed'].nunique()}",
        f"- vs linucb_mf_interaction wins: {int(mf_gap['standard_win'].sum())}/{mf_gap['seed'].nunique()}",
        "- row-level v3 status: partial until all required 20-seed prediction CSVs are generated.",
        "",
        "## Final Summary",
        "",
        md_table(summary[["claim_role", "variant", "seed_count", "standard_mask_rmse", "pdf_literal_norm_rmse", "cumulative_reward", "w_rmse"]]),
        "",
        "## Same-Config Figure 4a / 4b",
        "",
        md_table(one_config),
        "",
        "## Metric Audit Snapshot",
        "",
        md_table(metric_audit.head(12)),
        "",
        "## Row-Level Completion",
        "",
        md_table(row_status),
        "",
        "## Boundary",
        "",
        "This bundle freezes the strongest current paper-only clean-room mainline. It does not claim exact replication until metric, split/mask, Linear UCB protocol, and ELPH/backend gates are closed.",
        "",
        "## Manifest",
        "",
        md_table(manifest_frame),
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    ROW_LEVEL_ROOT.mkdir(parents=True, exist_ok=True)

    staged_row_level_paths = stage_available_row_level_predictions()
    summary = source_final_summary()
    metrics = source_metrics()
    final = final_rows(metrics)
    gaps = per_seed_gaps(final)
    curve = per_cycle_curve(metrics)
    one_config = one_config_figure_table(summary)
    metric_audit_path = ROOT / "outputs" / "tables" / f"{METRIC_AUDIT}_comparison.csv"
    metric_audit = pd.read_csv(metric_audit_path) if metric_audit_path.exists() else pd.DataFrame()
    row_status = row_level_status()

    final_summary_path = TABLE_ROOT / "tau1_final_summary.csv"
    gaps_path = TABLE_ROOT / "tau1_per_seed_gaps.csv"
    curve_path = TABLE_ROOT / "tau1_per_cycle_curve.csv"
    one_config_path = TABLE_ROOT / "tau1_one_config_figure4a_figure4b.csv"
    metric_audit_out = TABLE_ROOT / "tau1_metric_audit.csv"
    row_status_path = TABLE_ROOT / "tau1_row_level_status.csv"
    manifest_path = TABLE_ROOT / "tau1_manifest.csv"
    log_path = LOG_ROOT / "tau1_bundle_log.txt"

    summary.to_csv(final_summary_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    curve.to_csv(curve_path, index=False)
    one_config.to_csv(one_config_path, index=False)
    metric_audit.to_csv(metric_audit_out, index=False)
    row_status.to_csv(row_status_path, index=False)
    figure_paths = plot_curves(curve)
    log_path.write_text(
        f"Generated {PREFIX} at {datetime.now(timezone.utc).isoformat()}\n"
        "Source metrics are reused from completed 20-seed tau1/comparator runs.\n"
        "Row-level predictions remain partial until all required method/seed CSVs are generated.\n",
        encoding="utf-8",
    )

    manifest_frame = manifest(
        [
            (CONFIG_PATH, "canonical tau1 mainline config"),
            (final_summary_path, "20-seed final metrics"),
            (gaps_path, "per-seed comparator gaps"),
            (curve_path, "21-cycle mean curves"),
            (one_config_path, "same-config Figure 4a/Figure 4b comparison"),
            (metric_audit_out, "row-level metric audit snapshot"),
            (row_status_path, "row-level prediction completion status"),
            (ROW_LEVEL_ROOT, "canonical row-level prediction directory"),
            (log_path, "bundle generation log"),
        ]
        + [(path, f"staged row-level prediction {path.name}") for path in staged_row_level_paths]
        + [(path, f"figure {path.name}") for path in figure_paths]
    )
    manifest_frame.to_csv(manifest_path, index=False)
    write_report(summary, gaps, one_config, metric_audit, row_status, manifest_frame)
    manifest_frame = manifest([(manifest_path, "artifact manifest"), (REPORT_PATH, "mainline bundle report")])
    manifest_frame.to_csv(manifest_path, mode="a", header=False, index=False)
    print(REPORT_PATH)
    print(TABLE_ROOT)
    print(FIGURE_ROOT)
    print(ROW_LEVEL_ROOT)


if __name__ == "__main__":
    main()
