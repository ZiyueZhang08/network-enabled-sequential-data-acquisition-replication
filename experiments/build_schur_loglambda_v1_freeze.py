from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "outputs" / "logs"
CONFIG_DIR = ROOT / "configs"

SOURCE_PREFIX = "paper_faithful_movielens_schur_loglambda_tuning_standard_topk_20seeds"
SWEEP_PREFIX = "paper_faithful_movielens_schur_loglambda_sweep_3seeds"
FROZEN_PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_topk_v1"
DU_VARIANT = "schur_loglambda_tuned_standard_topk"
BASELINE_VARIANT = "linucb_concat"


def _ci95(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_with_frozen_prefix(source: Path) -> Path:
    target = TABLE_DIR / source.name.replace(SOURCE_PREFIX, FROZEN_PREFIX)
    shutil.copyfile(source, target)
    return target


def _write_frozen_config() -> Path:
    source = CONFIG_DIR / f"{SOURCE_PREFIX}.json"
    target = CONFIG_DIR / f"{FROZEN_PREFIX}.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["output_prefix"] = FROZEN_PREFIX
    config["freeze_name"] = FROZEN_PREFIX
    config["source_result_prefix"] = SOURCE_PREFIX
    config["freeze_note"] = (
        "Frozen Figure 4a clean-room mainline: Schur conditional variance "
        "entropy, log-lambda paper-style dynamic tuning, deterministic top-k."
    )
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return target


def _per_seed_gaps(metrics: pd.DataFrame) -> pd.DataFrame:
    final = metrics.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)
    rows: list[dict[str, float | int]] = []
    for seed, group in final.groupby("seed"):
        du = group[group["variant"] == DU_VARIANT]
        base = group[group["variant"] == BASELINE_VARIANT]
        if du.empty or base.empty:
            continue
        du_row = du.iloc[0]
        base_row = base.iloc[0]
        rows.append(
            {
                "seed": int(seed),
                "dunet_standard_mask_rmse": float(du_row["standard_mask_rmse"]),
                "linucb_standard_mask_rmse": float(base_row["standard_mask_rmse"]),
                "standard_mask_rmse_gap_linucb_minus_dunet": float(
                    base_row["standard_mask_rmse"] - du_row["standard_mask_rmse"]
                ),
                "dunet_pdf_literal_norm_rmse": float(du_row["pdf_literal_norm_rmse"]),
                "linucb_pdf_literal_norm_rmse": float(base_row["pdf_literal_norm_rmse"]),
                "pdf_literal_norm_rmse_gap_linucb_minus_dunet": float(
                    base_row["pdf_literal_norm_rmse"] - du_row["pdf_literal_norm_rmse"]
                ),
                "dunet_cumulative_reward": float(du_row["cumulative_reward"]),
                "linucb_cumulative_reward": float(base_row["cumulative_reward"]),
                "cumulative_reward_gap_dunet_minus_linucb": float(
                    du_row["cumulative_reward"] - base_row["cumulative_reward"]
                ),
                "standard_rmse_win": int(du_row["standard_mask_rmse"] < base_row["standard_mask_rmse"]),
                "pdf_literal_rmse_win": int(du_row["pdf_literal_norm_rmse"] < base_row["pdf_literal_norm_rmse"]),
                "reward_win": int(du_row["cumulative_reward"] > base_row["cumulative_reward"]),
            }
        )
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def _per_cycle_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["cycle", "variant"], as_index=False)
        .agg(
            standard_mask_rmse_mean=("standard_mask_rmse", "mean"),
            standard_mask_rmse_ci95=("standard_mask_rmse", _ci95),
            pdf_literal_norm_rmse_mean=("pdf_literal_norm_rmse", "mean"),
            pdf_literal_norm_rmse_ci95=("pdf_literal_norm_rmse", _ci95),
            cumulative_reward_mean=("cumulative_reward", "mean"),
            cumulative_reward_ci95=("cumulative_reward", _ci95),
            selected_avg_reward_percentile_mean=("selected_avg_reward_percentile", "mean"),
            selected_avg_entropy_percentile_mean=("selected_avg_entropy_percentile", "mean"),
        )
        .reset_index(drop=True)
    )
    du = grouped[grouped["variant"] == DU_VARIANT].set_index("cycle")
    base = grouped[grouped["variant"] == BASELINE_VARIANT].set_index("cycle")
    rows: list[dict[str, float | int]] = []
    for cycle in sorted(set(du.index).intersection(base.index)):
        du_row = du.loc[cycle]
        base_row = base.loc[cycle]
        rows.append(
            {
                "cycle": int(cycle),
                "dunet_standard_mask_rmse": float(du_row["standard_mask_rmse_mean"]),
                "dunet_standard_mask_rmse_ci95": float(du_row["standard_mask_rmse_ci95"]),
                "linucb_standard_mask_rmse": float(base_row["standard_mask_rmse_mean"]),
                "linucb_standard_mask_rmse_ci95": float(base_row["standard_mask_rmse_ci95"]),
                "standard_mask_rmse_gap_linucb_minus_dunet": float(
                    base_row["standard_mask_rmse_mean"] - du_row["standard_mask_rmse_mean"]
                ),
                "dunet_pdf_literal_norm_rmse": float(du_row["pdf_literal_norm_rmse_mean"]),
                "linucb_pdf_literal_norm_rmse": float(base_row["pdf_literal_norm_rmse_mean"]),
                "pdf_literal_norm_rmse_gap_linucb_minus_dunet": float(
                    base_row["pdf_literal_norm_rmse_mean"] - du_row["pdf_literal_norm_rmse_mean"]
                ),
                "dunet_cumulative_reward": float(du_row["cumulative_reward_mean"]),
                "linucb_cumulative_reward": float(base_row["cumulative_reward_mean"]),
                "cumulative_reward_gap_dunet_minus_linucb": float(
                    du_row["cumulative_reward_mean"] - base_row["cumulative_reward_mean"]
                ),
                "dunet_selected_reward_percentile": float(du_row["selected_avg_reward_percentile_mean"]),
                "dunet_selected_entropy_percentile": float(du_row["selected_avg_entropy_percentile_mean"]),
            }
        )
    return pd.DataFrame(rows)


def _tuning_logs(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "seed",
        "cycle",
        "sigma",
        "adaptive_lambda_entropy",
        "adaptive_diffusion_prob",
        "standard_mask_rmse",
        "cumulative_reward",
        "selected_avg_reward_percentile",
        "selected_avg_entropy_percentile",
        "selected_avg_reward_score",
        "selected_avg_entropy_score",
    ]
    logs = metrics[metrics["variant"] == DU_VARIANT][cols].copy()
    summary = (
        logs.groupby(["adaptive_lambda_entropy", "sigma"], as_index=False)
        .agg(
            count=("cycle", "size"),
            mean_cycle=("cycle", "mean"),
            mean_standard_mask_rmse=("standard_mask_rmse", "mean"),
            mean_cumulative_reward=("cumulative_reward", "mean"),
            mean_selected_reward_percentile=("selected_avg_reward_percentile", "mean"),
            mean_selected_entropy_percentile=("selected_avg_entropy_percentile", "mean"),
        )
        .sort_values(["count", "adaptive_lambda_entropy", "sigma"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return logs, summary


def _tradeoff_table() -> pd.DataFrame:
    sweep = pd.read_csv(TABLE_DIR / f"{SWEEP_PREFIX}_final_summary.csv")
    table = sweep[
        [
            "variant",
            "policy",
            "seed_count",
            "standard_mask_rmse",
            "standard_mask_rmse_ci95",
            "pdf_literal_norm_rmse",
            "cumulative_reward",
            "cumulative_reward_ci95",
            "w_rmse",
        ]
    ].copy()
    baseline = table[table["variant"] == BASELINE_VARIANT]
    if not baseline.empty:
        base_rmse = float(baseline.iloc[0]["standard_mask_rmse"])
        base_reward = float(baseline.iloc[0]["cumulative_reward"])
        table["standard_mask_rmse_gap_vs_linucb"] = base_rmse - table["standard_mask_rmse"]
        table["cumulative_reward_gap_vs_linucb"] = table["cumulative_reward"] - base_reward
    return table.sort_values(["policy", "standard_mask_rmse"]).reset_index(drop=True)


def _plot_curves(per_cycle: pd.DataFrame, tradeoff: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    curve_path = FIGURE_DIR / f"{FROZEN_PREFIX}_standard_rmse_curve.png"
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(per_cycle["cycle"], per_cycle["dunet_standard_mask_rmse"], marker="o", label="DU-Net Schur tuned")
    plt.plot(per_cycle["cycle"], per_cycle["linucb_standard_mask_rmse"], marker="s", label="Linear UCB concat")
    plt.xlabel("Acquisition cycle")
    plt.ylabel("Standard mask RMSE")
    plt.title("Figure 4a mainline RMSE curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=180)
    plt.close()
    paths.append(curve_path)

    tradeoff_path = FIGURE_DIR / f"{FROZEN_PREFIX}_reward_rmse_tradeoff_fixed_lambda_sweep.png"
    plt.figure(figsize=(7.5, 4.5))
    for _, row in tradeoff.iterrows():
        plt.scatter(row["cumulative_reward"], row["standard_mask_rmse"], s=55)
        plt.annotate(str(row["variant"]).replace("schur_", ""), (row["cumulative_reward"], row["standard_mask_rmse"]), fontsize=8)
    plt.xlabel("Cumulative reward")
    plt.ylabel("Standard mask RMSE")
    plt.title("Reward-RMSE trade-off, fixed log-lambda sweep")
    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=180)
    plt.close()
    paths.append(tradeoff_path)

    return paths


def _write_report(
    copied_metrics: Path,
    copied_summary: Path,
    per_seed: pd.DataFrame,
    per_cycle: pd.DataFrame,
    tuning_summary: pd.DataFrame,
    tradeoff: pd.DataFrame,
    artifacts: list[Path],
) -> Path:
    summary = pd.read_csv(copied_summary)
    du = summary[summary["variant"] == DU_VARIANT].iloc[0]
    base = summary[summary["variant"] == BASELINE_VARIANT].iloc[0]
    standard_gap = float(base["standard_mask_rmse"] - du["standard_mask_rmse"])
    visual_gap_recovery = standard_gap / 0.026
    relative_improvement = standard_gap / float(base["standard_mask_rmse"])
    early = per_cycle[per_cycle["cycle"].isin([0, 1, 2, 3, 4])]
    positive_cycle_share = float((per_cycle["standard_mask_rmse_gap_linucb_minus_dunet"] > 0).mean())

    report = REPORT_DIR / f"{FROZEN_PREFIX}.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {FROZEN_PREFIX}",
        "",
        "Frozen Figure 4a clean-room mainline.",
        "",
        "## Main 20-seed result",
        "",
        f"- DU-Net standard RMSE: {float(du['standard_mask_rmse']):.6f}",
        f"- Linear UCB concat standard RMSE: {float(base['standard_mask_rmse']):.6f}",
        f"- Gap, Linear UCB minus DU-Net: {standard_gap:.6f}",
        f"- Visual paper-gap recovery against 0.026 rough reference: {visual_gap_recovery:.1%}",
        f"- Relative comparator improvement: {relative_improvement:.2%}",
        f"- Standard RMSE win rate: {int(per_seed['standard_rmse_win'].sum())}/{len(per_seed)}",
        f"- PDF-literal RMSE win rate: {int(per_seed['pdf_literal_rmse_win'].sum())}/{len(per_seed)}",
        f"- Reward win rate: {int(per_seed['reward_win'].sum())}/{len(per_seed)}",
        f"- DU-Net cumulative reward: {float(du['cumulative_reward']):.2f}",
        f"- Linear UCB cumulative reward: {float(base['cumulative_reward']):.2f}",
        "",
        "## Per-cycle check",
        "",
        f"- Positive RMSE gap cycles: {int((per_cycle['standard_mask_rmse_gap_linucb_minus_dunet'] > 0).sum())}/{len(per_cycle)} ({positive_cycle_share:.1%})",
        f"- Early-cycle mean gap, cycles 0-4: {float(early['standard_mask_rmse_gap_linucb_minus_dunet'].mean()):.6f}",
        f"- Final-cycle gap: {float(per_cycle.iloc[-1]['standard_mask_rmse_gap_linucb_minus_dunet']):.6f}",
        "",
        "## Tuning signature",
        "",
        "```",
        tuning_summary.head(10).to_string(index=False),
        "```",
        "",
        "## Trade-off signature",
        "",
        "```",
        tradeoff[
            [
                "variant",
                "standard_mask_rmse",
                "cumulative_reward",
                "standard_mask_rmse_gap_vs_linucb",
                "cumulative_reward_gap_vs_linucb",
            ]
        ].to_string(index=False),
        "```",
        "",
        "## Artifact index",
        "",
        f"- Metrics: `{copied_metrics.relative_to(ROOT)}`",
        f"- Final summary: `{copied_summary.relative_to(ROOT)}`",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact.relative_to(ROOT)}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config_path = _write_frozen_config()
    source_metrics = TABLE_DIR / f"{SOURCE_PREFIX}_metrics.csv"
    source_summary = TABLE_DIR / f"{SOURCE_PREFIX}_final_summary.csv"
    copied_metrics = _copy_with_frozen_prefix(source_metrics)
    copied_summary = _copy_with_frozen_prefix(source_summary)

    metrics = pd.read_csv(copied_metrics)
    per_seed = _per_seed_gaps(metrics)
    per_cycle = _per_cycle_curve(metrics)
    tuning_logs, tuning_summary = _tuning_logs(metrics)
    tradeoff = _tradeoff_table()

    artifacts = [
        config_path,
        copied_metrics,
        copied_summary,
        TABLE_DIR / f"{FROZEN_PREFIX}_per_seed_gaps.csv",
        TABLE_DIR / f"{FROZEN_PREFIX}_per_cycle_rmse_curve.csv",
        TABLE_DIR / f"{FROZEN_PREFIX}_lambda_sigma_logs.csv",
        TABLE_DIR / f"{FROZEN_PREFIX}_lambda_sigma_summary.csv",
        TABLE_DIR / f"{FROZEN_PREFIX}_fixed_lambda_reward_rmse_tradeoff.csv",
    ]
    pytest_log = LOG_DIR / f"{FROZEN_PREFIX}_pytest_core.txt"
    if pytest_log.exists():
        artifacts.append(pytest_log)

    per_seed.to_csv(artifacts[3], index=False)
    per_cycle.to_csv(artifacts[4], index=False)
    tuning_logs.to_csv(artifacts[5], index=False)
    tuning_summary.to_csv(artifacts[6], index=False)
    tradeoff.to_csv(artifacts[7], index=False)
    artifacts.extend(_plot_curves(per_cycle, tradeoff))

    report = _write_report(copied_metrics, copied_summary, per_seed, per_cycle, tuning_summary, tradeoff, artifacts)
    artifacts.append(report)

    manifest_rows = []
    for artifact in artifacts:
        if artifact.exists():
            manifest_rows.append(
                {
                    "artifact": str(artifact.relative_to(ROOT)),
                    "bytes": artifact.stat().st_size,
                    "sha256": _file_sha256(artifact),
                }
            )
    manifest = TABLE_DIR / f"{FROZEN_PREFIX}_artifact_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    print(f"Wrote frozen config: {config_path}")
    print(f"Wrote report: {report}")
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
