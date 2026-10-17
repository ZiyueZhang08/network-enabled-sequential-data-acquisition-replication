from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "outputs" / "logs"
CONFIG_DIR = ROOT / "configs"

PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1"
SOFTMAX = "schur_loglambda_tuned_full_pool_softmax_tau0p3"
TOPK = "schur_tuned_topk"
TOPK_SOURCE = "schur_loglambda_tuned_standard_topk"
CONCAT = "linucb_concat"
MF = "linucb_mf_interaction"


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


def final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)


def load_combined_metrics() -> pd.DataFrame:
    soft = pd.read_csv(TABLE_DIR / f"{PREFIX}_metrics.csv")
    topk = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_schur_loglambda_tuned_topk_v1_metrics.csv")
    topk = topk[topk["variant"] == TOPK_SOURCE].copy()
    topk["variant"] = TOPK
    comp = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_comparator_sensitivity_v1_20seeds_metrics.csv")
    comp = comp[comp["variant"].isin([CONCAT, MF])].copy()
    return pd.concat([soft, topk, comp], ignore_index=True, sort=False)


def final_summary(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in final.groupby("variant", sort=False):
        row = {"variant": variant, "seed_count": int(group["seed"].nunique())}
        for metric in [
            "standard_mask_rmse",
            "pdf_literal_norm_rmse",
            "mild_reward_w_rmse",
            "medium_reward_w_rmse",
            "sigmoid_reward_w_rmse",
            "w_rmse",
            "reward_tail_proxy_rmse",
            "cumulative_reward",
            "centered_cumulative_reward",
        ]:
            if metric not in group.columns:
                continue
            row[metric] = float(group[metric].mean())
            row[f"{metric}_ci95"] = ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gaps(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline in [TOPK, CONCAT, MF]:
        for seed, group in final.groupby("seed"):
            soft = group[group["variant"] == SOFTMAX]
            base = group[group["variant"] == baseline]
            if soft.empty or base.empty:
                continue
            s = soft.iloc[0]
            b = base.iloc[0]
            rows.append(
                {
                    "seed": int(seed),
                    "baseline": baseline,
                    "softmax_standard_mask_rmse": float(s["standard_mask_rmse"]),
                    "baseline_standard_mask_rmse": float(b["standard_mask_rmse"]),
                    "standard_gap_baseline_minus_softmax": float(b["standard_mask_rmse"] - s["standard_mask_rmse"]),
                    "softmax_pdf_literal_norm_rmse": float(s["pdf_literal_norm_rmse"]),
                    "baseline_pdf_literal_norm_rmse": float(b["pdf_literal_norm_rmse"]),
                    "pdf_gap_baseline_minus_softmax": float(b["pdf_literal_norm_rmse"] - s["pdf_literal_norm_rmse"]),
                    "softmax_cumulative_reward": float(s["cumulative_reward"]),
                    "baseline_cumulative_reward": float(b["cumulative_reward"]),
                    "reward_gap_softmax_minus_baseline": float(s["cumulative_reward"] - b["cumulative_reward"]),
                    "standard_win": int(float(s["standard_mask_rmse"]) < float(b["standard_mask_rmse"])),
                }
            )
    return pd.DataFrame(rows)


def gap_summary(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby("baseline", as_index=False)
        .agg(
            mean_standard_gap=("standard_gap_baseline_minus_softmax", "mean"),
            standard_win_rate=("standard_win", "mean"),
            wins=("standard_win", "sum"),
            seeds=("seed", "nunique"),
            mean_pdf_gap=("pdf_gap_baseline_minus_softmax", "mean"),
            mean_reward_gap=("reward_gap_softmax_minus_baseline", "mean"),
        )
        .sort_values("mean_standard_gap", ascending=False)
        .reset_index(drop=True)
    )


def per_cycle(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = (
        metrics.groupby(["variant", "cycle"], as_index=False)
        .agg(
            standard_mask_rmse=("standard_mask_rmse", "mean"),
            pdf_literal_norm_rmse=("pdf_literal_norm_rmse", "mean"),
            cumulative_reward=("cumulative_reward", "mean"),
            selected_avg_rating=("selected_avg_rating", "mean"),
            selected_rating_5_share=("selected_rating_5_share", "mean"),
            selected_rating_ge4_share=("selected_rating_ge4_share", "mean"),
            selected_unique_users=("selected_unique_users", "mean"),
            selected_unique_items=("selected_unique_items", "mean"),
            selected_avg_item_popularity=("selected_avg_item_popularity", "mean"),
            selected_entropy_quantile=("selected_avg_entropy_percentile", "mean"),
            selected_reward_quantile=("selected_avg_reward_percentile", "mean"),
        )
        .sort_values(["variant", "cycle"])
    )
    soft = grouped[grouped["variant"] == SOFTMAX].set_index("cycle")
    rows = []
    for baseline in [TOPK, CONCAT, MF]:
        base = grouped[grouped["variant"] == baseline].set_index("cycle")
        for cycle in sorted(set(soft.index).intersection(base.index)):
            rows.append(
                {
                    "cycle": int(cycle),
                    "baseline": baseline,
                    "softmax_standard_mask_rmse": float(soft.loc[cycle, "standard_mask_rmse"]),
                    "baseline_standard_mask_rmse": float(base.loc[cycle, "standard_mask_rmse"]),
                    "standard_gap_baseline_minus_softmax": float(
                        base.loc[cycle, "standard_mask_rmse"] - soft.loc[cycle, "standard_mask_rmse"]
                    ),
                    "softmax_pdf_literal_norm_rmse": float(soft.loc[cycle, "pdf_literal_norm_rmse"]),
                    "baseline_pdf_literal_norm_rmse": float(base.loc[cycle, "pdf_literal_norm_rmse"]),
                    "pdf_gap_baseline_minus_softmax": float(
                        base.loc[cycle, "pdf_literal_norm_rmse"] - soft.loc[cycle, "pdf_literal_norm_rmse"]
                    ),
                    "softmax_cumulative_reward": float(soft.loc[cycle, "cumulative_reward"]),
                    "baseline_cumulative_reward": float(base.loc[cycle, "cumulative_reward"]),
                    "reward_gap_softmax_minus_baseline": float(
                        soft.loc[cycle, "cumulative_reward"] - base.loc[cycle, "cumulative_reward"]
                    ),
                }
            )
    return grouped.reset_index(drop=True), pd.DataFrame(rows)


def metric_sensitivity(final: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "standard_mask_rmse",
        "pdf_literal_norm_rmse",
        "mild_reward_w_rmse",
        "medium_reward_w_rmse",
        "sigmoid_reward_w_rmse",
        "w_rmse",
        "reward_tail_proxy_rmse",
    ]
    soft = final[final["variant"] == SOFTMAX]
    rows = []
    for metric in metrics:
        if metric not in final.columns:
            continue
        soft_value = float(soft[metric].mean())
        row = {"metric": metric, "softmax": soft_value}
        for baseline in [TOPK, CONCAT, MF]:
            base = final[final["variant"] == baseline]
            base_value = float(base[metric].mean())
            row[baseline] = base_value
            row[f"gap_{baseline}_minus_softmax"] = base_value - soft_value
            row[f"relative_improvement_vs_{baseline}"] = (base_value - soft_value) / base_value if base_value else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def selection_logs(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "seed",
        "cycle",
        "sigma",
        "adaptive_lambda_entropy",
        "adaptive_diffusion_prob",
        "standard_mask_rmse",
        "pdf_literal_norm_rmse",
        "cumulative_reward",
        "selected_avg_rating",
        "selected_rating_5_share",
        "selected_rating_ge4_share",
        "selected_unique_users",
        "selected_unique_items",
        "selected_avg_item_popularity",
        "selected_avg_reward_score",
        "selected_avg_entropy_score",
        "selected_avg_reward_percentile",
        "selected_avg_entropy_percentile",
    ]
    available = [col for col in cols if col in metrics.columns]
    return metrics[metrics["variant"] == SOFTMAX][available].copy()


def plot_curves(cycle_table: pd.DataFrame) -> list[Path]:
    paths = []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rmse_path = FIGURE_DIR / f"{PREFIX}_standard_rmse_curve.png"
    plt.figure(figsize=(8.5, 5))
    for variant in [SOFTMAX, TOPK, CONCAT, MF]:
        cur = cycle_table[cycle_table["variant"] == variant]
        plt.plot(cur["cycle"], cur["standard_mask_rmse"], marker="o", label=variant)
    plt.xlabel("Acquisition cycle")
    plt.ylabel("Standard mask RMSE")
    plt.tight_layout()
    plt.savefig(rmse_path, dpi=180)
    plt.close()
    paths.append(rmse_path)

    reward_path = FIGURE_DIR / f"{PREFIX}_reward_rmse_tradeoff.png"
    final = cycle_table.sort_values("cycle").groupby("variant", as_index=False).tail(1)
    plt.figure(figsize=(7.5, 4.8))
    for _, row in final.iterrows():
        plt.scatter(row["cumulative_reward"], row["standard_mask_rmse"], s=70)
        plt.annotate(str(row["variant"]), (row["cumulative_reward"], row["standard_mask_rmse"]), fontsize=8)
    plt.xlabel("Cumulative reward")
    plt.ylabel("Standard mask RMSE")
    plt.tight_layout()
    plt.savefig(reward_path, dpi=180)
    plt.close()
    paths.append(reward_path)
    return paths


def write_report(
    summary: pd.DataFrame,
    gaps: pd.DataFrame,
    per_cycle_gap: pd.DataFrame,
    metric_table: pd.DataFrame,
    artifacts: list[Path],
) -> Path:
    report = REPORT_DIR / f"{PREFIX}.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    soft = summary[summary["variant"] == SOFTMAX].iloc[0]
    topk_gap = gaps[gaps["baseline"] == TOPK].iloc[0]
    concat_gap = gaps[gaps["baseline"] == CONCAT].iloc[0]
    mf_gap = gaps[gaps["baseline"] == MF].iloc[0]
    cycle_summary = (
        per_cycle_gap.assign(positive=per_cycle_gap["standard_gap_baseline_minus_softmax"] > 0)
        .groupby("baseline", as_index=False)
        .agg(
            positive_cycles=("positive", "sum"),
            cycles=("cycle", "nunique"),
            mean_gap=("standard_gap_baseline_minus_softmax", "mean"),
            final_gap=("standard_gap_baseline_minus_softmax", "last"),
        )
    )
    lines = [
        f"# {PREFIX}",
        "",
        "Frozen stochastic main candidate for MovieLens Figure 4a clean-room reproduction.",
        "",
        "## Config",
        "",
        f"- selection: full-pool softmax (`selection='exp'`)",
        "- tau: 0.3",
        "- entropy: Schur conditional variance",
        "- dynamic tuning: log-lambda grid, standard RMSE inner validation",
        "- seeds: 0-19",
        "",
        "## Final Result",
        "",
        summary.to_string(index=False),
        "",
        "## Paired 20-Seed Gaps",
        "",
        gaps.to_string(index=False),
        "",
        f"Softmax standard RMSE is {soft['standard_mask_rmse']:.6f}. It beats top-k by {topk_gap['mean_standard_gap']:.6f} ({topk_gap['wins']:.0f}/{topk_gap['seeds']:.0f}), concat by {concat_gap['mean_standard_gap']:.6f} ({concat_gap['wins']:.0f}/{concat_gap['seeds']:.0f}), and mf_interaction by {mf_gap['mean_standard_gap']:.6f} ({mf_gap['wins']:.0f}/{mf_gap['seeds']:.0f}).",
        "",
        "## Per-Cycle Gap Summary",
        "",
        cycle_summary.to_string(index=False),
        "",
        "## Metric Sensitivity",
        "",
        metric_table.to_string(index=False),
        "",
        "The stochastic softmax result is strongest on standard/PDF-literal RMSE. Reward-weighted metrics still favor high-reward Linear UCB variants, so they remain diagnostic rather than the Figure 4a main claim.",
        "",
        "## Positioning",
        "",
        "Keep deterministic top-k v1 as the frozen deterministic approximation. Promote this softmax tau=0.3 run as the stochastic paper-style main candidate because it uses the same Schur/log-lambda core and only switches batch selection to the paper-style sampling family.",
        "",
        "Keep lambda=0.3 as the balanced reward/RMSE diagnostic: entropy-dominant softmax best matches Figure 4a, while lambda=0.3 explains the reward trade-off.",
        "",
        "## Artifact Index",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact.relative_to(ROOT)}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    metrics = load_combined_metrics()
    final = final_rows(metrics)
    summary = final_summary(final)
    paired = paired_gaps(final)
    gaps = gap_summary(paired)
    cycle_table, cycle_gap = per_cycle(metrics)
    metric_table = metric_sensitivity(final)
    logs = selection_logs(metrics)

    artifacts: list[Path] = [
        CONFIG_DIR / f"{PREFIX}.json",
        TABLE_DIR / f"{PREFIX}_metrics.csv",
        TABLE_DIR / f"{PREFIX}_final_summary.csv",
        TABLE_DIR / f"{PREFIX}_paired_gaps.csv",
        TABLE_DIR / f"{PREFIX}_gap_summary.csv",
        TABLE_DIR / f"{PREFIX}_per_cycle.csv",
        TABLE_DIR / f"{PREFIX}_per_cycle_gaps.csv",
        TABLE_DIR / f"{PREFIX}_metric_sensitivity.csv",
        TABLE_DIR / f"{PREFIX}_selection_logs.csv",
        LOG_DIR / f"{PREFIX}_pytest_core.txt",
    ]
    summary.to_csv(artifacts[2], index=False)
    paired.to_csv(artifacts[3], index=False)
    gaps.to_csv(artifacts[4], index=False)
    cycle_table.to_csv(artifacts[5], index=False)
    cycle_gap.to_csv(artifacts[6], index=False)
    metric_table.to_csv(artifacts[7], index=False)
    logs.to_csv(artifacts[8], index=False)
    artifacts.extend(plot_curves(cycle_table))
    report = write_report(summary, gaps, cycle_gap, metric_table, artifacts)
    artifacts.append(report)

    manifest = TABLE_DIR / f"{PREFIX}_artifact_manifest.csv"
    rows = []
    for artifact in artifacts:
        if artifact.exists():
            rows.append({"artifact": str(artifact.relative_to(ROOT)), "bytes": artifact.stat().st_size, "sha256": sha256(artifact)})
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(report)
    print(manifest)


if __name__ == "__main__":
    main()
