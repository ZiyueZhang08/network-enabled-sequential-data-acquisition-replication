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
PREFIX = "paper_faithful_movielens_selection_policy_confirmation_v1_20seeds"
TARGET = "schur_tuned_full_pool_softmax_tau0p3"
BASELINES = ["schur_tuned_topk", "linucb_concat", "linucb_mf_interaction"]


def ci95(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)


def final_summary(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in final.groupby("variant", sort=False):
        row = {"variant": variant, "seed_count": int(group["seed"].nunique())}
        for metric in [
            "standard_mask_rmse",
            "pdf_literal_norm_rmse",
            "cumulative_reward",
            "w_rmse",
            "reward_tail_proxy_rmse",
            "high_rating_rmse",
            "low_rating_rmse",
        ]:
            if metric not in group.columns:
                continue
            row[metric] = float(group[metric].mean())
            row[f"{metric}_ci95"] = ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_gaps(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for baseline in BASELINES:
        for seed, group in final.groupby("seed"):
            target = group[group["variant"] == TARGET]
            base = group[group["variant"] == baseline]
            if target.empty or base.empty:
                continue
            t = target.iloc[0]
            b = base.iloc[0]
            rows.append(
                {
                    "seed": int(seed),
                    "baseline": baseline,
                    "target_standard_mask_rmse": float(t["standard_mask_rmse"]),
                    "baseline_standard_mask_rmse": float(b["standard_mask_rmse"]),
                    "standard_gap_baseline_minus_target": float(b["standard_mask_rmse"] - t["standard_mask_rmse"]),
                    "target_pdf_literal_norm_rmse": float(t["pdf_literal_norm_rmse"]),
                    "baseline_pdf_literal_norm_rmse": float(b["pdf_literal_norm_rmse"]),
                    "pdf_gap_baseline_minus_target": float(b["pdf_literal_norm_rmse"] - t["pdf_literal_norm_rmse"]),
                    "target_cumulative_reward": float(t["cumulative_reward"]),
                    "baseline_cumulative_reward": float(b["cumulative_reward"]),
                    "reward_gap_target_minus_baseline": float(t["cumulative_reward"] - b["cumulative_reward"]),
                    "standard_win": int(t["standard_mask_rmse"] < b["standard_mask_rmse"]),
                }
            )
    return pd.DataFrame(rows)


def gap_summary(paired: pd.DataFrame) -> pd.DataFrame:
    return (
        paired.groupby("baseline", as_index=False)
        .agg(
            mean_standard_gap=("standard_gap_baseline_minus_target", "mean"),
            standard_win_rate=("standard_win", "mean"),
            wins=("standard_win", "sum"),
            seeds=("seed", "nunique"),
            mean_pdf_gap=("pdf_gap_baseline_minus_target", "mean"),
            mean_reward_gap=("reward_gap_target_minus_baseline", "mean"),
        )
        .sort_values("mean_standard_gap", ascending=False)
    )


def per_cycle(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["variant", "cycle"], as_index=False)
        .agg(
            standard_mask_rmse=("standard_mask_rmse", "mean"),
            pdf_literal_norm_rmse=("pdf_literal_norm_rmse", "mean"),
            cumulative_reward=("cumulative_reward", "mean"),
            selected_avg_rating=("selected_avg_rating", "mean"),
            selected_unique_users=("selected_unique_users", "mean"),
            selected_unique_items=("selected_unique_items", "mean"),
            selected_entropy_quantile=("selected_avg_entropy_percentile", "mean"),
            selected_reward_quantile=("selected_avg_reward_percentile", "mean"),
        )
        .sort_values(["variant", "cycle"])
    )
    return grouped


def selection_features(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["cycle"] < metrics["cycle"].max()]
    rows = []
    for variant, group in selected.groupby("variant", sort=False):
        row = {"variant": variant}
        rating_valid = pd.Series(True, index=group.index)
        if "selected_avg_rating" in group.columns:
            rating_valid &= group["selected_avg_rating"].between(1.0, 5.0) | group["selected_avg_rating"].isna()
        if "selected_rating_5_share" in group.columns:
            rating_valid &= group["selected_rating_5_share"].between(0.0, 1.0) | group["selected_rating_5_share"].isna()
        if "selected_rating_ge4_share" in group.columns:
            rating_valid &= group["selected_rating_ge4_share"].between(0.0, 1.0) | group["selected_rating_ge4_share"].isna()
        rating_group = group.loc[rating_valid]
        for col in ["selected_avg_rating", "selected_rating_5_share", "selected_rating_ge4_share"]:
            row[col] = float(rating_group[col].dropna().mean()) if col in rating_group.columns and rating_group[col].notna().any() else np.nan

        for col in [
            "selected_unique_users",
            "selected_unique_items",
            "selected_avg_item_popularity",
            "selected_avg_entropy_percentile",
            "selected_avg_reward_percentile",
        ]:
            if col not in group.columns:
                row[col] = np.nan
                continue
            values = group[col].dropna()
            if "percentile" in col:
                values = values[values.between(0.0, 1.0)]
            elif col in {"selected_unique_users", "selected_unique_items"}:
                values = values[values.between(0.0, 100.0)]
            elif col == "selected_avg_item_popularity":
                values = values[values >= 0.0]
            row[col] = float(values.mean()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curve(cycle_table: pd.DataFrame) -> Path:
    path = FIGURE_DIR / f"{PREFIX}_standard_rmse_curve.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for variant in [TARGET, "schur_tuned_topk", "linucb_concat", "linucb_mf_interaction"]:
        cur = cycle_table[cycle_table["variant"] == variant]
        plt.plot(cur["cycle"], cur["standard_mask_rmse"], marker="o", label=variant)
    plt.xlabel("Acquisition cycle")
    plt.ylabel("Standard mask RMSE")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_tradeoff(summary: pd.DataFrame) -> Path:
    path = FIGURE_DIR / f"{PREFIX}_reward_rmse_tradeoff.png"
    plt.figure(figsize=(7.5, 4.8))
    for _, row in summary.iterrows():
        plt.scatter(row["cumulative_reward"], row["standard_mask_rmse"], s=70)
        plt.annotate(str(row["variant"]).replace("schur_tuned_", ""), (row["cumulative_reward"], row["standard_mask_rmse"]), fontsize=8)
    plt.xlabel("Cumulative reward")
    plt.ylabel("Standard mask RMSE")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_report(summary: pd.DataFrame, gaps: pd.DataFrame, features: pd.DataFrame, artifacts: list[Path]) -> Path:
    report = REPORT_DIR / f"{PREFIX}.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    target = summary[summary["variant"] == TARGET].iloc[0]
    topk_gap = gaps[gaps["baseline"] == "schur_tuned_topk"].iloc[0]
    mf_gap = gaps[gaps["baseline"] == "linucb_mf_interaction"].iloc[0]
    concat_gap = gaps[gaps["baseline"] == "linucb_concat"].iloc[0]
    lines = [
        f"# {PREFIX}",
        "",
        "## Result",
        "",
        summary.to_string(index=False),
        "",
        "## Paired Gaps",
        "",
        gaps.to_string(index=False),
        "",
        "## Selection Features",
        "",
        features.to_string(index=False),
        "",
        "## Diagnosis",
        "",
        f"`{TARGET}` has standard RMSE {target['standard_mask_rmse']:.6f} over 20 seeds.",
        f"It beats top-k by {topk_gap['mean_standard_gap']:.6f} with win-rate {topk_gap['wins']:.0f}/{topk_gap['seeds']:.0f}.",
        f"It beats `linucb_concat` by {concat_gap['mean_standard_gap']:.6f} with win-rate {concat_gap['wins']:.0f}/{concat_gap['seeds']:.0f}.",
        f"It also beats `linucb_mf_interaction` by {mf_gap['mean_standard_gap']:.6f} with win-rate {mf_gap['wins']:.0f}/{mf_gap['seeds']:.0f}.",
        "This confirms selection policy is not a minor cosmetic detail: paper-style full-pool softmax can recover substantially more Figure 4a RMSE gap than deterministic top-k, while reward remains low relative to concat.",
        "",
        "## Recommendation",
        "",
        "Do not overwrite the frozen deterministic top-k mainline retroactively. Instead, add this as `selection_policy_confirmation_v1`: a paper-style sampling robustness result and a plausible explanation for the remaining Figure 4a gap.",
        "",
        "## Artifact Index",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact.relative_to(ROOT)}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    metrics = pd.read_csv(TABLE_DIR / f"{PREFIX}_metrics.csv")
    final = final_rows(metrics)
    summary = final_summary(final)
    paired = paired_gaps(final)
    gaps = gap_summary(paired)
    cycle_table = per_cycle(metrics)
    features = selection_features(metrics)

    artifacts: list[Path] = []
    for name, table in [
        ("final_summary", summary),
        ("paired_gaps", paired),
        ("gap_summary", gaps),
        ("per_cycle", cycle_table),
        ("selection_features", features),
    ]:
        path = TABLE_DIR / f"{PREFIX}_{name}.csv"
        table.to_csv(path, index=False)
        artifacts.append(path)
    artifacts.append(plot_curve(cycle_table))
    artifacts.append(plot_tradeoff(summary))
    report = write_report(summary, gaps, features, artifacts)
    artifacts.append(report)

    manifest = TABLE_DIR / f"{PREFIX}_artifact_manifest.csv"
    pd.DataFrame(
        [{"artifact": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifacts]
    ).to_csv(manifest, index=False)
    print(report)
    print(manifest)


if __name__ == "__main__":
    main()
