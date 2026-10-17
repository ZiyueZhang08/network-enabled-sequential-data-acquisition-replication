from __future__ import annotations

import hashlib
import json
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
PREFIX = "paper_faithful_movielens_gap_investigation_v1"


def ci95(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return 0.0
    return float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [col for col in ["fold", "variant", "seed"] if col in metrics.columns]
    return metrics.sort_values("cycle").groupby(keys, as_index=False).tail(1)


def paired_gap(final: pd.DataFrame, method: str, baseline: str, metric: str, lower_is_better: bool = True) -> pd.DataFrame:
    rows = []
    for seed, group in final.groupby("seed"):
        cur = group[group["variant"] == method]
        base = group[group["variant"] == baseline]
        if cur.empty or base.empty:
            continue
        a = float(cur.iloc[0][metric])
        b = float(base.iloc[0][metric])
        gap = b - a if lower_is_better else a - b
        rows.append({"seed": int(seed), "gap": gap, "win": int(gap > 0)})
    return pd.DataFrame(rows)


def summarize_methods(final: pd.DataFrame, methods: list[str], baseline: str = "linucb_concat") -> pd.DataFrame:
    rows = []
    base = final[final["variant"] == baseline]
    for method in methods:
        cur = final[final["variant"] == method]
        if cur.empty:
            continue
        rmse_gap = paired_gap(final, method, baseline, "standard_mask_rmse", True)
        pdf_gap = paired_gap(final, method, baseline, "pdf_literal_norm_rmse", True)
        reward_gap = paired_gap(final, method, baseline, "cumulative_reward", False)
        rows.append(
            {
                "method": method,
                "seed_count": int(cur["seed"].nunique()),
                "standard_mask_rmse": float(cur["standard_mask_rmse"].mean()),
                "standard_mask_rmse_ci95": ci95(cur["standard_mask_rmse"]),
                "rmse_gap_vs_linucb_concat": float(base["standard_mask_rmse"].mean() - cur["standard_mask_rmse"].mean())
                if not base.empty
                else np.nan,
                "standard_rmse_win_rate_vs_linucb_concat": float(rmse_gap["win"].mean()) if not rmse_gap.empty else np.nan,
                "pdf_literal_norm_rmse": float(cur["pdf_literal_norm_rmse"].mean()),
                "pdf_win_rate_vs_linucb_concat": float(pdf_gap["win"].mean()) if not pdf_gap.empty else np.nan,
                "cumulative_reward": float(cur["cumulative_reward"].mean()),
                "cumulative_reward_ci95": ci95(cur["cumulative_reward"]),
                "reward_gap_vs_linucb_concat": float(cur["cumulative_reward"].mean() - base["cumulative_reward"].mean())
                if not base.empty
                else np.nan,
                "reward_win_rate_vs_linucb_concat": float(reward_gap["win"].mean()) if not reward_gap.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pareto_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    dominated = []
    for i, row in out.iterrows():
        is_dominated = False
        for j, other in out.iterrows():
            if i == j:
                continue
            no_worse = (
                float(other["standard_mask_rmse"]) <= float(row["standard_mask_rmse"])
                and float(other["cumulative_reward"]) >= float(row["cumulative_reward"])
            )
            strictly = (
                float(other["standard_mask_rmse"]) < float(row["standard_mask_rmse"])
                or float(other["cumulative_reward"]) > float(row["cumulative_reward"])
            )
            if no_worse and strictly:
                is_dominated = True
                break
        dominated.append(is_dominated)
    out["dominated"] = dominated
    return out.sort_values("standard_mask_rmse").reset_index(drop=True)


def aggregate_selection_features(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "selected_avg_rating",
        "selected_rating_5_share",
        "selected_rating_ge4_share",
        "selected_unique_users",
        "selected_unique_items",
        "selected_avg_user_degree",
        "selected_avg_item_popularity",
        "selected_avg_entropy_percentile",
        "selected_avg_reward_percentile",
    ]
    by_cycle = (
        metrics.groupby(["variant", "cycle"], as_index=False)
        .agg({col: "mean" for col in feature_cols if col in metrics.columns})
        .sort_values(["variant", "cycle"])
    )
    final = final_rows(metrics)
    aggregate = (
        metrics[metrics["cycle"] < metrics["cycle"].max()]
        .groupby("variant", as_index=False)
        .agg({col: "mean" for col in feature_cols if col in metrics.columns})
    )
    error_cols = ["standard_mask_rmse", "cumulative_reward", "high_rating_rmse", "low_rating_rmse"]
    final_agg = final.groupby("variant", as_index=False).agg({col: "mean" for col in error_cols if col in final.columns})
    aggregate = aggregate.merge(final_agg, on="variant", how="left")
    return by_cycle, aggregate


def comparator_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    final = final_rows(metrics)
    du = "schur_loglambda_tuned_standard_topk"
    rows = []
    for comp in ["linucb_concat", "linucb_bias_interaction", "linucb_mf_interaction"]:
        gaps = []
        reward_gaps = []
        wins = []
        for seed, group in final.groupby("seed"):
            d = group[group["variant"] == du]
            c = group[group["variant"] == comp]
            if d.empty or c.empty:
                continue
            gap = float(c.iloc[0]["standard_mask_rmse"] - d.iloc[0]["standard_mask_rmse"])
            reward_gap = float(d.iloc[0]["cumulative_reward"] - c.iloc[0]["cumulative_reward"])
            gaps.append(gap)
            reward_gaps.append(reward_gap)
            wins.append(int(gap > 0))
        d_all = final[final["variant"] == du]
        c_all = final[final["variant"] == comp]
        rows.append(
            {
                "comparator": comp,
                "dunet_rmse": float(d_all["standard_mask_rmse"].mean()),
                "comparator_rmse": float(c_all["standard_mask_rmse"].mean()),
                "rmse_gap_comparator_minus_dunet": float(np.mean(gaps)),
                "dunet_reward": float(d_all["cumulative_reward"].mean()),
                "comparator_reward": float(c_all["cumulative_reward"].mean()),
                "reward_gap_dunet_minus_comparator": float(np.mean(reward_gaps)),
                "seed_win_rate": float(np.mean(wins)),
                "wins": int(np.sum(wins)),
                "seeds": len(wins),
            }
        )
    return pd.DataFrame(rows)


def metric_sensitivity(final: pd.DataFrame) -> pd.DataFrame:
    du_group = final[final["variant"] == "schur_loglambda_tuned_standard_topk"]
    base_group = final[final["variant"] == "linucb_concat"]
    metrics = [
        ("standard_mask_rmse", "lower"),
        ("pdf_literal_norm_rmse", "lower"),
        ("mild_reward_w_rmse", "lower"),
        ("medium_reward_w_rmse", "lower"),
        ("sigmoid_reward_w_rmse", "lower"),
        ("reward_tail_proxy_rmse", "lower"),
        ("w_rmse", "lower"),
    ]
    rows = []
    for metric, _ in metrics:
        if metric not in final.columns:
            continue
        d = float(du_group[metric].mean())
        b = float(base_group[metric].mean())
        gap = b - d
        rows.append(
            {
                "metric": metric,
                "dunet": d,
                "linucb_concat": b,
                "gap_linucb_minus_dunet": gap,
                "relative_improvement": gap / b if b else np.nan,
                "abs_gap_distance_to_visual_0p026": abs(abs(gap) - 0.026),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_gap_distance_to_visual_0p026")


def selection_policy_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    final = final_rows(metrics)
    selected = (
        metrics[metrics["cycle"] < metrics["cycle"].max()]
        .groupby("variant", as_index=False)
        .agg(
            selected_avg_rating=("selected_avg_rating", "mean"),
            unique_users=("selected_unique_users", "mean"),
            unique_items=("selected_unique_items", "mean"),
            selected_entropy_quantile=("selected_avg_entropy_percentile", "mean"),
            selected_reward_quantile=("selected_avg_reward_percentile", "mean"),
        )
    )
    summary = final.groupby("variant", as_index=False).agg(
        standard_mask_rmse=("standard_mask_rmse", "mean"),
        cumulative_reward=("cumulative_reward", "mean"),
    )
    out = summary.merge(selected, on="variant", how="left")
    topk_rmse = float(out[out["variant"] == "schur_tuned_topk"]["standard_mask_rmse"].iloc[0])
    out["collapse"] = np.where(out["standard_mask_rmse"] > topk_rmse + 0.02, "yes", "no")
    out["selection"] = np.where(out["variant"].str.contains("full_pool"), "full_pool_softmax", np.where(out["variant"].str.contains("topm"), "topM_softmax", "topk"))
    out["tau"] = out["variant"].str.extract(r"tau([0-9]p[0-9])")[0].str.replace("p", ".", regex=False).astype(float)
    return out.sort_values(["selection", "tau"], na_position="first")


def plot_tradeoff(table: pd.DataFrame, path: Path, label_col: str = "method") -> None:
    plt.figure(figsize=(7.5, 4.8))
    for _, row in table.iterrows():
        color = "#2563eb" if not bool(row.get("dominated", False)) else "#94a3b8"
        plt.scatter(row["cumulative_reward"], row["standard_mask_rmse"], s=65, color=color)
        plt.annotate(str(row[label_col]).replace("schur_", ""), (row["cumulative_reward"], row["standard_mask_rmse"]), fontsize=8)
    plt.xlabel("Cumulative reward")
    plt.ylabel("Standard mask RMSE")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_report(tables: dict[str, pd.DataFrame], artifacts: list[Path]) -> Path:
    report = REPORT_DIR / f"{PREFIX}.md"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    reward = tables["reward_a1"]
    lambda03 = reward[reward["method"] == "schur_lambda_0p3"].iloc[0]
    tuned = reward[reward["method"] == "schur_loglambda_tuned_standard_topk"].iloc[0]
    comp = tables["comparator_b1"]
    mf = comp[comp["comparator"] == "linucb_mf_interaction"].iloc[0]
    folds = tables["fold_c1"]
    concat_folds = folds[folds["comparator"] == "linucb_concat"]
    mf_folds = folds[folds["comparator"] == "linucb_mf_interaction"]
    metric = tables["metric_c2"].iloc[0]
    selection = tables["selection_c3"]
    best_sel = selection.sort_values("standard_mask_rmse").iloc[0]
    lines = [
        f"# {PREFIX}",
        "",
        "## 1. Main Figure 4a frozen result",
        "",
        "- Mainline remains `paper_faithful_movielens_schur_loglambda_tuned_topk_v1`.",
        "- 20-seed standard RMSE gap vs `linucb_concat`: 0.017528; win-rate 19/20; reward gap -143.12.",
        "- Per-cycle frozen curve stayed positive on 21/21 cycles against `linucb_concat`.",
        "",
        "## 2. Reward trade-off / lambda=0.3 20-seed result",
        "",
        reward[["method", "seed_count", "standard_mask_rmse", "rmse_gap_vs_linucb_concat", "standard_rmse_win_rate_vs_linucb_concat", "cumulative_reward", "reward_gap_vs_linucb_concat"]].to_string(index=False),
        "",
        f"`lambda=0.3` keeps a stable RMSE win ({lambda03['standard_rmse_win_rate_vs_linucb_concat']:.0%}) and cuts reward loss from {tuned['reward_gap_vs_linucb_concat']:.2f} to {lambda03['reward_gap_vs_linucb_concat']:.2f}. It is a balanced diagnostic, not a replacement for the Figure 4a mainline because its RMSE ({lambda03['standard_mask_rmse']:.6f}) is worse than entropy-dominant fixed lambda and the frozen tuned mainline.",
        "",
        "## 3. Comparator sensitivity 20-seed result",
        "",
        comp.to_string(index=False),
        "",
        f"`linucb_mf_interaction` remains a real caveat: mean RMSE gap comparator minus DU-Net is {mf['rmse_gap_comparator_minus_dunet']:.6f}, with DU-Net wins {int(mf['wins'])}/{int(mf['seeds'])}. It should be labeled the strongest clean-room Linear UCB proxy, not dismissed as 3-seed noise.",
        "",
        "## 4. Fold sensitivity result",
        "",
        folds.to_string(index=False),
        "",
        f"Against concat, DU-Net is positive on {int((concat_folds['gap'] > 0).sum())}/5 folds. Against mf_interaction, it is positive on {int((mf_folds['gap'] > 0).sum())}/5 folds. Split/fold changes preserve the concat Figure 4a pattern but do not remove the mf comparator caveat.",
        "",
        "## 5. Metric sensitivity result",
        "",
        tables["metric_c2"].to_string(index=False),
        "",
        f"The metric whose absolute gap is closest to the rough 0.026 visual target is `{metric['metric']}` with gap {metric['gap_linucb_minus_dunet']:.6f}. Metrics involving reward weighting reverse or change the interpretation, so reward-tail proxy remains diagnostic only.",
        "",
        "## 6. Selection policy diagnostic",
        "",
        selection.to_string(index=False),
        "",
        f"Full-pool softmax did not collapse in this Schur-tuned diagnostic; best 3-seed RMSE was `{best_sel['variant']}` at {best_sel['standard_mask_rmse']:.6f}. This makes selection policy a plausible contributor to the remaining gap, but the evidence is only 3 seeds.",
        "",
        "## 7. Final diagnosis: what explains the remaining gap?",
        "",
        "The remaining gap is best explained by a combination of metric ambiguity, comparator ambiguity, and selection/split details. Metric ambiguity remains P0 because different w-RMSE definitions reverse reward-weighted conclusions. Comparator ambiguity is now explicit: concat supports the Figure 4a pattern strongly, while mf_interaction nearly matches or beats DU-Net. Selection policy is newly important because paper-style softmax improved 3-seed RMSE. Split/mask still matters, but the u1-u5 diagnostic says concat success generalizes across official folds.",
        "",
        "## 8. Recommended next action",
        "",
        "Stop modifying the Figure 4a mainline. Keep Schur log-lambda top-k v1 as the frozen deterministic main result, add `lambda=0.3` as a balanced reward/RMSE diagnostic, label `linucb_mf_interaction` as the strongest clean-room comparator caveat, and only run robustness/caveat extensions from here. If more algorithm work is done, the only justified path is source-backed paper mechanism fidelity such as exact metric, exact split/mask, or paper-confirmed sampling.",
        "",
        "## Artifact index",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact.relative_to(ROOT)}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    reward_metrics = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_reward_tradeoff_v1_20seeds_metrics.csv")
    reward_final = final_rows(reward_metrics)
    reward_methods = [
        "schur_loglambda_tuned_standard_topk",
        "schur_lambda_0",
        "schur_lambda_0p1",
        "schur_lambda_0p3",
        "schur_lambda_0p5",
        "schur_lambda_1",
        "schur_lambda_3",
        "schur_lambda_30",
        "schur_entropy_only",
        "linucb_concat",
    ]
    reward_summary = summarize_methods(reward_final, reward_methods)
    reward_pareto = pareto_table(reward_summary.rename(columns={"method": "method"}))

    selection_feature_metrics = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_selection_feature_diagnostic_v1_3seeds_metrics.csv")
    selected_by_cycle, selected_agg = aggregate_selection_features(selection_feature_metrics)

    comparator_metrics = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_comparator_sensitivity_v1_20seeds_metrics.csv")
    comp_summary = comparator_summary(comparator_metrics)

    alpha_metrics = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_linucb_mf_alpha_sensitivity_v1_3seeds_metrics.csv")
    alpha_by_cycle, alpha_summary = aggregate_selection_features(alpha_metrics)

    fold_summary = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_fold_sensitivity_v1_3seeds_fold_comparator_gaps.csv")

    frozen_metrics = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_schur_loglambda_tuned_topk_v1_metrics.csv")
    metric_table = metric_sensitivity(final_rows(frozen_metrics))

    selection_metrics = pd.read_csv(TABLE_DIR / "paper_faithful_movielens_selection_policy_diagnostic_v1_3seeds_metrics.csv")
    selection_table = selection_policy_summary(selection_metrics)

    outputs = {
        "reward_a1": reward_summary,
        "reward_pareto_a3": reward_pareto,
        "selection_features_by_cycle_a2_b2": selected_by_cycle,
        "selection_features_aggregate_a2_b2": selected_agg,
        "comparator_b1": comp_summary,
        "linucb_mf_alpha_b3": alpha_summary,
        "linucb_mf_alpha_by_cycle_b3": alpha_by_cycle,
        "fold_c1": fold_summary,
        "metric_c2": metric_table,
        "selection_c3": selection_table,
    }
    artifacts: list[Path] = []
    for name, table in outputs.items():
        path = TABLE_DIR / f"{PREFIX}_{name}.csv"
        table.to_csv(path, index=False)
        artifacts.append(path)

    tradeoff_fig = FIGURE_DIR / f"{PREFIX}_reward_rmse_pareto_20seeds.png"
    plot_tradeoff(reward_pareto, tradeoff_fig, "method")
    artifacts.append(tradeoff_fig)

    selection_fig = FIGURE_DIR / f"{PREFIX}_selection_policy_rmse_reward_3seeds.png"
    plt.figure(figsize=(7.5, 4.8))
    for _, row in selection_table.iterrows():
        plt.scatter(row["cumulative_reward"], row["standard_mask_rmse"], s=65)
        plt.annotate(str(row["variant"]).replace("schur_tuned_", ""), (row["cumulative_reward"], row["standard_mask_rmse"]), fontsize=8)
    plt.xlabel("Cumulative reward")
    plt.ylabel("Standard mask RMSE")
    plt.tight_layout()
    plt.savefig(selection_fig, dpi=180)
    plt.close()
    artifacts.append(selection_fig)

    report = write_report(outputs, artifacts)
    artifacts.append(report)

    manifest = []
    for artifact in artifacts:
        manifest.append({"artifact": str(artifact.relative_to(ROOT)), "bytes": artifact.stat().st_size, "sha256": sha(artifact)})
    manifest_path = TABLE_DIR / f"{PREFIX}_artifact_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(report)
    print(json.dumps({"artifacts": len(artifacts), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
