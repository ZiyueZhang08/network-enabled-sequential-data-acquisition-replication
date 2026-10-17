from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir


POLICY_LABELS = {
    "myopic": "Myopic",
    "du_net": "DU-Net",
    "du_net_no_dls": "No DLS",
    "du_net_shrinkage_dls": "Reliability DLS",
    "du_net_logdet_shrinkage_dls": "Weighted Logdet + Reliability DLS",
    "current_du_net": "DU-Net",
    "no_dls": "No DLS",
    "reliability_dls": "Reliability DLS",
    "weighted_logdet_reliability": "Weighted Logdet + Reliability DLS",
    "netobs_reliability_spectral": "NetObs + Reliability DLS",
}


def _label(identifier: str) -> str:
    return POLICY_LABELS.get(identifier, identifier)


def _group_column(metrics: pd.DataFrame) -> str:
    return "variant" if "variant" in metrics.columns else "policy"


def _cycle_summary(metrics: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    present = [col for col in columns if col in metrics.columns]
    if not present:
        return pd.DataFrame()
    group_col = _group_column(metrics)
    summary = metrics.groupby([group_col, "cycle"], as_index=False)[present].mean(numeric_only=True)
    return summary.rename(columns={group_col: "group"})


def _plot_lines(summary: pd.DataFrame, y_cols: list[str], ylabel: str, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for group_name, group in summary.groupby("group", sort=False):
        for col in y_cols:
            if col not in group.columns:
                continue
            label = f"{_label(group_name)}: {col.replace('_', ' ')}"
            plt.plot(group["cycle"], group[col], linewidth=2, label=label)
    plt.xlabel("Acquisition cycle")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_selected_degree_bins(summary: pd.DataFrame, path: Path) -> None:
    cols = ["selected_low_degree_share", "selected_mid_degree_share", "selected_high_degree_share"]
    present = [col for col in cols if col in summary.columns]
    if not present:
        return
    final = summary[summary["cycle"] < summary["cycle"].max()].groupby("group", as_index=False)[present].mean()
    x = np.arange(len(final))
    bottom = np.zeros(len(final))
    plt.figure(figsize=(8, 5))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    labels = ["Low degree", "Mid degree", "High degree"]
    for col, color, label in zip(present, colors, labels):
        values = final[col].to_numpy(dtype=float)
        plt.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    plt.xticks(x, [_label(group_name) for group_name in final["group"]], rotation=20, ha="right")
    plt.ylabel("Share of selected users")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _write_report(prefix: str, metrics: pd.DataFrame, report_path: Path, figure_paths: list[Path]) -> None:
    group_col = _group_column(metrics)
    final = metrics.sort_values("cycle").groupby([group_col, "seed"], as_index=False).tail(1)
    final_summary = final.groupby(group_col, as_index=False).agg(
        reward=("cumulative_reward", "mean"),
        w_rmse=("w_rmse", "mean"),
        cold_w_rmse=("cold_user_w_rmse", "mean"),
        warm_w_rmse=("warm_user_w_rmse", "mean"),
        low_degree_w_rmse=("low_degree_w_rmse", "mean"),
        mid_degree_w_rmse=("mid_degree_w_rmse", "mean"),
        high_degree_w_rmse=("high_degree_w_rmse", "mean"),
    )
    selection_summary = metrics[metrics["cycle"] < metrics["cycle"].max()].groupby(group_col, as_index=False).agg(
        reward_pct=("selected_avg_reward_percentile", "mean"),
        entropy_pct=("selected_avg_entropy_percentile", "mean"),
        avg_degree=("selected_avg_user_degree", "mean"),
        low_degree_share=("selected_low_degree_share", "mean"),
        mid_degree_share=("selected_mid_degree_share", "mean"),
        high_degree_share=("selected_high_degree_share", "mean"),
        avg_shrinkage=("avg_dls_shrinkage", "mean"),
    )
    merged = final_summary.merge(selection_summary, on=group_col, how="left")
    lines = [
        f"# Mechanism Diagnostics: {prefix}",
        "",
        "## Final and Selection Summary",
        "",
        f"| {group_col} | reward | w-RMSE | cold w-RMSE | warm w-RMSE | low-degree w-RMSE | high-degree w-RMSE | reward pct | entropy pct | selected avg degree |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in merged.sort_values("w_rmse").itertuples(index=False):
        group_name = getattr(row, group_col)
        lines.append(
            f"| {_label(group_name)} | {row.reward:.2f} | {row.w_rmse:.4f} | "
            f"{row.cold_w_rmse:.4f} | {row.warm_w_rmse:.4f} | "
            f"{row.low_degree_w_rmse:.4f} | {row.high_degree_w_rmse:.4f} | "
            f"{row.reward_pct:.4f} | {row.entropy_pct:.4f} | {row.avg_degree:.4f} |"
        )
    lines.extend(["", "## Figures", ""])
    for path in figure_paths:
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use the final w-RMSE column to identify whether a mechanism improves implementation quality, not only whether it changes selection behavior.",
            "- Reward and entropy percentiles show whether a policy remains myopic or meaningfully shifts toward information-oriented acquisition.",
            "- Selected average degree and degree-binned w-RMSE diagnose centrality bias and sparse-graph DLS noise.",
            "- Treat 3-seed pilots as mechanism evidence; promote a setting only after 20-seed confirmation.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    table_dir = ROOT / "outputs" / "tables"
    figure_dir = ensure_dir(ROOT / "outputs" / "figures")
    report_dir = ensure_dir(ROOT / "reports")
    path = table_dir / f"{args.prefix}_metrics.csv"
    metrics = pd.read_csv(path)

    figures: list[Path] = []
    group_summary = _cycle_summary(
        metrics,
        [
            "cold_user_w_rmse",
            "warm_user_w_rmse",
            "low_degree_w_rmse",
            "mid_degree_w_rmse",
            "high_degree_w_rmse",
            "selected_low_degree_share",
            "selected_mid_degree_share",
            "selected_high_degree_share",
            "avg_dls_shrinkage",
            "selected_avg_reward_percentile",
            "selected_avg_entropy_percentile",
        ],
    )

    cold_path = figure_dir / f"{args.prefix}_cold_warm_w_rmse.png"
    _plot_lines(group_summary, ["cold_user_w_rmse", "warm_user_w_rmse"], "w-RMSE", cold_path)
    figures.append(cold_path)

    centrality_path = figure_dir / f"{args.prefix}_centrality_binned_w_rmse.png"
    _plot_lines(group_summary, ["low_degree_w_rmse", "mid_degree_w_rmse", "high_degree_w_rmse"], "w-RMSE", centrality_path)
    figures.append(centrality_path)

    selected_bins_path = figure_dir / f"{args.prefix}_selected_degree_bins.png"
    _plot_selected_degree_bins(group_summary, selected_bins_path)
    figures.append(selected_bins_path)

    shrinkage_path = figure_dir / f"{args.prefix}_dls_shrinkage_over_cycles.png"
    _plot_lines(group_summary, ["avg_dls_shrinkage"], "Average DLS shrinkage", shrinkage_path)
    figures.append(shrinkage_path)

    percentile_path = figure_dir / f"{args.prefix}_reward_entropy_percentiles.png"
    _plot_lines(
        group_summary,
        ["selected_avg_reward_percentile", "selected_avg_entropy_percentile"],
        "Selected candidate percentile",
        percentile_path,
    )
    figures.append(percentile_path)

    summary_path = table_dir / f"{args.prefix}_mechanism_summary.csv"
    group_summary.to_csv(summary_path, index=False)
    _write_report(args.prefix, metrics, report_dir / f"{args.prefix}_mechanism_diagnostics.md", figures)
    print(f"Wrote mechanism summary: {summary_path}")
    for figure in figures:
        print(f"Wrote figure: {figure}")


if __name__ == "__main__":
    main()
