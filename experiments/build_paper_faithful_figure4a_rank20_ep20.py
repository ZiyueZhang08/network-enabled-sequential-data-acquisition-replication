"""Build Figure 4a-style artifacts for the paper-faithful rank20/ep20 run."""

from __future__ import annotations

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

PREFIX = "paper_faithful_movielens_rank20_ep20_v4_20seeds"
METRICS_PATH = TABLE_DIR / f"{PREFIX}_metrics.csv"
SUMMARY_PATH = TABLE_DIR / f"{PREFIX}_final_summary.csv"
GAPS_PATH = TABLE_DIR / f"{PREFIX}_gaps.csv"

OUT_TRAJECTORY = TABLE_DIR / "paper_faithful_figure4a_rank20_ep20_trajectory.csv"
OUT_SENSITIVITY = TABLE_DIR / "paper_faithful_figure4a_rank20_ep20_metric_sensitivity.csv"
OUT_RMSE_FIG = FIGURE_DIR / "paper_faithful_figure4a_rank20_ep20_standard_rmse.png"
OUT_REWARD_FIG = FIGURE_DIR / "paper_faithful_figure4a_rank20_ep20_cumulative_reward.png"
OUT_SENSITIVITY_FIG = FIGURE_DIR / "paper_faithful_figure4a_rank20_ep20_metric_sensitivity.png"
OUT_REPORT = REPORT_DIR / "paper_faithful_figure4a_rank20_ep20_report.md"
EXP_PROBE_SUMMARY = TABLE_DIR / "paper_faithful_movielens_rank20_ep20_exp_comparator_20seeds_final_summary.csv"
EXP_PROBE_GAPS = TABLE_DIR / "paper_faithful_movielens_rank20_ep20_exp_comparator_20seeds_gaps.csv"
FALLBACK_EXP_PROBE_SUMMARY = TABLE_DIR / "paper_faithful_movielens_rank20_ep20_exp_comparator_3seeds_final_summary.csv"
FALLBACK_EXP_PROBE_GAPS = TABLE_DIR / "paper_faithful_movielens_rank20_ep20_exp_comparator_3seeds_gaps.csv"

LABELS = {
    "paper_faithful_dunet": "DU-Net",
    "linucb_concat": "Linear UCB concat",
    "linucb_mf_interaction": "Linear UCB MF",
}
COLORS = {
    "paper_faithful_dunet": "#2563eb",
    "linucb_concat": "#dc2626",
    "linucb_mf_interaction": "#059669",
}


def ci95(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def summarize_trajectory(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    metric_cols = [
        "standard_mask_rmse",
        "pdf_literal_norm_rmse",
        "cumulative_reward",
        "reward_tail_proxy_rmse",
        "w_rmse",
    ]
    for (variant, cycle), group in metrics.groupby(["variant", "cycle"], sort=False):
        row: dict[str, float | int | str] = {
            "variant": variant,
            "method": LABELS.get(str(variant), str(variant)),
            "cycle": int(cycle),
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[metric] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_ci95"] = ci95(values)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("standard_mask_rmse", "Standard RMSE", "lower"),
        ("pdf_literal_norm_rmse", "PDF-literal norm RMSE", "lower"),
        ("reward_tail_proxy_rmse", "Reward-tail proxy RMSE", "lower"),
        ("w_rmse", "Local weighted RMSE", "lower"),
        ("cumulative_reward", "Cumulative reward", "higher"),
    ]
    rows: list[dict[str, float | str]] = []
    du = summary[summary["variant"] == "paper_faithful_dunet"].iloc[0]
    for comparator in ["linucb_concat", "linucb_mf_interaction"]:
        comp = summary[summary["variant"] == comparator].iloc[0]
        for metric, label, direction in metrics:
            du_value = float(du[metric])
            comp_value = float(comp[metric])
            gap = du_value - comp_value
            favors_du = gap < 0.0 if direction == "lower" else gap > 0.0
            rows.append(
                {
                    "comparator": comparator,
                    "comparator_label": LABELS[comparator],
                    "metric": metric,
                    "metric_label": label,
                    "direction": direction,
                    "du_value": du_value,
                    "comparator_value": comp_value,
                    "du_minus_comparator": gap,
                    "favors_du": bool(favors_du),
                }
            )
    return pd.DataFrame(rows)


def plot_trajectory(trajectory: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for variant in LABELS:
        frame = trajectory[trajectory["variant"] == variant].sort_values("cycle")
        x = frame["cycle"].to_numpy(dtype=float)
        y = frame[metric].to_numpy(dtype=float)
        ci = frame[f"{metric}_ci95"].to_numpy(dtype=float)
        ax.plot(x, y, label=LABELS[variant], color=COLORS[variant], linewidth=2.2)
        ax.fill_between(x, y - ci, y + ci, color=COLORS[variant], alpha=0.12, linewidth=0)
    ax.set_xlabel("Acquisition cycle")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 20)
    ax.grid(True, axis="y", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_sensitivity(sensitivity: pd.DataFrame, path: Path) -> None:
    display = sensitivity[sensitivity["comparator"] == "linucb_mf_interaction"].copy()
    order = [
        "standard_mask_rmse",
        "pdf_literal_norm_rmse",
        "reward_tail_proxy_rmse",
        "w_rmse",
        "cumulative_reward",
    ]
    display["metric"] = pd.Categorical(display["metric"], categories=order, ordered=True)
    display = display.sort_values("metric")
    colors = ["#2563eb" if value else "#9ca3af" for value in display["favors_du"]]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y_pos = np.arange(len(display))
    ax.barh(y_pos, display["du_minus_comparator"].to_numpy(dtype=float), color=colors)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display["metric_label"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel("DU-Net minus Linear UCB MF")
    ax.set_title("Metric sensitivity of the Figure 4a-style conclusion")
    ax.grid(True, axis="x", alpha=0.24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, gaps: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    display = summary[
        [
            "variant",
            "standard_mask_rmse",
            "pdf_literal_norm_rmse",
            "cumulative_reward",
            "reward_tail_proxy_rmse",
            "w_rmse",
        ]
    ].copy()
    display["method"] = display["variant"].map(LABELS)
    display = display[
        [
            "method",
            "standard_mask_rmse",
            "pdf_literal_norm_rmse",
            "cumulative_reward",
            "reward_tail_proxy_rmse",
            "w_rmse",
        ]
    ]
    gap_display = gaps.copy()
    gap_display["comparator"] = gap_display["comparator"].map(LABELS)
    gap_display = gap_display[
        [
            "comparator",
            "du_minus_comp_standard_mask_rmse",
            "du_minus_comp_pdf_literal_norm_rmse",
            "du_minus_comp_cumulative_reward",
            "du_minus_comp_reward_tail_proxy_rmse",
            "du_minus_comp_w_rmse",
        ]
    ]
    sensitivity_display = sensitivity[
        [
            "comparator_label",
            "metric_label",
            "du_value",
            "comparator_value",
            "du_minus_comparator",
            "favors_du",
        ]
    ]
    du = summary[summary["variant"] == "paper_faithful_dunet"].iloc[0]
    linear = summary[summary["variant"] == "linucb_mf_interaction"].iloc[0]
    standard_gap = float(du["standard_mask_rmse"] - linear["standard_mask_rmse"])
    exp_probe_section = ""
    exp_summary_path = EXP_PROBE_SUMMARY if EXP_PROBE_SUMMARY.exists() else FALLBACK_EXP_PROBE_SUMMARY
    exp_gaps_path = EXP_PROBE_GAPS if EXP_PROBE_GAPS.exists() else FALLBACK_EXP_PROBE_GAPS
    if exp_summary_path.exists() and exp_gaps_path.exists():
        exp_summary = pd.read_csv(exp_summary_path)
        exp_gaps = pd.read_csv(exp_gaps_path)
        exp_seed_count = int(exp_summary["seed_count"].max())
        exp_display = exp_summary[
            [
                "variant",
                "standard_mask_rmse",
                "pdf_literal_norm_rmse",
                "cumulative_reward",
                "reward_tail_proxy_rmse",
                "w_rmse",
            ]
        ].copy()
        exp_display["method"] = exp_display["variant"].map(LABELS)
        exp_display = exp_display[
            [
                "method",
                "standard_mask_rmse",
                "pdf_literal_norm_rmse",
                "cumulative_reward",
                "reward_tail_proxy_rmse",
                "w_rmse",
            ]
        ]
        exp_gap_display = exp_gaps.copy()
        exp_gap_display["comparator"] = exp_gap_display["comparator"].map(LABELS)
        exp_gap_display = exp_gap_display[
            [
                "comparator",
                "du_minus_comp_standard_mask_rmse",
                "du_minus_comp_pdf_literal_norm_rmse",
                "du_minus_comp_cumulative_reward",
                "du_minus_comp_reward_tail_proxy_rmse",
                "du_minus_comp_w_rmse",
            ]
        ]
        exp_probe_section = f"""
## Comparator Sampling Sensitivity

The paper states that DU-Net and two policies use exponential weighting for
batch selection, but the PDF text does not identify those two policies clearly
enough to prove that Linear UCB must also use exponential sampling. Because
this is a plausible fidelity concern, I ran a {exp_seed_count}-seed probe that forces
`selection=exp` for DU-Net and both Linear UCB comparators under the same
rank20/ep20 backend.

Forced-exp {exp_seed_count}-seed summary:

{markdown_table(exp_display, list(exp_display.columns), digits=6)}

Forced-exp DU-Net minus comparator gaps:

{markdown_table(exp_gap_display, list(exp_gap_display.columns), digits=6)}

This weakens the strongest conclusion: under forced exponential sampling,
DU-Net is slightly better than Linear UCB MF on standard RMSE, but it is worse
than Linear UCB concat on standard RMSE. I therefore do not promote the
top-k-comparator result as an exact Figure 4a reproduction. The safest claim is
metric- and comparator-protocol-specific: the rank20/ep20 mainline is
directionally consistent with Linear UCB MF and with top-k Linear UCB concat,
but not with forced-exp Linear UCB concat.
"""
    report = f"""# Paper-faithful Figure 4a-style MovieLens result

Generated by `experiments/build_paper_faithful_figure4a_rank20_ep20.py`.

## Main Result

This report uses the current paper-faithful mainline:

- SVD++ rank 20, 20 epochs, learning rate 0.015, regularization 0.06.
- MovieLens auxiliary user features as `u_sn`.
- Full-user Gaussian DLS for Eq. 3.3.
- Schur-style observational entropy for Eq. 3.6.
- MovieLens `p=0`, dynamic tuning over `lambda_H` and `sigma`.
- Full-pool exponential sampling for DU-Net.

Final 20-seed summary:

{markdown_table(display, list(display.columns), digits=6)}

DU-Net minus comparator gaps:

{markdown_table(gap_display, list(gap_display.columns), digits=6)}

Negative RMSE gaps favor DU-Net. On the main standard-RMSE proxy, DU-Net beats
the MF-interaction Linear UCB comparator by {standard_gap:.6f}. This is the
cleanest current evidence that the reproduction is directionally consistent
with Figure 4a's DU-Net-vs-Linear-UCB ordering.

## Metric Sensitivity

{markdown_table(sensitivity_display, list(sensitivity_display.columns), digits=6)}

The conclusion is metric-sensitive. DU-Net is better than Linear UCB MF on
standard RMSE, PDF-literal RMSE, reward-tail proxy, and local weighted RMSE in
this rank20/ep20 confirmation. Against Linear UCB concat, DU-Net wins RMSE
metrics but gives up cumulative reward and local weighted RMSE.

{exp_probe_section}

## Figures

- `{OUT_RMSE_FIG.relative_to(ROOT)}`
- `{OUT_REWARD_FIG.relative_to(ROOT)}`
- `{OUT_SENSITIVITY_FIG.relative_to(ROOT)}`

## Reflection

This step improves the paper-facing claim more than it improves DU-Net's
absolute error. The main gain is consistency: under a shared, stronger SVD++
backend, DU-Net is now lower error than the Linear UCB comparators on the
RMSE-style metrics. The remaining gap to the paper's visual Figure 4a scale
cannot be closed responsibly without exact metric/split/backend details from
the authors or supplementary material.

## Next Step

Do not open another broad local hyperparameter sweep. The next faithful action
should be source-backed, not another local proxy search:

1. If source-backed details arrive, implement the exact Figure 4a metric, split,
   backend, or Linear UCB adapter and run the minimal 3-seed diagnostic first.
2. Without new source-backed details, keep the current rank20/ep20 result as a
   strong partial reproduction: it is consistent with Linear UCB MF and top-k
   Linear UCB concat, but the completed 20-seed forced-exp check shows it is not
   robust to forcing exponential sampling on Linear UCB concat.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(METRICS_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    gaps = pd.read_csv(GAPS_PATH)
    trajectory = summarize_trajectory(metrics)
    sensitivity = metric_sensitivity(summary)
    trajectory.to_csv(OUT_TRAJECTORY, index=False)
    sensitivity.to_csv(OUT_SENSITIVITY, index=False)
    plot_trajectory(trajectory, "standard_mask_rmse", "Standard RMSE", OUT_RMSE_FIG)
    plot_trajectory(trajectory, "cumulative_reward", "Cumulative reward", OUT_REWARD_FIG)
    plot_metric_sensitivity(sensitivity, OUT_SENSITIVITY_FIG)
    write_report(summary, gaps, sensitivity)
    print(f"Wrote {OUT_TRAJECTORY}")
    print(f"Wrote {OUT_SENSITIVITY}")
    print(f"Wrote {OUT_RMSE_FIG}")
    print(f"Wrote {OUT_REWARD_FIG}")
    print(f"Wrote {OUT_SENSITIVITY_FIG}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
