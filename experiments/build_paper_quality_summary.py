"""Build a frozen paper-quality summary from confirmed reproduction runs.

This script intentionally reads only final summary CSVs. It does not rerun or
overwrite experiments, so the current best evidence remains stable.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class ComparisonSpec:
    dataset: str
    setting: str
    source_table: str
    candidate: str
    comparator: str
    candidate_filter: dict[str, str]
    comparator_filter: dict[str, str]
    status: str
    note: str


SPECS = [
    ComparisonSpec(
        dataset="MovieLens 100K",
        setting="homogeneous response",
        source_table="movielens_100k_svdpp_optimized_20seeds_final_summary.csv",
        candidate="DU-Net",
        comparator="myopic",
        candidate_filter={"policy": "du_net"},
        comparator_filter={"policy": "myopic"},
        status="main frozen",
        note="Figure 4-style core MovieLens run; DU-Net trades slightly lower reward for lower weighted RMSE.",
    ),
    ComparisonSpec(
        dataset="MovieLens 100K",
        setting="heterogeneous response alpha=0.02",
        source_table="movielens_100k_svdpp_robustness_20seeds_final_summary.csv",
        candidate="DU-Net",
        comparator="myopic",
        candidate_filter={"policy": "du_net", "alpha": "0.02"},
        comparator_filter={"policy": "myopic", "alpha": "0.02"},
        status="main frozen",
        note="Figure 5-style response heterogeneity run; DU-Net keeps a weighted-RMSE advantage but lower observed reward.",
    ),
    ComparisonSpec(
        dataset="Flixster",
        setting="homogeneous response",
        source_table="flixster_svdpp_tuned_20seeds_20cycles_final_summary.csv",
        candidate="DU-Net",
        comparator="myopic",
        candidate_filter={"policy": "du_net"},
        comparator_filter={"policy": "myopic"},
        status="main frozen",
        note="Best current social-graph reproduction: DU-Net improves both reward and weighted RMSE.",
    ),
    ComparisonSpec(
        dataset="Flixster",
        setting="heterogeneous response alpha=0.02",
        source_table="flixster_svdpp_robustness_20seeds_final_summary.csv",
        candidate="DU-Net",
        comparator="myopic",
        candidate_filter={"policy": "du_net", "alpha": "0.02"},
        comparator_filter={"policy": "myopic", "alpha": "0.02"},
        status="main frozen",
        note="Newly completed 20-seed Flixster response-bias extension; DU-Net remains ahead on both reward and weighted RMSE.",
    ),
    ComparisonSpec(
        dataset="Flixster",
        setting="baseline matrix homogeneous",
        source_table="flixster_svdpp_baselines_20seeds_final_summary.csv",
        candidate="DU-Net",
        comparator="myopic",
        candidate_filter={"variant": "du_net"},
        comparator_filter={"variant": "myopic"},
        status="baseline matrix partial",
        note="DU-Net leads the implemented Flixster baseline matrix; graph_social is close and second-best among implemented baselines.",
    ),
    ComparisonSpec(
        dataset="Flixster",
        setting="DLS/network ablation homogeneous",
        source_table="flixster_svdpp_ablation_20seeds_final_summary.csv",
        candidate="DU-Net",
        comparator="no-network",
        candidate_filter={"variant": "current_du_net"},
        comparator_filter={"variant": "no_network"},
        status="mechanism evidence",
        note="Current DU-Net and no-network are numerically identical in this config, showing that Flixster's gain is not coming from the current diffusion proxy.",
    ),
    ComparisonSpec(
        dataset="Douban",
        setting="legacy tuned homogeneous",
        source_table="douban_svdpp_tuned_20seeds_20cycles_final_summary.csv",
        candidate="current DU-Net",
        comparator="myopic",
        candidate_filter={"policy": "du_net"},
        comparator_filter={"policy": "myopic"},
        status="legacy near-tie",
        note="Original tuned proxy shows why Douban needed mechanism work: DU-Net and myopic are effectively tied.",
    ),
    ComparisonSpec(
        dataset="Douban",
        setting="homogeneous final candidate",
        source_table="douban_svdpp_network_entropy_spectral_20seeds_final_summary.csv",
        candidate="NetObs + reliability-DLS",
        comparator="myopic",
        candidate_filter={"variant": "netobs_reliability_spectral"},
        comparator_filter={"variant": "myopic"},
        status="main candidate",
        note="Most faithful current Douban homogeneous candidate; positive reward and weighted-RMSE gap over myopic.",
    ),
    ComparisonSpec(
        dataset="Douban",
        setting="heterogeneous response alpha=0.02",
        source_table="douban_svdpp_response_aware_netobs_20seeds_final_summary.csv",
        candidate="Response-aware NetObs",
        comparator="myopic",
        candidate_filter={"variant": "netobs_response_centered_str2"},
        comparator_filter={"variant": "myopic"},
        status="main candidate",
        note="Response-aware network observational entropy is the current best heterogeneous Douban candidate.",
    ),
    ComparisonSpec(
        dataset="Douban",
        setting="baseline matrix homogeneous",
        source_table="douban_svdpp_baselines_netobs_20seeds_final_summary.csv",
        candidate="NetObs + reliability-DLS",
        comparator="myopic",
        candidate_filter={"variant": "netobs_reliability_spectral"},
        comparator_filter={"variant": "myopic"},
        status="baseline matrix partial",
        note="NetObs candidate leads the implemented Douban baseline matrix on weighted RMSE; paper's full 16-baseline suite remains incomplete.",
    ),
    ComparisonSpec(
        dataset="Douban",
        setting="DLS/network ablation homogeneous",
        source_table="douban_svdpp_ablation_netobs_20seeds_final_summary.csv",
        candidate="no-DLS",
        comparator="NetObs + reliability-DLS",
        candidate_filter={"variant": "no_dls"},
        comparator_filter={"variant": "netobs_reliability_spectral"},
        status="mechanism evidence",
        note="No-DLS slightly wins on sparse Douban, supporting the noise-propagation diagnosis for the current DLS proxy.",
    ),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matches(row: dict[str, str], filters: dict[str, str]) -> bool:
    for key, value in filters.items():
        if str(row.get(key, "")) != value:
            return False
    return True


def find_row(rows: list[dict[str, str]], filters: dict[str, str], source_name: str) -> dict[str, str]:
    found = [row for row in rows if matches(row, filters)]
    if len(found) != 1:
        raise ValueError(f"{source_name}: expected one row for {filters}, found {len(found)}")
    return found[0]


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def build_summary() -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for spec in SPECS:
        source_path = TABLE_DIR / spec.source_table
        rows = read_rows(source_path)
        candidate_row = find_row(rows, spec.candidate_filter, spec.source_table)
        comparator_row = find_row(rows, spec.comparator_filter, spec.source_table)
        candidate_reward = f(candidate_row, "cumulative_reward")
        comparator_reward = f(comparator_row, "cumulative_reward")
        candidate_w_rmse = f(candidate_row, "w_rmse")
        comparator_w_rmse = f(comparator_row, "w_rmse")
        rows_out.append(
            {
                "dataset": spec.dataset,
                "setting": spec.setting,
                "candidate": spec.candidate,
                "comparator": spec.comparator,
                "candidate_reward": candidate_reward,
                "comparator_reward": comparator_reward,
                "reward_delta": candidate_reward - comparator_reward,
                "candidate_w_rmse": candidate_w_rmse,
                "comparator_w_rmse": comparator_w_rmse,
                "w_rmse_delta": candidate_w_rmse - comparator_w_rmse,
                "candidate_reward_ci95": f(candidate_row, "cumulative_reward_ci95"),
                "comparator_reward_ci95": f(comparator_row, "cumulative_reward_ci95"),
                "candidate_w_rmse_ci95": f(candidate_row, "w_rmse_ci95"),
                "comparator_w_rmse_ci95": f(comparator_row, "w_rmse_ci95"),
                "candidate_response_rate": f(candidate_row, "response_rate"),
                "comparator_response_rate": f(comparator_row, "response_rate"),
                "status": spec.status,
                "source_table": spec.source_table,
                "note": spec.note,
            }
        )
    return rows_out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return ""
    return f"{numeric:.{digits}f}"


def plot_w_rmse(rows: list[dict[str, object]], path: Path) -> None:
    main_rows = [
        row
        for row in rows
        if row["setting"]
        in {
            "homogeneous response",
            "heterogeneous response alpha=0.02",
            "legacy tuned homogeneous",
            "homogeneous final candidate",
        }
    ]
    labels = [
        f"{row['dataset']}\n{str(row['setting']).replace(' response ', ' ')}"
        for row in main_rows
    ]
    x = list(range(len(main_rows)))
    width = 0.36
    candidate_vals = [float(row["candidate_w_rmse"]) for row in main_rows]
    comparator_vals = [float(row["comparator_w_rmse"]) for row in main_rows]

    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.bar([i - width / 2 for i in x], candidate_vals, width, label="candidate", color="#2f6f73")
    ax.bar([i + width / 2 for i in x], comparator_vals, width, label="comparator", color="#c57b57")
    ax.set_ylabel("weighted RMSE (lower is better)")
    ax.set_title("Current paper-quality reproduction snapshot")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "dataset",
        "setting",
        "candidate",
        "comparator",
        "reward delta",
        "w-RMSE delta",
        "status",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["setting"]),
                    str(row["candidate"]),
                    str(row["comparator"]),
                    fmt(row["reward_delta"], 2),
                    fmt(row["w_rmse_delta"], 4),
                    str(row["status"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]], csv_path: Path, figure_path: Path, report_path: Path) -> None:
    report = f"""# Paper-quality reproduction status and next plan

Generated by `experiments/build_paper_quality_summary.py`.

## Current progress

The reproduction has moved from a MovieLens-only MVP to a three-dataset, real-data replication package with 20-seed evidence for the main MovieLens, Flixster, and Douban settings. The current frozen comparison table is:

{markdown_table(rows)}

The key interpretation is now stable:

- **MovieLens**: the DU-Net-style policy has a robust weighted-RMSE advantage, while cumulative observed reward is slightly below myopic in the current SVD++ reproduction. This is consistent with a reward-information tradeoff rather than a pure reward-maximization win.
- **Flixster**: the tuned social-graph reproduction is the cleanest current match to the paper's qualitative claim, with DU-Net improving both cumulative reward and weighted RMSE over myopic in homogeneous and heterogeneous-response settings. The new 20-seed baseline matrix also keeps DU-Net ahead of the implemented baselines.
- **Douban**: the original tuned proxy was essentially tied with myopic. Mechanism work changed the state of the reproduction: network observational entropy plus reliability-DLS now gives a positive homogeneous gap, and response-aware network observational entropy is the current best heterogeneous Douban candidate.
- **Douban mechanism evidence**: no-DLS remains slightly stronger than reliability-DLS on homogeneous sparse Douban, which supports the diagnosis that naive DLS can propagate graph noise. Reliability-DLS reduces centrality bias and becomes useful when paired with stronger network entropy or response-aware entropy.

## Recent baseline screens

The latest MovieLens baseline-upgrade work is useful evidence, but not promotion-worthy final evidence:

| branch | best new setting | reward | w-RMSE | decision |
|---|---|---:|---:|---|
| MF-centered bandits | MF linear UCB, alpha=0.05 | 2702.33 | 0.8306 | Fixes old bandit reward collapse but does not beat DU-Net or graph_social. |
| residual MF bandits | residual UCB, weight=0.25 alpha=0.02 | 2705.33 | 0.8318 | Does not beat MF linear UCB. |
| active-learning baselines | active hybrid, alpha=0.20 | 267.67 | 1.1637 | Negative; uncertainty-only selection moves far off the reward frontier. |
| social-rec baselines | social MF hybrid, weight=0.50 | 2719.67 | 0.8301 | Useful social-rec comparator, but still below the DU-Net/graph_social w-RMSE frontier. |

These screens close part of the implemented-baseline coverage gap. They do **not** close the paper's full 16-baseline gap, because the remaining issue is paper-specific baseline strength and exact definitions rather than simply adding more weak rows.

## Distance to full paper reproduction

The project is approximately at a **strong partial reproduction / pre-paper-quality rerun** stage. The main empirical direction is reproduced on MovieLens and Flixster, and Douban is no longer just a failed near-tie case; it has a documented sparse-network mechanism story and a stronger final candidate. The largest remaining gap is not ordinary tuning, but faithfulness:

- the current DU-Net is still a modular proxy rather than the exact paper architecture;
- observational entropy is closer than before but still not the exact paper estimator;
- graph embeddings include spectral and lightweight sparse proxies, not a fully validated ELPH/Node2Vec implementation;
- the complete paper baseline matrix is only partially implemented;
- Douban and Flixster now both have additional 20-seed heterogeneous and baseline/ablation evidence, but the full paper-level baseline suite is still incomplete and the current network diffusion proxy remains weaker than the paper's intended network module.

## Basic next steps

1. Freeze the current candidate configs and outputs as the handoff baseline. Do not overwrite the 20-seed summary CSVs listed in `{csv_path.relative_to(ROOT)}`.
2. Use the generated summary table and figure as the current paper-quality entry point:
   - `{csv_path.relative_to(ROOT)}`
   - `{figure_path.relative_to(ROOT)}`
3. Keep `reports/three_dataset_reproduction_summary.md` synchronized with this generated snapshot and `reports/paper_alignment_gap_matrix.md`.
4. Treat current Flixster/Douban 20-seed ablation and baseline matrices as implemented-baseline evidence, not as the full paper benchmark suite.
5. Add any new experiment only through a new output prefix so the frozen 20-seed tables are not overwritten.

## Long-term plan

1. Replace lightweight structural embeddings with a true sparse Node2Vec implementation and, if feasible, a closer ELPH/role-structure embedding.
2. Implement a more faithful network observational entropy estimator that combines uncertainty, graph neighborhood structure, and observed-history effects rather than treating entropy as a scalar add-on.
3. Complete the full benchmark matrix with the remaining paper-specific baseline definitions. MF-bandit, active-learning, and social-rec pilot branches should be promoted only if a focused screen reaches the DU-Net/graph_social frontier.
4. Extend mechanism diagnostics to all three datasets: cold-start vs warm-user, centrality bias, graph-density sensitivity, response bias, DLS effective neighborhood size, and reward-information tradeoff over acquisition cycles.
5. After the above is stable, run the final paper-quality matrix: MovieLens / Flixster / Douban main results, heterogeneous response, ablations, and baselines, all at 20 seeds or higher.

## Generated artifacts

- Summary CSV: `{csv_path.relative_to(ROOT)}`
- Figure: `{figure_path.relative_to(ROOT)}`
- This report: `{report_path.relative_to(ROOT)}`
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_summary()
    csv_path = TABLE_DIR / "paper_quality_current_summary.csv"
    figure_path = FIGURE_DIR / "paper_quality_current_summary_w_rmse.png"
    report_path = REPORT_DIR / "paper_quality_reproduction_status_and_plan.md"
    write_csv(rows, csv_path)
    plot_w_rmse(rows, figure_path)
    write_report(rows, csv_path, figure_path, report_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
