"""Build a paper/plan alignment matrix for the current reproduction state."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"


ROWS = [
    {
        "paper_requirement": "Network-enabled sequential data acquisition problem",
        "plan_reference": "Level 1/2 acquisition loop with initial, candidate, and test sets",
        "current_status": "complete",
        "evidence": "src/acquisition.py; src/data_loader.py; tests/test_acquisition_loop.py",
        "gap": "No major structural gap; exact paper hyperparameters may still differ by dataset.",
        "basic_next": "Keep this as frozen infrastructure and avoid unplanned refactors.",
        "long_term_next": "Only revisit if exact paper budget/split details are recovered from supplementary material.",
    },
    {
        "paper_requirement": "Three real datasets: MovieLens, Flixster, Douban",
        "plan_reference": "Use MovieLens / Flixster / Douban with true social networks where available",
        "current_status": "complete",
        "evidence": "data/raw/gcmc/*/training_test_dataset.mat; reports/three_dataset_reproduction_summary.md",
        "gap": "MovieLens uses a constructed similarity network, consistent with the plan; Flixster/Douban use GCMC social graphs.",
        "basic_next": "Keep dataset loader tests as the guardrail.",
        "long_term_next": "Document any scale differences from the paper's exact preprocessing.",
    },
    {
        "paper_requirement": "20-cycle, batch-size-100, 20-seed main experiments",
        "plan_reference": "Paper-quality reproduction Figure 4/Figure 5 runs",
        "current_status": "mostly complete",
        "evidence": "outputs/tables/paper_quality_current_summary.csv; outputs/tables/final_paper_quality_results.csv; reports/final_paper_quality_results.md",
        "gap": "Main MovieLens/Flixster/Douban candidates are 20-seed and now synthesized in the final caveated report; a new full rerun should wait for primary-source or faithful-mechanism upgrades.",
        "basic_next": "Treat current 20-seed summaries and final caveated synthesis as the frozen snapshot.",
        "long_term_next": "Rerun all final settings only after exact entropy/embedding/DLS/diffusion or author-code benchmark upgrades.",
    },
    {
        "paper_requirement": "DU-Net combines reward and information value",
        "plan_reference": "z-normalized reward + lambda * information score; batch reward/information composition",
        "current_status": "complete proxy",
        "evidence": "src/acquisition.py; outputs/tables/*diagnostics_summary.csv; douban_svdpp_network_batch_entropy_3seeds_final_summary.csv; douban_svdpp_stochastic_network_entropy_3seeds_final_summary.csv; douban_svdpp_reliability_stochastic_entropy_3seeds_final_summary.csv; douban_svdpp_joint_score_gate_3seeds_final_summary.csv; douban_svdpp_weighted_network_entropy_3seeds_final_summary.csv; douban_svdpp_reward_frontier_guard_3seeds_final_summary.csv",
        "gap": "Scalar mixture, batch composition, an optional reward-information joint gate, probability-weighted expected network log-det entropy, and reward-frontier entropy gating are implemented. Network-batch, stochastic entropy, weighted expected entropy, joint-gate, and reward-frontier variants change behavior but current information pressure still costs w-RMSE on Douban.",
        "basic_next": "Report reward-information diagnostics alongside headline results; do not promote network-batch, stochastic entropy, weighted expected entropy, joint-gate, or reward-frontier variants without a stronger representation signal.",
        "long_term_next": "Move from local entropy-gate tuning to stronger structural representation or paper-specific benchmark definitions.",
    },
    {
        "paper_requirement": "Network-amplified reward via diffusion",
        "plan_reference": "network reward and diffusion_prob, with normalization variants",
        "current_status": "partial",
        "evidence": "src/reward.py; flixster_svdpp_ablation_20seeds_final_summary.csv; douban_svdpp_ablation_netobs_20seeds_final_summary.csv",
        "gap": "Current Flixster DU-Net equals no-network; Douban no-DLS/no-network evidence suggests raw proxy diffusion is not the full paper mechanism.",
        "basic_next": "Keep network ablations in the main report, not hidden as tuning failures.",
        "long_term_next": "Implement a more faithful independent-cascade expected reward approximation with calibrated degree normalization.",
    },
    {
        "paper_requirement": "DLS dynamic locally smooth user representation",
        "plan_reference": "Gaussian kernel over network embedding neighborhoods; dynamic/static/shrinkage ablations",
        "current_status": "partial positive mechanism",
        "evidence": "src/dls.py; outputs/tables/douban_svdpp_ablation_netobs_20seeds_final_summary.csv; outputs/tables/flixster_svdpp_ablation_20seeds_final_summary.csv",
        "gap": "DLS exists, but Douban shows sparse-graph noise; reliability shrinkage helps, while exact paper DLS architecture is not fully reproduced.",
        "basic_next": "Use reliability-DLS and no-DLS as formal ablations.",
        "long_term_next": "Rebuild DLS around a stronger structural embedding and exact dynamic-neighborhood update rule.",
    },
    {
        "paper_requirement": "Observational entropy / value of information",
        "plan_reference": "Move from coverage proxy to observational/log-det/network entropy",
        "current_status": "partial",
        "evidence": "src/entropy.py; douban_svdpp_network_entropy_spectral_20seeds_final_summary.csv; douban_svdpp_network_expected_entropy_3seeds_final_summary.csv; douban_svdpp_stochastic_network_entropy_3seeds_final_summary.csv; douban_svdpp_reliability_stochastic_entropy_3seeds_final_summary.csv; douban_svdpp_weighted_network_entropy_3seeds_final_summary.csv; douban_svdpp_reward_frontier_guard_3seeds_final_summary.csv",
        "gap": "Single-entry log-det, network-observational, first-order expected-network, stochastic subset, reliability-calibrated stochastic, probability-weighted expected network log-det, reward-frontier gated entropy, and network-batch approximations are implemented, but not the exact paper estimator.",
        "basic_next": "Clearly label current entropy as proxy; do not promote expected, stochastic, weighted expected, or reward-frontier variants because current 3-seed pilots lose or match current netobs.",
        "long_term_next": "Stop local entropy-gate sweeps until structural embedding or the paper estimator itself is improved.",
    },
    {
        "paper_requirement": "ELPH / higher-order graph representation",
        "plan_reference": "Use Node2Vec/GraphSAGE initially, replace with ELPH for faithful version",
        "current_status": "proxy only",
        "evidence": "src/graph_embedding.py; reports/douban_embedding_upgrade_findings.md; reports/douban_node2vec_sgns_findings.md; reports/embedding_fidelity_feasibility.md",
        "gap": "Spectral, dependency-free Node2Vec/ELPH-style proxies, and a local torch SGNS Node2Vec path exist; true ELPH or validated third-party Node2Vec is still not integrated.",
        "basic_next": "Keep spectral as frozen best candidate, report proxy status, and do not promote SGNS Node2Vec to 20 seeds.",
        "long_term_next": "Add a validated sparse Node2Vec package or external ELPH implementation only if dependencies are explicitly allowed, then rerun Douban embedding matrix.",
    },
    {
        "paper_requirement": "16 benchmark comparison",
        "plan_reference": "Random, myopic, entropy, bandit, network/social recommendation, DU-Net ablations",
        "current_status": "partial",
        "evidence": "outputs/tables/*baselines*20seeds_final_summary.csv; outputs/tables/baseline_coverage_audit.csv; outputs/tables/paper_benchmark_definition_audit.csv; reports/paper_benchmark_slot_recovery_audit.md; reports/author_bandit_source_comparison.md; reports/final_benchmark_claim_freeze.md; outputs/tables/flixster_svdpp_oful_3seeds_final_summary.csv; outputs/tables/flixster_svdpp_paper_bandit_repair_3seeds_final_summary.csv; outputs/tables/flixster_svdpp_mf_oful_beta_zoom_3seeds_final_summary.csv; outputs/tables/flixster_svdpp_paper_bandit_gap_3seeds_final_summary.csv; outputs/tables/flixster_svdpp_logistic_bandit_scale_3seeds_final_summary.csv; outputs/tables/movielens_100k_svdpp_bandit_upgrade_3seeds_final_summary.csv; outputs/tables/movielens_100k_svdpp_mf_linear_ucb_alpha_grid_3seeds_final_summary.csv; outputs/tables/movielens_100k_svdpp_residual_bandit_3seeds_final_summary.csv; outputs/tables/movielens_100k_svdpp_active_learning_baselines_3seeds_final_summary.csv; outputs/tables/movielens_100k_svdpp_social_rec_baselines_3seeds_final_summary.csv",
        "gap": "Implemented baselines now cover the shared cross-dataset families, including the MovieLens graph_social/Thompson extension. The paper-specific audit recovers 14 explicit slots under the paper's 16-benchmark claim; all 14 explicit slots are now implemented or proxy-covered after adding OFUL, logistic, and finite-arm rows. Local source/footnote review plus public footnote-code comparison still cannot confirm the two paper-count slots. contextualbandits exposes plausible extra methods such as active-learning exploration and SoftMax chooser, but the paper text does not name them, so the benchmark claim is frozen as 14 covered/proxy-covered explicit slots plus 2 unresolved slots. Raw OFUL is weak; MF-centered OFUL repairs much of the collapse but still remains below myopic, graph_social, and DU-Net. Logistic UCB/TS need probability-scale exploration and finite-arm proxies are competitive only as local stress rows.",
        "basic_next": "Use paper_benchmark_definition_audit, paper_benchmark_slot_recovery_audit, author_bandit_source_comparison, and final_benchmark_claim_freeze as the tracking artifacts.",
        "long_term_next": "Run final synthesis or final 20-seed rerun with the frozen 14+2 benchmark caveat unless original PDF, supplement, or author implementation becomes available.",
    },
    {
        "paper_requirement": "Ablation studies: DLS, reward, entropy, network components",
        "plan_reference": "Full DU-Net vs w/o DLS, entropy, reward, network, dynamic sigma",
        "current_status": "mostly complete",
        "evidence": "outputs/tables/flixster_svdpp_ablation_20seeds_final_summary.csv; outputs/tables/douban_svdpp_ablation_netobs_20seeds_final_summary.csv",
        "gap": "MovieLens/Flixster/Douban ablations exist, but exact paper ablation definitions may differ.",
        "basic_next": "Promote Flixster/Douban 20-seed ablations into final result chapter.",
        "long_term_next": "Repeat ablations after faithful entropy/embedding upgrades.",
    },
    {
        "paper_requirement": "Heterogeneous response robustness",
        "plan_reference": "Response probability increases with true rating; Figure 5-style robustness",
        "current_status": "mostly complete",
        "evidence": "movielens_100k_svdpp_robustness_20seeds_final_summary.csv; flixster_svdpp_robustness_20seeds_final_summary.csv; douban_svdpp_response_aware_netobs_20seeds_final_summary.csv",
        "gap": "Main three-dataset evidence exists; final unified response grid with all baselines remains open.",
        "basic_next": "Report current 20-seed response-aware candidates.",
        "long_term_next": "Rerun heterogeneous response for full benchmark matrix.",
    },
    {
        "paper_requirement": "Mechanism diagnosis: cold-start, centrality, graph density, reward-information tradeoff",
        "plan_reference": "Extensions E1-E6 and DLS mechanism diagnostics",
        "current_status": "partial and growing",
        "evidence": "outputs/figures/*mechanism*.png; outputs/tables/*diagnostics*.csv; douban_svdpp_graph_density_final_candidates_20seeds_final_summary.csv; douban_svdpp_network_batch_entropy_3seeds_final_summary.csv; douban_svdpp_stochastic_network_entropy_3seeds_final_summary.csv; douban_svdpp_reliability_stochastic_entropy_3seeds_final_summary.csv; douban_svdpp_weighted_network_entropy_3seeds_final_summary.csv; douban_svdpp_reward_frontier_guard_3seeds_final_summary.csv",
        "gap": "Douban and Flixster now have diagnostics; Douban has focused graph-density, network-batch, stochastic-entropy, reliability-calibration, weighted expected-network, and reward-frontier tradeoff evidence; MovieLens mechanism diagnostics are less complete.",
        "basic_next": "Use diagnostics to explain why Douban is hard and why Flixster network proxy is weak.",
        "long_term_next": "Run graph-density sensitivity and cold/warm diagnostics for final candidates across all datasets.",
    },
]


def write_csv(path: Path) -> None:
    fieldnames = list(ROWS[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ROWS)


def markdown_table() -> str:
    headers = ["paper requirement", "status", "gap", "basic next", "long-term next"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in ROWS:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["paper_requirement"],
                    row["current_status"],
                    row["gap"],
                    row["basic_next"],
                    row["long_term_next"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(path: Path, csv_path: Path) -> None:
    complete = sum(1 for row in ROWS if row["current_status"] == "complete")
    mostly = sum(1 for row in ROWS if row["current_status"].startswith("mostly"))
    partial = sum(1 for row in ROWS if row["current_status"].startswith("partial"))
    proxy = sum(1 for row in ROWS if "proxy" in row["current_status"])
    report = f"""# Paper Alignment Gap Matrix

This report aligns the reproduction plan and the paper's empirical claims with the current code/results state.

## Summary Judgment

The project is no longer an MVP. It is a **strong partial reproduction with paper-style 20-seed evidence**, especially on MovieLens and Flixster. Douban has moved from a near-tie to a diagnosed sparse-network case with positive network-observational candidates. The remaining distance to the paper is now mostly **faithfulness and completeness**, not basic execution.

Status counts: complete={complete}, mostly complete={mostly}, partial={partial}, proxy-only={proxy}.

## Gap Matrix

{markdown_table()}

## Basic Next Plan

1. Freeze the current 20-seed summary files and preserve them as the baseline snapshot.
2. Keep `reports/three_dataset_reproduction_summary.md` synchronized with the generated alignment matrix and remove stale 3-seed-only language.
3. Promote Flixster and Douban mechanism diagnostics into the result chapter.
4. Keep current Douban conclusions conservative: no-DLS is the strongest numerical sparse-graph fallback; netobs reliability-DLS is the most faithful positive candidate.

## Long-Term Plan

1. Upgrade the baseline matrix before claiming paper-level benchmark coverage.
2. Add a validated external Node2Vec/ELPH implementation only if dependencies are explicitly allowed later.
3. Rerun the full paper-quality matrix after the mechanism and benchmark definitions stabilize: main results, heterogeneous response, ablations, and baselines across MovieLens, Flixster, and Douban.

CSV artifact: `{csv_path.relative_to(ROOT)}`.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "paper_alignment_gap_matrix.csv"
    report_path = REPORT_DIR / "paper_alignment_gap_matrix.md"
    write_csv(csv_path)
    write_report(report_path, csv_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
