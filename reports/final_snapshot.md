# Final Snapshot: Deliverable Freeze

Freeze date: 2026-07-06

This snapshot freezes the strongest current reproducible artifacts. These files should be treated as the current deliverable baseline and should not be casually overwritten by exploratory runs.

## Environment and Verification

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Core checks:

```powershell
python -m compileall src experiments
python -m unittest discover -s tests
```

Latest verification: core tests passed; 12 unit tests passed.

## Frozen Main Results

| dataset | setting | config | output table | figure | conclusion |
|---|---|---|---|---|---|
| MovieLens | homogeneous response, 20 seeds | `configs/movielens_100k_svdpp_optimized_20seeds.json` | `outputs/tables/movielens_100k_svdpp_optimized_20seeds_final_summary.csv` | `outputs/figures/figure4_movielens_main_20seeds.png` | DU-Net lowers w-RMSE versus myopic while preserving near-myopic reward. |
| MovieLens | heterogeneous response alpha=0.02, 20 seeds | `configs/movielens_100k_svdpp_robustness_20seeds.json` | `outputs/tables/movielens_100k_svdpp_robustness_20seeds_final_summary.csv` | `outputs/figures/figure5_movielens_heterogeneous_20seeds.png` | DU-Net keeps a w-RMSE advantage under response bias. |
| Flixster | true social graph, tuned, 20 seeds | `configs/flixster_svdpp_tuned_20seeds_20cycles.json` | `outputs/tables/flixster_svdpp_tuned_20seeds_20cycles_final_summary.csv` | `outputs/figures/flixster_svdpp_tuned_20seeds_20cycles_w_rmse.png` | Cleanest current true-social-network reproduction; DU-Net beats myopic on reward and w-RMSE. |
| Flixster | heterogeneous response alpha=0.02, 20 seeds | `configs/flixster_svdpp_robustness_20seeds.json` | `outputs/tables/flixster_svdpp_robustness_20seeds_final_summary.csv` | `outputs/figures/flixster_svdpp_robustness_20seeds_w_rmse_by_alpha.png` | DU-Net still beats myopic on reward and w-RMSE under response bias. |
| Flixster | DLS/network ablation, 20 seeds | `configs/flixster_svdpp_ablation_20seeds.json` | `outputs/tables/flixster_svdpp_ablation_20seeds_final_summary.csv` | `outputs/figures/flixster_svdpp_ablation_20seeds_w_rmse.png` | Current DU-Net and no-network are numerically identical, so the current diffusion proxy is not driving the Flixster gain. |
| Flixster | baseline matrix, 20 seeds | `configs/flixster_svdpp_baselines_20seeds.json` | `outputs/tables/flixster_svdpp_baselines_20seeds_final_summary.csv` | `outputs/figures/flixster_svdpp_baselines_20seeds_w_rmse.png` | DU-Net is strongest among implemented baselines; graph_social is close. |
| Douban | true social graph, tuned, 20 seeds | `configs/douban_svdpp_tuned_20seeds_20cycles.json` | `outputs/tables/douban_svdpp_tuned_20seeds_20cycles_final_summary.csv` | `outputs/figures/douban_svdpp_tuned_20seeds_20cycles_w_rmse.png` | DU-Net is close to myopic but not stably better. This is the main remaining fidelity gap. |
| Paper-quality current summary | frozen cross-run snapshot | `experiments/build_paper_quality_summary.py` | `outputs/tables/paper_quality_current_summary.csv` | `outputs/figures/paper_quality_current_summary_w_rmse.png` | Unified current status and next-plan report. |
| Paper alignment gap matrix | paper/plan coverage audit | `experiments/build_paper_alignment_gap_matrix.py` | `outputs/tables/paper_alignment_gap_matrix.csv` | n/a | Tracks complete, partial, and proxy-only reproduction components. |
| Baseline coverage audit | implemented-vs-paper benchmark audit | `experiments/build_baseline_coverage_audit.py` | `outputs/tables/baseline_coverage_audit.csv` | n/a | Separates implemented baseline rows from the paper's full 16-baseline benchmark claim. |
| Douban graph-density final-candidate confirmation | focused mechanism check, 20 seeds | `configs/douban_svdpp_graph_density_final_candidates_20seeds.json` | `outputs/tables/douban_svdpp_graph_density_final_candidates_20seeds_final_summary.csv` | `outputs/figures/douban_svdpp_graph_density_final_candidates_20seeds_w_rmse.png` | No-DLS is numerically best, netobs reliability nearly ties with much lower centrality, and dropout/rewiring hurt. |
| Exact-vs-proxy appendix | faithfulness boundary | `reports/exact_vs_proxy_appendix.md` | `reports/exact_vs_proxy_appendix.md` | n/a | Separates close paper matches from documented proxies. |
| Embedding fidelity feasibility | next-step diagnosis | `reports/embedding_fidelity_feasibility.md` | `reports/embedding_fidelity_feasibility.md` | n/a | Shows that local Node2Vec/ELPH proxy sweeps are exhausted enough; entropy fidelity should be prioritized next. |
| Next experiment queue | execution roadmap | `outputs/tables/next_experiment_queue.csv` | `outputs/tables/next_experiment_queue.csv` | n/a | Concrete basic and long-term queue after the current snapshot. |
| Three-dataset summary | paper-style overview | generated from frozen tables | see above | `outputs/figures/three_dataset_main_reproduction.png` | Main cross-dataset reproduction figure. |

## Main Numeric Summary

| dataset / setting | DU-Net reward | myopic reward | DU-Net w-RMSE | myopic w-RMSE |
|---|---:|---:|---:|---:|
| MovieLens homogeneous | 2714.55 | 2717.90 | 0.8264 | 0.8348 |
| MovieLens heterogeneous alpha=0.02 | 2268.05 | 2290.05 | 0.8328 | 0.8364 |
| Flixster tuned | 2919.65 | 2845.38 | 0.6309 | 0.6338 |
| Flixster heterogeneous alpha=0.02 | 2407.10 | 2352.48 | 0.6334 | 0.6375 |
| Douban tuned | 2836.90 | 2844.60 | 0.6805 | 0.6801 |
| Douban netobs reliability-DLS | 2861.80 | 2844.60 | 0.6778 | 0.6801 |
| Douban response-aware heterogeneous alpha=0.02 | 2360.35 | 2354.70 | 0.6934 | 0.6953 |

## Recent Baseline Screens

These are evidence for the benchmark-coverage audit, not frozen headline results.

| branch | best new setting | seeds | reward | w-RMSE | decision |
|---|---|---:|---:|---:|---|
| MovieLens MF-centered bandits | MF linear UCB alpha=0.05 | 3 | 2702.33 | 0.8306 | Useful but not promotion-worthy. |
| MovieLens residual MF bandits | residual UCB weight=0.25 alpha=0.02 | 3 | 2705.33 | 0.8318 | Does not beat MF linear UCB. |
| MovieLens active-learning baselines | active hybrid alpha=0.20 | 3 | 267.67 | 1.1637 | Negative; do not promote. |
| MovieLens social-rec baselines | social MF hybrid weight=0.50 | 3 | 2719.67 | 0.8301 | Useful comparator, still below DU-Net/graph_social. |

## Commands

```powershell
python experiments\run_movielens.py --config configs\movielens_100k_svdpp_optimized_20seeds.json
python experiments\run_movielens_robustness.py --config configs\movielens_100k_svdpp_robustness_20seeds.json
python experiments\run_movielens.py --config configs\flixster_svdpp_tuned_20seeds_20cycles.json
python experiments\run_movielens_robustness.py --config configs\flixster_svdpp_robustness_20seeds.json --resume
python experiments\run_policy_variants.py --config configs\flixster_svdpp_ablation_20seeds.json --resume
python experiments\run_policy_variants.py --config configs\flixster_svdpp_baselines_20seeds.json --resume
python experiments\run_movielens.py --config configs\douban_svdpp_tuned_20seeds_20cycles.json
python experiments\build_paper_quality_summary.py
python experiments\build_paper_alignment_gap_matrix.py
python experiments\build_baseline_coverage_audit.py
python experiments\plot_movielens_figures.py
python experiments\plot_three_dataset_main.py
```

## Douban Next-Step Results

These are not frozen as final paper results; they are the next decision evidence for attacking Douban.

| experiment | best setting | reward | w-RMSE | interpretation |
|---|---|---:|---:|---|
| temperature-controlled exponential sampling, 3 seeds | `selection_temperature=0.5` | 2874.00 | 0.6757 | Much better than untempered `exp`, but still not clearly better than the strongest top-k pilots. |
| observational entropy lambda, 3 seeds | `lambda_entropy=0.2` | 2875.67 | 0.6757 | Moderate entropy helps; large entropy weights hurt reward and w-RMSE. |
| observational-hybrid coverage, 3 seeds | `entropy_coverage_weight=0.1` | 2837.67 | 0.6796 | Coverage weight has only a small effect in the current proxy. |
| dynamic sigma / no-DLS, 20 seeds | `du_net_no_dls` | 2863.05 | 0.6777 | No-DLS beats current DU-Net and myopic on w-RMSE, suggesting sparse-graph DLS noise. |
| dynamic sigma / no-DLS, 20 seeds | `du_net_dynamic_sigma` | 2849.20 | 0.6791 | Dynamic sigma helps slightly versus current DU-Net but is not enough. |
| reliability-DLS frontier batch, 3 seeds | 70% reward + 30% information within reward frontier | 3012.00 | 0.6586 | Strongest current Douban pilot; makes selection less myopic without a scalar entropy sweep. |
| reliability-DLS frontier batch, 20 seeds | 70% reward + 30% information within reward frontier | 2813.95 | 0.6833 | Negative confirmation; entropy percentile rises but w-RMSE is worse than myopic. |
| graph-density sensitivity, 3 seeds | original reliability-DLS | 2977.33 | 0.6642 | Reliability-DLS and no-DLS are strongest on original sparse graph; rewiring hurts DLS sharply. |
| network observational entropy, 3 seeds | spectral reliability-DLS | 2996.67 | 0.6626 | Promising mechanism evidence; beats myopic in 3 seeds and was promoted to a narrow 20-seed confirmation. |
| network observational entropy, 20 seeds | spectral reliability-DLS | 2861.80 | 0.6778 | First 20-seed Douban follow-up that modestly beats myopic while using network-aware entropy and reliability-DLS. |
| formal Douban mechanism matrix, 20 seeds | no-DLS / reliability / netobs | 2863.05 / 2861.65 / 2861.80 | 0.6777 / 0.6778 / 0.6778 | No-DLS remains numerically best; reliability and netobs are essentially tied and both beat myopic/current DU-Net on w-RMSE. |
| Douban heterogeneous alpha=0.02, 20 seeds | reliability-DLS | 2353.30 | 0.6948 | Earlier heterogeneous check; superseded by response-aware netobs. |
| Douban DLS/network ablation, 20 seeds | no-DLS / reliability / netobs | 2863.05 / 2861.65 / 2861.80 | 0.6777 / 0.6778 / 0.6778 | Confirms sparse-graph DLS noise and reliability shrinkage. |
| Douban baseline matrix, 20 seeds | netobs reliability-DLS | 2861.80 | 0.6778 | Netobs reliability-DLS is strongest in the current baseline matrix. |
| Douban embedding upgrade, 3 seeds | spectral still best | 2996.67 | 0.6626 | Biased Node2Vec and ELPH-role improve over older proxies but do not beat spectral. |
| Douban response-aware heterogeneous alpha=0.02, 20 seeds | response-aware netobs | 2360.35 | 0.6934 | Response-aware entropy fixes the earlier netobs heterogeneous weakness and beats reliability-DLS/myopic/plain netobs. |

Commands:

```powershell
python experiments\run_movielens_grid.py --config configs\douban_svdpp_tau_grid_3seeds.json
python experiments\run_movielens_grid.py --config configs\douban_svdpp_entropy_lambda_grid_3seeds.json
python experiments\run_movielens_grid.py --config configs\douban_svdpp_entropy_coverage_grid_3seeds.json
python experiments\run_movielens.py --config configs\douban_svdpp_dynamic_no_dls_20seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_batch_composition_3seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_batch_composition_20seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_graph_density_sensitivity_3seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_network_entropy_embedding_3seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_network_entropy_spectral_20seeds.json
python experiments\run_movielens_robustness.py --config configs\douban_svdpp_robustness_netobs_20seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_ablation_netobs_20seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_baselines_netobs_20seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_embedding_upgrade_3seeds.json
python experiments\run_movielens_robustness.py --config configs\douban_svdpp_response_aware_netobs_20seeds.json
python experiments\run_policy_variants.py --config configs\douban_svdpp_graph_density_final_candidates_20seeds.json --resume
```

The formal Douban mechanism table is stored at `outputs/tables/douban_svdpp_formal_mechanism_20seeds_final_summary.csv`. The unified 6-variant rerun exceeded the local command timeout, so the table was synthesized from two matching 20-seed runs with identical Douban split, seeds, SVD++ settings, and acquisition baseline:

- `douban_svdpp_reliability_dls_20seeds`
- `douban_svdpp_network_entropy_spectral_20seeds`

Extension report: `reports/douban_netobs_20seed_extension.md`.
Embedding upgrade report: `reports/douban_embedding_upgrade_findings.md`.
Response-aware netobs report: `reports/douban_response_aware_netobs_findings.md`.
Unified paper-quality status report: `reports/paper_quality_reproduction_status_and_plan.md`.
Paper alignment gap matrix: `reports/paper_alignment_gap_matrix.md`.
Progress and next execution queue: `reports/reproduction_progress_next_execution.md`.
Exact-vs-proxy appendix: `reports/exact_vs_proxy_appendix.md`.

## Current Scientific Reading

The implementation now reproduces the paper's core direction on MovieLens and Flixster, including Flixster heterogeneous response, ablation, and baseline-matrix extensions. Douban has moved from a pure near-tie/fidelity-gap case to a diagnosed sparse-graph case with positive 20-seed network-observational candidates in both homogeneous and heterogeneous-response settings. The latest MovieLens bandit, active-learning, and social-rec screens improve baseline audit coverage but do not close the paper's full benchmark gap. The embedding feasibility check shows that another local Node2Vec p/q sweep is unlikely to move the result; the highest-value next local work is now a more faithful observational entropy estimator, followed by true structural embeddings if dependency or heavier local implementation work is allowed.
