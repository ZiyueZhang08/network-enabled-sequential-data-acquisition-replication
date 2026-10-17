# Three-Dataset Main Reproduction Results

This chapter consolidates the current paper-style reproduction across MovieLens, Flixster, and Douban. It is intended as the formal results chapter for the current replication state.

## Scope

The current reproduction covers:

- Network-enabled sequential data acquisition with initial/candidate/test splits.
- 20 acquisition cycles and batch size 100.
- 20 random seeds for the main MovieLens, Flixster, and Douban runs.
- SVD++-style user/item representation.
- DLS smoothing over user-network embeddings.
- Network-amplified reward and network-amplified information value.
- Random, myopic, entropy, and DU-Net comparisons.
- Heterogeneous response robustness on MovieLens, Flixster, and the current best Douban network-observational candidate.
- 20-seed Flixster DLS/network ablation and implemented baseline matrix.

## Main Results

This table is the frozen cross-dataset result table for the current deliverable version.

| dataset | setting | DU-Net reward | myopic reward | DU-Net w-RMSE | myopic w-RMSE | status |
|---|---|---:|---:|---:|---:|---|
| MovieLens | homogeneous response | 2714.55 | 2717.90 | 0.8264 | 0.8348 | DU-Net improves w-RMSE with near-myopic reward |
| MovieLens | heterogeneous alpha=0.02 | 2268.05 | 2290.05 | 0.8328 | 0.8364 | DU-Net improves w-RMSE under response bias |
| Flixster | tuned true social graph | 2919.65 | 2845.38 | 0.6309 | 0.6338 | DU-Net wins both reward and w-RMSE |
| Flixster | heterogeneous alpha=0.02 | 2407.10 | 2352.48 | 0.6334 | 0.6375 | DU-Net wins both reward and w-RMSE under response bias |
| Douban | tuned true social graph | 2836.90 | 2844.60 | 0.6805 | 0.6801 | DU-Net and myopic are effectively tied |
| Douban | netobs reliability-DLS | 2861.80 | 2844.60 | 0.6778 | 0.6801 | Current faithful candidate modestly beats myopic |
| Douban | response-aware netobs alpha=0.02 | 2360.35 | 2354.70 | 0.6934 | 0.6953 | Best current heterogeneous Douban candidate |

The headline interpretation is intentionally conservative: the current reproduction has strong positive evidence on MovieLens and Flixster, including a newly completed Flixster heterogeneous-response extension and implemented 20-seed baseline/ablation matrix. Douban started as a near-tie/fidelity-gap case, and the newest network-observational follow-up now gives a modest 20-seed Douban win over myopic, but not yet a full paper-quality matrix.

## MovieLens

MovieLens uses a demographic similarity network because the 100K release does not include a true social graph. The reproduction is strongest on the intended reward-information tradeoff: DU-Net lowers weighted implementation error while preserving almost all myopic acquisition reward.

Primary artifacts:

- `outputs/tables/movielens_100k_svdpp_optimized_20seeds_final_summary.csv`
- `outputs/figures/figure4_movielens_main_20seeds.png`
- `outputs/tables/movielens_100k_svdpp_robustness_20seeds_final_summary.csv`
- `outputs/figures/figure5_movielens_heterogeneous_20seeds.png`

Recent MovieLens baseline-upgrade screens clarify the remaining benchmark gap:

| branch | best new setting | reward | w-RMSE | decision |
|---|---|---:|---:|---|
| graph/Thompson 20-seed extension | graph_social | 2751.50 | 0.8300 | Highest reward, but DU-Net remains best implemented baseline by w-RMSE in the 20-seed matrix. |
| MF-centered bandits, 3 seeds | MF linear UCB alpha=0.05 | 2702.33 | 0.8306 | Fixes the old bandit reward collapse but does not beat DU-Net or graph_social. |
| residual MF bandits, 3 seeds | residual UCB weight=0.25 alpha=0.02 | 2705.33 | 0.8318 | Does not improve over MF linear UCB. |
| active-learning baselines, 3 seeds | active hybrid alpha=0.20 | 267.67 | 1.1637 | Negative; uncertainty-only acquisition selects low-reward ambiguous ratings. |
| social-rec baselines, 3 seeds | social MF hybrid weight=0.50 | 2719.67 | 0.8301 | Useful comparator; still below the DU-Net/graph_social w-RMSE frontier. |

These results mean the MovieLens benchmark gap is now better diagnosed: adding more naive baseline rows is not enough. The remaining work is to reproduce the paper-specific strong baseline definitions and only promote a pilot to 20 seeds if it reaches the DU-Net/graph_social frontier.

## Flixster

Flixster is currently the cleanest true-social-network reproduction. After dataset-specific tuning, DU-Net beats myopic on both final cumulative reward and w-RMSE across 20 seeds. The new heterogeneous-response extension confirms the same direction under `alpha=0.02`: DU-Net reward is 2407.10 versus myopic 2352.48, and DU-Net w-RMSE is 0.6334 versus myopic 0.6375.

Tuned setting:

- `lambda_entropy=0.05`
- `diffusion_prob=0.0`
- `selection=topk`
- `entropy_mode=hybrid`
- `graph_embedding_method=spectral`

The `diffusion_prob=0.0` result does not mean the network is unused: DLS still uses the user graph. The 20-seed Flixster ablation now makes the mechanism reading sharper: current DU-Net and `no_network` are numerically identical in this config, while `no_dls` is worse. So the current Flixster gain is robust, but it is not yet evidence that the implemented network diffusion proxy is doing the paper's full job.

Flixster 20-seed extension results:

| experiment | best / comparison | reward | w-RMSE | reading |
|---|---|---:|---:|---|
| heterogeneous response alpha=0.02 | DU-Net vs myopic | 2407.10 vs 2352.48 | 0.6334 vs 0.6375 | DU-Net remains ahead under response bias. |
| DLS/network ablation | DU-Net vs no-network | 2919.65 vs 2919.65 | 0.6309 vs 0.6309 | Current network diffusion does not change this tuned Flixster result. |
| baseline matrix | DU-Net vs graph_social / myopic | 2919.65 vs 2911.68 / 2845.38 | 0.6309 vs 0.6323 / 0.6338 | DU-Net is best among implemented baselines; graph_social is close. |

Primary artifacts:

- `outputs/tables/flixster_svdpp_tuned_20seeds_20cycles_final_summary.csv`
- `outputs/figures/flixster_svdpp_tuned_20seeds_20cycles_w_rmse.png`
- `outputs/figures/flixster_svdpp_tuned_20seeds_20cycles_cumulative_reward.png`
- `outputs/tables/flixster_svdpp_robustness_20seeds_final_summary.csv`
- `outputs/tables/flixster_svdpp_ablation_20seeds_final_summary.csv`
- `outputs/tables/flixster_svdpp_baselines_20seeds_final_summary.csv`

## Douban

Douban is the main remaining challenge. Tuned DU-Net is very close to myopic and clearly better than random/entropy, but it does not yet deliver a stable win over myopic in the 20-seed result.

Tuned setting:

- `lambda_entropy=0.05`
- `diffusion_prob=0.01`
- `selection=topk`
- `entropy_mode=hybrid`
- `graph_embedding_method=spectral`

Likely causes:

- Douban's user graph is much sparser than Flixster's, so local smoothing has fewer reliable neighbors.
- The current DU-Net score is still highly reward-driven; diagnostics show entropy percentiles near 0.53, so information value is not yet strongly shaping selection.
- Deterministic top-k over-concentrates on high predicted reward and can collapse back toward myopic behavior.
- Current observational entropy and graph embedding are efficient proxies, not the full paper machinery.
- Stronger SVD++ training improves both DU-Net and myopic, so the bottleneck is not only factorization capacity.

Primary artifacts:

- `outputs/tables/douban_svdpp_tuned_20seeds_20cycles_final_summary.csv`
- `outputs/figures/douban_svdpp_tuned_20seeds_20cycles_w_rmse.png`
- `outputs/figures/douban_svdpp_tuned_20seeds_20cycles_cumulative_reward.png`

After the first tuned 20-seed run, I ran a second Douban pass with stronger SVD++ training and a larger information weight: `epochs=6`, `lambda_entropy=0.10`. This improved DU-Net's absolute w-RMSE from 0.6805 to 0.6732, but myopic also improved to 0.6701. The result is still not a stable DU-Net win, but it narrows the diagnosis: Douban is not mainly blocked by factorization strength; it is blocked by how the network/information terms interact with the sparse user graph.

| Douban setting | policy | reward | w-RMSE | status |
|---|---|---:|---:|---|
| tuned epochs=2 lambda=0.05 | DU-Net | 2836.90 | 0.6805 | near-tie |
| tuned epochs=2 lambda=0.05 | myopic | 2844.60 | 0.6801 | near-tie |
| epochs=6 lambda=0.10 | DU-Net | 2987.05 | 0.6732 | better absolute error |
| epochs=6 lambda=0.10 | myopic | 3023.80 | 0.6701 | still slightly ahead |

The newest Douban mechanism checks sharpen this diagnosis:

| experiment | best policy/setting | reward | w-RMSE | reading |
|---|---|---:|---:|---|
| temperature-controlled exponential sampling, 3 seeds | tau=0.5 | 2874.00 | 0.6757 | Softmax temperature fixes the worst `exp` behavior, but does not yet dominate top-k. |
| observational entropy lambda, 3 seeds | lambda=0.2 | 2875.67 | 0.6757 | Moderate entropy helps; large entropy weights damage the reward-information balance. |
| observational-hybrid coverage, 3 seeds | coverage=0.1 | 2837.67 | 0.6796 | Coverage contributes little under the current proxy. |
| dynamic sigma / no-DLS, 20 seeds | no-DLS | 2863.05 | 0.6777 | Removing DLS beats current DU-Net, supporting the sparse-graph-noise hypothesis. |
| dynamic sigma / no-DLS, 20 seeds | dynamic sigma | 2849.20 | 0.6791 | Dynamic sigma helps slightly, but is not enough by itself. |

This means the most valuable Douban next step is mechanism redesign, not simply more epochs/rank. The goal should be to make DU-Net genuinely less myopic through stronger information value and better sparse-graph smoothing.

After adding log-det entropy, adaptive/reliability DLS shrinkage, and weighted log-det variants, the confirmed Douban interpretation is now sharper:

| confirmed Douban variant | seeds | reward | w-RMSE | selection mechanism reading |
|---|---:|---:|---:|---|
| `du_net_no_dls` | 20 | 2863.05 | 0.6777 | Best confirmed w-RMSE; sparse-graph DLS is the main failure mode. |
| `du_net_shrinkage_dls` reliability | 20 | 2861.65 | 0.6778 | Nearly ties no-DLS while lowering selected-user degree from 1.8632 to 1.2997. |
| weighted logdet + reliability DLS | 20 | 2860.05 | 0.6780 | Raises entropy percentile to 0.6073 but does not improve final w-RMSE. |
| current DU-Net | 20 | 2838.65 | 0.6801 | Still too close to myopic selection. |

Mechanism diagnostics show that reliability DLS reduces centrality bias without paying much accuracy cost, but no-DLS remains the safest confirmed Douban setting. This should be reported as a formal Douban ablation rather than hidden as a failed tuning attempt.

Mechanism diagnostic artifacts:

- `reports/douban_svdpp_reliability_dls_20seeds_mechanism_diagnostics.md`
- `outputs/figures/douban_svdpp_reliability_dls_20seeds_cold_warm_w_rmse.png`
- `outputs/figures/douban_svdpp_reliability_dls_20seeds_centrality_binned_w_rmse.png`
- `outputs/figures/douban_svdpp_reliability_dls_20seeds_selected_degree_bins.png`
- `outputs/figures/douban_svdpp_reliability_dls_20seeds_dls_shrinkage_over_cycles.png`
- `outputs/figures/douban_svdpp_reliability_dls_20seeds_reward_entropy_percentiles.png`

The next Douban mechanism step replaces scalar z-score mixing with reward-frontier batch composition. The first 3-seed pilot was positive:

| Douban batch-composition variant | seeds | reward | w-RMSE | reward pct | entropy pct | reading |
|---|---:|---:|---:|---:|---:|---|
| reliability-DLS frontier batch 70/30 | 3 | 3012.00 | 0.6586 | 0.9991 | 0.6898 | Best current Douban pilot; much less myopic while preserving high reward. |
| logdet + reliability frontier batch 70/30 | 3 | 2929.00 | 0.6677 | 0.9991 | 0.7412 | Strong information orientation, but lower reward than reliability batch. |
| myopic | 3 | 2921.33 | 0.6710 | 0.9996 | 0.5406 | Strong baseline, but clearly less information-oriented. |
| no-DLS frontier batch 70/30 | 3 | 2918.00 | 0.6713 | 0.9991 | 0.6822 | Batch composition helps exploration, but no-DLS no longer dominates. |

The 20-seed confirmation is negative:

| Douban batch-composition variant | seeds | reward | w-RMSE | reward pct | entropy pct | reading |
|---|---:|---:|---:|---:|---:|---|
| myopic | 20 | 2844.60 | 0.6801 | 0.9996 | 0.5435 | Best in this confirmation. |
| no-DLS frontier batch 70/30 | 20 | 2818.00 | 0.6832 | 0.9991 | 0.6864 | Less myopic, but worse w-RMSE. |
| reliability-DLS frontier batch 70/30 | 20 | 2813.95 | 0.6833 | 0.9991 | 0.6889 | Centrality bias is lower, but no accuracy gain. |
| logdet + reliability frontier batch 70/30 | 20 | 2804.95 | 0.6843 | 0.9991 | 0.7258 | Highest information percentile, weakest final w-RMSE. |

This changes the interpretation: batch composition is useful as a diagnostic because it proves selection can be made less myopic, but the current information proxy is not yet predictive enough on Douban.

Graph-density sensitivity further supports the sparse-graph diagnosis: original reliability-DLS and no-DLS are strongest, edge rewiring hurts DLS sharply, and densification increases selected-user centrality rather than producing a clear gain. This implies the Douban graph contains useful structure, but the current proxy DLS must be reliability-controlled and paired with stronger structural embeddings and more faithful network observational entropy.

The newest focused graph-density test reruns this question only on the final meaningful candidates: no-DLS and netobs reliability-DLS. The 20-seed confirmation strengthens the same diagnosis:

| Douban focused graph-density setting | seeds | reward | w-RMSE | selected degree | reading |
|---|---:|---:|---:|---:|---|
| original no-DLS | 20 | 2863.05 | 0.6777 | 1.8632 | Best numerical w-RMSE, but removes the DLS mechanism. |
| original netobs reliability | 20 | 2861.80 | 0.6778 | 1.2373 | Nearly ties no-DLS while preserving network-observational entropy and reliability-DLS. |
| densify50 netobs reliability | 20 | 2858.50 | 0.6781 | 2.4379 | Accuracy nearly holds, but centrality bias rises sharply. |
| drop50 netobs reliability | 20 | 2841.10 | 0.6803 | 0.5292 | Edge removal hurts final w-RMSE despite higher entropy percentile. |
| rewire50 netobs reliability | 20 | 2825.20 | 0.6826 | 0.9945 | Rewiring damages graph signal. |

This is now 20-seed mechanism evidence: Douban's observed graph is not pure noise. The original sparse graph helps netobs reliability-DLS; deleting or rewiring edges harms it; densifying the graph risks centrality bias.

The first network-observational entropy / sparse-embedding pass adds `sparse_node2vec`, `elph_struct`, and `network_observational` entropy. It shows useful signal:

| Douban network entropy variant | seeds | reward | w-RMSE | entropy pct | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability spectral | 3 | 2996.67 | 0.6626 | 0.5467 | Best new network-observational setting; beats myopic in 3 seeds. |
| netobs reliability sparse Node2Vec | 3 | 2949.00 | 0.6672 | 0.5735 | Beats myopic but not spectral netobs. |
| netobs reliability ELPH-struct | 3 | 2946.67 | 0.6681 | 0.5515 | Beats myopic but not spectral netobs. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | Strong baseline. |

The narrowed 20-seed confirmation for `netobs_reliability_spectral` is now positive:

| Douban network entropy variant | seeds | reward | w-RMSE | entropy pct | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability spectral | 20 | 2861.80 | 0.6778 | 0.5672 | Modestly beats myopic while retaining network-observational entropy and reliability-DLS. |
| myopic | 20 | 2844.60 | 0.6801 | 0.5435 | Strong reward baseline. |

This is the first 20-seed Douban follow-up that is both positive against myopic and closer to the paper mechanism than the no-DLS fallback. It is still a small separation, so it should be treated as a new candidate for a formal Douban matrix rather than as a final paper-level result.

The next fidelity step implements `network_observational_expected`, which replaces the set-max network log-det proxy with first-order expected marginal gains from same-item social-neighbor observations. In the narrow Douban 3-seed check it does not change the selected batches relative to current `network_observational`:

| Douban expected-network entropy variant | seeds | reward | w-RMSE | entropy pct | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability spectral | 3 | 2996.67 | 0.6626 | 0.5467 | Current faithful Douban candidate. |
| expected-netobs reliability spectral | 3 | 2996.67 | 0.6626 | 0.5467 | Same selections/results; tested fidelity path, not a promoted main result. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | Strong reward baseline. |

This suggests the next entropy upgrade should be stochastic subset/log-det or batched network-observation entropy, not another lambda sweep or immediate 20-seed confirmation of the identical first-order expected variant.

The first batch-level network selector is now implemented as `selection="network_batch_composition"`. It uses reward/information batch shares but makes information slots network-aware by penalizing repeated users/items and same-item social neighbors already in the batch. The 3-seed Douban pilot shows that this can change selection behavior, but not yet improve accuracy:

| Douban network-batch variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains best. |
| expected-netobs network-batch p=0.10 | 3 | 2939.33 | 0.6661 | 0.6520 | 1.1333 | More information-oriented and lower centrality, weaker w-RMSE. |
| netobs network-batch p=0.02 | 3 | 2936.33 | 0.6661 | 0.6631 | 1.0698 | Strongest information shift, but not accuracy-positive. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong baseline. |

This is useful mechanism evidence: the selector can push the system away from myopic high-degree acquisitions, but the current entropy signal is still not predictive enough. It should not be promoted to 20 seeds until the entropy or embedding representation improves.

The stochastic subset log-det entropy pilot then tests a more faithful network-observation estimator directly:

| Douban stochastic network entropy variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains best. |
| stochastic netobs p=0.01 | 3 | 2981.33 | 0.6640 | 0.5653 | 1.2582 | Slightly more information-oriented, but weaker w-RMSE. |
| stochastic netobs p=0.05 | 3 | 2661.67 | 0.7032 | 0.3733 | 4.7875 | Over-amplifies network observations and creates centrality bias. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong baseline. |

This closes an important branch: simply making entropy stochastic or increasing diffusion probability does not solve Douban. The next candidate must calibrate neighbor inclusion by degree/reliability or improve the graph embedding before stochastic entropy is worth a 20-seed run.

The follow-up reliability-calibrated stochastic entropy test closes that immediate calibration branch:

| Douban reliability stochastic variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| reliability stochastic p=0.05 + degree norm | 3 | 2864.33 | 0.6773 | 0.4212 | 3.8183 | Degree normalization partly fixes the stochastic collapse, but not enough. |
| reliability stochastic p=0.05 | 3 | 2661.33 | 0.7039 | 0.3703 | 4.8827 | Reliability damping alone does not control high-degree selection. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |

This result is important precisely because it is negative: stochastic network entropy should not be promoted to 20 seeds yet. The next high-value step is stronger structural embedding or a more faithful network observational estimator, with degree normalization treated as a required guardrail.

The follow-up probability-weighted expected network log-det estimator is now implemented as `network_observational_weighted`. It makes the same-item social-neighbor contribution deterministic and probability-weighted rather than sampled. This is closer in spirit to expected network observations, but the 3-seed Douban pilot does not beat the current top-k `network_observational` candidate:

| Douban weighted network entropy variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| weighted netobs BFS p=0.02 | 3 | 2966.33 | 0.6662 | 0.5090 | 2.3765 | Best weighted variant; beats myopic but loses to top-k netobs. |
| weighted netobs spectral p=0.001 | 3 | 2956.67 | 0.6667 | 0.5707 | 1.2877 | Low-probability weighting is stable but not promotion-worthy. |
| weighted netobs spectral p=0.002 | 3 | 2951.00 | 0.6678 | 0.5024 | 2.2562 | More graph exposure weakens w-RMSE. |
| weighted netobs spectral p=0.005 + degree norm | 3 | 2865.67 | 0.6770 | 0.4217 | 3.7360 | Degree normalization helps but still leaves high-degree drift. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |

This closes the immediate “more faithful expected network entropy” branch: the infrastructure is useful, but another probability sweep is not the right next step. The next entropy attempt should change the information target itself or use a stricter reward-frontier guard.

The reward-frontier guard follow-up then tests that stricter guard directly. It
allows positive entropy bonuses only for candidates above a reward-percentile
frontier, while preserving entropy penalties for low-information candidates:

| Douban reward-frontier guard variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| weighted guard q99 lambda=0.20 | 3 | 2950.00 | 0.6657 | 0.6467 | 1.0052 | Best guarded variant; improves weighted entropy but does not beat top-k netobs. |
| netobs guard q99 lambda=0.20 | 3 | 2925.67 | 0.6687 | 0.6447 | 0.9920 | More information-oriented and lower-degree, but weaker w-RMSE. |
| netobs guard q99 lambda=0.10 | 3 | 2924.00 | 0.6705 | 0.6003 | 1.2283 | Only slightly above myopic. |
| weighted guard q99 lambda=0.10 | 3 | 2922.33 | 0.6706 | 0.5902 | 1.2222 | Only slightly above myopic. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |
| netobs guard q95 lambda=0.20 | 3 | 2899.67 | 0.6715 | 0.6756 | 0.9803 | Strongest information shift, but over-guards reward. |

This is the second consecutive entropy-fidelity branch that is useful but not
promotion-worthy. The immediate next work should stop local entropy-gate sweeps
and move to structural representation fidelity or paper-specific benchmark
definitions.

The structural representation follow-up then adds a local SGNS Node2Vec backend.
The environment lacks validated external stacks such as `node2vec`, `pecanpy`,
`gensim`, `torch_geometric`, `dgl`, and `ogb`, but it does have `torch`, so the
new `sparse_node2vec_sgns_bfs` and `sparse_node2vec_sgns_dfs` methods train
second-order random-walk skip-gram embeddings without adding dependencies:

| Douban SGNS embedding variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability spectral | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| netobs reliability ELPH-role | 3 | 2987.00 | 0.6633 | 0.5731 | 1.3427 | Best non-spectral local structural proxy. |
| Node2Vec SGNS BFS | 3 | 2978.00 | 0.6643 | 0.5623 | 1.2702 | Positive against myopic, but weaker than spectral/ELPH-role. |
| Node2Vec SGNS DFS | 3 | 2949.33 | 0.6676 | 0.5949 | 1.2677 | More information-oriented, weaker accuracy. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |

This closes the local no-new-dependency embedding branch. SGNS Node2Vec is useful
infrastructure, but it is not promotion-worthy. The next local branch should move
to paper-specific benchmark definitions rather than another embedding-only sweep.

The formal Douban mechanism matrix now combines matching 20-seed runs into one table:

| Douban formal mechanism variant | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---|
| no-DLS | 2863.05 | 0.6777 | 0.5689 | 1.8632 | Best numerical w-RMSE, but removes the DLS mechanism. |
| reliability-DLS | 2861.65 | 0.6778 | 0.5720 | 1.2997 | Nearly ties no-DLS and reduces centrality bias. |
| netobs reliability spectral | 2861.80 | 0.6778 | 0.5672 | 1.2373 | Most faithful positive candidate so far; keeps network-observational entropy and reliability-DLS. |
| weighted logdet reliability | 2860.05 | 0.6780 | 0.6073 | 1.2590 | Strongest information shift, but not best w-RMSE. |
| current DU-Net | 2838.65 | 0.6801 | 0.5385 | 1.8359 | Too close to myopic and too centrality-biased. |
| myopic | 2844.60 | 0.6801 | 0.5435 | 1.2465 | Strong reward baseline. |

This matrix is the current best Douban explanation: the graph is useful but sparse, full DLS over-smooths, reliability shrinkage fixes much of the damage, and network-observational entropy gives a more paper-faithful positive result without yet creating a large margin.

The follow-up 20-seed extension adds heterogeneous response, ablation, and baseline evidence:

| Douban extension | best variant | reward | w-RMSE | reading |
|---|---|---:|---:|---|
| heterogeneous alpha=0.02 | reliability-DLS | 2353.30 | 0.6948 | Earlier check; superseded by response-aware netobs. |
| DLS/network ablation | no-DLS | 2863.05 | 0.6777 | Confirms sparse-graph DLS noise; reliability/netobs nearly tie. |
| baseline matrix | netobs reliability spectral | 2861.80 | 0.6778 | Netobs is strongest among current random/myopic/entropy/bandit/social/DU-Net baselines. |
| response-aware heterogeneous alpha=0.02 | response-aware netobs | 2360.35 | 0.6934 | Fixes the earlier netobs weakness under response bias. |

New formal Douban ablation artifacts:

- `reports/douban_formal_ablation_batch_density.md`
- `reports/douban_network_entropy_embedding_findings.md`
- `outputs/tables/douban_svdpp_batch_composition_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_batch_composition_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_graph_density_sensitivity_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_graph_density_final_candidates_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_graph_density_final_candidates_20seeds_final_summary.csv`
- `reports/douban_svdpp_graph_density_final_candidates_3seeds_mechanism_diagnostics.md`
- `reports/douban_svdpp_graph_density_final_candidates_20seeds_mechanism_diagnostics.md`
- `outputs/tables/douban_svdpp_network_entropy_embedding_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_network_entropy_spectral_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_formal_mechanism_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_robustness_netobs_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_ablation_netobs_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_baselines_netobs_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_response_aware_netobs_20seeds_final_summary.csv`
- `outputs/figures/douban_svdpp_batch_composition_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_batch_composition_20seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_graph_density_sensitivity_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_network_entropy_embedding_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_network_entropy_spectral_20seeds_w_rmse.png`
- `reports/douban_svdpp_formal_mechanism_20seeds_mechanism_diagnostics.md`
- `reports/douban_netobs_20seed_extension.md`
- `reports/douban_response_aware_netobs_findings.md`
- `reports/douban_expected_network_entropy_findings.md`
- `reports/douban_network_batch_entropy_findings.md`
- `reports/douban_stochastic_network_entropy_findings.md`
- `reports/douban_reliability_stochastic_entropy_findings.md`
- `reports/douban_weighted_network_entropy_findings.md`
- `reports/douban_reward_frontier_guard_findings.md`
- `reports/douban_node2vec_sgns_findings.md`
- `reports/embedding_fidelity_feasibility.md`

## Current Validity Assessment

What is already strong:

- The full acquisition loop works on all three paper datasets.
- MovieLens and Flixster reproduce the central DU-Net direction.
- Douban no longer fails; it is close to myopic and far above weak baselines.
- Results use 20 seeds and confidence intervals for the main tables.

What remains less faithful:

- ELPH is approximated by spectral / proxy / local SGNS graph embeddings.
- Observational entropy is still an efficient single-entry approximation.
- Paper-scale benchmark coverage is incomplete. Recent MovieLens MF-bandit, active-learning, and social-rec screens improve the audit trail, but none should be promoted as a final 20-seed paper baseline yet.
- Flixster and Douban now have 20-seed heterogeneous-response, ablation, and implemented-baseline evidence, but the exact paper benchmark matrix and final faithful rerun remain incomplete.

## Difference From the Paper

The current reproduction follows the paper's network-enabled sequential acquisition setup, but it is not an exact DU-Net reimplementation. Important differences are:

- The paper's observational information criterion is represented here by efficient proxy entropy scores: coverage, subspace, hybrid, observational, log-det, network-observational, expected-network, stochastic-network, weighted expected-network, and reward-frontier gated variants.
- ELPH/Node2Vec-style structural representations are currently lightweight dependency-free or local SGNS proxies; spectral embedding is the main stable embedding used in the frozen results.
- The current DLS implementation smooths matrix-factorization user factors over graph-embedding neighborhoods, but it does not yet reproduce every architectural detail of the paper's DU-Net.
- The benchmark matrix is partial: random, myopic, entropy, Linear UCB, low-rank bandit, Thompson sampling, graph-social, and DU-Net ablations exist, but the full 16-paper-baseline matrix is not yet complete.
- The current experiments use 20 cycles and batch size 100 across datasets for comparability; exact paper hyperparameters may differ by dataset and response setting.

These differences are why the current output should be described as a strong reproduction scaffold with two successful headline datasets and one diagnosed gap, not as a final exact replication.

What has now been added beyond the first main results:

- Flixster 20-seed ablations for DLS, network diffusion, reward, entropy, and dynamic sigma.
- Flixster 20-seed heterogeneous response robustness and implemented baseline matrix.
- Douban 20-seed DLS/network ablation, netobs baseline matrix, and response-aware heterogeneous robustness.
- Mechanism diagnostics for reward-information percentile, cold-user share, user-degree centrality, cold/warm w-RMSE, degree-binned w-RMSE, and DLS shrinkage.

## Next Research-Grade Milestones

1. Add remaining paper-specific baseline definitions before claiming paper-level benchmark coverage; MF-bandit, active-learning, and social-rec pilots are documented but not promotion-worthy yet.
2. Revisit validated external Node2Vec/ELPH only if new dependencies are explicitly allowed; the local no-new-dependency SGNS branch is exhausted.
3. Stop local entropy-gate sweeps until the structural representation or exact paper estimator is improved; expected, stochastic, batch-level, weighted network, and reward-frontier variants are implemented but not yet sufficient on Douban.
4. Extend graph-density sensitivity and mechanism diagnostics to final candidates across all three datasets.
5. Run the final paper-quality matrix only after the mechanism upgrades: main results, heterogeneous response, ablations, and baselines for MovieLens, Flixster, and Douban.

Immediate completed diagnostic milestone:

- Cold-start vs warm-user w-RMSE: implemented and plotted for Douban.
- Centrality-binned w-RMSE: implemented and plotted for Douban.
- Selected user degree bins: implemented and plotted for Douban.
- DLS shrinkage over cycles: implemented and plotted for Douban.
- Reward percentile vs entropy percentile over cycles: implemented and plotted for Douban.

## Mechanism Diagnostics

Selection diagnostics show that the current DU-Net implementation is still highly exploitative:

| dataset | DU-Net reward percentile | DU-Net entropy percentile | DU-Net avg user degree | DU-Net cold-user share |
|---|---:|---:|---:|---:|
| Flixster | 0.9977 | 0.5425 | 8.3952 | 0.1973 |
| Douban | 0.9996 | 0.5321 | 1.6682 | 0.1483 |

This explains why the current model is close to myopic, especially on Douban. The entropy term is present but not dominating selection; future faithful reproduction should focus on observational entropy and stochastic sampling rather than only reward diffusion.

Diagnostic artifacts:

- `reports/flixster_svdpp_baselines_20seeds_diagnostics.md`
- `reports/flixster_svdpp_baselines_20seeds_mechanism_diagnostics.md`
- `reports/flixster_svdpp_ablation_20seeds_diagnostics.md`
- `reports/flixster_svdpp_ablation_20seeds_mechanism_diagnostics.md`
- `reports/douban_svdpp_baselines_netobs_20seeds_diagnostics.md`
- `reports/douban_svdpp_baselines_netobs_20seeds_mechanism_diagnostics.md`
- `outputs/tables/flixster_svdpp_baselines_20seeds_diagnostics_summary.csv`
- `outputs/tables/flixster_svdpp_ablation_20seeds_diagnostics_summary.csv`
- `outputs/tables/douban_svdpp_baselines_netobs_20seeds_diagnostics_summary.csv`

## Paper Alignment Matrix

The current paper/plan alignment matrix is available at:

- `outputs/tables/paper_alignment_gap_matrix.csv`
- `reports/paper_alignment_gap_matrix.md`
- `reports/reproduction_progress_next_execution.md`
- `reports/exact_vs_proxy_appendix.md`
- `outputs/tables/next_experiment_queue.csv`

## Three-Dataset Main Figure

The combined three-dataset figure is available at:

- `outputs/figures/three_dataset_main_reproduction.png`
