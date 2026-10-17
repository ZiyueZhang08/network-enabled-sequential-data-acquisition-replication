# Config Catalog

## Status Legend

- `MAIN/FROZEN`: current deliverable result; do not overwrite casually.
- `FOLLOW-UP`: scientifically important next-step result, but not final paper-quality evidence yet.
- `EXPLORATION`: tuning, smoke, pilot, or diagnostic run.
- `LEGACY`: earlier run retained for traceability; not recommended as headline evidence.

## Frozen Deliverable Set

| status | config | output prefix | role |
|---|---|---|---|
| MAIN/FROZEN | `movielens_100k_svdpp_optimized_20seeds.json` | `movielens_100k_svdpp_optimized_20seeds` | MovieLens homogeneous main result |
| MAIN/FROZEN | `movielens_100k_svdpp_robustness_20seeds.json` | `movielens_100k_svdpp_robustness_20seeds` | MovieLens heterogeneous response result |
| MAIN/FROZEN | `flixster_svdpp_tuned_20seeds_20cycles.json` | `flixster_svdpp_tuned_20seeds_20cycles` | Flixster main true-social result |
| MAIN/FROZEN | `flixster_svdpp_robustness_20seeds.json` | `flixster_svdpp_robustness_20seeds` | Flixster heterogeneous response alpha=0.02 result |
| MAIN/FROZEN | `flixster_svdpp_ablation_20seeds.json` | `flixster_svdpp_ablation_20seeds` | Flixster DLS/network ablation confirmation |
| MAIN/FROZEN | `flixster_svdpp_baselines_20seeds.json` | `flixster_svdpp_baselines_20seeds` | Flixster implemented baseline matrix |
| FOLLOW-UP | `flixster_svdpp_oful_3seeds.json` | `flixster_svdpp_oful_3seeds` | Paper-specific OFUL benchmark screen; implemented but negative for promotion |
| FOLLOW-UP | `flixster_svdpp_paper_bandit_repair_3seeds.json` | `flixster_svdpp_paper_bandit_repair_3seeds` | MF-centered OFUL repair screen; improves raw OFUL but remains below frontier |
| FOLLOW-UP | `flixster_svdpp_mf_oful_beta_zoom_3seeds.json` | `flixster_svdpp_mf_oful_beta_zoom_3seeds` | Focused MF-OFUL beta zoom; smallest bonus is best but still below myopic |
| FOLLOW-UP | `flixster_svdpp_paper_bandit_gap_3seeds.json` | `flixster_svdpp_paper_bandit_gap_3seeds` | Logistic and finite-arm paper benchmark proxy rows; finite-arm rows are useful but not frontier-level |
| FOLLOW-UP | `flixster_svdpp_logistic_bandit_scale_3seeds.json` | `flixster_svdpp_logistic_bandit_scale_3seeds` | Focused logistic bandit scale check; low exploration scale is required and still does not beat myopic |
| MAIN/FROZEN | `douban_svdpp_tuned_20seeds_20cycles.json` | `douban_svdpp_tuned_20seeds_20cycles` | Douban current main result; near-tie with myopic |
| FOLLOW-UP | `douban_svdpp_dynamic_no_dls_20seeds.json` | `douban_svdpp_dynamic_no_dls_20seeds` | Douban DLS-noise confirmation |
| FOLLOW-UP | `douban_svdpp_mechanism_matrix_3seeds.json` | `douban_svdpp_mechanism_matrix_3seeds` | Douban log-det entropy / shrinkage-DLS mechanism matrix |
| FOLLOW-UP | `douban_svdpp_logdet_shrinkage_stage1_3seeds.json` | `douban_svdpp_logdet_shrinkage_stage1_3seeds` | Douban shrinkage x lambda screen for log-det entropy |
| FOLLOW-UP | `douban_svdpp_logdet_shrinkage_stage2_reg_3seeds.json` | `douban_svdpp_logdet_shrinkage_stage2_reg_3seeds` | Douban log-det regularization screen for top shrinkage/lambda candidates |
| FOLLOW-UP | `douban_svdpp_logdet_shrinkage_best_20seeds.json` | `douban_svdpp_logdet_shrinkage_best_20seeds` | 20-seed confirmation; logdet+shrinkage did not beat no-DLS/current DU-Net |
| FOLLOW-UP | `douban_svdpp_weighted_logdet_reliability_3seeds.json` | `douban_svdpp_weighted_logdet_reliability_3seeds` | Weighted logdet / reliability DLS 3-seed mechanism matrix |
| FOLLOW-UP | `douban_svdpp_reliability_dls_20seeds.json` | `douban_svdpp_reliability_dls_20seeds` | 20-seed confirmation; reliability DLS nearly ties no-DLS and lowers centrality bias |
| FOLLOW-UP | `douban_svdpp_batch_composition_3seeds.json` | `douban_svdpp_batch_composition_3seeds` | Douban reward-frontier batch composition pilot; reliability-DLS 70/30 is the strongest current pilot |
| FOLLOW-UP | `douban_svdpp_batch_composition_20seeds.json` | `douban_svdpp_batch_composition_20seeds` | 20-seed negative confirmation; batch composition raises entropy percentile but does not beat myopic |
| FOLLOW-UP | `douban_svdpp_graph_density_sensitivity_3seeds.json` | `douban_svdpp_graph_density_sensitivity_3seeds` | Douban graph-density sensitivity; rewiring hurts DLS and reliability-DLS stabilizes sparse graph use |
| FOLLOW-UP | `douban_svdpp_network_entropy_embedding_3seeds.json` | `douban_svdpp_network_entropy_embedding_3seeds` | First network-observational entropy and sparse embedding mechanism pass |
| FOLLOW-UP | `douban_svdpp_network_entropy_compact_3seeds.json` | `douban_svdpp_network_entropy_compact_3seeds` | Narrow cached network-observational confirmation for spectral and sparse Node2Vec |
| FOLLOW-UP | `douban_svdpp_network_entropy_spectral_20seeds.json` | `douban_svdpp_network_entropy_spectral_20seeds` | Positive 20-seed Douban confirmation for network-observational entropy with reliability-DLS |
| FOLLOW-UP | `douban_svdpp_formal_mechanism_20seeds.json` | `douban_svdpp_formal_mechanism_20seeds` | Formal Douban mechanism matrix synthesized from matching 20-seed reliability-DLS and netobs runs after full unified rerun exceeded runtime |
| FOLLOW-UP | `douban_svdpp_robustness_netobs_20seeds.json` | `douban_svdpp_robustness_netobs_20seeds` | Douban heterogeneous response alpha=0.02 with netobs reliability-DLS included |
| FOLLOW-UP | `douban_svdpp_ablation_netobs_20seeds.json` | `douban_svdpp_ablation_netobs_20seeds` | Douban DLS/network ablation with netobs reliability-DLS included |
| FOLLOW-UP | `douban_svdpp_baselines_netobs_20seeds.json` | `douban_svdpp_baselines_netobs_20seeds` | Douban baseline matrix with netobs reliability-DLS included |
| FOLLOW-UP | `douban_svdpp_embedding_upgrade_3seeds.json` | `douban_svdpp_embedding_upgrade_3seeds` | Douban upgraded Node2Vec/ELPH proxy check; improved proxies still do not beat spectral |
| FOLLOW-UP | `douban_svdpp_node2vec_pq_grid_3seeds.json` | `douban_svdpp_node2vec_pq_grid_3seeds` | Douban Node2Vec p/q bias grid; BFS-like walks are closest but spectral remains best |
| FOLLOW-UP | `douban_svdpp_response_aware_netobs_3seeds.json` | `douban_svdpp_response_aware_netobs_3seeds` | Response-aware netobs pilot for Douban heterogeneous response |
| FOLLOW-UP | `douban_svdpp_response_aware_netobs_20seeds.json` | `douban_svdpp_response_aware_netobs_20seeds` | Positive 20-seed Douban heterogeneous response confirmation for response-aware netobs |
| FOLLOW-UP | `douban_svdpp_graph_density_final_candidates_3seeds.json` | `douban_svdpp_graph_density_final_candidates_3seeds` | Focused graph-density pilot for no-DLS and netobs reliability final candidates |
| FOLLOW-UP | `douban_svdpp_graph_density_final_candidates_20seeds.json` | `douban_svdpp_graph_density_final_candidates_20seeds` | Completed 20-seed focused graph-density confirmation for Douban final candidates |
| FOLLOW-UP | `douban_svdpp_network_expected_entropy_3seeds.json` | `douban_svdpp_network_expected_entropy_3seeds` | First expected-network observational entropy pilot; same selections as current netobs under this setting |
| FOLLOW-UP | `douban_svdpp_network_batch_entropy_3seeds.json` | `douban_svdpp_network_batch_entropy_3seeds` | Batch-level network information selector; raises entropy percentile and lowers selected degree but does not beat netobs top-k |
| FOLLOW-UP | `douban_svdpp_stochastic_network_entropy_3seeds.json` | `douban_svdpp_stochastic_network_entropy_3seeds` | Stochastic subset log-det entropy pilot; implemented but not promoted because p=0.01 slightly loses and p=0.05 creates centrality bias |
| FOLLOW-UP | `douban_svdpp_reliability_stochastic_entropy_3seeds.json` | `douban_svdpp_reliability_stochastic_entropy_3seeds` | Reliability-calibrated stochastic entropy pilot; negative for promotion, but confirms degree normalization is required |
| FOLLOW-UP | `douban_svdpp_embedding_entropy_bridge_3seeds.json` | `douban_svdpp_embedding_entropy_bridge_3seeds` | Embedding x expected-netobs bridge; expected entropy matches current netobs and spectral remains best |
| FOLLOW-UP | `douban_svdpp_joint_score_gate_3seeds.json` | `douban_svdpp_joint_score_gate_3seeds` | Reward-information joint gate pilot for closer DU-Net acquisition scoring |
| FOLLOW-UP | `douban_svdpp_weighted_network_entropy_3seeds.json` | `douban_svdpp_weighted_network_entropy_3seeds` | Probability-weighted expected network log-det entropy pilot; useful fidelity infrastructure but weaker than current netobs top-k |
| FOLLOW-UP | `douban_svdpp_reward_frontier_guard_3seeds.json` | `douban_svdpp_reward_frontier_guard_3seeds` | Reward-frontier guarded entropy pilot; mechanism-positive but not promoted because current netobs top-k remains best |
| FOLLOW-UP | `douban_svdpp_node2vec_sgns_3seeds.json` | `douban_svdpp_node2vec_sgns_3seeds` | Local SGNS Node2Vec fidelity pilot; positive versus myopic but weaker than spectral/ELPH-role |

## Synthetic

| status | config | purpose |
|---|---|---|
| EXPLORATION | `mvp_synthetic.json` | Minimal synthetic acquisition loop validation |
| EXPLORATION | `ablation_synthetic.json` | Synthetic DU-Net ablations |
| EXPLORATION | `robustness_synthetic.json` | Synthetic heterogeneous response robustness |

## MovieLens

| status | config | purpose |
|---|---|---|
| MAIN/FROZEN | `movielens_100k_svdpp_optimized_20seeds.json` | Main MovieLens homogeneous 20-seed result |
| MAIN/FROZEN | `movielens_100k_svdpp_robustness_20seeds.json` | MovieLens heterogeneous response alpha=0.02 |
| FOLLOW-UP | `movielens_100k_svdpp_baselines_20seeds.json` | MovieLens random/myopic/entropy/bandit/DU-Net comparison |
| FOLLOW-UP | `movielens_100k_svdpp_baselines_missing_20seeds.json` | MovieLens graph_social and Thompson baseline extension under a new prefix |
| FOLLOW-UP | `movielens_100k_svdpp_bandit_upgrade_3seeds.json` | Conservative MF-centered bandit baseline upgrade pilot |
| FOLLOW-UP | `movielens_100k_svdpp_mf_linear_ucb_alpha_grid_3seeds.json` | Focused MF-linear-UCB alpha grid; alpha=0.05 is best but does not beat DU-Net |
| FOLLOW-UP | `movielens_100k_svdpp_residual_bandit_3seeds.json` | Residual MF-bandit pilot; residual correction does not beat MF-linear-UCB |
| FOLLOW-UP | `movielens_100k_svdpp_active_learning_baselines_3seeds.json` | Active-learning baseline pilot; pure uncertainty selection is negative for promotion |
| FOLLOW-UP | `movielens_100k_svdpp_social_rec_baselines_3seeds.json` | Social-rec baseline pilot; damped social hybrid improves over myopic but does not beat DU-Net/graph_social frontier |
| FOLLOW-UP | `movielens_100k_svdpp_dls_ablation_20seeds.json` | MovieLens DLS/network/dynamic sigma ablation |
| EXPLORATION | `movielens_100k_svdpp_lambda_grid.json` | MovieLens lambda diagnostic |
| LEGACY | `movielens_100k_svdpp_optimized_10seeds.json` | Earlier 10-seed stability check |
| LEGACY | `movielens_100k_svdpp_optimized_3seeds.json` | Earlier 3-seed diagnostic |
| LEGACY | `movielens_100k.json` | Early plain-MF MovieLens baseline |
| LEGACY | `movielens_100k_optimized.json` | Early plain-MF tuned MovieLens baseline |

## Flixster

| status | config | purpose |
|---|---|---|
| EXPLORATION | `flixster_svdpp_smoke.json` | GCMC loader smoke test |
| LEGACY | `flixster_svdpp_1seed_20cycles.json` | Untuned single-seed 20-cycle baseline |
| EXPLORATION | `flixster_svdpp_lambda_grid_1seed.json` | Single-seed lambda tuning |
| EXPLORATION | `flixster_svdpp_diffusion_grid_1seed.json` | Single-seed diffusion tuning |
| LEGACY | `flixster_svdpp_tuned_1seed_20cycles.json` | Tuned single-seed comparison |
| LEGACY | `flixster_svdpp_tuned_3seeds_20cycles.json` | Tuned 3-seed stability check |
| MAIN/FROZEN | `flixster_svdpp_tuned_20seeds_20cycles.json` | Main tuned Flixster 20-seed result |
| FOLLOW-UP | `flixster_svdpp_ablation_3seeds.json` | Flixster DLS/network/reward/entropy/dynamic-sigma ablation |
| FOLLOW-UP | `flixster_svdpp_baselines_3seeds.json` | Flixster extended baseline matrix |
| FOLLOW-UP | `flixster_svdpp_robustness_3seeds.json` | Flixster heterogeneous response alpha=0.02 pilot |
| MAIN/FROZEN | `flixster_svdpp_robustness_20seeds.json` | Flixster heterogeneous response alpha=0.02 confirmation |
| MAIN/FROZEN | `flixster_svdpp_ablation_20seeds.json` | Flixster DLS/network/reward/entropy/dynamic-sigma ablation confirmation |
| MAIN/FROZEN | `flixster_svdpp_baselines_20seeds.json` | Flixster implemented baseline matrix confirmation |
| FOLLOW-UP | `flixster_svdpp_oful_3seeds.json` | Paper-specific OFUL benchmark screen; current OFUL is far below DU-Net/graph-social/myopic and should not be promoted |
| FOLLOW-UP | `flixster_svdpp_paper_bandit_repair_3seeds.json` | MF-centered OFUL repair; large improvement over raw OFUL but still below myopic/frontier |
| FOLLOW-UP | `flixster_svdpp_mf_oful_beta_zoom_3seeds.json` | Focused MF-OFUL beta zoom; confirms OFUL bonus remains harmful in this local adaptation |
| FOLLOW-UP | `flixster_svdpp_paper_bandit_gap_3seeds.json` | Logistic and finite-arm benchmark proxy screen; closes explicit catalog rows but not paper-level benchmark strength |
| FOLLOW-UP | `flixster_svdpp_logistic_bandit_scale_3seeds.json` | Logistic UCB/Thompson scale zoom; best rows approach myopic but do not beat the frontier |

## Douban

| status | config | purpose |
|---|---|---|
| EXPLORATION | `douban_svdpp_smoke.json` | GCMC loader smoke test |
| LEGACY | `douban_svdpp_1seed_20cycles.json` | Untuned single-seed 20-cycle baseline |
| EXPLORATION | `douban_svdpp_lambda_grid_1seed.json` | Single-seed lambda tuning |
| EXPLORATION | `douban_svdpp_diffusion_grid_1seed.json` | Single-seed diffusion tuning |
| LEGACY | `douban_svdpp_tuned_1seed_20cycles.json` | Tuned single-seed comparison |
| LEGACY | `douban_svdpp_tuned_3seeds_20cycles.json` | Tuned 3-seed stability check |
| MAIN/FROZEN | `douban_svdpp_tuned_20seeds_20cycles.json` | Main tuned Douban 20-seed result |
| FOLLOW-UP | `douban_svdpp_dynamic_no_dls_20seeds.json` | 20-seed dynamic sigma / no-DLS confirmation |
| FOLLOW-UP | `douban_svdpp_tau_grid_3seeds.json` | Temperature-controlled exponential sampling |
| FOLLOW-UP | `douban_svdpp_entropy_lambda_grid_3seeds.json` | Observational entropy scaling |
| FOLLOW-UP | `douban_svdpp_entropy_coverage_grid_3seeds.json` | Observational-hybrid coverage scaling |
| FOLLOW-UP | `douban_svdpp_mechanism_matrix_3seeds.json` | Log-det entropy and shrinkage-DLS mechanism matrix |
| FOLLOW-UP | `douban_svdpp_logdet_shrinkage_stage1_3seeds.json` | Log-det + shrinkage DLS screen over shrinkage and lambda |
| FOLLOW-UP | `douban_svdpp_logdet_shrinkage_stage2_reg_3seeds.json` | Log-det regularization screen for top stage-1 candidates |
| FOLLOW-UP | `douban_svdpp_logdet_shrinkage_best_20seeds.json` | 20-seed confirmation of best log-det + shrinkage candidate; negative confirmation |
| FOLLOW-UP | `douban_svdpp_weighted_logdet_reliability_3seeds.json` | Weighted logdet and reliability-DLS screen |
| FOLLOW-UP | `douban_svdpp_reliability_dls_20seeds.json` | 20-seed reliability-DLS confirmation |
| FOLLOW-UP | `douban_svdpp_batch_composition_3seeds.json` | Reward-frontier batch composition, 70% reward / 30% information |
| FOLLOW-UP | `douban_svdpp_batch_composition_20seeds.json` | 20-seed batch-composition confirmation; negative for final w-RMSE |
| FOLLOW-UP | `douban_svdpp_graph_density_sensitivity_3seeds.json` | Edge dropout, rewiring, and densification sensitivity |
| FOLLOW-UP | `douban_svdpp_network_entropy_embedding_3seeds.json` | Sparse Node2Vec / ELPH-struct and network-observational entropy pilot |
| FOLLOW-UP | `douban_svdpp_network_entropy_compact_3seeds.json` | Cached narrow network-observational 3-seed confirmation |
| FOLLOW-UP | `douban_svdpp_network_entropy_spectral_20seeds.json` | 20-seed network-observational spectral reliability-DLS confirmation; beats myopic modestly |
| FOLLOW-UP | `douban_svdpp_formal_mechanism_20seeds.json` | Formal 20-seed mechanism matrix across myopic, current DU-Net, no-DLS, reliability-DLS, weighted logdet, and netobs reliability |
| EXPLORATION | `douban_svdpp_selection_grid_3seeds.json` | Douban top-k vs untempered exponential selection tuning |
| EXPLORATION | `douban_svdpp_sigma_grid_3seeds.json` | Douban DLS sigma tuning |
| EXPLORATION | `douban_svdpp_dls_topk_grid_3seeds.json` | Douban DLS top-k tuning |
| EXPLORATION | `douban_svdpp_diffusion_norm_grid_3seeds.json` | Douban degree-normalized diffusion tuning |
| EXPLORATION | `douban_svdpp_entropy_grid_3seeds.json` | Douban observational entropy mode tuning |
| EXPLORATION | `douban_svdpp_embedding_grid_3seeds.json` | Douban spectral vs ELPH-proxy embedding tuning |
| EXPLORATION | `douban_svdpp_rank_grid_3seeds.json` | Douban SVD++ rank tuning |
| EXPLORATION | `douban_svdpp_epochs_grid_3seeds.json` | Douban SVD++ epochs tuning |
| EXPLORATION | `douban_svdpp_epochs6_lambda_grid_3seeds.json` | Douban lambda tuning under epochs=6 |
| EXPLORATION | `douban_svdpp_epochs6_sigma_grid_3seeds.json` | Douban sigma tuning under epochs=6 |
| EXPLORATION | `douban_svdpp_epochs6_entropy_grid_3seeds.json` | Douban entropy tuning under epochs=6 |
| EXPLORATION | `douban_svdpp_tuned_epochs6_lambda10_3seeds.json` | Douban stronger 3-seed setting, lambda=0.10 epochs=6 |
| FOLLOW-UP | `douban_svdpp_tuned_epochs6_lambda10_20seeds.json` | Stronger 20-seed Douban diagnostic; not headline because myopic improves too |
| FOLLOW-UP | `douban_svdpp_ablation_3seeds.json` | Douban DLS/network/reward/entropy/dynamic-sigma ablation |
| EXPLORATION | `douban_svdpp_ablation_epochs6_3seeds.json` | Douban ablation under epochs=6 |
| FOLLOW-UP | `douban_svdpp_baselines_3seeds.json` | Douban extended baseline matrix |
| FOLLOW-UP | `douban_svdpp_robustness_3seeds.json` | Douban heterogeneous response alpha=0.02 pilot |
| FOLLOW-UP | `douban_svdpp_robustness_netobs_20seeds.json` | 20-seed heterogeneous response alpha=0.02 with myopic/current/no-DLS/reliability/netobs |
| FOLLOW-UP | `douban_svdpp_ablation_netobs_20seeds.json` | 20-seed DLS/network ablation with netobs reliability-DLS |
| FOLLOW-UP | `douban_svdpp_baselines_netobs_20seeds.json` | 20-seed baseline matrix with netobs reliability-DLS |
| FOLLOW-UP | `douban_svdpp_embedding_upgrade_3seeds.json` | 3-seed check for biased sparse Node2Vec and ELPH-role embeddings |
| FOLLOW-UP | `douban_svdpp_node2vec_pq_grid_3seeds.json` | 3-seed Node2Vec p/q bias grid; local BFS-like walks are closest but still do not beat spectral |
| FOLLOW-UP | `douban_svdpp_response_aware_netobs_3seeds.json` | 3-seed response-aware netobs pilot under heterogeneous response |
| FOLLOW-UP | `douban_svdpp_response_aware_netobs_20seeds.json` | 20-seed response-aware netobs confirmation under heterogeneous response |
| FOLLOW-UP | `douban_svdpp_graph_density_final_candidates_3seeds.json` | Focused 3-seed graph-density pilot for no-DLS vs netobs reliability final candidates |
| FOLLOW-UP | `douban_svdpp_graph_density_final_candidates_20seeds.json` | Completed 20-seed graph-density confirmation for no-DLS vs netobs reliability final candidates |
| FOLLOW-UP | `douban_svdpp_network_expected_entropy_3seeds.json` | Expected-network observational entropy pilot; tested but not promoted because it matches current netobs selections |
| FOLLOW-UP | `douban_svdpp_network_batch_entropy_3seeds.json` | Network-batch information selection pilot; mechanism-positive but not promoted to 20 seeds |
| FOLLOW-UP | `douban_svdpp_stochastic_network_entropy_3seeds.json` | Stochastic subset log-det entropy pilot; faithful infrastructure, current negative confirmation |
| FOLLOW-UP | `douban_svdpp_reliability_stochastic_entropy_3seeds.json` | Reliability-calibrated stochastic subset log-det pilot; degree-normalized variant partly fixes centrality bias but still loses to netobs top-k |
| FOLLOW-UP | `douban_svdpp_embedding_entropy_bridge_3seeds.json` | Embedding x expected-network entropy bridge; expected entropy does not change selections and spectral remains best |
| FOLLOW-UP | `douban_svdpp_joint_score_gate_3seeds.json` | Reward-information joint gate pilot; tests whether a closer acquisition scoring interaction improves the spectral-netobs candidate |
| FOLLOW-UP | `douban_svdpp_weighted_network_entropy_3seeds.json` | Probability-weighted expected network log-det entropy pilot; tested but not promoted because it does not beat current netobs top-k |
| FOLLOW-UP | `douban_svdpp_reward_frontier_guard_3seeds.json` | Reward-frontier guarded entropy pilot; reduces selected degree and raises entropy percentile but does not beat current netobs top-k |
| FOLLOW-UP | `douban_svdpp_node2vec_sgns_3seeds.json` | Local torch SGNS Node2Vec pilot; implemented but not promoted because spectral and ELPH-role remain stronger |

## New Experimental Knobs

These fields are supported by `AcquisitionConfig` for follow-up experiments:

- `selection`: `topk`, `exp`, `batch_composition`, `network_batch_composition`, or `random`.
- `selection_temperature`: softmax temperature for `selection=exp`; lower values make exponential sampling closer to top-k.
- `batch_composition_reward_fraction`, `batch_composition_entropy_fraction`, `batch_composition_diversity_fraction`: fixed within-batch shares for reward, information, and diversity selection.
- `batch_composition_pool_multiplier`: optional reward-frontier pool size for information/diversity slots; `5.0` means those slots are chosen from the top `5 x batch_size` reward candidates.
- `batch_composition_network_penalty`: redundancy penalty for `network_batch_composition` information/diversity slots.
- `batch_composition_network_scan_multiplier`: top-candidate scan width for network-aware greedy batch selection.
- `diffusion_normalization`: `none`, `degree`, or `sqrt_degree`.
- `graph_embedding_method`: `spectral`, `node2vec`, `sparse_node2vec`, `elph`, or `elph_struct`.
- `graph_embedding_method`: also supports `sparse_node2vec_biased` and `elph_role`.
- `graph_embedding_method`: also supports `sparse_node2vec_bfs`, `sparse_node2vec_dfs`, `sparse_node2vec_return`, and `sparse_node2vec_outward` for fixed Node2Vec p/q bias checks.
- `graph_embedding_method`: also supports `sparse_node2vec_sgns_bfs` and `sparse_node2vec_sgns_dfs` for local torch skip-gram negative-sampling Node2Vec checks.
- `entropy_mode`: `coverage`, `subspace`, `hybrid`, `observational`, or `observational_hybrid`.
- `entropy_mode`: also supports `observational_logdet` for one-step log-det information gain.
- `entropy_mode`: also supports `network_observational` for set-level network log-det information gain.
- `entropy_mode`: also supports `network_observational_expected` for first-order expected same-item neighbor log-det gain.
- `entropy_mode`: also supports `network_observational_stochastic` for Monte Carlo same-item neighbor subset log-det gain.
- `entropy_mode`: also supports `network_observational_stochastic_reliability` for reliability-calibrated Monte Carlo same-item neighbor subset log-det gain.
- `entropy_mode`: also supports `network_observational_weighted` for deterministic probability-weighted expected same-item social-neighbor log-det gain.
- `entropy_logdet_regularization`: ridge term for `observational_logdet`.
- `entropy_logdet_weighting`: `none`, `reward`, `cold`, or `reward_cold`.
- `entropy_logdet_weight_strength`: multiplier for weighted log-det candidate weights.
- `entropy_response_weighting`: `none`, `expected_response`, or `centered_expected_response`; discounts information value by predicted response probability when `response_alpha > 0`.
- `entropy_response_weight_strength`: exponent/strength for response-aware entropy weighting.
- `dls_shrinkage`: adaptive max shrinkage weight for sparse-graph DLS; `0.0` keeps the original full DLS behavior.
- `dls_shrinkage_mode`: `adaptive` or `reliability`.
- `network_entropy_max_neighbors`: maximum same-item social-neighbor candidates included in `network_observational` set gain.
- `network_entropy_stochastic_samples`: Monte Carlo subset count for `network_observational_stochastic`.
- `network_entropy_reliability_strength`: reliability damping strength for `network_observational_stochastic_reliability` and `network_observational_weighted`.
- `score_interaction_weight`: optional DU-Net score interaction term; when positive, boosts candidates that are jointly above-average in reward and entropy. The default `0.0` preserves the existing linear score.
- `entropy_reward_frontier_quantile`: optional reward-percentile guard for positive entropy bonuses. The default `0.0` disables the guard.
- `entropy_reward_frontier_floor`: residual positive-entropy bonus multiplier outside the reward frontier.
- `entropy_reward_frontier_softness`: optional logistic smoothing around the reward-frontier threshold.
- `policy`: also supports conservative MF-centered bandit baselines `mf_linear_ucb`, `mf_low_rank_ucb`, and `mf_thompson_sampling`.
- `policy`: also supports residual MF-bandit baselines `mf_residual_ucb` and `mf_residual_thompson_sampling`.
- `residual_bandit_weight`: residual correction strength for residual MF-bandit policies.
- `policy`: also supports `oful` as a separate ridge-linear optimistic bandit benchmark.
- `policy`: also supports `mf_oful`, which uses SVD++ predicted reward as the mean and an OFUL confidence bonus for exploration.
- `oful_delta`, `oful_noise_scale`, `oful_param_bound`, and `oful_beta_scale`: confidence-radius controls for the OFUL benchmark screen.
- `policy`: also supports paper-benchmark proxy rows `logistic_ucb`, `logistic_thompson_sampling`, `finite_ucb`, `epsilon_greedy`, `adaptive_greedy`, and `explore_then_exploit`.
- `logistic_bandit_temperature`: scales the logistic preference transform for logistic bandit proxy rows.
- `finite_bandit_epsilon`, `finite_bandit_explore_fraction`, and `adaptive_greedy_min_epsilon`: exploration controls for finite-arm proxy rows.
- `policy`: also supports active-learning baselines `prediction_uncertainty`, `leverage_uncertainty`, and `active_learning_hybrid`.
- `policy`: also supports social-rec baselines `social_neighbor_item` and `social_mf_hybrid`; `social_mf_hybrid` uses `residual_bandit_weight` as the social correction strength.

## Mechanism Diagnostics

| artifact | purpose |
|---|---|
| `experiments/plot_mechanism_diagnostics.py` | Produces cold/warm, centrality, selected-degree, DLS-shrinkage, and reward-vs-entropy diagnostic figures from metrics files. |
| `reports/douban_svdpp_reliability_dls_20seeds_mechanism_diagnostics.md` | Formal Douban mechanism diagnosis for no-DLS, reliability-DLS, weighted-logdet reliability-DLS, current DU-Net, and myopic. |
| `reports/douban_formal_ablation_batch_density.md` | Formal Douban ablation note covering no-DLS, reliability-DLS, reward-frontier batch composition, and graph-density sensitivity. |
| `reports/douban_network_entropy_embedding_findings.md` | First Douban network-observational entropy and sparse embedding mechanism findings. |
| `reports/douban_node2vec_pq_grid_findings.md` | Douban Node2Vec p/q bias grid; local BFS-like walks are closest but spectral remains best. |
| `reports/douban_response_aware_netobs_findings.md` | Response-aware network-observational entropy for Douban heterogeneous response. |
| `reports/douban_expected_network_entropy_findings.md` | First expected-network observational entropy implementation and Douban 3-seed pilot. |
| `reports/douban_network_batch_entropy_findings.md` | First batch-level network information selection pilot on Douban. |
| `reports/douban_stochastic_network_entropy_findings.md` | First stochastic subset log-det network entropy pilot on Douban. |
| `reports/douban_reliability_stochastic_entropy_findings.md` | Reliability-calibrated stochastic network entropy pilot; confirms degree normalization is necessary but not sufficient. |
| `reports/douban_embedding_entropy_bridge_findings.md` | Embedding x expected-network entropy bridge; expected-netobs is not promoted and spectral remains the best homogeneous Douban candidate. |
| `reports/douban_joint_score_gate_findings.md` | Reward-information joint gate pilot; useful diagnostic but not promoted because it worsens w-RMSE as information pressure increases. |
| `reports/douban_weighted_network_entropy_findings.md` | Probability-weighted expected network log-det entropy pilot; useful infrastructure but not promoted because current netobs top-k remains best. |
| `reports/douban_reward_frontier_guard_findings.md` | Reward-frontier guarded entropy pilot; mechanism-positive but not promoted because current netobs top-k remains best. |
| `reports/douban_node2vec_sgns_findings.md` | Local SGNS Node2Vec fidelity pilot; positive against myopic but not promoted because spectral and ELPH-role remain stronger. |
| `reports/movielens_baseline_extension_findings.md` | MovieLens baseline extension; graph_social wins reward but DU-Net remains best implemented baseline by w-RMSE. |
| `reports/movielens_bandit_upgrade_findings.md` | MovieLens conservative bandit upgrade pilot; MF-centered bandits fix reward collapse but do not beat DU-Net yet. |
| `reports/movielens_active_learning_baseline_findings.md` | MovieLens active-learning baseline pilot; pure uncertainty selection is negative for promotion. |
| `reports/movielens_social_rec_baseline_findings.md` | MovieLens social-rec baseline pilot; moderate social hybrid helps but does not beat DU-Net/graph_social. |
| `reports/long_term_reproduction_roadmap.md` | Roadmap for batch composition, sparse graph embedding, graph-density sensitivity, and final paper-quality rerun. |
- `dls_top_k`, `sigma`, `sigma_grid`.
