# Final Clean-Room Reproduction Report: paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1

## 1. Objective

This package freezes a clean-room reproduction of the MovieLens Figure 4a-style result under unavailable author source code, exact Monti split/mask, exact Linear UCB implementation, and exact w-RMSE convention.

The intended claim is strong Figure 4a-style RMSE recovery, not exact author-result replication.

## 2. Paper Protocol Target

- MovieLens matrix: 943 users x 1681 items.
- Initial density target: 0.086%, implemented as 1364 initial observations.
- Test density target: 1.26%, implemented through the MovieLens u1 fallback test fold.
- 20 acquisition cycles.
- Batch size 100.
- 20 random initializations.
- DU-Net-style acquisition combines reward and information value.
- Information value is implemented through Schur conditional-variance observational entropy.
- Batch selection uses full-pool exponential weighting, reported here as tau=0.3 main and tau=1 literal diagnostic.

## 3. Implementation Evolution

The reproduction path moved from high-level proxies to the paper-core computation path:

old proxy -> Gaussian DLS -> Schur entropy -> log-lambda tuning -> full-pool softmax

The final result is not claimed on reward-tail metric tuning or non-paper heuristics. The main winning configuration uses the recovered Schur/log-lambda core and a stochastic softmax selection policy.

## 4. Main Result: Figure 4a-Style RMSE

Frozen config: `paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1`

- Softmax standard RMSE: 1.070025
- LinUCB concat standard RMSE: 1.106389
- Gap vs concat: +0.036364
- Win rate vs concat: 20/20
- Gap vs mf-interaction: +0.017397
- Win rate vs mf-interaction: 19/20
- Gap vs deterministic top-k: +0.018836
- Win rate vs deterministic top-k: 20/20

Final summary:

| variant | seed_count | standard_mask_rmse | standard_mask_rmse_ci95 | pdf_literal_norm_rmse | pdf_literal_norm_rmse_ci95 | mild_reward_w_rmse | mild_reward_w_rmse_ci95 | medium_reward_w_rmse | medium_reward_w_rmse_ci95 | sigmoid_reward_w_rmse | sigmoid_reward_w_rmse_ci95 | w_rmse | w_rmse_ci95 | reward_tail_proxy_rmse | reward_tail_proxy_rmse_ci95 | cumulative_reward | cumulative_reward_ci95 | centered_cumulative_reward | centered_cumulative_reward_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linucb_bias_interaction | 20 | 1.13033 | 0.00441287 | 0.00799265 | 3.12037e-05 | 1.07363 | 0.00247879 | 1.04049 | 0.0026596 | 0.928794 | 0.00288801 | 0.940149 | 0.00751323 | 1.17896 | 0.0113357 | 1521.65 | 4.57647 | 2622.55 | 23.9174 |
| linucb_mf_interaction | 20 | 1.08742 | 0.00346816 | 0.00768923 | 2.45236e-05 | 1.07516 | 0.00350894 | 1.06822 | 0.0041269 | 0.964855 | 0.00423732 | 1.04826 | 0.00703642 | 1.33672 | 0.00988172 | 1206.37 | 16.6578 | 1005.65 | 84.4908 |
| schur_tuned_topk | 20 | 1.08886 | 0.00443579 | 0.00769941 | 3.13657e-05 | 1.07375 | 0.00420863 | 1.06515 | 0.00596376 | 0.9595 | 0.00632057 | 1.0402 | 0.0129773 | 1.32671 | 0.0180558 | 1250.56 | 33.058 | 1219.85 | 170.956 |
| linucb_concat | 20 | 1.10639 | 0.00506687 | 0.00782335 | 3.58282e-05 | 1.06376 | 0.00328681 | 1.03907 | 0.0040832 | 0.931942 | 0.00419489 | 0.965524 | 0.0102467 | 1.21603 | 0.0152499 | 1393.68 | 8.91685 | 1949.1 | 45.8365 |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | 20 | 1.07003 | 0.00345767 | 0.00756622 | 2.44494e-05 | 1.05704 | 0.00294611 | 1.04967 | 0.00375452 | 0.950392 | 0.00416359 | 1.02844 | 0.0077191 | 1.30848 | 0.0108085 | 1245.7 | 14.986 | 1188.7 | 77.5254 |

Per-cycle gap summary:

| baseline | positive_cycles | cycles | early_cycles_0_5_avg_gap | middle_cycles_6_15_avg_gap | late_cycles_16_20_avg_gap | final_gap | max_gap | min_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| schur_tuned_topk | 21 | 21 | 0.00422272 | 0.0124578 | 0.0174142 | 0.0188363 | 0.0188363 | 2.90075e-05 |
| linucb_concat | 21 | 21 | 0.00575268 | 0.019234 | 0.0327656 | 0.036364 | 0.036364 | 0.00283252 |
| linucb_mf_interaction | 21 | 21 | 0.00448753 | 0.0101722 | 0.0161695 | 0.0173968 | 0.0173968 | 0.00283252 |
| linucb_bias_interaction | 21 | 21 | 0.0106222 | 0.0369698 | 0.0561929 | 0.0603066 | 0.0603066 | 0.00283252 |

The curve result is the strongest evidence: softmax is lower than concat, mf-interaction, and top-k in every acquisition cycle where the variants overlap.

## 5. Same-Config Figure 4b-Style Reward Result

The same softmax config is reported for reward. No second config is spliced in to mimic Figure 4b.

| metric | softmax_dunet_style | linucb_concat | gap | interpretation |
| --- | --- | --- | --- | --- |
| standard RMSE | 1.07003 | 1.10639 | 0.036364 | strong Figure 4a recovery |
| cumulative reward | 1245.7 | 1393.68 | -147.979 | reward-side caveat vs concat |
| RMSE vs mf-interaction | 1.07003 | 1.08742 | 0.0173968 | strongest clean-room comparator reduced |
| reward vs mf-interaction | 1245.7 | 1206.37 | 39.3312 | favorable against mf-interaction |
| win rate vs concat | 20 | 20 | 1 | 20/20 seed wins in standard RMSE |

Interpretation: the same config strongly recovers the Figure 4a-style RMSE ranking, but it does not fully recover the reward-side Figure 4b result relative to high-reward linucb_concat. Against mf-interaction, it is favorable on both RMSE and reward.

## 6. Reward-RMSE Trade-off Diagnostic

| config | standard_mask_rmse | cumulative_reward | rmse_gap_vs_concat | reward_gap_vs_concat | use |
| --- | --- | --- | --- | --- | --- |
| softmax_tau0p3 | 1.07003 | 1245.7 | 0.036364 | -147.979 | Figure 4a main |
| schur_lambda_0p3 | 1.09196 | 1353.19 | 0.0144331 | -40.4866 | balanced diagnostic |
| schur_lambda_0 | 1.12682 | 1547.58 | -0.0204328 | 153.895 | reward-heavy diagnostic |
| schur_lambda_0p1 | 1.12218 | 1535.95 | -0.0157945 | 142.274 | reward-heavy diagnostic |
| schur_lambda_30 | 1.08246 | 1231.86 | 0.0239304 | -161.82 | entropy-heavy diagnostic |
| schur_entropy_only | 1.08222 | 1233.01 | 0.0241693 | -160.665 | entropy-only diagnostic |

The fixed lambda=0.3 diagnostic has RMSE gap +0.014433 and reward gap -40.49 vs concat. This supports the interpretation that reward loss is not inherent to DU-Net-style acquisition; it reflects using an entropy-dominant Figure 4a-oriented main config.

## 7. Comparator Sensitivity

| baseline | mean_standard_gap | standard_win_rate | wins | seeds | mean_pdf_gap | mean_reward_gap |
| --- | --- | --- | --- | --- | --- | --- |
| linucb_bias_interaction | 0.0603066 | 1 | 20 | 20 | 0.000426432 | -275.945 |
| linucb_concat | 0.036364 | 1 | 20 | 20 | 0.000257133 | -147.979 |
| schur_tuned_topk | 0.0188363 | 1 | 20 | 20 | 0.000133193 | -4.85546 |
| linucb_mf_interaction | 0.0173968 | 0.95 | 19 | 20 | 0.000123014 | 39.3312 |

Earlier, mf_interaction was the main comparator caveat. Under paper-style full-pool softmax selection, the DU-Net-style policy now also beats that strongest clean-room Linear UCB proxy in RMSE.

## 8. Metric Sensitivity

| metric | softmax | schur_tuned_topk | gap_schur_tuned_topk_minus_softmax | relative_improvement_vs_schur_tuned_topk | linucb_concat | gap_linucb_concat_minus_softmax | relative_improvement_vs_linucb_concat | linucb_mf_interaction | gap_linucb_mf_interaction_minus_softmax | relative_improvement_vs_linucb_mf_interaction | linucb_bias_interaction | gap_linucb_bias_interaction_minus_softmax | relative_improvement_vs_linucb_bias_interaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_mask_rmse | 1.07003 | 1.08886 | 0.0188363 | 0.0172991 | 1.10639 | 0.036364 | 0.0328673 | 1.08742 | 0.0173968 | 0.0159982 | 1.13033 | 0.0603066 | 0.053353 |
| pdf_literal_norm_rmse | 0.00756622 | 0.00769941 | 0.000133193 | 0.0172991 | 0.00782335 | 0.000257133 | 0.0328673 | 0.00768923 | 0.000123014 | 0.0159982 | 0.00799265 | 0.000426432 | 0.053353 |
| mild_reward_w_rmse | 1.05704 | 1.07375 | 0.016718 | 0.0155697 | 1.06376 | 0.00672042 | 0.00631763 | 1.07516 | 0.0181272 | 0.01686 | 1.07363 | 0.0165943 | 0.0154563 |
| medium_reward_w_rmse | 1.04967 | 1.06515 | 0.0154808 | 0.0145339 | 1.03907 | -0.0106012 | -0.0102026 | 1.06822 | 0.0185518 | 0.017367 | 1.04049 | -0.00918233 | -0.00882502 |
| sigmoid_reward_w_rmse | 0.950392 | 0.9595 | 0.00910854 | 0.009493 | 0.931942 | -0.0184496 | -0.019797 | 0.964855 | 0.0144638 | 0.0149907 | 0.928794 | -0.0215974 | -0.0232531 |
| w_rmse | 1.02844 | 1.0402 | 0.0117526 | 0.0112984 | 0.965524 | -0.0629201 | -0.0651668 | 1.04826 | 0.0198189 | 0.0189065 | 0.940149 | -0.0882954 | -0.0939164 |
| reward_tail_proxy_rmse | 1.30848 | 1.32671 | 0.018228 | 0.0137393 | 1.21603 | -0.0924495 | -0.0760255 | 1.33672 | 0.028237 | 0.0211241 | 1.17896 | -0.129525 | -0.109864 |
| cumulative_reward | 1245.7 | 1250.56 | 4.85546 | 0.00388264 | 1393.68 | 147.979 | 0.106179 | 1206.37 | -39.3312 | -0.0326029 | 1521.65 | 275.945 | 0.181346 |
| centered_cumulative_reward | 1188.7 | 1219.85 | 31.15 | 0.0255359 | 1949.1 | 760.4 | 0.390129 | 1005.65 | -183.05 | -0.182022 | 2622.55 | 1433.85 | 0.546739 |

Since the paper's w-RMSE formula and wording leave ambiguity about exact weighting and normalization, standard_mask_rmse is reported as the main executable metric. PDF-literal and reward-weighted variants are diagnostics rather than exact w-RMSE claims.

## 9. Literal Softmax Diagnostic

| selection | tau | seed_count | standard_mask_rmse | cumulative_reward | rmse_gap_vs_concat | reward_gap_vs_concat | concat_reference | collapse | role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top-k | n/a | 20 | 1.08886 | 1250.56 | 0.0175277 | -143.124 | 20-seed mean | no | deterministic ablation |
| full-pool softmax | 0.3 | 20 | 1.07003 | 1245.7 | 0.036364 | -147.979 | 20-seed mean | no | stochastic main |
| full-pool softmax | 1.0 | 3 | 1.07228 | 1245.94 | 0.0263143 | -140.559 | seed-matched 0-2 | no | literal-softmax diagnostic |

The paper specifies exponential weighting but does not provide a temperature or score-scaling convention. Tau=0.3 is therefore reported as a calibrated full-pool softmax implementation, with tau=1 included as a literal-softmax diagnostic. The 3-seed tau=1 run did not collapse.

## 10. Remaining Caveats

1. Exact w-RMSE remains unresolved.
2. Exact Monti/author split and mask remain unresolved.
3. Exact Linear UCB protocol remains unresolved.
4. Figure 4b reward side is not fully recovered under the same main config relative to linucb_concat.

## 11. Final Conclusion

The final full-pool softmax Schur-DU-Net clean-room implementation strongly reproduces the qualitative Figure 4a pattern and achieves a DU-Net-vs-Linear-UCB gap comparable to or larger than the visually estimated paper gap under the local standard-RMSE protocol. It remains a clean-room partial reproduction rather than an exact replication because the original metric, split/mask, and benchmark implementation are unavailable.

## 12. Artifact Manifest

| artifact_path | description | sha256 |
| --- | --- | --- |
| configs/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1.json | Frozen 20-seed stochastic softmax main config | 6375b70500c1d1d61865a3558c1cce18b0247aac99b56b592a148f206d9d74ed |
| configs/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds.json | Literal tau=1 softmax 3-seed diagnostic config | 2aa082afeaca84f33b0ca8b3b2407a6437958d46453668f54b61d9bd4353b18d |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_metrics.csv | Raw per-cycle frozen softmax metrics | 7de41181c832f794eff7ede5158d31d01099e756a1943d89b1fe85bd1c268369 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_final_summary.csv | Final 20-seed summary for softmax and comparators | beb2a0e2c2e0dd686f78a41bc00036f458c6634a400454486aa47d3e45902c0e |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_gap_summary.csv | Paired gap summary | 8125e1df574cfc4c93835863d7412e62f67e9e6fa0cfeede077b6076ce7b4522 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_per_seed_gaps.csv | Per-seed paired gaps | a99b17e09229a98fff3555952e172030be6403925c4ac0cdd4c34f53f9e534e8 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_per_cycle_rmse_curve.csv | Per-cycle mean RMSE/reward curves with CI columns | 736a9a2e999c40c0faed381ed8ccfc9c2047be9f31464056e34647335c6e6c19 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_per_cycle_gaps.csv | Per-cycle paired gaps | 5471cc101d201a6475f1ec3788f6e028b5ae56bd260c148d505804414add6c33 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_per_cycle_gap_summary.csv | Early/middle/late cycle gap summary | 5f91d104544d27456521e2b4995b99c74ee42a59671a3da14e4383722db15c1f |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_selected_batch_diagnostics.csv | Per-seed selected-batch diagnostics | edd78582f80fe8dae6f5aedb043c682d1f59820b54f4332e776ca56777458fe2 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_selected_batch_diagnostics_summary.csv | Per-cycle selected-batch diagnostic summary | 24b875759c2ea4a12dd715efd90f167d7cff9692e0c35e8fc1a3e25fc2a611d9 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_lambda_sigma_logs.csv | Dynamic lambda/sigma logs | 518f73b4dca94b5856833c90478bce758deb40e81119a3c2d01ee6844aa5a327 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_metric_sensitivity.csv | Metric sensitivity table | 57e9272fc4defc4097b5d8bdc95c7d2ec1ddaa64fcc8acde242a6ce88d621a69 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_one_config_figure4a_figure4b.csv | Same-config Figure 4a/Figure 4b table | 23bb3ec28c3940b7d954ec63263ce677aae521efe07fb4ae1f861eeceb26092b |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_reward_rmse_tradeoff_diagnostic.csv | Reward-RMSE trade-off diagnostic including lambda=0.3 | 8dc49e520bdf47729646a342a78521a499b6ccb611318ff08ef3e8d616c4015b |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_literal_softmax_tau1_diagnostic.csv | Tau=1 literal-softmax diagnostic | 7dddb01637d39dcb8b9678977ae2a66c8bd72e143cdf3bcdceefb91e009dd6d9 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_comparator_sensitivity.csv | Comparator sensitivity summary | 8125e1df574cfc4c93835863d7412e62f67e9e6fa0cfeede077b6076ce7b4522 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds_metrics.csv | Raw tau=1 literal-softmax metrics | 6027bceb624148707bf098cdace3fef95ada1d922153974263f11145384d097b |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds_final_summary.csv | Tau=1 literal-softmax final summary | 5e116cc2efd3b462a3fe0f8795666ff761c3cfd6b96f95a245995a94cd016d0f |
| outputs/figures/figure4a_rmse_curve_softmax_vs_linucb_concat.png | Figure 4a-style curve vs linucb_concat | 4361e2ba7cc9a3a8d6f99496e32dbb814c2efb3dae8a0a9efb4352ed9685c2dd |
| outputs/figures/figure4a_rmse_curve_softmax_vs_mf_interaction.png | Figure 4a-style curve vs linucb_mf_interaction | ee1f6a31c3b89c75410043c9aab81d8b940d7771304e1bbd0f422cf389b734a8 |
| outputs/figures/selection_policy_ablation_topk_vs_softmax.png | Selection policy ablation curve | 7d4fe20d9008f8e820112f978e8fb196d69807fc891446acbb1f4267d7da5d8c |
| outputs/figures/reward_rmse_pareto_lambda_tradeoff.png | Reward-RMSE Pareto/trade-off diagnostic | 287c3fe318642fa71492f81c4a193d259a592317ef93d203b85c4f5868b1d55b |
| outputs/logs/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_pytest_core.txt | Core pytest log | 1c90a1bb6ef1af21ab19eaf422ffde93a61ab6f564a06b14af85b614e4c27883 |
| outputs/logs/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_run_command_log.txt | Run command log | d0cd1aa88fee9a4ba834638f9b0566ebad5600b199ba86fa9e3a252a34b3235f |

