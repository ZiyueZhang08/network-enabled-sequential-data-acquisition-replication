# paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1

Frozen stochastic main candidate for MovieLens Figure 4a clean-room reproduction.

## Config

- selection: full-pool softmax (`selection='exp'`)
- tau: 0.3
- entropy: Schur conditional variance
- dynamic tuning: log-lambda grid, standard RMSE inner validation
- seeds: 0-19

## Final Result

                                       variant  seed_count  standard_mask_rmse  standard_mask_rmse_ci95  pdf_literal_norm_rmse  pdf_literal_norm_rmse_ci95  mild_reward_w_rmse  mild_reward_w_rmse_ci95  medium_reward_w_rmse  medium_reward_w_rmse_ci95  sigmoid_reward_w_rmse  sigmoid_reward_w_rmse_ci95   w_rmse  w_rmse_ci95  reward_tail_proxy_rmse  reward_tail_proxy_rmse_ci95  cumulative_reward  cumulative_reward_ci95  centered_cumulative_reward  centered_cumulative_reward_ci95
                                 linucb_concat          20            1.106389                 0.005067               0.007823                    0.000036            1.063757                 0.003287              1.039070                   0.004083               0.931942                    0.004195 0.965524     0.010247                1.216032                     0.015250        1393.680246                8.916846                     1949.10                        45.836549
                         linucb_mf_interaction          20            1.087422                 0.003468               0.007689                    0.000025            1.075164                 0.003509              1.068223                   0.004127               0.964855                    0.004237 1.048264     0.007036                1.336719                     0.009882        1206.370096               16.657837                     1005.65                        84.490763
                              schur_tuned_topk          20            1.088861                 0.004436               0.007699                    0.000031            1.073754                 0.004209              1.065152                   0.005964               0.959500                    0.006321 1.040197     0.012977                1.326710                     0.018056        1250.556709               33.057971                     1219.85                       170.956279
schur_loglambda_tuned_full_pool_softmax_tau0p3          20            1.070025                 0.003458               0.007566                    0.000024            1.057036                 0.002946              1.049671                   0.003755               0.950392                    0.004164 1.028445     0.007719                1.308482                     0.010809        1245.701252               14.986018                     1188.70                        77.525384

## Paired 20-Seed Gaps

             baseline  mean_standard_gap  standard_win_rate  wins  seeds  mean_pdf_gap  mean_reward_gap
        linucb_concat           0.036364               1.00    20     20      0.000257      -147.978994
     schur_tuned_topk           0.018836               1.00    20     20      0.000133        -4.855456
linucb_mf_interaction           0.017397               0.95    19     20      0.000123        39.331156

Softmax standard RMSE is 1.070025. It beats top-k by 0.018836 (20/20), concat by 0.036364 (20/20), and mf_interaction by 0.017397 (19/20).

## Per-Cycle Gap Summary

             baseline  positive_cycles  cycles  mean_gap  final_gap
        linucb_concat               21      21  0.018604   0.036364
linucb_mf_interaction               21      21  0.009976   0.017397
     schur_tuned_topk               21      21  0.011285   0.018836

## Metric Sensitivity

                metric  softmax  schur_tuned_topk  gap_schur_tuned_topk_minus_softmax  relative_improvement_vs_schur_tuned_topk  linucb_concat  gap_linucb_concat_minus_softmax  relative_improvement_vs_linucb_concat  linucb_mf_interaction  gap_linucb_mf_interaction_minus_softmax  relative_improvement_vs_linucb_mf_interaction
    standard_mask_rmse 1.070025          1.088861                            0.018836                                  0.017299       1.106389                         0.036364                               0.032867               1.087422                                 0.017397                                       0.015998
 pdf_literal_norm_rmse 0.007566          0.007699                            0.000133                                  0.017299       0.007823                         0.000257                               0.032867               0.007689                                 0.000123                                       0.015998
    mild_reward_w_rmse 1.057036          1.073754                            0.016718                                  0.015570       1.063757                         0.006720                               0.006318               1.075164                                 0.018127                                       0.016860
  medium_reward_w_rmse 1.049671          1.065152                            0.015481                                  0.014534       1.039070                        -0.010601                              -0.010203               1.068223                                 0.018552                                       0.017367
 sigmoid_reward_w_rmse 0.950392          0.959500                            0.009109                                  0.009493       0.931942                        -0.018450                              -0.019797               0.964855                                 0.014464                                       0.014991
                w_rmse 1.028445          1.040197                            0.011753                                  0.011298       0.965524                        -0.062920                              -0.065167               1.048264                                 0.019819                                       0.018906
reward_tail_proxy_rmse 1.308482          1.326710                            0.018228                                  0.013739       1.216032                        -0.092450                              -0.076026               1.336719                                 0.028237                                       0.021124

The stochastic softmax result is strongest on standard/PDF-literal RMSE. Reward-weighted metrics still favor high-reward Linear UCB variants, so they remain diagnostic rather than the Figure 4a main claim.

## Positioning

Keep deterministic top-k v1 as the frozen deterministic approximation. Promote this softmax tau=0.3 run as the stochastic paper-style main candidate because it uses the same Schur/log-lambda core and only switches batch selection to the paper-style sampling family.

Keep lambda=0.3 as the balanced reward/RMSE diagnostic: entropy-dominant softmax best matches Figure 4a, while lambda=0.3 explains the reward trade-off.

## Artifact Index

- `configs\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1.json`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_metrics.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_final_summary.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_paired_gaps.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_gap_summary.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_per_cycle.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_per_cycle_gaps.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_metric_sensitivity.csv`
- `outputs\tables\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_selection_logs.csv`
- `outputs\logs\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_pytest_core.txt`
- `outputs\figures\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_standard_rmse_curve.png`
- `outputs\figures\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_reward_rmse_tradeoff.png`
