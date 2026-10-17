# paper_faithful_movielens_gap_investigation_v1

## 1. Main Figure 4a frozen result

- Mainline remains `paper_faithful_movielens_schur_loglambda_tuned_topk_v1`.
- 20-seed standard RMSE gap vs `linucb_concat`: 0.017528; win-rate 19/20; reward gap -143.12.
- Per-cycle frozen curve stayed positive on 21/21 cycles against `linucb_concat`.

## 2. Reward trade-off / lambda=0.3 20-seed result

                             method  seed_count  standard_mask_rmse  rmse_gap_vs_linucb_concat  standard_rmse_win_rate_vs_linucb_concat  cumulative_reward  reward_gap_vs_linucb_concat
schur_loglambda_tuned_standard_topk          20            1.088861                   0.017528                                     0.95        1250.556709                  -143.123538
                     schur_lambda_0          20            1.126822                  -0.020433                                     0.00        1547.575358                   153.895112
                   schur_lambda_0p1          20            1.122184                  -0.015794                                     0.05        1535.953951                   142.273705
                   schur_lambda_0p3          20            1.091956                   0.014433                                     0.95        1353.193679                   -40.486568
                   schur_lambda_0p5          20            1.085949                   0.020440                                     1.00        1284.759752                  -108.920494
                     schur_lambda_1          20            1.084353                   0.022036                                     1.00        1255.556781                  -138.123466
                     schur_lambda_3          20            1.083094                   0.023295                                     1.00        1241.063406                  -152.616841
                    schur_lambda_30          20            1.082459                   0.023930                                     1.00        1231.859943                  -161.820303
                 schur_entropy_only          20            1.082220                   0.024169                                     1.00        1233.014982                  -160.665264
                      linucb_concat          20            1.106389                   0.000000                                     0.00        1393.680246                     0.000000

`lambda=0.3` keeps a stable RMSE win (95%) and cuts reward loss from -143.12 to -40.49. It is a balanced diagnostic, not a replacement for the Figure 4a mainline because its RMSE (1.091956) is worse than entropy-dominant fixed lambda and the frozen tuned mainline.

## 3. Comparator sensitivity 20-seed result

             comparator  dunet_rmse  comparator_rmse  rmse_gap_comparator_minus_dunet  dunet_reward  comparator_reward  reward_gap_dunet_minus_comparator  seed_win_rate  wins  seeds
          linucb_concat    1.088861         1.106389                         0.017528   1250.556709        1393.680246                        -143.123538           0.95    19     20
linucb_bias_interaction    1.088861         1.130332                         0.041470   1250.556709        1521.645951                        -271.089243           1.00    20     20
  linucb_mf_interaction    1.088861         1.087422                        -0.001439   1250.556709        1206.370096                          44.186612           0.55    11     20

`linucb_mf_interaction` remains a real caveat: mean RMSE gap comparator minus DU-Net is -0.001439, with DU-Net wins 11/20. It should be labeled the strongest clean-room Linear UCB proxy, not dismissed as 3-seed noise.

## 4. Fold sensitivity result

fold            comparator  dunet_rmse  comparator_rmse       gap  reward_gap  win_rate  seeds
  u1         linucb_concat    1.087699         1.102384  0.014685 -133.542224  1.000000      3
  u1 linucb_mf_interaction    1.087699         1.081287 -0.006412   37.997865  0.333333      3
  u2         linucb_concat    1.072843         1.083085  0.010242  -87.251921  1.000000      3
  u2 linucb_mf_interaction    1.072843         1.068653 -0.004189   77.370093  0.333333      3
  u3         linucb_concat    1.070688         1.077344  0.006656  -35.317168  0.666667      3
  u3 linucb_mf_interaction    1.070688         1.064191 -0.006498  149.882484  0.000000      3
  u4         linucb_concat    1.056182         1.076863  0.020680 -172.121841  1.000000      3
  u4 linucb_mf_interaction    1.056182         1.062118  0.005936   18.355147  0.666667      3
  u5         linucb_concat    1.070905         1.088786  0.017881 -114.072441  1.000000      3
  u5 linucb_mf_interaction    1.070905         1.065973 -0.004933  121.966357  0.333333      3

Against concat, DU-Net is positive on 5/5 folds. Against mf_interaction, it is positive on 1/5 folds. Split/fold changes preserve the concat Figure 4a pattern but do not remove the mf comparator caveat.

## 5. Metric sensitivity result

                metric    dunet  linucb_concat  gap_linucb_minus_dunet  relative_improvement  abs_gap_distance_to_visual_0p026
  medium_reward_w_rmse 1.065152       1.039070               -0.026082             -0.025101                          0.000082
 sigmoid_reward_w_rmse 0.959500       0.931942               -0.027558             -0.029571                          0.001558
    standard_mask_rmse 1.088861       1.106389                0.017528              0.015842                          0.008472
    mild_reward_w_rmse 1.073754       1.063757               -0.009998             -0.009398                          0.016002
 pdf_literal_norm_rmse 0.007699       0.007823                0.000124              0.015842                          0.025876
                w_rmse 1.040197       0.965524               -0.074673             -0.077339                          0.048673
reward_tail_proxy_rmse 1.326710       1.216032               -0.110678             -0.091015                          0.084678

The metric whose absolute gap is closest to the rough 0.026 visual target is `medium_reward_w_rmse` with gap -0.026082. Metrics involving reward weighting reverse or change the interpretation, so reward-tail proxy remains diagnostic only.

## 6. Selection policy diagnostic

                             variant  standard_mask_rmse  cumulative_reward  selected_avg_rating  unique_users  unique_items  selected_entropy_quantile  selected_reward_quantile collapse         selection  tau
schur_tuned_full_pool_softmax_tau0p1            1.068814        1221.039403             3.532167     90.366667     91.616667                   0.510168                  0.501870       no full_pool_softmax  0.1
schur_tuned_full_pool_softmax_tau0p3            1.064319        1214.567546             3.513000     89.683333     91.333333                   0.516725                  0.500970       no full_pool_softmax  0.3
schur_tuned_full_pool_softmax_tau1p0            1.072284        1245.938345             3.595667     86.966667     88.416667                   0.534183                  0.540737       no full_pool_softmax  1.0
     schur_tuned_topm_softmax_tau0p1            1.084651        1299.172916             3.735500     58.083333     68.366667                   0.871518                  0.623904       no      topM_softmax  0.1
     schur_tuned_topm_softmax_tau0p3            1.087327        1311.820540             3.768500     61.066667     70.283333                   0.846503                  0.662729       no      topM_softmax  0.3
     schur_tuned_topm_softmax_tau1p0            1.082377        1323.331767             3.796167     62.983333     70.383333                   0.805097                  0.693298       no      topM_softmax  1.0
                    schur_tuned_topk            1.088734        1284.705624             3.696333     35.150000     57.183333                   0.925209                  0.586774       no              topk  NaN

Full-pool softmax did not collapse in this Schur-tuned diagnostic; best 3-seed RMSE was `schur_tuned_full_pool_softmax_tau0p3` at 1.064319. This makes selection policy a plausible contributor to the remaining gap, but the evidence is only 3 seeds.

## 7. Final diagnosis: what explains the remaining gap?

The remaining gap is best explained by a combination of metric ambiguity, comparator ambiguity, and selection/split details. Metric ambiguity remains P0 because different w-RMSE definitions reverse reward-weighted conclusions. Comparator ambiguity is now explicit: concat supports the Figure 4a pattern strongly, while mf_interaction nearly matches or beats DU-Net. Selection policy is newly important because paper-style softmax improved 3-seed RMSE. Split/mask still matters, but the u1-u5 diagnostic says concat success generalizes across official folds.

## 8. Recommended next action

Stop modifying the Figure 4a mainline. Keep Schur log-lambda top-k v1 as the frozen deterministic main result, add `lambda=0.3` as a balanced reward/RMSE diagnostic, label `linucb_mf_interaction` as the strongest clean-room comparator caveat, and only run robustness/caveat extensions from here. If more algorithm work is done, the only justified path is source-backed paper mechanism fidelity such as exact metric, exact split/mask, or paper-confirmed sampling.

## Artifact index

- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_reward_a1.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_reward_pareto_a3.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_selection_features_by_cycle_a2_b2.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_selection_features_aggregate_a2_b2.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_comparator_b1.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_linucb_mf_alpha_b3.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_linucb_mf_alpha_by_cycle_b3.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_fold_c1.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_metric_c2.csv`
- `outputs\tables\paper_faithful_movielens_gap_investigation_v1_selection_c3.csv`
- `outputs\figures\paper_faithful_movielens_gap_investigation_v1_reward_rmse_pareto_20seeds.png`
- `outputs\figures\paper_faithful_movielens_gap_investigation_v1_selection_policy_rmse_reward_3seeds.png`
