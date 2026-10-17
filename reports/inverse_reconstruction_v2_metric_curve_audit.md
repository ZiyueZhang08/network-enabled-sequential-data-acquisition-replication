# inverse_reconstruction_v2 metric curve audit

Scope: all-cycle row-level metric reconstruction for Figure 4a. This upgrades the previous endpoint-only inverse reconstruction once the all-cycle shards are present.

- Row-level directory: `outputs/row_level_predictions/paper_exact_attempt_movielens_v2_allcycles`
- Target curves: `outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_digitized_targets.csv`
- All-cycle completeness: `complete`
- Metric gate promotion: `no`

## All-Cycle Row-Level Completeness

| method | seed_count | available_seeds | missing_seeds | cycle_count | available_cycles | missing_cycles | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | 20 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |  | 21 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 |  | complete_20seed_21cycle |
| linucb_concat | 20 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |  | 21 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 |  | complete_20seed_21cycle |
| linucb_mf_interaction | 20 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |  | 21 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 |  | complete_20seed_21cycle |
| lambda0p3_diagnostic | 20 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |  | 21 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 |  | complete_20seed_21cycle |

## Top Metric Curve Candidates

| du_method | linear_method | metric | figure4a_du_curve_error | figure4a_linear_curve_error | final_gap_error | endpoint_scale_error | local_final_gap | paper_final_gap | direction_match | text_defensibility | seed_count | cycle_count | curve_score | paper_wrmse_reconstructed_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | reward_tail_proxy | 0.481465 | 0.485886 | 0.00168421 | 0.478123 | 0.0276842 | 0.026 | 1 | defensible | 20 | 21 | 1.45389 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | reward_tail_proxy | 0.481465 | 0.485886 | 0.00168421 | 0.478123 | 0.0276842 | 0.026 | 1 | defensible | 20 | 21 | 1.45389 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | centered_rating_rmse | 0.734508 | 0.740031 | 0.00966297 | 0.712442 | 0.035663 | 0.026 | 1 | defensible | 20 | 21 | 2.2353 | no |
| lambda0p3_diagnostic | linucb_concat | centered_rating_rmse | 0.734508 | 0.740031 | 0.00966297 | 0.712442 | 0.035663 | 0.026 | 1 | defensible | 20 | 21 | 2.2353 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | standard_rmse | 0.734508 | 0.740031 | 0.00966297 | 0.712442 | 0.035663 | 0.026 | 1 | defensible | 20 | 21 | 2.2353 | no |
| lambda0p3_diagnostic | linucb_concat | standard_rmse | 0.734508 | 0.740031 | 0.00966297 | 0.712442 | 0.035663 | 0.026 | 1 | defensible | 20 | 21 | 2.2353 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | unclipped_standard_rmse | 0.734494 | 0.740003 | 0.0096936 | 0.712403 | 0.0356936 | 0.026 | 1 | defensible | 20 | 21 | 2.23537 | no |
| lambda0p3_diagnostic | linucb_concat | unclipped_standard_rmse | 0.734494 | 0.740003 | 0.0096936 | 0.712403 | 0.0356936 | 0.026 | 1 | defensible | 20 | 21 | 2.23537 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | unclipped_standard_rmse | 0.734494 | 0.748393 | 0.00928787 | 0.721893 | 0.0167121 | 0.026 | 1 | defensible | 20 | 21 | 2.25122 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | unclipped_standard_rmse | 0.734494 | 0.748393 | 0.00928787 | 0.721893 | 0.0167121 | 0.026 | 1 | defensible | 20 | 21 | 2.25122 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | centered_rating_rmse | 0.734508 | 0.748427 | 0.00930425 | 0.721926 | 0.0166957 | 0.026 | 1 | defensible | 20 | 21 | 2.25138 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | centered_rating_rmse | 0.734508 | 0.748427 | 0.00930425 | 0.721926 | 0.0166957 | 0.026 | 1 | defensible | 20 | 21 | 2.25138 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | standard_rmse | 0.734508 | 0.748427 | 0.00930425 | 0.721926 | 0.0166957 | 0.026 | 1 | defensible | 20 | 21 | 2.25138 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | standard_rmse | 0.734508 | 0.748427 | 0.00930425 | 0.721926 | 0.0166957 | 0.026 | 1 | defensible | 20 | 21 | 2.25138 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | mild_reward_w_rmse | 0.745833 | 0.758573 | 0.00853455 | 0.734569 | 0.0174655 | 0.026 | 1 | defensible | 20 | 21 | 2.28165 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | mild_reward_w_rmse | 0.745833 | 0.758573 | 0.00853455 | 0.734569 | 0.0174655 | 0.026 | 1 | defensible | 20 | 21 | 2.28165 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | medium_reward_w_rmse | 0.752269 | 0.764329 | 0.00808251 | 0.741736 | 0.0179175 | 0.026 | 1 | defensible | 20 | 21 | 2.29875 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | medium_reward_w_rmse | 0.752269 | 0.764329 | 0.00808251 | 0.741736 | 0.0179175 | 0.026 | 1 | defensible | 20 | 21 | 2.29875 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | current_w_rmse | 0.770876 | 0.78092 | 0.0067136 | 0.76238 | 0.0192864 | 0.026 | 1 | defensible | 20 | 21 | 2.34774 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | current_w_rmse | 0.770876 | 0.78092 | 0.0067136 | 0.76238 | 0.0192864 | 0.026 | 1 | defensible | 20 | 21 | 2.34774 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | mild_reward_w_rmse | 0.745833 | 0.767447 | 0.0199414 | 0.740273 | 0.00605865 | 0.026 | 1 | defensible | 20 | 21 | 2.35326 | no |
| lambda0p3_diagnostic | linucb_concat | mild_reward_w_rmse | 0.745833 | 0.767447 | 0.0199414 | 0.740273 | 0.00605865 | 0.026 | 1 | defensible | 20 | 21 | 2.35326 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | inverse_popularity_weighted_rmse | 0.688139 | 0.691022 | 0.0166866 | 0.648444 | 0.0426866 | 0.026 | 1 | weak | 20 | 21 | 2.36104 | no |
| lambda0p3_diagnostic | linucb_concat | inverse_popularity_weighted_rmse | 0.688139 | 0.691022 | 0.0166866 | 0.648444 | 0.0426866 | 0.026 | 1 | weak | 20 | 21 | 2.36104 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | inverse_popularity_weighted_rmse | 0.688139 | 0.706203 | 0.0165258 | 0.66505 | 0.00947423 | 0.026 | 1 | weak | 20 | 21 | 2.39202 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | inverse_popularity_weighted_rmse | 0.688139 | 0.706203 | 0.0165258 | 0.66505 | 0.00947423 | 0.026 | 1 | weak | 20 | 21 | 2.39202 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | rating_weighted_rmse | 0.832292 | 0.846162 | 0.011036 | 0.819316 | 0.014964 | 0.026 | 1 | defensible | 20 | 21 | 2.55295 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | rating_weighted_rmse | 0.832292 | 0.846162 | 0.011036 | 0.819316 | 0.014964 | 0.026 | 1 | defensible | 20 | 21 | 2.55295 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | sigmoid_w_rmse | 0.856591 | 0.870565 | 0.011649 | 0.84332 | 0.014351 | 0.026 | 1 | defensible | 20 | 21 | 2.62872 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | sigmoid_w_rmse | 0.856591 | 0.870565 | 0.011649 | 0.84332 | 0.014351 | 0.026 | 1 | defensible | 20 | 21 | 2.62872 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | reward_tail_proxy | 0.481465 | 0.557731 | 0.119002 | 0.538466 | -0.0930023 | 0.026 | 0 | defensible | 20 | 21 | 3.17267 | no |
| lambda0p3_diagnostic | linucb_concat | reward_tail_proxy | 0.481465 | 0.557731 | 0.119002 | 0.538466 | -0.0930023 | 0.026 | 0 | defensible | 20 | 21 | 3.17267 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | medium_reward_w_rmse | 0.752269 | 0.783313 | 0.0372355 | 0.756313 | -0.0112355 | 0.026 | 0 | defensible | 20 | 21 | 3.47807 | no |
| lambda0p3_diagnostic | linucb_concat | medium_reward_w_rmse | 0.752269 | 0.783313 | 0.0372355 | 0.756313 | -0.0112355 | 0.026 | 0 | defensible | 20 | 21 | 3.47807 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | rating_weighted_rmse | 0.832292 | 0.862536 | 0.0356679 | 0.831632 | -0.00966789 | 0.026 | 0 | defensible | 20 | 21 | 3.7048 | no |
| lambda0p3_diagnostic | linucb_concat | rating_weighted_rmse | 0.832292 | 0.862536 | 0.0356679 | 0.831632 | -0.00966789 | 0.026 | 0 | defensible | 20 | 21 | 3.7048 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | sigmoid_w_rmse | 0.856591 | 0.891677 | 0.0445625 | 0.859777 | -0.0185625 | 0.026 | 0 | defensible | 20 | 21 | 3.83086 | no |
| lambda0p3_diagnostic | linucb_concat | sigmoid_w_rmse | 0.856591 | 0.891677 | 0.0445625 | 0.859777 | -0.0185625 | 0.026 | 0 | defensible | 20 | 21 | 3.83086 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | current_w_rmse | 0.770876 | 0.830449 | 0.0894527 | 0.803749 | -0.0634527 | 0.026 | 0 | defensible | 20 | 21 | 3.85234 | no |
| lambda0p3_diagnostic | linucb_concat | current_w_rmse | 0.770876 | 0.830449 | 0.0894527 | 0.803749 | -0.0634527 | 0.026 | 0 | defensible | 20 | 21 | 3.85234 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | 0_1_normalized_rmse | 1.54724 | 1.56697 | 0.0170843 | 1.52886 | 0.00891574 | 0.026 | 1 | weak | 20 | 21 | 4.97849 | no |
| lambda0p3_diagnostic | linucb_concat | 0_1_normalized_rmse | 1.54724 | 1.56697 | 0.0170843 | 1.52886 | 0.00891574 | 0.026 | 1 | weak | 20 | 21 | 4.97849 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | 0_1_normalized_rmse | 1.54724 | 1.5691 | 0.0218261 | 1.53123 | 0.00417394 | 0.026 | 1 | weak | 20 | 21 | 5.0067 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | 0_1_normalized_rmse | 1.54724 | 1.5691 | 0.0218261 | 1.53123 | 0.00417394 | 0.026 | 1 | weak | 20 | 21 | 5.0067 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | pdf_literal_norm_rmse | 1.81053 | 1.8349 | 0.0257478 | 1.7933 | 0.000252175 | 0.026 | 1 | defensible | 20 | 21 | 5.56747 | no |
| lambda0p3_diagnostic | linucb_concat | pdf_literal_norm_rmse | 1.81053 | 1.8349 | 0.0257478 | 1.7933 | 0.000252175 | 0.026 | 1 | defensible | 20 | 21 | 5.56747 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | pdf_literal_norm_rmse | 1.81053 | 1.83496 | 0.0258819 | 1.79337 | 0.000118057 | 0.026 | 1 | defensible | 20 | 21 | 5.56827 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | pdf_literal_norm_rmse | 1.81053 | 1.83496 | 0.0258819 | 1.79337 | 0.000118057 | 0.026 | 1 | defensible | 20 | 21 | 5.56827 | no |

## Alignment Rows

| du_method | linear_method | metric | figure4a_du_curve_error | figure4a_linear_curve_error | final_gap_error | endpoint_scale_error | local_final_gap | paper_final_gap | direction_match | text_defensibility | seed_count | cycle_count | curve_score | paper_wrmse_reconstructed_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | reward_tail_proxy | 0.481465 | 0.485886 | 0.00168421 | 0.478123 | 0.0276842 | 0.026 | 1 | defensible | 20 | 21 | 1.45389 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | reward_tail_proxy | 0.481465 | 0.485886 | 0.00168421 | 0.478123 | 0.0276842 | 0.026 | 1 | defensible | 20 | 21 | 1.45389 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | centered_rating_rmse | 0.734508 | 0.740031 | 0.00966297 | 0.712442 | 0.035663 | 0.026 | 1 | defensible | 20 | 21 | 2.2353 | no |
| lambda0p3_diagnostic | linucb_concat | centered_rating_rmse | 0.734508 | 0.740031 | 0.00966297 | 0.712442 | 0.035663 | 0.026 | 1 | defensible | 20 | 21 | 2.2353 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | standard_rmse | 0.734508 | 0.740031 | 0.00966297 | 0.712442 | 0.035663 | 0.026 | 1 | defensible | 20 | 21 | 2.2353 | no |
| lambda0p3_diagnostic | linucb_concat | standard_rmse | 0.734508 | 0.740031 | 0.00966297 | 0.712442 | 0.035663 | 0.026 | 1 | defensible | 20 | 21 | 2.2353 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | unclipped_standard_rmse | 0.734494 | 0.740003 | 0.0096936 | 0.712403 | 0.0356936 | 0.026 | 1 | defensible | 20 | 21 | 2.23537 | no |
| lambda0p3_diagnostic | linucb_concat | unclipped_standard_rmse | 0.734494 | 0.740003 | 0.0096936 | 0.712403 | 0.0356936 | 0.026 | 1 | defensible | 20 | 21 | 2.23537 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | unclipped_standard_rmse | 0.734494 | 0.748393 | 0.00928787 | 0.721893 | 0.0167121 | 0.026 | 1 | defensible | 20 | 21 | 2.25122 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | unclipped_standard_rmse | 0.734494 | 0.748393 | 0.00928787 | 0.721893 | 0.0167121 | 0.026 | 1 | defensible | 20 | 21 | 2.25122 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | centered_rating_rmse | 0.734508 | 0.748427 | 0.00930425 | 0.721926 | 0.0166957 | 0.026 | 1 | defensible | 20 | 21 | 2.25138 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | centered_rating_rmse | 0.734508 | 0.748427 | 0.00930425 | 0.721926 | 0.0166957 | 0.026 | 1 | defensible | 20 | 21 | 2.25138 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | standard_rmse | 0.734508 | 0.748427 | 0.00930425 | 0.721926 | 0.0166957 | 0.026 | 1 | defensible | 20 | 21 | 2.25138 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | standard_rmse | 0.734508 | 0.748427 | 0.00930425 | 0.721926 | 0.0166957 | 0.026 | 1 | defensible | 20 | 21 | 2.25138 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | mild_reward_w_rmse | 0.745833 | 0.758573 | 0.00853455 | 0.734569 | 0.0174655 | 0.026 | 1 | defensible | 20 | 21 | 2.28165 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | mild_reward_w_rmse | 0.745833 | 0.758573 | 0.00853455 | 0.734569 | 0.0174655 | 0.026 | 1 | defensible | 20 | 21 | 2.28165 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | medium_reward_w_rmse | 0.752269 | 0.764329 | 0.00808251 | 0.741736 | 0.0179175 | 0.026 | 1 | defensible | 20 | 21 | 2.29875 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | medium_reward_w_rmse | 0.752269 | 0.764329 | 0.00808251 | 0.741736 | 0.0179175 | 0.026 | 1 | defensible | 20 | 21 | 2.29875 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | current_w_rmse | 0.770876 | 0.78092 | 0.0067136 | 0.76238 | 0.0192864 | 0.026 | 1 | defensible | 20 | 21 | 2.34774 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | current_w_rmse | 0.770876 | 0.78092 | 0.0067136 | 0.76238 | 0.0192864 | 0.026 | 1 | defensible | 20 | 21 | 2.34774 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | mild_reward_w_rmse | 0.745833 | 0.767447 | 0.0199414 | 0.740273 | 0.00605865 | 0.026 | 1 | defensible | 20 | 21 | 2.35326 | no |
| lambda0p3_diagnostic | linucb_concat | mild_reward_w_rmse | 0.745833 | 0.767447 | 0.0199414 | 0.740273 | 0.00605865 | 0.026 | 1 | defensible | 20 | 21 | 2.35326 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_concat | inverse_popularity_weighted_rmse | 0.688139 | 0.691022 | 0.0166866 | 0.648444 | 0.0426866 | 0.026 | 1 | weak | 20 | 21 | 2.36104 | no |
| lambda0p3_diagnostic | linucb_concat | inverse_popularity_weighted_rmse | 0.688139 | 0.691022 | 0.0166866 | 0.648444 | 0.0426866 | 0.026 | 1 | weak | 20 | 21 | 2.36104 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | inverse_popularity_weighted_rmse | 0.688139 | 0.706203 | 0.0165258 | 0.66505 | 0.00947423 | 0.026 | 1 | weak | 20 | 21 | 2.39202 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | inverse_popularity_weighted_rmse | 0.688139 | 0.706203 | 0.0165258 | 0.66505 | 0.00947423 | 0.026 | 1 | weak | 20 | 21 | 2.39202 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | rating_weighted_rmse | 0.832292 | 0.846162 | 0.011036 | 0.819316 | 0.014964 | 0.026 | 1 | defensible | 20 | 21 | 2.55295 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | rating_weighted_rmse | 0.832292 | 0.846162 | 0.011036 | 0.819316 | 0.014964 | 0.026 | 1 | defensible | 20 | 21 | 2.55295 | no |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | linucb_mf_interaction | sigmoid_w_rmse | 0.856591 | 0.870565 | 0.011649 | 0.84332 | 0.014351 | 0.026 | 1 | defensible | 20 | 21 | 2.62872 | no |
| lambda0p3_diagnostic | linucb_mf_interaction | sigmoid_w_rmse | 0.856591 | 0.870565 | 0.011649 | 0.84332 | 0.014351 | 0.026 | 1 | defensible | 20 | 21 | 2.62872 | no |

## Decision

A metric is promoted only if curve shape, endpoint scale, final gap, direction, text defensibility, 20 seeds, and 21 cycles all pass. If no metric is promoted after completeness is achieved, the metric gate remains failed with all-cycle evidence.
