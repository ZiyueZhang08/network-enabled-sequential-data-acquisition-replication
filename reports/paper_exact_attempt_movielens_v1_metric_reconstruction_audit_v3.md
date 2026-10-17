# Metric Reconstruction Audit v3

- Row-level source: `outputs\row_level_predictions\paper_exact_attempt_movielens_v1`
- Paper Figure 4a visual endpoints: DU-Net `1.788`, Linear UCB `1.814`
- Gate status: **fail**

## Row-Level Completion

| method | available_seed_count | available_seeds | available_cycles | test_row_count | required_seed_count | row_level_scope | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| linucb_concat | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| linucb_mf_interaction | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| schur_tuned_topk | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| lambda0p3_diagnostic | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |

## Best Metric Candidates

| metric | comparator | DU-Net endpoint | Linear UCB endpoint | gap | direction | scale_distance_to_paper | gap_distance_to_paper | definition_defensible_from_text | raw_match | affine_match | paper_wrmse_reconstructed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sqrt_mean_squared_weighted_max_rating_minus_3_eps_weighted_rmse | linucb_mf_interaction | 1.34016 | 1.36139 | 0.0212302 | DU-Net < Linear UCB | 0.249991 | 0.183456 | 0 | fail | not_testable_from_final_only_row_level | no |
| sqrt_mean_squared_weighted_max_rating_minus_3_eps_weighted_rmse | linucb_concat | 1.34016 | 1.24309 | -0.0970683 | not_matched | 0.282598 | 4.7334 | 0 | fail | not_testable_from_final_only_row_level | no |
| inverse_item_popularity_weighted_rmse | linucb_concat | 1.11571 | 1.15591 | 0.0402032 | DU-Net < Linear UCB | 0.369392 | 0.546278 | 0 | fail | not_testable_from_final_only_row_level | no |
| inverse_item_popularity_weighted_rmse | linucb_mf_interaction | 1.11571 | 1.13508 | 0.0193733 | DU-Net < Linear UCB | 0.375134 | 0.254872 | 0 | fail | not_testable_from_final_only_row_level | no |
| raw_unclipped_rmse | linucb_concat | 1.06964 | 1.09802 | 0.0283837 | DU-Net < Linear UCB | 0.398231 | 0.0916823 | 1 | fail | not_testable_from_final_only_row_level | no |
| centered_rating_rmse | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | DU-Net < Linear UCB | 0.398245 | 0.0903541 | 1 | fail | not_testable_from_final_only_row_level | no |
| clipped_rmse | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | DU-Net < Linear UCB | 0.398245 | 0.0903541 | 1 | fail | not_testable_from_final_only_row_level | no |
| standard_rmse | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | DU-Net < Linear UCB | 0.398245 | 0.0903541 | 1 | fail | not_testable_from_final_only_row_level | no |
| raw_unclipped_rmse | linucb_mf_interaction | 1.06964 | 1.08643 | 0.0167882 | DU-Net < Linear UCB | 0.401427 | 0.354299 | 1 | fail | not_testable_from_final_only_row_level | no |
| centered_rating_rmse | linucb_mf_interaction | 1.06963 | 1.0864 | 0.0167722 | DU-Net < Linear UCB | 0.401436 | 0.354916 | 1 | fail | not_testable_from_final_only_row_level | no |
| clipped_rmse | linucb_mf_interaction | 1.06963 | 1.0864 | 0.0167722 | DU-Net < Linear UCB | 0.401436 | 0.354916 | 1 | fail | not_testable_from_final_only_row_level | no |
| standard_rmse | linucb_mf_interaction | 1.06963 | 1.0864 | 0.0167722 | DU-Net < Linear UCB | 0.401436 | 0.354916 | 1 | fail | not_testable_from_final_only_row_level | no |
| user_degree_weighted_rmse | linucb_concat | 1.05781 | 1.07436 | 0.0165434 | DU-Net < Linear UCB | 0.408061 | 0.363715 | 0 | fail | not_testable_from_final_only_row_level | no |
| user_degree_weighted_rmse | linucb_mf_interaction | 1.05781 | 1.07251 | 0.0146953 | DU-Net < Linear UCB | 0.40857 | 0.434797 | 0 | fail | not_testable_from_final_only_row_level | no |
| max_rating_minus_3_eps_weighted_rmse | linucb_mf_interaction | 1.05098 | 1.06626 | 0.015277 | DU-Net < Linear UCB | 0.412203 | 0.412421 | 1 | fail | not_testable_from_final_only_row_level | no |
| item_popularity_weighted_rmse | linucb_mf_interaction | 1.03127 | 1.05433 | 0.0230517 | DU-Net < Linear UCB | 0.421005 | 0.113397 | 0 | fail | not_testable_from_final_only_row_level | no |
| sqrt_mean_squared_weighted_normalized_rating_weighted_rmse | linucb_mf_interaction | 1.03383 | 1.04834 | 0.0145089 | DU-Net < Linear UCB | 0.421941 | 0.441964 | 0 | fail | not_testable_from_final_only_row_level | no |
| item_popularity_weighted_rmse | linucb_concat | 1.03127 | 1.0078 | -0.0234723 | not_matched | 0.433828 | 1.90278 | 0 | fail | not_testable_from_final_only_row_level | no |
| max_rating_minus_3_eps_weighted_rmse | linucb_concat | 1.05098 | 0.983652 | -0.0673321 | not_matched | 0.434973 | 3.5897 | 1 | fail | not_testable_from_final_only_row_level | no |
| sqrt_mean_squared_weighted_normalized_rating_weighted_rmse | linucb_concat | 1.03383 | 0.990439 | -0.0433893 | not_matched | 0.437899 | 2.66882 | 0 | fail | not_testable_from_final_only_row_level | no |

## Curve Evidence

| figure | method | local_metric | cycles | endpoint_local | endpoint_paper_visual | endpoint_abs_error | raw_curve_rmse | curve_correlation | affine_intercept_to_paper | affine_slope_to_paper | affine_curve_rmse | curve_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | DU-Net | standard_mask_rmse | 21 | 1.07073 | 1.788 | 0.717274 | 0.734508 | 0.895132 | -1.9642 | 3.48947 | 0.0155324 | outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_curve_alignment_summary.csv |
| Figure 4a | Linear UCB | standard_mask_rmse | 21 | 1.10639 | 1.814 | 0.707611 | 0.740031 | -0.653278 | 17.147 | -13.8734 | 0.0250394 | outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_curve_alignment_summary.csv |
| Figure 4a | DU-Net | w_rmse | 21 | 1.02898 | 1.788 | 0.759023 | 0.770876 | 0.957599 | -0.498481 | 2.2118 | 0.010038 | outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_curve_alignment_summary.csv |
| Figure 4a | Linear UCB | w_rmse | 21 | 0.965524 | 1.814 | 0.848476 | 0.830449 | 0.965066 | 0.970152 | 0.861896 | 0.00866507 | outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_curve_alignment_summary.csv |
| Figure 4b | DU-Net | cumulative_reward | 21 | 1246.39 | 1325 | 78.6137 | 81.6146 | 0.998763 | 8.58889 | 1.09797 | 20.6368 | outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_curve_alignment_summary.csv |
| Figure 4b | Linear UCB | cumulative_reward | 21 | 1393.68 | 1390 | 3.68025 | 61.5131 | 0.99835 | 11.5403 | 1.05633 | 25.7838 | outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_curve_alignment_summary.csv |

## Decision

No metric is promoted to `paper_wrmse_reconstructed` unless endpoint scale, direction, gap, curve shape, and text-defensible definition all pass. With current final-cycle row-level evidence, no candidate passes. Exact w-RMSE remains blocked.
