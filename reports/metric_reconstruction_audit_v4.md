# Metric Reconstruction Audit v4

- Prefix: `paper_exact_attempt_movielens_v2`
- Row-level source: `outputs/row_level_predictions/paper_exact_attempt_movielens_v2`
- Figure 4a visual endpoints: DU-Net `1.788`, Linear UCB `1.814`
- Gate status: **fail**
- Plot: `outputs/figures/metric_curve_alignment_v4.png`

## Row-Level Completeness

| method | gate_role | required_for_metric_gate | available_seed_count | available_seeds | missing_seeds | available_cycles | test_row_count | required_seed_count | row_level_scope | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | metric_gate_required | 1 | 20 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |  | 20 | 400000 | 20 | final_cycle_only | complete_20seed |
| linucb_concat | metric_gate_required | 1 | 20 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |  | 20 | 400000 | 20 | final_cycle_only | complete_20seed |
| linucb_mf_interaction | metric_gate_required | 1 | 20 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |  | 20 | 400000 | 20 | final_cycle_only | complete_20seed |
| linucb_bias_interaction | metric_gate_required | 1 | 20 | 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 |  | 20 | 400000 | 20 | final_cycle_only | complete_20seed |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | diagnostic_tracked | 0 | 1 | 0 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 | 20 | 20000 | 20 | final_cycle_only | partial |
| schur_tuned_topk | diagnostic_tracked | 0 | 1 | 0 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 | 20 | 20000 | 20 | final_cycle_only | partial |
| lambda0p3_diagnostic | diagnostic_tracked | 0 | 1 | 0 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 | 20 | 20000 | 20 | final_cycle_only | partial |

## Missing Work Queue

_No rows._

## Diagnostic Branch Tracking

| method | gate_role | available_seed_count | missing_seeds | status |
| --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | diagnostic_tracked | 1 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 | partial |
| schur_tuned_topk | diagnostic_tracked | 1 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 | partial |
| lambda0p3_diagnostic | diagnostic_tracked | 1 | 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 | partial |

## Metric Candidates

- Row-level metric rows: `1411`
- Summary rows: `119`

| metric | comparator | DU-Net endpoint | Linear UCB endpoint | gap | direction | scale_distance_to_paper | gap_distance_to_paper | definition_defensible_from_text | row_level_seed_count | raw_match | paper_wrmse_reconstructed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reward_tail_proxy | linucb_mf_interaction | 1.30903 | 1.33672 | 0.0276842 | DU-Net < Linear UCB | 0.265494 | 0.0647771 | 1 | 20 | fail | no |
| reward_tail_proxy | linucb_concat | 1.30903 | 1.21603 | -0.0930023 | not_matched | 0.298759 | 4.57701 | 1 | 20 | fail | no |
| reward_tail_proxy | linucb_bias_interaction | 1.30903 | 1.17896 | -0.130078 | not_matched | 0.308978 | 6.00301 | 1 | 20 | fail | no |
| inverse_item_popularity_weighted_rmse | linucb_bias_interaction | 1.13121 | 1.20674 | 0.0755234 | DU-Net < Linear UCB | 0.351048 | 1.90475 | 1 | 20 | fail | no |
| inverse_item_popularity_weighted_rmse | linucb_concat | 1.13121 | 1.1739 | 0.0426866 | DU-Net < Linear UCB | 0.360099 | 0.64179 | 1 | 20 | fail | no |
| inverse_item_popularity_weighted_rmse | linucb_mf_interaction | 1.13121 | 1.14069 | 0.00947423 | DU-Net < Linear UCB | 0.369253 | 0.635606 | 1 | 20 | fail | no |
| raw_prediction_standard_rmse | linucb_bias_interaction | 1.07075 | 1.13034 | 0.0595849 | DU-Net < Linear UCB | 0.389014 | 1.29173 | 0 | 20 | fail | no |
| unclipped_standard_rmse | linucb_bias_interaction | 1.07075 | 1.13034 | 0.0595849 | DU-Net < Linear UCB | 0.389014 | 1.29173 | 1 | 20 | fail | no |
| centered_rating_rmse | linucb_bias_interaction | 1.07073 | 1.13033 | 0.0596056 | DU-Net < Linear UCB | 0.389022 | 1.29252 | 1 | 20 | fail | no |
| standard_rmse | linucb_bias_interaction | 1.07073 | 1.13033 | 0.0596056 | DU-Net < Linear UCB | 0.389022 | 1.29252 | 1 | 20 | fail | no |
| raw_prediction_standard_rmse | linucb_concat | 1.07075 | 1.10644 | 0.0356936 | DU-Net < Linear UCB | 0.3956 | 0.372831 | 0 | 20 | fail | no |
| unclipped_standard_rmse | linucb_concat | 1.07075 | 1.10644 | 0.0356936 | DU-Net < Linear UCB | 0.3956 | 0.372831 | 1 | 20 | fail | no |
| centered_rating_rmse | linucb_concat | 1.07073 | 1.10639 | 0.035663 | DU-Net < Linear UCB | 0.395622 | 0.371653 | 1 | 20 | fail | no |
| standard_rmse | linucb_concat | 1.07073 | 1.10639 | 0.035663 | DU-Net < Linear UCB | 0.395622 | 0.371653 | 1 | 20 | fail | no |
| raw_prediction_standard_rmse | linucb_mf_interaction | 1.07075 | 1.08746 | 0.0167121 | DU-Net < Linear UCB | 0.400831 | 0.357226 | 0 | 20 | fail | no |
| unclipped_standard_rmse | linucb_mf_interaction | 1.07075 | 1.08746 | 0.0167121 | DU-Net < Linear UCB | 0.400831 | 0.357226 | 1 | 20 | fail | no |
| centered_rating_rmse | linucb_mf_interaction | 1.07073 | 1.08742 | 0.0166957 | DU-Net < Linear UCB | 0.40085 | 0.357856 | 1 | 20 | fail | no |
| standard_rmse | linucb_mf_interaction | 1.07073 | 1.08742 | 0.0166957 | DU-Net < Linear UCB | 0.40085 | 0.357856 | 1 | 20 | fail | no |
| user_degree_weighted_rmse | linucb_bias_interaction | 1.05225 | 1.10588 | 0.053627 | DU-Net < Linear UCB | 0.400928 | 1.06258 | 0 | 20 | fail | no |
| user_degree_weighted_rmse | linucb_concat | 1.05225 | 1.08308 | 0.0308251 | DU-Net < Linear UCB | 0.407213 | 0.185581 | 0 | 20 | fail | no |
| mild_reward_w_rmse | linucb_mf_interaction | 1.0577 | 1.07516 | 0.0174655 | DU-Net < Linear UCB | 0.407872 | 0.328252 | 1 | 20 | fail | no |
| mild_reward_w_rmse | linucb_bias_interaction | 1.0577 | 1.07363 | 0.0159326 | DU-Net < Linear UCB | 0.408294 | 0.387209 | 1 | 20 | fail | no |
| mild_reward_w_rmse | linucb_concat | 1.0577 | 1.06376 | 0.00605865 | DU-Net < Linear UCB | 0.411016 | 0.766975 | 1 | 20 | fail | no |
| user_degree_weighted_rmse | linucb_mf_interaction | 1.05225 | 1.06925 | 0.0170012 | DU-Net < Linear UCB | 0.411024 | 0.346109 | 0 | 20 | fail | no |
| medium_reward_w_rmse | linucb_mf_interaction | 1.05031 | 1.06822 | 0.0179175 | DU-Net < Linear UCB | 0.411852 | 0.310866 | 1 | 20 | fail | no |

## Curve Alignment

| status | figure | metric | comparator | curve_rmse | endpoint_abs_error_mean | local_gap | paper_gap | gap_abs_error | points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| computed | Figure 4a | reward_tail_proxy_rmse | linucb_mf_interaction | 0.48368 | 0.478123 | 0.0276842 | 0.026 | 0.00168421 | 42 |
| computed | Figure 4a | reward_tail_proxy_rmse | linucb_concat | 0.520995 | 0.538466 | -0.0930023 | 0.026 | 0.119002 | 42 |
| computed | Figure 4a | reward_tail_proxy_rmse | linucb_bias_interaction | 0.540673 | 0.557004 | -0.130078 | 0.026 | 0.156078 | 42 |
| computed | Figure 4a | standard_mask_rmse | linucb_bias_interaction | 0.729781 | 0.700471 | 0.0596056 | 0.026 | 0.0336056 | 42 |
| computed | Figure 4a | standard_mask_rmse | linucb_concat | 0.737274 | 0.712442 | 0.035663 | 0.026 | 0.00966297 | 42 |
| computed | Figure 4a | standard_mask_rmse | linucb_mf_interaction | 0.7415 | 0.721926 | 0.0166957 | 0.026 | 0.00930425 | 42 |
| computed | Figure 4a | mild_reward_w_rmse | linucb_mf_interaction | 0.75223 | 0.734569 | 0.0174655 | 0.026 | 0.00853455 | 42 |
| computed | Figure 4a | mild_reward_w_rmse | linucb_bias_interaction | 0.755023 | 0.735336 | 0.0159326 | 0.026 | 0.0100674 | 42 |
| computed | Figure 4a | mild_reward_w_rmse | linucb_concat | 0.756717 | 0.740273 | 0.00605865 | 0.026 | 0.0199414 | 42 |
| computed | Figure 4a | medium_reward_w_rmse | linucb_mf_interaction | 0.758323 | 0.741736 | 0.0179175 | 0.026 | 0.00808251 | 42 |
| computed | Figure 4a | medium_reward_w_rmse | linucb_concat | 0.767948 | 0.756313 | -0.0112355 | 0.026 | 0.0372355 | 42 |
| computed | Figure 4a | medium_reward_w_rmse | linucb_bias_interaction | 0.769796 | 0.755603 | -0.00981666 | 0.026 | 0.0358167 | 42 |
| computed | Figure 4a | w_rmse | linucb_mf_interaction | 0.775914 | 0.76238 | 0.0192864 | 0.026 | 0.0067136 | 42 |
| computed | Figure 4a | w_rmse | linucb_concat | 0.801216 | 0.803749 | -0.0634527 | 0.026 | 0.0894527 | 42 |
| computed | Figure 4a | w_rmse | linucb_bias_interaction | 0.814435 | 0.816437 | -0.0888279 | 0.026 | 0.114828 | 42 |
| computed | Figure 4a | sigmoid_reward_w_rmse | linucb_mf_interaction | 0.863606 | 0.84332 | 0.014351 | 0.026 | 0.011649 | 42 |
| computed | Figure 4a | sigmoid_reward_w_rmse | linucb_concat | 0.87431 | 0.859777 | -0.0185625 | 0.026 | 0.0445625 | 42 |
| computed | Figure 4a | sigmoid_reward_w_rmse | linucb_bias_interaction | 0.877798 | 0.861351 | -0.0217102 | 0.026 | 0.0477102 | 42 |
| computed | Figure 4a | pdf_literal_norm_rmse | linucb_bias_interaction | 1.8227 | 1.79322 | 0.000421475 | 0.026 | 0.0255785 | 42 |
| computed | Figure 4a | pdf_literal_norm_rmse | linucb_concat | 1.82276 | 1.7933 | 0.000252175 | 0.026 | 0.0257478 | 42 |

## Decision

No candidate is promoted to `paper_wrmse_reconstructed` unless row-level completeness is 20 seeds for the metric-gate-required methods and endpoint scale, gap, direction, and text-defensible formula all pass. Diagnostic branches are tracked but cannot promote or block the Figure 4a metric gate.
