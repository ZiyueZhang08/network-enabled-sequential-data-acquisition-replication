# Inverse Reconstruction Leaderboard v1

Purpose: convert the Figure 4 replication gap into a latent-variable reconstruction surface. This report ranks what can be ranked from existing artifacts, generates Table-1-compatible random masks, and keeps unrun/source-blocked gates explicit.

## Inputs

- Digitized target curves: `outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_digitized_targets.csv`
- Metric alignment: `outputs/tables/metric_curve_alignment_v4.csv`
- Linear UCB alpha/reg metrics: `outputs/tables/linucb_alpha_reg_sensitivity_v2_metrics.csv`
- Generated mask index archive: `outputs/masks/inverse_reconstruction_v1/table1_random_masks_v1.npz`

## Target Endpoints

| figure | paper_metric | du_endpoint_visual | linear_endpoint_visual | linear_minus_du_visual_gap | digitization_source | digitization_status | conservative_endpoint_stress_band | conservative_gap_stress_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4b | cumulative_reward | 1325 | 1390 | 65 | tmp/pdfs/figure4b_movielens_cumulative_crop.png | visual_approximation_not_author_data | 10 | 20 |
| Figure 4a | w_rmse | 1.788 | 1.814 | 0.026 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data | 0.005 | 0.01 |

## Task 1: Row-Level Metric Family Endpoint Top 20

| rank | metric | definition | comparator | du_endpoint | linear_endpoint | local_gap | scale_distance | gap_distance | direction | seed_count | row_level_scope | curve_search_status | implausibility_penalty | endpoint_family_score | paper_wrmse_reconstructed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | clipped_prediction_error\|sqrt_mean_weighted_e2\|unit_plus_1_sigmoid | sqrt_mean_weighted_e2; error=clipped_prediction_error; weight=1 + 1 * sigmoid(rating - 3) | linucb_mf_interaction | 1.62353 | 1.64865 | 0.0251221 | 0.0915703 | 0.0337658 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.26 | 0.360012 | no |
| 2 | clipped_prediction_error\|sqrt_mean_weighted_e2\|unit_plus_0.5_positive_tail | sqrt_mean_weighted_e2; error=clipped_prediction_error; weight=1 + 0.5 * max(rating - 3, 0) | linucb_mf_interaction | 1.534 | 1.56125 | 0.027247 | 0.140695 | 0.0479617 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.25 | 0.402686 | no |
| 3 | unclipped_prediction_error\|sqrt_mean_weighted_e2\|unit_plus_1_sigmoid | sqrt_mean_weighted_e2; error=unclipped_prediction_error; weight=1 + 1 * sigmoid(rating - 3) | linucb_mf_interaction | 1.62357 | 1.64872 | 0.0251515 | 0.0915373 | 0.0326346 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.31 | 0.409696 | no |
| 4 | unclipped_prediction_error\|sqrt_mean_weighted_e2\|unit_plus_0.5_positive_tail | sqrt_mean_weighted_e2; error=unclipped_prediction_error; weight=1 + 0.5 * max(rating - 3, 0) | linucb_mf_interaction | 1.53404 | 1.56131 | 0.0272759 | 0.140668 | 0.0490716 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.3 | 0.452936 | no |
| 5 | clipped_prediction_error\|sqrt_mean_weighted_e2\|affine_tail_0.5_1 | sqrt_mean_weighted_e2; error=clipped_prediction_error; weight=0.5 + 1 * max(rating - 3, 0) | linucb_mf_interaction | 1.62677 | 1.65943 | 0.0326547 | 0.0876904 | 0.255949 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.32 | 0.471678 | no |
| 6 | clipped_prediction_error\|sqrt_mean_weighted_e2\|affine_tail_1_0.5 | sqrt_mean_weighted_e2; error=clipped_prediction_error; weight=1 + 0.5 * max(rating - 3, 0) | linucb_mf_interaction | 1.534 | 1.56125 | 0.027247 | 0.140695 | 0.0479617 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.32 | 0.472686 | no |
| 7 | clipped_prediction_error\|sqrt_mean_weighted_e2\|positive_tail_eps | sqrt_mean_weighted_e2; error=clipped_prediction_error; weight=max(rating - 3, 0) + 0.1 | linucb_mf_interaction | 1.30903 | 1.33672 | 0.0276842 | 0.265494 | 0.0647771 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.2 | 0.481688 | no |
| 8 | clipped_prediction_error\|sqrt_mean_e2\|unit | sqrt_mean_e2; error=clipped_prediction_error; weight=1 | linucb_concat | 1.07073 | 1.10639 | 0.035663 | 0.395622 | 0.371653 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0 | 0.488535 | no |
| 9 | clipped_prediction_error\|sqrt_mean_e2\|unit | sqrt_mean_e2; error=clipped_prediction_error; weight=1 | linucb_mf_interaction | 1.07073 | 1.08742 | 0.0166957 | 0.40085 | 0.357856 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0 | 0.490313 | no |
| 10 | clipped_prediction_error\|sqrt_mean_weighted_e2\|unit_plus_1_sigmoid | sqrt_mean_weighted_e2; error=clipped_prediction_error; weight=1 + 1 * sigmoid(rating - 3) | linucb_bias_interaction | 1.62353 | 1.63305 | 0.00952065 | 0.0958706 | 0.633821 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.26 | 0.514326 | no |
| 11 | normalized_clipped_error\|sqrt_mean_weighted_e2\|unit_plus_5_positive_tail | sqrt_mean_weighted_e2; error=normalized_clipped_error; weight=1 + 5 * max(rating - 3, 0) | linucb_mf_interaction | 1.7286 | 1.7647 | 0.0360913 | 0.0301995 | 0.388127 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.39 | 0.517231 | no |
| 12 | unclipped_prediction_error\|sqrt_mean_weighted_e2\|affine_tail_0.5_1 | sqrt_mean_weighted_e2; error=unclipped_prediction_error; weight=0.5 + 1 * max(rating - 3, 0) | linucb_mf_interaction | 1.62681 | 1.6595 | 0.0326887 | 0.0876626 | 0.25726 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.37 | 0.521977 | no |
| 13 | unclipped_prediction_error\|sqrt_mean_weighted_e2\|affine_tail_1_0.5 | sqrt_mean_weighted_e2; error=unclipped_prediction_error; weight=1 + 0.5 * max(rating - 3, 0) | linucb_mf_interaction | 1.53404 | 1.56131 | 0.0272759 | 0.140668 | 0.0490716 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.37 | 0.522936 | no |
| 14 | unclipped_prediction_error\|sqrt_mean_weighted_e2\|positive_tail_eps | sqrt_mean_weighted_e2; error=unclipped_prediction_error; weight=max(rating - 3, 0) + 0.1 | linucb_mf_interaction | 1.30906 | 1.33677 | 0.027712 | 0.265473 | 0.065845 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.25 | 0.531934 | no |
| 15 | clipped_prediction_error\|sqrt_mean_e2\|rating | sqrt_mean_e2; error=clipped_prediction_error; weight=rating | linucb_concat | 1.07073 | 1.10639 | 0.035663 | 0.395622 | 0.371653 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.05 | 0.538535 | no |
| 16 | clipped_prediction_error\|sqrt_mean_e2\|rating_mean_norm | sqrt_mean_e2; error=clipped_prediction_error; weight=rating / mean(rating) | linucb_concat | 1.07073 | 1.10639 | 0.035663 | 0.395622 | 0.371653 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.05 | 0.538535 | no |
| 17 | clipped_prediction_error\|sqrt_weighted_mean_e2\|unit | sqrt_weighted_mean_e2; error=clipped_prediction_error; weight=1 | linucb_concat | 1.07073 | 1.10639 | 0.035663 | 0.395622 | 0.371653 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.05 | 0.538535 | no |
| 18 | unclipped_prediction_error\|sqrt_mean_e2\|unit | sqrt_mean_e2; error=unclipped_prediction_error; weight=1 | linucb_concat | 1.07075 | 1.10644 | 0.0356936 | 0.3956 | 0.372831 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.05 | 0.538807 | no |
| 19 | unclipped_prediction_error\|sqrt_mean_e2\|unit | sqrt_mean_e2; error=unclipped_prediction_error; weight=1 | linucb_mf_interaction | 1.07075 | 1.08746 | 0.0167121 | 0.400831 | 0.357226 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.05 | 0.540138 | no |
| 20 | clipped_prediction_error\|sqrt_mean_e2\|rating | sqrt_mean_e2; error=clipped_prediction_error; weight=rating | linucb_mf_interaction | 1.07073 | 1.08742 | 0.0166957 | 0.40085 | 0.357856 | DU-Net < Linear UCB | 20 | final_cycle_only | blocked_requires_all_cycle_row_level_predictions | 0.05 | 0.540313 | no |

## Task 1b: Existing Curve-Level Metric Alignment Top 20

| rank | metric | comparator | curve_rmse | endpoint_abs_error_mean | gap_abs_error | metric_family_score | direction | scale_distance | gap_distance | text_defensible | row_level_seed_count | paper_wrmse_reconstructed | implausibility_penalty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | reward_tail_proxy_rmse | linucb_mf_interaction | 0.48368 | 0.478123 | 0.00168421 | 0.970224 | DU-Net < Linear UCB | 0.265494 | 0.0647771 | 1 | 20 | no | 0 |
| 2 | standard_mask_rmse | linucb_concat | 0.737274 | 0.712442 | 0.00966297 | 1.49803 | DU-Net < Linear UCB | 0.395622 | 0.371653 | 1 | 20 | no | 0 |
| 3 | standard_mask_rmse | linucb_mf_interaction | 0.7415 | 0.721926 | 0.00930425 | 1.50995 | DU-Net < Linear UCB | 0.40085 | 0.357856 | 1 | 20 | no | 0 |
| 4 | mild_reward_w_rmse | linucb_mf_interaction | 0.75223 | 0.734569 | 0.00853455 | 1.52947 | DU-Net < Linear UCB | 0.407872 | 0.328252 | 1 | 20 | no | 0 |
| 5 | medium_reward_w_rmse | linucb_mf_interaction | 0.758323 | 0.741736 | 0.00808251 | 1.54047 | DU-Net < Linear UCB | 0.411852 | 0.310866 | 1 | 20 | no | 0 |
| 6 | mild_reward_w_rmse | linucb_bias_interaction | 0.755023 | 0.735336 | 0.0100674 | 1.5407 | DU-Net < Linear UCB | 0.408294 | 0.387209 | 1 | 20 | no | 0 |
| 7 | w_rmse | linucb_mf_interaction | 0.775914 | 0.76238 | 0.0067136 | 1.57186 | DU-Net < Linear UCB | 0.423318 | 0.258215 | 1 | 20 | no | 0 |
| 8 | mild_reward_w_rmse | linucb_concat | 0.756717 | 0.740273 | 0.0199414 | 1.5967 | DU-Net < Linear UCB | 0.411016 | 0.766975 | 1 | 20 | no | 0 |
| 9 | standard_mask_rmse | linucb_bias_interaction | 0.729781 | 0.700471 | 0.0336056 | 1.59828 | DU-Net < Linear UCB | 0.389022 | 1.29252 | 1 | 20 | no | 0 |
| 10 | reward_tail_proxy_rmse | linucb_concat | 0.520995 | 0.538466 | 0.119002 | 1.65447 | not_matched | 0.298759 | 4.57701 | 1 | 20 | no | 0 |
| 11 | medium_reward_w_rmse | linucb_bias_interaction | 0.769796 | 0.755603 | 0.0358167 | 1.70448 | not_matched | 0.419497 | 1.37756 | 1 | 20 | no | 0 |
| 12 | medium_reward_w_rmse | linucb_concat | 0.767948 | 0.756313 | 0.0372355 | 1.71044 | not_matched | 0.419888 | 1.43214 | 1 | 20 | no | 0 |
| 13 | sigmoid_reward_w_rmse | linucb_mf_interaction | 0.863606 | 0.84332 | 0.011649 | 1.76517 | DU-Net < Linear UCB | 0.468252 | 0.448039 | 1 | 20 | no | 0 |
| 14 | reward_tail_proxy_rmse | linucb_bias_interaction | 0.540673 | 0.557004 | 0.156078 | 1.87807 | not_matched | 0.308978 | 6.00301 | 1 | 20 | no | 0 |
| 15 | sigmoid_reward_w_rmse | linucb_concat | 0.87431 | 0.859777 | 0.0445625 | 1.9569 | not_matched | 0.477324 | 1.71394 | 1 | 20 | no | 0 |
| 16 | sigmoid_reward_w_rmse | linucb_bias_interaction | 0.877798 | 0.861351 | 0.0477102 | 1.9777 | not_matched | 0.478192 | 1.83501 | 1 | 20 | no | 0 |
| 17 | w_rmse | linucb_concat | 0.801216 | 0.803749 | 0.0894527 | 2.05223 | not_matched | 0.446123 | 3.44049 | 1 | 20 | no | 0 |
| 18 | w_rmse | linucb_bias_interaction | 0.814435 | 0.816437 | 0.114828 | 2.20501 | not_matched | 0.453118 | 4.41646 | 1 | 20 | no | 0 |
| 19 | pdf_literal_norm_rmse | linucb_bias_interaction | 1.8227 | 1.79322 | 0.0255785 | 3.74381 | DU-Net < Linear UCB | 0.99568 | 0.983789 | 1 | 20 | no | 0 |
| 20 | pdf_literal_norm_rmse | linucb_concat | 1.82276 | 1.7933 | 0.0257478 | 3.7448 | DU-Net < Linear UCB | 0.995726 | 0.990301 | 1 | 20 | no | 0 |

## Task 2: Linear UCB Candidate Family

| rank | candidate | feature | alpha | reg | target | update | seed_count | fig4a_linear_curve_error | fig4a_endpoint_error | fig4b_linear_curve_error | fig4b_endpoint_error | protocol_penalty | linear_ucb_score | paper_linear_ucb_reconstructed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | linucb_mf_alpha_0p1_reg_1 | mf_interaction | 0.1 | 1 | rating | batch_update_per_cycle | 3 | 0.576023 | 0.63306 | 35.7322 | 130.364 | 0.15 | 1.08334 | no |
| 2 | linucb_mf_alpha_0p1_reg_10 | mf_interaction | 0.1 | 10 | rating | batch_update_per_cycle | 3 | 0.581541 | 0.637602 | 35.5971 | 129.608 | 0.15 | 1.08751 | no |
| 3 | linucb_mf_alpha_0p1_reg_0p1 | mf_interaction | 0.1 | 0.1 | rating | batch_update_per_cycle | 3 | 0.577487 | 0.627055 | 36.6615 | 132.906 | 0.15 | 1.0941 | no |
| 4 | linucb_mf_alpha_0p3_reg_0p1 | mf_interaction | 0.3 | 0.1 | rating | batch_update_per_cycle | 3 | 0.570215 | 0.607075 | 38.3371 | 26.2169 | 0.15 | 1.10359 | no |
| 5 | linucb_mf_alpha_0p05_reg_10 | mf_interaction | 0.05 | 10 | rating | batch_update_per_cycle | 3 | 0.573749 | 0.619854 | 44.3486 | 150.732 | 0.15 | 1.16723 | no |
| 6 | linucb_mf_alpha_0p01_reg_0p1 | mf_interaction | 0.01 | 0.1 | rating | batch_update_per_cycle | 3 | 0.567297 | 0.608345 | 46.1977 | 156.344 | 0.15 | 1.17927 | no |
| 7 | linucb_mf_alpha_0p05_reg_1 | mf_interaction | 0.05 | 1 | rating | batch_update_per_cycle | 3 | 0.576306 | 0.620868 | 45.5197 | 153.201 | 0.15 | 1.1815 | no |
| 8 | linucb_mf_alpha_0p3_reg_1 | mf_interaction | 0.3 | 1 | rating | batch_update_per_cycle | 3 | 0.565694 | 0.601824 | 46.925 | 5.16427 | 0.15 | 1.18494 | no |
| 9 | linucb_mf_alpha_0p05_reg_0p1 | mf_interaction | 0.05 | 0.1 | rating | batch_update_per_cycle | 3 | 0.575499 | 0.62638 | 46.387 | 156.977 | 0.15 | 1.18937 | no |
| 10 | linucb_mf_alpha_0p02_reg_1 | mf_interaction | 0.02 | 1 | rating | batch_update_per_cycle | 3 | 0.572228 | 0.626213 | 47.2335 | 160.888 | 0.15 | 1.19456 | no |
| 11 | linucb_mf_alpha_0p02_reg_10 | mf_interaction | 0.02 | 10 | rating | batch_update_per_cycle | 3 | 0.575194 | 0.624032 | 47.0268 | 157.646 | 0.15 | 1.19546 | no |
| 12 | linucb_mf_alpha_0p01_reg_10 | mf_interaction | 0.01 | 10 | rating | batch_update_per_cycle | 3 | 0.572589 | 0.616542 | 47.4221 | 157.17 | 0.15 | 1.19681 | no |
| 13 | linucb_mf_alpha_0p005_reg_1 | mf_interaction | 0.005 | 1 | rating | batch_update_per_cycle | 3 | 0.571026 | 0.620149 | 47.7726 | 160.131 | 0.15 | 1.19875 | no |
| 14 | linucb_mf_alpha_0p02_reg_0p1 | mf_interaction | 0.02 | 0.1 | rating | batch_update_per_cycle | 3 | 0.575243 | 0.619797 | 47.8517 | 158.299 | 0.15 | 1.20376 | no |
| 15 | linucb_mf_alpha_0p005_reg_10 | mf_interaction | 0.005 | 10 | rating | batch_update_per_cycle | 3 | 0.571915 | 0.616734 | 49.0411 | 162.33 | 0.15 | 1.21233 | no |
| 16 | linucb_mf_alpha_0p005_reg_0p1 | mf_interaction | 0.005 | 0.1 | rating | batch_update_per_cycle | 3 | 0.574082 | 0.622389 | 48.8633 | 160.84 | 0.15 | 1.21271 | no |
| 17 | linucb_mf_alpha_0p01_reg_1 | mf_interaction | 0.01 | 1 | rating | batch_update_per_cycle | 3 | 0.57286 | 0.620579 | 49.7149 | 165.451 | 0.15 | 1.22001 | no |
| 18 | linucb_mf_alpha_0p3_reg_10 | mf_interaction | 0.3 | 10 | rating | batch_update_per_cycle | 3 | 0.557797 | 0.579435 | 52.22 | 13.5735 | 0.15 | 1.23 | no |
| 19 | linucb_mf_alpha_0p8_reg_0p1 | mf_interaction | 0.8 | 0.1 | rating | batch_update_per_cycle | 3 | 0.492668 | 0.50476 | 135.453 | 146.95 | 0.15 | 1.99719 | no |
| 20 | linucb_mf_alpha_0p8_reg_1 | mf_interaction | 0.8 | 1 | rating | batch_update_per_cycle | 3 | 0.484816 | 0.484212 | 140.129 | 158.283 | 0.15 | 2.0361 | no |

## Task 3: Random Table-1-Compatible Masks

| mask_id | type | seed | users | items | initial_size | candidate_size | test_size | matrix_density | initial_rating_mean | candidate_rating_mean | test_rating_mean | initial_user_coverage | candidate_user_coverage | test_user_coverage | initial_item_coverage | candidate_item_coverage | test_item_coverage | rating_distribution_l1_to_full | density_match | selected | candidate_xor_hash |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| table1_random_000 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.53519 | 3.53191 | 3.52145 | 572 | 943 | 941 | 670 | 1651 | 1397 | 0.00750483 | pass | queued_for_3seed_screen | 86467 |
| table1_random_001 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.51686 | 3.52797 | 3.5382 | 582 | 943 | 943 | 655 | 1659 | 1393 | 0.00300061 | pass | queued_for_3seed_screen | 75282 |
| table1_random_002 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.51026 | 3.52992 | 3.531 | 579 | 943 | 942 | 670 | 1650 | 1413 | 0.00438613 | pass | queued_for_3seed_screen | 130160 |
| table1_random_003 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.53079 | 3.53134 | 3.524 | 581 | 943 | 941 | 667 | 1651 | 1397 | 0.00383613 | pass | queued_for_3seed_screen | 58488 |
| table1_random_004 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.50806 | 3.5316 | 3.52455 | 575 | 943 | 940 | 653 | 1647 | 1420 | 0.00866387 | pass | queued_for_3seed_screen | 17249 |
| table1_random_005 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.53886 | 3.52984 | 3.52935 | 578 | 943 | 942 | 662 | 1646 | 1422 | 0.00755483 | pass | queued_for_3seed_screen | 128696 |
| table1_random_006 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.51979 | 3.53124 | 3.52515 | 554 | 943 | 942 | 638 | 1648 | 1411 | 0.00171212 | pass | queued_for_3seed_screen | 37675 |
| table1_random_007 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.53519 | 3.5289 | 3.5333 | 579 | 943 | 943 | 645 | 1646 | 1396 | 0.00240114 | pass | queued_for_3seed_screen | 43584 |
| table1_random_008 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.54985 | 3.52755 | 3.5376 | 562 | 943 | 942 | 649 | 1651 | 1414 | 0.00663674 | pass | queued_for_3seed_screen | 129085 |
| table1_random_009 | random_table1_global | 2.02607e+07 | 943 | 1681 | 1364 | 78635 | 20000 | 0.0630836 | 3.46921 | 3.53349 | 3.51975 | 552 | 943 | 942 | 662 | 1654 | 1405 | 0.00584342 | pass | queued_for_3seed_screen | 69159 |

### Mask Run Queue

| mask_id | method | seeds | cycles | status | reason | suggested_output_prefix |
| --- | --- | --- | --- | --- | --- | --- |
| table1_random_000 | schur_loglambda_tuned_full_pool_softmax_tau1_literal | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_000_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds |
| table1_random_000 | linucb_concat | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_000_linucb_concat_3seeds |
| table1_random_000 | linucb_mf_interaction | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_000_linucb_mf_interaction_3seeds |
| table1_random_001 | schur_loglambda_tuned_full_pool_softmax_tau1_literal | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_001_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds |
| table1_random_001 | linucb_concat | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_001_linucb_concat_3seeds |
| table1_random_001 | linucb_mf_interaction | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_001_linucb_mf_interaction_3seeds |
| table1_random_002 | schur_loglambda_tuned_full_pool_softmax_tau1_literal | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_002_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds |
| table1_random_002 | linucb_concat | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_002_linucb_concat_3seeds |
| table1_random_002 | linucb_mf_interaction | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_002_linucb_mf_interaction_3seeds |
| table1_random_003 | schur_loglambda_tuned_full_pool_softmax_tau1_literal | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_003_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds |
| table1_random_003 | linucb_concat | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_003_linucb_concat_3seeds |
| table1_random_003 | linucb_mf_interaction | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_003_linucb_mf_interaction_3seeds |
| table1_random_004 | schur_loglambda_tuned_full_pool_softmax_tau1_literal | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_004_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds |
| table1_random_004 | linucb_concat | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_004_linucb_concat_3seeds |
| table1_random_004 | linucb_mf_interaction | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_004_linucb_mf_interaction_3seeds |
| table1_random_005 | schur_loglambda_tuned_full_pool_softmax_tau1_literal | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_005_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds |
| table1_random_005 | linucb_concat | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_005_linucb_concat_3seeds |
| table1_random_005 | linucb_mf_interaction | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_005_linucb_mf_interaction_3seeds |
| table1_random_006 | schur_loglambda_tuned_full_pool_softmax_tau1_literal | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_006_schur_loglambda_tuned_full_pool_softmax_tau1_literal_3seeds |
| table1_random_006 | linucb_concat | 0 1 2 | 20 | queued_not_run | requires mask-aware acquisition runner; generated mask indices are available | inverse_reconstruction_v1_table1_random_006_linucb_concat_3seeds |

## Task 4: Schur Shortlist vs Full

| cycle | seeds | min_top100_overlap | min_selected_batch_overlap | max_selected_changed_count | max_abs_one_step_rmse_diff | mean_abs_rating_diff | selection | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | 1 | 1 | 0 | 0 | 0 | shortlist20k_vs_all_candidate_tiled | screen_pass_low_effect |
| 5 | 3 | 1 | 1 | 0 | 0 | 0 | shortlist20k_vs_all_candidate_tiled | screen_pass_low_effect |
| 10 | 3 | 0.97 | 0.97 | 6 | 5.17833e-05 | 0.0133333 | shortlist20k_vs_all_candidate_tiled | screen_pass_low_effect |
| 15 | 3 | 0.98 | 0.98 | 4 | 5.31685e-05 | 0.0133333 | shortlist20k_vs_all_candidate_tiled | screen_pass_low_effect |
| 20 | 3 | 0.99 | 0.99 | 2 | 6.45779e-06 | 0.0166667 | shortlist20k_vs_all_candidate_tiled | screen_pass_low_effect |

## Task 5: Lambda Noise and Tuning Objective

| lambda_noise | seeds | min_top100_overlap | min_selected_batch_overlap | max_selected_changed_count | mean_selected_rating | mean_selected_entropy | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 3 | 0.99 | 0.99 | 2 | 3.69667 | 0.00607811 | cycle0_low_effect |
| 0.3 | 3 | 1 | 1 | 0 | 3.7 | 0.00612352 | cycle0_low_effect |
| 1 | 3 | 1 | 1 | 0 | 3.7 | 0.00614064 | cycle0_low_effect |
| 3 | 3 | 1 | 1 | 0 | 3.7 | 0.00614558 | cycle0_low_effect |
| 10 | 3 | 1 | 1 | 0 | 3.7 | 0.00614732 | cycle0_low_effect |

| knob_family | candidate | best_available_evidence | fig4a_effect | fig4b_effect | status | protocol_penalty |
| --- | --- | --- | --- | --- | --- | --- |
| selection_pool | shortlist20k_vs_all_candidate | min selected overlap=0.97; max changed=6 | one_step_rmse_screen_only | selected_rating_screen_only | screened_not_end_to_end_full_pool | 0.55 |
| lambda_noise | lambda_noise=0.3 | cycle0 selected overlap=1; changed=0 | cycle0_selection_screen_only | selected_rating_screen_only | screened_not_curve_confirmed | 0.3 |
| tuning_objective | retuning_frequency | prior 5-seed every-cycle vs every-5 diagnostic: standard_mask_rmse_delta=0.00056484; w_rmse_delta=-0.00854367; reward_tail_proxy_rmse_delta=-0.0129611; cumulative_reward_delta=-3.32959; source=outputs/tables/paper_faithful_tune_every_cycle_5seed_diagnostic_20260709.csv | audit | audit | source_blocked_not_closed | 0.3 |
| tuning_objective | inner_objective | mainline seed_count=5; standard_mask_rmse=1.069519; w_rmse=1.026202; reward_tail_proxy_rmse=1.304808; cumulative_reward=1241.076444; vs linucb_concat: standard_gap_linear_minus_du=0.032862, reward_gap_du_minus_linear=-148.336, reward_tail_gap_linear_minus_du=-0.0903332; vs linucb_mf_interaction: standard_gap_linear_minus_du=0.0188889, reward_gap_du_minus_linear=29.9259, reward_tail_gap_linear_minus_du=0.028623; vs linucb_bias_interaction: standard_gap_linear_minus_du=0.0583272, reward_gap_du_minus_linear=-276.25, reward_tail_gap_linear_minus_du=-0.123278 | audit | audit | source_blocked_not_closed | 0.3 |
| tuning_objective | lambda_sigma_p_grid | v2 config grid is explicit and reproducible, but not source-verified. | audit | audit | source_blocked_not_closed | 0.3 |
| tuning_objective | metric_gate_interaction | best_metric=reward_tail_proxy; comparator=linucb_mf_interaction; row_level_seed_count=5; scale_distance=0.267582; gap_distance=0.100883; paper_wrmse_reconstructed=no | audit | audit | blocked_by_metric_gate | 0 |
| tuning_objective | joint_figure4_objective | mainline seed_count=5; standard_mask_rmse=1.069519; w_rmse=1.026202; reward_tail_proxy_rmse=1.304808; cumulative_reward=1241.076444; vs linucb_concat: standard_gap_linear_minus_du=0.032862, reward_gap_du_minus_linear=-148.336, reward_tail_gap_linear_minus_du=-0.0903332; vs linucb_mf_interaction: standard_gap_linear_minus_du=0.0188889, reward_gap_du_minus_linear=29.9259, reward_tail_gap_linear_minus_du=0.028623; vs linucb_bias_interaction: standard_gap_linear_minus_du=0.0583272, reward_gap_du_minus_linear=-276.25, reward_tail_gap_linear_minus_du=-0.123278 | audit | audit | not_jointly_closed | 0 |
| same_config_figure4a_figure4b | tau0p3_current_best_same_config | standard RMSE: gap=0.036364; cumulative reward: gap=-147.979; RMSE vs mf-interaction: gap=0.0173968; reward vs mf-interaction: gap=39.3312; win rate vs concat: gap=1 | strong_rmse_recovery | comparator_dependent_reward_gap | not_jointly_closed | 0.3 |

## Task 6: Joint Candidate Leaderboard

| rank | pipeline_id | metric | mask | linear_ucb | du_net_tuning | selection | fig4a_du_error | fig4a_linear_error | fig4b_du_error | fig4b_linear_error | protocol_penalty | total_error | claim_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | candidate_0003 | reward_tail_proxy_rmse | table1_random_026 | linucb_mf_alpha_0p1_reg_1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.576023 | 0 | 35.7322 | 0.7 | 1.12449 | candidate_not_confirmed_20seed |
| 2 | candidate_0012 | reward_tail_proxy_rmse | table1_random_007 | linucb_mf_alpha_0p1_reg_1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.576023 | 0 | 35.7322 | 0.7 | 1.12449 | candidate_not_confirmed_20seed |
| 3 | candidate_0015 | reward_tail_proxy_rmse | table1_random_042 | linucb_mf_alpha_0p1_reg_1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.576023 | 0 | 35.7322 | 0.7 | 1.12449 | candidate_not_confirmed_20seed |
| 4 | candidate_0006 | reward_tail_proxy_rmse | table1_random_006 | linucb_mf_alpha_0p1_reg_1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.576023 | 0 | 35.7322 | 0.7 | 1.12449 | candidate_not_confirmed_20seed |
| 5 | candidate_0009 | reward_tail_proxy_rmse | table1_random_043 | linucb_mf_alpha_0p1_reg_1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.576023 | 0 | 35.7322 | 0.7 | 1.12449 | candidate_not_confirmed_20seed |
| 6 | candidate_0021 | reward_tail_proxy_rmse | table1_random_006 | linucb_mf_alpha_0p1_reg_10 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.581541 | 0 | 35.5971 | 0.7 | 1.12622 | candidate_not_confirmed_20seed |
| 7 | candidate_0027 | reward_tail_proxy_rmse | table1_random_007 | linucb_mf_alpha_0p1_reg_10 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.581541 | 0 | 35.5971 | 0.7 | 1.12622 | candidate_not_confirmed_20seed |
| 8 | candidate_0024 | reward_tail_proxy_rmse | table1_random_043 | linucb_mf_alpha_0p1_reg_10 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.581541 | 0 | 35.5971 | 0.7 | 1.12622 | candidate_not_confirmed_20seed |
| 9 | candidate_0030 | reward_tail_proxy_rmse | table1_random_042 | linucb_mf_alpha_0p1_reg_10 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.581541 | 0 | 35.5971 | 0.7 | 1.12622 | candidate_not_confirmed_20seed |
| 10 | candidate_0018 | reward_tail_proxy_rmse | table1_random_026 | linucb_mf_alpha_0p1_reg_10 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.581541 | 0 | 35.5971 | 0.7 | 1.12622 | candidate_not_confirmed_20seed |
| 11 | candidate_0051 | reward_tail_proxy_rmse | table1_random_006 | linucb_mf_alpha_0p3_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.570215 | 0 | 38.3371 | 0.7 | 1.12637 | candidate_not_confirmed_20seed |
| 12 | candidate_0060 | reward_tail_proxy_rmse | table1_random_042 | linucb_mf_alpha_0p3_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.570215 | 0 | 38.3371 | 0.7 | 1.12637 | candidate_not_confirmed_20seed |
| 13 | candidate_0057 | reward_tail_proxy_rmse | table1_random_007 | linucb_mf_alpha_0p3_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.570215 | 0 | 38.3371 | 0.7 | 1.12637 | candidate_not_confirmed_20seed |
| 14 | candidate_0048 | reward_tail_proxy_rmse | table1_random_026 | linucb_mf_alpha_0p3_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.570215 | 0 | 38.3371 | 0.7 | 1.12637 | candidate_not_confirmed_20seed |
| 15 | candidate_0054 | reward_tail_proxy_rmse | table1_random_043 | linucb_mf_alpha_0p3_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.570215 | 0 | 38.3371 | 0.7 | 1.12637 | candidate_not_confirmed_20seed |
| 16 | candidate_0036 | reward_tail_proxy_rmse | table1_random_006 | linucb_mf_alpha_0p1_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.577487 | 0 | 36.6615 | 0.7 | 1.1264 | candidate_not_confirmed_20seed |
| 17 | candidate_0042 | reward_tail_proxy_rmse | table1_random_007 | linucb_mf_alpha_0p1_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.577487 | 0 | 36.6615 | 0.7 | 1.1264 | candidate_not_confirmed_20seed |
| 18 | candidate_0039 | reward_tail_proxy_rmse | table1_random_043 | linucb_mf_alpha_0p1_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.577487 | 0 | 36.6615 | 0.7 | 1.1264 | candidate_not_confirmed_20seed |
| 19 | candidate_0033 | reward_tail_proxy_rmse | table1_random_026 | linucb_mf_alpha_0p1_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.577487 | 0 | 36.6615 | 0.7 | 1.1264 | candidate_not_confirmed_20seed |
| 20 | candidate_0045 | reward_tail_proxy_rmse | table1_random_042 | linucb_mf_alpha_0p1_reg_0p1 | tau0p3_current_best_same_config | same_config_figure4a_figure4b | 0.48368 | 0.577487 | 0 | 36.6615 | 0.7 | 1.1264 | candidate_not_confirmed_20seed |

## Stop Rules

| stop_rule | status | evidence | required_next |
| --- | --- | --- | --- |
| best_attainable_paper_only_reconstruction | not_met | metric, Linear UCB, mask, and DU-Net latent-variable gates are represented, but at least one remains source-blocked or queued. | run queued mask screens and promote only defensible candidates |
| near_perfect_empirical_reconstruction | not_met | metric_closed=False; linucb_closed=False; masks_screened=False; same_config_confirmed=False | top 5 joint candidates require 20-seed confirmation under the same config |
| exact_replication | not_claimable | author metric code, exact masks, Linear UCB protocol, and numeric Figure 4 data are still absent | recover source artifacts or keep exactness claim disabled |

## Decision Boundary

No row is promoted to exact replication. Current outputs support a paper-only reconstruction workflow and a mask-screening queue; they do not close the missing author metric, mask, Linear UCB, backend, or numeric Figure 4 gates.
