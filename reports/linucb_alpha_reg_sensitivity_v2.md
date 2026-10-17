# linucb_alpha_reg_sensitivity_v2

Purpose: close the local Linear UCB alpha/regularization sensitivity gate without promoting any proxy to a paper-exact comparator.

## Protocol

- Source config: `configs/paper_exact_attempt_movielens_v2.json`
- Policy: `linucb_mf_interaction`
- Seeds: `[0, 1, 2]`
- Alpha grid: `[0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.8, 1.0]`
- Regularization grid: `[0.1, 1.0, 10.0]`
- Completed grid points: `24/24`
- Minimum seed count per point: `3`

## Sensitivity Ranges

- reward-tail proxy RMSE range: `1.1764` to `1.37469`
- standard-mask RMSE range: `1.08` to `1.12982`
- cumulative reward range: `1169.35` to `1555.45`

## Current Local Default

| variant | policy | source | feature_map | ucb_alpha | ucb_regularization | seed_count | paper_linear_ucb_reconstructed | pdf_literal_norm_rmse | pdf_literal_norm_rmse_ci95 | standard_mask_rmse | standard_mask_rmse_ci95 | cumulative_reward | cumulative_reward_ci95 | centered_cumulative_reward | centered_cumulative_reward_ci95 | reward_tail_proxy_rmse | reward_tail_proxy_rmse_ci95 | rmse | rmse_ci95 | w_rmse | w_rmse_ci95 | observed_size | observed_size_ci95 | candidate_size | candidate_size_ci95 | linear_minus_tau1_du_pdf_literal_norm_rmse | linear_minus_tau1_du_standard_mask_rmse | linear_minus_tau1_du_reward_tail_proxy_rmse | linear_minus_tau1_du_rmse | linear_minus_tau1_du_w_rmse | linear_minus_tau1_du_cumulative_reward | linear_minus_tau1_du_centered_cumulative_reward | figure4a_standard_direction | figure4a_reward_tail_direction | figure4b_reward_direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linucb_mf_alpha_0p8_reg_1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.8 | 1 | 3 | no | 0.00767169 | 1.21443e-05 | 1.08494 | 0.00171746 | 1231.72 | 68.2365 | 1129.67 | 345.867 | 1.32979 | 0.0384101 | 1.08494 | 0.00171746 | 1.0428 | 0.0277139 | 3364 | 0 | 76635 | 0 | 8.94981e-05 | 0.0126569 | 0.0101861 | 0.0126569 | 0.00595472 | -14.2215 | -61.6667 | du_lower_rmse | du_lower_rmse | du_higher_or_tie_reward |

## Lowest Reward-Tail Proxy RMSE

| variant | policy | source | feature_map | ucb_alpha | ucb_regularization | seed_count | paper_linear_ucb_reconstructed | pdf_literal_norm_rmse | pdf_literal_norm_rmse_ci95 | standard_mask_rmse | standard_mask_rmse_ci95 | cumulative_reward | cumulative_reward_ci95 | centered_cumulative_reward | centered_cumulative_reward_ci95 | reward_tail_proxy_rmse | reward_tail_proxy_rmse_ci95 | rmse | rmse_ci95 | w_rmse | w_rmse_ci95 | observed_size | observed_size_ci95 | candidate_size | candidate_size_ci95 | linear_minus_tau1_du_pdf_literal_norm_rmse | linear_minus_tau1_du_standard_mask_rmse | linear_minus_tau1_du_reward_tail_proxy_rmse | linear_minus_tau1_du_rmse | linear_minus_tau1_du_w_rmse | linear_minus_tau1_du_cumulative_reward | linear_minus_tau1_du_centered_cumulative_reward | figure4a_standard_direction | figure4a_reward_tail_direction | figure4b_reward_direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linucb_mf_alpha_0p1_reg_10 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.1 | 10 | 3 | no | 0.00793406 | 2.54035e-05 | 1.12205 | 0.0035926 | 1519.61 | 1.63482 | 2620 | 5.65803 | 1.1764 | 0.0323888 | 1.12205 | 0.0035926 | 0.938521 | 0.0223404 | 3364 | 0 | 76635 | 0 | 0.000351873 | 0.0497623 | -0.143204 | 0.0497623 | -0.0983276 | 273.67 | 1428.67 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |
| linucb_mf_alpha_0p1_reg_1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.1 | 1 | 3 | no | 0.0079201 | 1.96009e-05 | 1.12007 | 0.00277198 | 1520.36 | 7.93779 | 2622.33 | 40.9625 | 1.18094 | 0.0341587 | 1.12007 | 0.00277198 | 0.940982 | 0.0237159 | 3364 | 0 | 76635 | 0 | 0.000337915 | 0.0477884 | -0.138662 | 0.0477884 | -0.0958666 | 274.426 | 1431 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |
| linucb_mf_alpha_0p1_reg_0p1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.1 | 0.1 | 3 | no | 0.00791567 | 3.54107e-05 | 1.11945 | 0.00500783 | 1522.91 | 5.7159 | 2633.33 | 31.6513 | 1.18694 | 0.027608 | 1.11945 | 0.00500783 | 0.944456 | 0.0200262 | 3364 | 0 | 76635 | 0 | 0.000333482 | 0.0471615 | -0.132657 | 0.0471615 | -0.092393 | 276.967 | 1442 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |
| linucb_mf_alpha_0p05_reg_0p1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.05 | 0.1 | 3 | no | 0.00797072 | 2.0456e-05 | 1.12723 | 0.00289292 | 1546.98 | 1.98052 | 2759.33 | 12.719 | 1.18762 | 0.0413851 | 1.12723 | 0.00289292 | 0.945455 | 0.0288492 | 3364 | 0 | 76635 | 0 | 0.00038853 | 0.0549464 | -0.131982 | 0.0549464 | -0.091394 | 301.039 | 1568 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |
| linucb_mf_alpha_0p02_reg_1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.02 | 1 | 3 | no | 0.00798905 | 2.8407e-05 | 1.12982 | 0.00401736 | 1550.89 | 7.25417 | 2777.67 | 37.639 | 1.18779 | 0.0471094 | 1.12982 | 0.00401736 | 0.945928 | 0.0332558 | 3364 | 0 | 76635 | 0 | 0.000406863 | 0.0575391 | -0.131815 | 0.0575391 | -0.0909211 | 304.95 | 1586.33 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |

## Highest Cumulative Reward

| variant | policy | source | feature_map | ucb_alpha | ucb_regularization | seed_count | paper_linear_ucb_reconstructed | pdf_literal_norm_rmse | pdf_literal_norm_rmse_ci95 | standard_mask_rmse | standard_mask_rmse_ci95 | cumulative_reward | cumulative_reward_ci95 | centered_cumulative_reward | centered_cumulative_reward_ci95 | reward_tail_proxy_rmse | reward_tail_proxy_rmse_ci95 | rmse | rmse_ci95 | w_rmse | w_rmse_ci95 | observed_size | observed_size_ci95 | candidate_size | candidate_size_ci95 | linear_minus_tau1_du_pdf_literal_norm_rmse | linear_minus_tau1_du_standard_mask_rmse | linear_minus_tau1_du_reward_tail_proxy_rmse | linear_minus_tau1_du_rmse | linear_minus_tau1_du_w_rmse | linear_minus_tau1_du_cumulative_reward | linear_minus_tau1_du_centered_cumulative_reward | figure4a_standard_direction | figure4a_reward_tail_direction | figure4b_reward_direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linucb_mf_alpha_0p01_reg_1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.01 | 1 | 3 | no | 0.00797519 | 2.22728e-05 | 1.12786 | 0.00314985 | 1555.45 | 1.71989 | 2798 | 12.2402 | 1.19342 | 0.0436461 | 1.12786 | 0.00314985 | 0.949627 | 0.0301268 | 3364 | 0 | 76635 | 0 | 0.000393005 | 0.0555792 | -0.126181 | 0.0555792 | -0.0872213 | 309.512 | 1606.67 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |
| linucb_mf_alpha_0p005_reg_10 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.005 | 10 | 3 | no | 0.00796605 | 3.9338e-05 | 1.12657 | 0.00556324 | 1552.33 | 7.9268 | 2785.67 | 42.4817 | 1.19727 | 0.031164 | 1.12657 | 0.00556324 | 0.951714 | 0.0224961 | 3364 | 0 | 76635 | 0 | 0.000383861 | 0.0542862 | -0.122336 | 0.0542862 | -0.085135 | 306.391 | 1594.33 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |
| linucb_mf_alpha_0p02_reg_1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.02 | 1 | 3 | no | 0.00798905 | 2.8407e-05 | 1.12982 | 0.00401736 | 1550.89 | 7.25417 | 2777.67 | 37.639 | 1.18779 | 0.0471094 | 1.12982 | 0.00401736 | 0.945928 | 0.0332558 | 3364 | 0 | 76635 | 0 | 0.000406863 | 0.0575391 | -0.131815 | 0.0575391 | -0.0909211 | 304.95 | 1586.33 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |
| linucb_mf_alpha_0p005_reg_0p1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.005 | 0.1 | 3 | no | 0.00797492 | 2.28678e-05 | 1.12782 | 0.00323399 | 1550.84 | 5.88529 | 2773 | 29.6602 | 1.19161 | 0.040535 | 1.12782 | 0.00323399 | 0.948076 | 0.028871 | 3364 | 0 | 76635 | 0 | 0.000392726 | 0.0555399 | -0.127991 | 0.0555399 | -0.0887722 | 304.901 | 1581.67 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |
| linucb_mf_alpha_0p005_reg_1 | linucb_mf_interaction | local_clean_room_proxy | mf_interaction | 0.005 | 1 | 3 | no | 0.00796782 | 2.70995e-05 | 1.12682 | 0.00383245 | 1550.13 | 3.87498 | 2773.33 | 19.6761 | 1.19385 | 0.0359513 | 1.12682 | 0.00383245 | 0.949273 | 0.0257168 | 3364 | 0 | 76635 | 0 | 0.000385628 | 0.054536 | -0.125751 | 0.054536 | -0.0875754 | 304.192 | 1582 | du_lower_rmse | linear_lower_or_tie_rmse | linear_higher_reward |

## Claim Boundary

This grid is sensitivity evidence only. The paper does not expose the Linear UCB feature map, alpha, regularization, reward transform, or update protocol, so no row is promoted to `paper_linear_ucb_reconstructed`.
