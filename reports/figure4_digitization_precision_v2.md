# figure4_digitization_precision_v2

This audit formalizes the Figure 4 visual-target boundary for the v2 exact-attempt gates. It does not introduce new tuning, and it does not treat visually digitized values as author data.

## Inputs

- Digitized target table: `outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_figure4_digitized_targets.csv`
- Same-config alignment table: `outputs/tables/metric_curve_alignment_v4.csv`
- Stress bands: Figure 4a endpoint +/-0.005 w-RMSE; Figure 4b endpoint +/-10 reward units; gap band is twice the endpoint band.

## Visual Targets

| figure | paper_metric | du_endpoint_visual | linear_endpoint_visual | linear_minus_du_visual_gap | digitization_source | digitization_status | conservative_endpoint_stress_band | conservative_gap_stress_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4b | cumulative_reward | 1325 | 1390 | 65 | tmp/pdfs/figure4b_movielens_cumulative_crop.png | visual_approximation_not_author_data | 10 | 20 |
| Figure 4a | w_rmse | 1.788 | 1.814 | 0.026 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data | 0.005 | 0.01 |

## Precision Stress Verdict

| figure | precision_status | rows |
| --- | --- | --- |
| Figure 4a | gap_within_stress_but_endpoint_scale_not | 6 |
| Figure 4a | not_explained_by_digitization_stress | 15 |
| Figure 4b | not_explained_by_digitization_stress | 3 |

## Best Local Alignments Under Current Visual Targets

| status | figure | metric | comparator | curve_rmse | endpoint_abs_error_mean | local_gap | paper_gap | gap_abs_error | points | paper_metric | du_endpoint_visual | linear_endpoint_visual | linear_minus_du_visual_gap | digitization_source | digitization_status | conservative_endpoint_stress_band | conservative_gap_stress_band | endpoint_error_to_stress_ratio | gap_error_to_stress_ratio | precision_status | paper_numeric_source_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| computed | Figure 4a | reward_tail_proxy_rmse | linucb_mf_interaction | 0.48368 | 0.478123 | 0.0276842 | 0.026 | 0.00168421 | 42 | w_rmse | 1.788 | 1.814 | 0.026 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data | 0.005 | 0.01 | 95.6246 | 0.168421 | gap_within_stress_but_endpoint_scale_not | visual_digitization_only_not_author_data |
| computed | Figure 4a | w_rmse | linucb_mf_interaction | 0.775914 | 0.76238 | 0.0192864 | 0.026 | 0.0067136 | 42 | w_rmse | 1.788 | 1.814 | 0.026 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data | 0.005 | 0.01 | 152.476 | 0.67136 | gap_within_stress_but_endpoint_scale_not | visual_digitization_only_not_author_data |
| computed | Figure 4a | medium_reward_w_rmse | linucb_mf_interaction | 0.758323 | 0.741736 | 0.0179175 | 0.026 | 0.00808251 | 42 | w_rmse | 1.788 | 1.814 | 0.026 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data | 0.005 | 0.01 | 148.347 | 0.808251 | gap_within_stress_but_endpoint_scale_not | visual_digitization_only_not_author_data |
| computed | Figure 4b | cumulative_reward | linucb_concat | 72.2662 | 41.147 | 147.294 | 65 | 82.294 | 42 | cumulative_reward | 1325 | 1390 | 65 | tmp/pdfs/figure4b_movielens_cumulative_crop.png | visual_approximation_not_author_data | 10 | 20 | 4.1147 | 4.1147 | not_explained_by_digitization_stress | visual_digitization_only_not_author_data |
| computed | Figure 4b | cumulative_reward | linucb_mf_interaction | 127.674 | 131.122 | -40.0162 | 65 | 105.016 | 42 | cumulative_reward | 1325 | 1390 | 65 | tmp/pdfs/figure4b_movielens_cumulative_crop.png | visual_approximation_not_author_data | 10 | 20 | 13.1122 | 5.25081 | not_explained_by_digitization_stress | visual_digitization_only_not_author_data |
| computed | Figure 4b | cumulative_reward | linucb_bias_interaction | 62.9814 | 105.13 | 275.26 | 65 | 210.26 | 42 | cumulative_reward | 1325 | 1390 | 65 | tmp/pdfs/figure4b_movielens_cumulative_crop.png | visual_approximation_not_author_data | 10 | 20 | 10.513 | 10.513 | not_explained_by_digitization_stress | visual_digitization_only_not_author_data |

## Claim Boundary

Digitization precision can explain small gap-level differences in some Figure 4a rows, but it cannot explain the endpoint-scale mismatch. For Figure 4b, the same-config reward gap error remains larger than the conservative digitization stress band. Exact Figure 4 reproduction therefore still requires author numeric data or source code.
