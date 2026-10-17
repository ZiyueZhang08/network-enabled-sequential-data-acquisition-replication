# Figure 4 Digitized Alignment v1

This audit uses visually digitized points from the local PDF crops. These values are not author data and are used only to test endpoint direction, curve shape, and affine scale diagnostics.

## Endpoint Gate

| figure | local_metric | du_endpoint_local | linear_endpoint_local | du_endpoint_paper_visual | linear_endpoint_paper_visual | local_gap_linear_minus_du | paper_gap_linear_minus_du | gap_abs_error | direction_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | standard_mask_rmse | 1.07073 | 1.10639 | 1.788 | 1.814 | 0.035663 | 0.026 | 0.00966297 | 1 |
| Figure 4a | w_rmse | 1.02898 | 0.965524 | 1.788 | 1.814 | -0.0634527 | 0.026 | 0.0894527 | 0 |
| Figure 4b | cumulative_reward | 1246.39 | 1393.68 | 1325 | 1390 | 147.294 | 65 | 82.294 | 1 |

## Curve Diagnostics

| figure | method | local_metric | cycles | endpoint_local | endpoint_paper_visual | endpoint_abs_error | raw_curve_rmse | curve_correlation | affine_intercept_to_paper | affine_slope_to_paper | affine_curve_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | DU-Net | standard_mask_rmse | 21 | 1.07073 | 1.788 | 0.717274 | 0.734508 | 0.895132 | -1.9642 | 3.48947 | 0.0155324 |
| Figure 4a | Linear UCB | standard_mask_rmse | 21 | 1.10639 | 1.814 | 0.707611 | 0.740031 | -0.653278 | 17.147 | -13.8734 | 0.0250394 |
| Figure 4a | DU-Net | w_rmse | 21 | 1.02898 | 1.788 | 0.759023 | 0.770876 | 0.957599 | -0.498481 | 2.2118 | 0.010038 |
| Figure 4a | Linear UCB | w_rmse | 21 | 0.965524 | 1.814 | 0.848476 | 0.830449 | 0.965066 | 0.970152 | 0.861896 | 0.00866507 |
| Figure 4b | DU-Net | cumulative_reward | 21 | 1246.39 | 1325 | 78.6137 | 81.6146 | 0.998763 | 8.58889 | 1.09797 | 20.6368 |
| Figure 4b | Linear UCB | cumulative_reward | 21 | 1393.68 | 1390 | 3.68025 | 61.5131 | 0.99835 | 11.5403 | 1.05633 | 25.7838 |

## Target Provenance

| figure | method | cycle | paper_metric | paper_value | digitization_source | digitization_status |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | DU-Net | 0 | w_rmse | 1.925 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data |
| Figure 4a | DU-Net | 1 | w_rmse | 1.885 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data |
| Figure 4a | DU-Net | 2 | w_rmse | 1.862 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data |
| Figure 4a | DU-Net | 3 | w_rmse | 1.846 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data |
| Figure 4a | DU-Net | 4 | w_rmse | 1.836 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data |
| Figure 4a | DU-Net | 5 | w_rmse | 1.828 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data |
| Figure 4a | DU-Net | 6 | w_rmse | 1.82 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data |
| Figure 4a | DU-Net | 7 | w_rmse | 1.814 | tmp/pdfs/figure4a_movielens_wrmse.png | visual_approximation_not_author_data |

## Interpretation

- Figure 4a is evaluated under both local `w_rmse` and `standard_mask_rmse` because the paper text/plot ambiguity remains unresolved.
- Figure 4b uses cumulative reward. The tau1 mainline matches the paper ordering against `linucb_concat`, but the local reward gap is larger than the visual paper gap.
- Affine calibration is diagnostic only. A good affine RMSE cannot close an exact-replication gate unless the metric, split/mask, and Linear UCB protocol are also source-grounded.

## Figures

- `outputs\figures\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1\tau1_figure4a_standard_mask_rmse_digitized_alignment.png`
- `outputs\figures\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1\tau1_figure4a_w_rmse_digitized_alignment.png`
- `outputs\figures\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1\tau1_figure4b_cumulative_reward_digitized_alignment.png`
