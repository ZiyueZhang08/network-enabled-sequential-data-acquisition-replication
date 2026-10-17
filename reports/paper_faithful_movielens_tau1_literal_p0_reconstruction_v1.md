# Tau1 Literal P0 Reconstruction Bundle: paper_faithful_movielens_tau1_literal_p0_reconstruction_v1

## Mainline Decision

`tau1_literal` is now the paper-faithful mainline because the paper states exponential weighting without a temperature. `tau0.3` remains a calibrated robustness result, not the main claim.

| version | status | seed_count | standard_mask_rmse | pdf_literal_norm_rmse | cumulative_reward | claim_role |
| --- | --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | literal paper-style mainline | 20 | 1.07073 | 0.00757118 | 1246.39 | mainline |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | calibrated robustness / best-RMSE robustness | 20 | 1.07003 | 0.00756622 | 1245.7 | supporting |
| schur_tuned_topk | deterministic ablation | 20 | 1.08886 | 0.00769941 | 1250.56 | supporting |

## Same-Config Figure 4a / Figure 4b

Paper endpoints are approximate visual anchors from the supplied Figure 4 visual extraction. They are not a replacement for full WebPlotDigitizer curve extraction.

| figure | metric | DU-Net local | Linear local | DU-Net paper | Linear paper | local_gap_linear_minus_du | paper_gap_linear_minus_du | ranking_match | scale_match | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | standard_mask_rmse executable proxy | 1.07073 | 1.10639 | 1.788 | 1.814 | 0.035663 | 0.026 | yes | no | metric_unresolved |
| Figure 4a | pdf_literal_norm_rmse | 0.00757118 | 0.00782335 | 1.788 | 1.814 | 0.000252175 | 0.026 | yes | no | metric_unresolved |
| Figure 4b | cumulative_reward | 1246.39 | 1393.68 | 1325 | 1390 | 147.294 | 65 | yes | partial | partial_recovery_gap_too_large |
| Figure 4b | cumulative_reward vs mf_interaction | 1246.39 | 1206.37 | 1325 | 1390 | -40.0162 | 65 | not_author_linear_ucb | diagnostic_only | comparator_proxy_only |

## Tau1 Final Results

- tau1 standard RMSE: 1.070726
- linucb_concat standard RMSE: 1.106389
- linucb_mf_interaction standard RMSE: 1.087422
- tau1 cumulative reward: 1246.386
- linucb_concat cumulative reward: 1393.680

| variant | seed_count | standard_mask_rmse | pdf_literal_norm_rmse | cumulative_reward | w_rmse |
| --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | 20 | 1.07073 | 0.00757118 | 1246.39 | 1.02898 |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | 20 | 1.07003 | 0.00756622 | 1245.7 | 1.02844 |
| schur_tuned_topk | 20 | 1.08886 | 0.00769941 | 1250.56 | 1.0402 |
| linucb_bias_interaction | 20 | 1.13033 | 0.00799265 | 1521.65 | 0.940149 |
| linucb_mf_interaction | 20 | 1.08742 | 0.00768923 | 1206.37 | 1.04826 |
| linucb_concat | 20 | 1.10639 | 0.00782335 | 1393.68 | 0.965524 |

## Comparator Gaps

| comparator | mean_standard_gap | standard_win_rate | wins | seeds | mean_pdf_gap | mean_reward_gap |
| --- | --- | --- | --- | --- | --- | --- |
| linucb_bias_interaction | 0.0596056 | 1 | 20 | 20 | 0.000421475 | -275.26 |
| linucb_concat | 0.035663 | 1 | 20 | 20 | 0.000252175 | -147.294 |
| schur_tuned_topk | 0.0181352 | 1 | 20 | 20 | 0.000128235 | -4.17041 |
| linucb_mf_interaction | 0.0166957 | 0.95 | 19 | 20 | 0.000118057 | 40.0162 |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | -0.000701059 | 0.4 | 8 | 20 | -4.95724e-06 | 0.685044 |

## Metric Reconstruction Gate

| metric | weight | comparator | DU-Net | Linear UCB | gap | scale_distance_to_paper | gap_distance_to_paper | direction | metric_gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M5_squared_weighted | w=item_popularity/mean | linucb_concat | 1.40653 | 2.1393 | 0.732762 | 0.196336 | 27.1831 | DU-Net < Linear UCB | unresolved |
| M5_squared_weighted | w=max(rating-midpoint,0)+eps | linucb_concat | 1.34016 | 1.24309 | -0.0970683 | 0.282598 | 4.7334 | not_matched | unresolved |
| M5_squared_weighted | w=max(rating-midpoint,0) | linucb_concat | 1.26966 | 1.17735 | -0.0923166 | 0.320433 | 4.55064 | not_matched | unresolved |
| M4_weighted_rmse | w=inverse_item_popularity | linucb_concat | 1.11571 | 1.15591 | 0.0402032 | 0.369392 | 0.546278 | DU-Net < Linear UCB | unresolved |
| M8_raw_unclipped_rmse | w=1 | linucb_concat | 1.06964 | 1.09802 | 0.0283837 | 0.398231 | 0.0916823 | DU-Net < Linear UCB | unresolved |
| M1_standard_rmse | w=1 | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | 0.398245 | 0.0903541 | DU-Net < Linear UCB | unresolved |
| M4_weighted_rmse | w=1 | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | 0.398245 | 0.0903541 | DU-Net < Linear UCB | unresolved |
| M4_weighted_rmse | w=test_weight_candidates | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | 0.398245 | 0.0903541 | DU-Net < Linear UCB | unresolved |
| M5_squared_weighted | w=1 | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | 0.398245 | 0.0903541 | DU-Net < Linear UCB | unresolved |
| M5_squared_weighted | w=test_weight_candidates | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | 0.398245 | 0.0903541 | DU-Net < Linear UCB | unresolved |
| M6_centered_rmse | w=1 | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | 0.398245 | 0.0903541 | DU-Net < Linear UCB | unresolved |
| M9_clipped_rmse | w=1 | linucb_concat | 1.06963 | 1.09798 | 0.0283492 | 0.398245 | 0.0903541 | DU-Net < Linear UCB | unresolved |

## Fold Sensitivity Gate

_No tau1 fold-sensitivity rows yet. The fold runner now uses tau1 full-pool softmax, but the u1-u5 x 3-seed rerun still needs to be executed._

## Gate Summary

| gate | status | evidence |
| --- | --- | --- |
| Metric | unresolved | metric reconstruction audit v2 |
| Entropy Eq3.6 | closed_cleanroom | explicit observed-kernel Schur complement equivalence note and unit test |
| Split | unresolved | fold sensitivity not yet run |
| Linear UCB | proxy_set | concat, mf_interaction, bias_interaction comparators; author library protocol unavailable |
| DU-Net | partial | Schur Eq3.6-equivalent conditional variance + DLS + tau1 selection; ELPH/exact author backend still open |
| Figure 4a | strong_pattern_recovery | tau1 lower than linucb_concat and mf_interaction under executable standard RMSE |
| Figure 4b | partial | ranking vs Linear UCB matches paper direction, local reward gap is too large |

## Conclusion

This is the strongest current paper-only clean-room reproduction path, not an exact replication. The mainline now removes the controllable selection inconsistency by using tau1 literal softmax, and the Schur entropy implementation is locked against the explicit observed-kernel Eq3.6 form. Metric, exact split/mask, author Linear UCB, and ELPH/exact author backend remain the limiting ambiguities.

## Artifact Manifest

| artifact_path | description | created_at | bytes | sha256 |
| --- | --- | --- | --- | --- |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_20seeds_metrics.csv | tau1 literal 20-seed metrics | 2026-07-10T19:32:14.357738+00:00 | 273230 | 3fef691b0ae981e8a1eda9da8e9498d87d0bd1df4f397bc6e949e459e9310b36 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_20seeds_final_summary.csv | tau1 literal 20-seed final summary | 2026-07-10T19:32:14.357738+00:00 | 744 | bd84000be5babb294d4db937e4b3f668ab4faeb1904c332be31e7df6b726aa0e |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau0p3_v1_final_summary.csv | tau0.3 robustness final summary | 2026-07-10T19:32:14.357738+00:00 | 2299 | beb2a0e2c2e0dd686f78a41bc00036f458c6634a400454486aa47d3e45902c0e |
| outputs/tables/paper_faithful_movielens_comparator_sensitivity_v1_20seeds_final_summary.csv | Linear UCB proxy comparator summary | 2026-07-10T19:32:14.357738+00:00 | 1727 | 7e3dac3e7bd16a9c190b47c35b82c57599e84880861bad17561343ef72ae3042 |
| outputs/tables/paper_faithful_movielens_tau1_literal_metric_reconstruction_audit_v2_comparison.csv | metric reconstruction audit v2 comparison | 2026-07-10T19:32:14.357738+00:00 | 5512 | f33ad438480101013290bf8ac037344c69b8f2e3313dd439e5ee093bff759784 |
| outputs/tables/paper_faithful_movielens_tau1_literal_p0_reconstruction_v1_selection_position.csv | selection role table | 2026-07-10T19:32:14.357738+00:00 | 525 | d6e836bad9c65ef0bcaac5c8dea40c38c1015127dc459f12d8579ae170eb8c9d |
| outputs/tables/paper_faithful_movielens_tau1_literal_p0_reconstruction_v1_one_config_figure4a_figure4b.csv | same-config Figure 4a/Figure 4b comparison table | 2026-07-10T19:32:14.357738+00:00 | 795 | 61df703013194df5b6777030b5d5492b4179f99c6f7696daf04b430bd661781b |
| outputs/tables/paper_faithful_movielens_tau1_literal_p0_reconstruction_v1_gate_summary.csv | P0 gate summary | 2026-07-10T19:32:14.357738+00:00 | 672 | 184d59b6939b4625e6fd83f0c34c043b524b674db7f92a09773fd60a6bc4ef0b |
