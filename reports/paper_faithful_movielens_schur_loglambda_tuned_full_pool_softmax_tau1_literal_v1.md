# Tau1 Literal Mainline Bundle: paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1

## Decision

`tau1 literal softmax` is the canonical mainline. `tau0.3` is retained only as a robustness/diagnostic branch.

## Gate Snapshot

- tau1 final standard RMSE: 1.070726
- vs linucb_concat wins: 20/20
- vs linucb_mf_interaction wins: 19/20
- row-level v3 status: partial until all required 20-seed prediction CSVs are generated.

## Final Summary

| claim_role | variant | seed_count | standard_mask_rmse | pdf_literal_norm_rmse | cumulative_reward | w_rmse |
| --- | --- | --- | --- | --- | --- | --- |
| mainline | schur_loglambda_tuned_full_pool_softmax_tau1_literal | 20 | 1.07073 | 0.00757118 | 1246.39 | 1.02898 |
| linear_ucb_proxy | linucb_bias_interaction | 20 | 1.13033 | 0.00799265 | 1521.65 | 0.940149 |
| linear_ucb_proxy | linucb_mf_interaction | 20 | 1.08742 | 0.00768923 | 1206.37 | 1.04826 |
| linear_ucb_proxy | linucb_concat | 20 | 1.10639 | 0.00782335 | 1393.68 | 0.965524 |
| robustness_diagnostic | schur_loglambda_tuned_full_pool_softmax_tau0p3 | 20 | 1.07003 | 0.00756622 | 1245.7 | 1.02844 |
| deterministic_ablation | schur_tuned_topk | 20 | 1.08886 | 0.00769941 | 1250.56 | 1.0402 |

## Same-Config Figure 4a / 4b

| figure | metric | tau1_du_net | linear_ucb | paper_du_net_visual | paper_linear_ucb_visual | local_gap_linear_minus_tau1 | paper_gap_linear_minus_du | match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | standard_mask_rmse executable proxy | 1.07073 | 1.10639 | 1.788 | 1.814 | 0.035663 | 0.026 | direction_yes_scale_no |
| Figure 4b | cumulative_reward | 1246.39 | 1393.68 | 1325 | 1390 | 147.294 | 65 | ranking_yes_magnitude_partial |
| Figure 4b | cumulative_reward_lambda0p3_diagnostic | 1245.7 | 1393.68 | 1325 | 1390 | 147.979 | 65 | diagnostic_only |
| Figure 4a | standard_mask_rmse_vs_mf_interaction | 1.07073 | 1.08742 | 1.788 | 1.814 | 0.0166957 | 0.026 | proxy_direction_yes |

## Metric Audit Snapshot

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

## Row-Level Completion

| method | available_seed_count | available_seeds | required_seed_count | status | source_dirs |
| --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | 1 | 0 | 20 | partial | outputs\row_level_predictions\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1;outputs\row_level_predictions\paper_faithful_movielens_tau1_literal_p0_row_level_seed0 |
| linucb_concat | 1 | 0 | 20 | partial | outputs\row_level_predictions\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1;outputs\row_level_predictions\paper_faithful_movielens_tau1_literal_p0_row_level_seed0 |
| linucb_mf_interaction | 0 |  | 20 | missing | outputs\row_level_predictions\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1;outputs\row_level_predictions\paper_faithful_movielens_tau1_literal_p0_row_level_seed0 |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | 0 |  | 20 | missing | outputs\row_level_predictions\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1;outputs\row_level_predictions\paper_faithful_movielens_tau1_literal_p0_row_level_seed0 |
| schur_tuned_topk | 0 |  | 20 | missing | outputs\row_level_predictions\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1;outputs\row_level_predictions\paper_faithful_movielens_tau1_literal_p0_row_level_seed0 |
| lambda0p3_diagnostic | 0 |  | 20 | missing | outputs\row_level_predictions\paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1;outputs\row_level_predictions\paper_faithful_movielens_tau1_literal_p0_row_level_seed0 |

## Boundary

This bundle freezes the strongest current paper-only clean-room mainline. It does not claim exact replication until metric, split/mask, Linear UCB protocol, and ELPH/backend gates are closed.

## Manifest

| artifact_path | description | status | bytes | sha256 | created_at |
| --- | --- | --- | --- | --- | --- |
| configs/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1.json | canonical tau1 mainline config | available | 2075 | 27963fd5d9f8009c0d7446fdeaa2183787615548c92b2be2c2fab0a9676a8f70 | 2026-07-10T20:01:34.606089+00:00 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_final_summary.csv | 20-seed final metrics | available | 2698 | 15c248bfa80b498f8153df28e13fc2f5d002d23a58806f6f7634edda9a9b6293 | 2026-07-10T20:01:34.606089+00:00 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_per_seed_gaps.csv | per-seed comparator gaps | available | 16966 | ab76f0f90a1f24885b2e3205bd56b4a317c0e7293f504d67b8fe98e0f7fb5de8 | 2026-07-10T20:01:34.606089+00:00 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_per_cycle_curve.csv | 21-cycle mean curves | available | 16824 | bc36288d9a6a05c005d74b2d1cc311a5f56dc5198b6e68aa1a830117d0b2b18f | 2026-07-10T20:01:34.606089+00:00 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_one_config_figure4a_figure4b.csv | same-config Figure 4a/Figure 4b comparison | available | 705 | 9618efcdc747a282ab96783b986bf765703d58ac900201c94abd8193c946d722 | 2026-07-10T20:01:34.606089+00:00 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_metric_audit.csv | row-level metric audit snapshot | available | 5423 | f48368a67a6889d384a09fb129a62e5ef72eae803abdc785193f7546b0f04f50 | 2026-07-10T20:01:34.606089+00:00 |
| outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_row_level_status.csv | row-level prediction completion status | available | 1538 | a119310a97ba9ba472fb0ee5f481fc1b021e2e453950b759bfd05e087ca862eb | 2026-07-10T20:01:34.606089+00:00 |
| outputs/row_level_predictions/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1 | canonical row-level prediction directory | available_directory | 0 |  | 2026-07-10T20:01:34.606089+00:00 |
| outputs/logs/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_bundle_log.txt | bundle generation log | available | 289 | c3c02ffca6934329a15e4e61825486edfc799b60ac558b9a0a4a2f3f18dff1ce | 2026-07-10T20:01:34.606089+00:00 |
| outputs/row_level_predictions/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/schur_loglambda_tuned_full_pool_softmax_tau1_literal__seed_00.csv | staged row-level prediction schur_loglambda_tuned_full_pool_softmax_tau1_literal__seed_00.csv | available | 7.77207e+06 | 9e909810ff80b03025134c58c2a925f5fa3fe5abe98d371d7abc031c7fb01495 | 2026-07-10T20:01:34.606089+00:00 |
| outputs/row_level_predictions/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/linucb_concat__seed_00.csv | staged row-level prediction linucb_concat__seed_00.csv | available | 6.02107e+06 | 0438d0b2663e99bcd2581222c3642c79e364aea48d7cbda1d1d60595c313e1ff | 2026-07-10T20:01:34.606089+00:00 |
| outputs/figures/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_standard_rmse_curve.png | figure tau1_standard_rmse_curve.png | available | 106433 | 3e64e04bd638e23c71cb6c66d18ed1991244cf16b9d6f7161b8cd309d8d06e0e | 2026-07-10T20:01:34.606089+00:00 |
| outputs/figures/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/tau1_cumulative_reward_curve.png | figure tau1_cumulative_reward_curve.png | available | 95731 | cabd9d3c27ae60fa4465550c13136617702452eaed98eec552708cf7dbe6ad39 | 2026-07-10T20:01:34.606089+00:00 |
