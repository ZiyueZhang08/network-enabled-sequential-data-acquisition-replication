# Metric Reconstruction Audit v2

- Row-level source: `outputs\row_level_predictions\paper_faithful_movielens_tau1_literal_p0_row_level_seed0`
- Paper endpoint targets: DU-Net `1.788`, Linear UCB `1.814`
- Status: **unresolved**

## Gate Interpretation

A metric passes only if endpoint scale is within 15%, DU-Net is lower than Linear UCB, and the local gap is within 20% of the paper visual gap. If no row passes, metric exactness remains a P0 caveat.

## Best Candidates

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
