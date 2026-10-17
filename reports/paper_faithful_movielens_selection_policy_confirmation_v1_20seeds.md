# paper_faithful_movielens_selection_policy_confirmation_v1_20seeds

## Result

                             variant  seed_count  standard_mask_rmse  standard_mask_rmse_ci95  pdf_literal_norm_rmse  pdf_literal_norm_rmse_ci95  cumulative_reward  cumulative_reward_ci95   w_rmse  w_rmse_ci95  reward_tail_proxy_rmse  reward_tail_proxy_rmse_ci95  high_rating_rmse  high_rating_rmse_ci95  low_rating_rmse  low_rating_rmse_ci95
               linucb_mf_interaction          20            1.087422                 0.003468               0.007689                    0.000025        1206.370096               16.657837 1.048264     0.007036                1.336719                     0.009882               NaN                    0.0              NaN                   0.0
schur_tuned_full_pool_softmax_tau0p3          20            1.070025                 0.003458               0.007566                    0.000024        1245.701252               14.986018 1.028445     0.007719                1.308482                     0.010809               NaN                    0.0              NaN                   0.0
                       linucb_concat          20            1.106389                 0.005067               0.007823                    0.000036        1393.680246                8.916846 0.965524     0.010247                1.216032                     0.015250               NaN                    0.0              NaN                   0.0
                    schur_tuned_topk          20            1.088861                 0.004436               0.007699                    0.000031        1250.556709               33.057971 1.040197     0.012977                1.326710                     0.018056               NaN                    0.0              NaN                   0.0

## Paired Gaps

             baseline  mean_standard_gap  standard_win_rate  wins  seeds  mean_pdf_gap  mean_reward_gap
        linucb_concat           0.036364               1.00    20     20      0.000257      -147.978994
     schur_tuned_topk           0.018836               1.00    20     20      0.000133        -4.855456
linucb_mf_interaction           0.017397               0.95    19     20      0.000123        39.331156

## Selection Features

                             variant  selected_avg_rating  selected_rating_5_share  selected_rating_ge4_share  selected_unique_users  selected_unique_items  selected_avg_item_popularity  selected_avg_entropy_percentile  selected_avg_reward_percentile
                    schur_tuned_topk                  NaN                      NaN                        NaN                    NaN                    NaN                           NaN                         0.917998                        0.565989
                       linucb_concat                  NaN                      NaN                        NaN                    NaN                    NaN                           NaN                              NaN                             NaN
               linucb_mf_interaction                  NaN                      NaN                        NaN                    NaN                    NaN                           NaN                              NaN                             NaN
schur_tuned_full_pool_softmax_tau0p3                3.513                 0.198333                   0.550833              18.158225               13.79585                       1.13085                         0.077984                        0.500970

## Diagnosis

`schur_tuned_full_pool_softmax_tau0p3` has standard RMSE 1.070025 over 20 seeds.
It beats top-k by 0.018836 with win-rate 20/20.
It beats `linucb_concat` by 0.036364 with win-rate 20/20.
It also beats `linucb_mf_interaction` by 0.017397 with win-rate 19/20.
This confirms selection policy is not a minor cosmetic detail: paper-style full-pool softmax can recover substantially more Figure 4a RMSE gap than deterministic top-k, while reward remains low relative to concat.

## Recommendation

Do not overwrite the frozen deterministic top-k mainline retroactively. Instead, add this as `selection_policy_confirmation_v1`: a paper-style sampling robustness result and a plausible explanation for the remaining Figure 4a gap.

## Artifact Index

- `outputs\tables\paper_faithful_movielens_selection_policy_confirmation_v1_20seeds_final_summary.csv`
- `outputs\tables\paper_faithful_movielens_selection_policy_confirmation_v1_20seeds_paired_gaps.csv`
- `outputs\tables\paper_faithful_movielens_selection_policy_confirmation_v1_20seeds_gap_summary.csv`
- `outputs\tables\paper_faithful_movielens_selection_policy_confirmation_v1_20seeds_per_cycle.csv`
- `outputs\tables\paper_faithful_movielens_selection_policy_confirmation_v1_20seeds_selection_features.csv`
- `outputs\figures\paper_faithful_movielens_selection_policy_confirmation_v1_20seeds_standard_rmse_curve.png`
- `outputs\figures\paper_faithful_movielens_selection_policy_confirmation_v1_20seeds_reward_rmse_tradeoff.png`
