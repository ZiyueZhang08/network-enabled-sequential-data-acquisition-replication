# Schur Shortlist vs Full Audit v2

Scope: Phase 4 diagnostic for the 20k Schur shortlist approximation. This replays the frozen v2 tau=1 mainline and compares shortlist-vs-full Schur scoring at selected cycles.

## Summary

| cycle | seeds | min_top100_overlap | min_selected_batch_overlap | max_selected_changed_count | max_abs_one_step_rmse_diff | max_abs_rating_diff |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 3 | 1 | 1 | 0 | 0 | 0 |
| 5 | 3 | 1 | 1 | 0 | 0 | 0 |
| 10 | 3 | 0.97 | 0.97 | 6 | 5.17833e-05 | 0.04 |
| 15 | 3 | 0.98 | 0.98 | 4 | 5.31685e-05 | 0.02 |
| 20 | 3 | 0.99 | 0.99 | 2 | 6.45779e-06 | 0.03 |

## Results

| seed | cycle | candidate_count | shortlist_size | top100_overlap | selected_batch_overlap | selected_changed_count | mean_entropy_short_selected | mean_entropy_full_selected | selected_entropy_diff_full_minus_short | mean_rating_short_selected | mean_rating_full_selected | selected_rating_diff_full_minus_short | one_step_standard_rmse_short | one_step_standard_rmse_full | one_step_standard_rmse_diff_full_minus_short | mean_score_abs_diff | max_score_abs_diff | mean_entropy_abs_diff | max_entropy_abs_diff | full_entropy_nonzero_share | short_entropy_nonzero_share | lambda_entropy | sigma | audit_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 78635 | 20000 | 1 | 1 | 0 | 0.0084104 | 0.0084104 | 0 | 3.56 | 3.56 | 0 | 0 | 0 | 0 | 0.0961884 | 1.39231 | 8.47793e-05 | 0.000262513 | 1 | 0.25434 | 1 | 0.1 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 0 | 5 | 78135 | 20000 | 1 | 1 | 0 | 0.00606608 | 0.00606608 | 0 | 3.38 | 3.38 | 0 | 0 | 0 | 0 | 13.9278 | 166.458 | 8.69421e-05 | 0.000268407 | 1 | 0.255967 | 100 | 0.4 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 0 | 10 | 77635 | 20000 | 0.97 | 0.97 | 6 | 0.000269172 | 0.000401728 | 0.000132556 | 4.44 | 4.4 | -0.04 | 1.09249 | 1.09244 | -5.17833e-05 | 0.0222027 | 0.214528 | 0.000130672 | 0.000359449 | 1 | 0.257616 | 0.1 | 1.6 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 0 | 15 | 77135 | 20000 | 0.99 | 0.99 | 2 | 0.00349766 | 0.00349822 | 5.64523e-07 | 3.45 | 3.43 | -0.02 | 1.0963 | 1.09625 | -5.31685e-05 | 2.30819 | 16.3407 | 0.000130509 | 0.000361191 | 1 | 0.259286 | 10 | 1.6 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 0 | 20 | 76635 | 20000 | 0.99 | 0.99 | 2 | 0.00329985 | 0.00330058 | 7.32272e-07 | 3.22 | 3.2 | -0.02 | 1.09984 | 1.09983 | -6.45779e-06 | 2.23359 | 35.35 | 0.000128985 | 0.000363091 | 1 | 0.260977 | 10 | 1.6 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 1 | 0 | 78635 | 20000 | 1 | 1 | 0 | 0.000158508 | 0.00022752 | 6.9012e-05 | 4.49 | 4.49 | 0 | 0 | 0 | 0 | 0 | 0 | 8.65754e-05 | 0.000292567 | 1 | 0.25434 | 0 | 0.1 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 1 | 5 | 78135 | 20000 | 1 | 1 | 0 | 0.00810379 | 0.00810379 | 0 | 3.4 | 3.4 | 0 | 0 | 0 | 0 | 32.7425 | 313.197 | 7.90685e-05 | 0.000284556 | 1 | 0.255967 | 300 | 0.1 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 1 | 10 | 77635 | 20000 | 1 | 1 | 0 | 0.00666144 | 0.00666144 | 0 | 3.34 | 3.34 | 0 | 0 | 0 | 0 | 2.9679 | 32.363 | 6.87978e-05 | 0.000228764 | 1 | 0.257616 | 30 | 0.2 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 1 | 15 | 77135 | 20000 | 1 | 1 | 0 | 0.00345176 | 0.00345176 | 0 | 3.55 | 3.55 | 0 | 0 | 0 | 0 | 22.0719 | 191.125 | 0.000129636 | 0.000375323 | 1 | 0.259286 | 100 | 1.6 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 1 | 20 | 76635 | 20000 | 1 | 1 | 0 | 0.00291638 | 0.00291638 | 0 | 3.54 | 3.54 | 0 | 0 | 0 | 0 | 22.1226 | 135.43 | 0.000129315 | 0.000371261 | 1 | 0.260977 | 100 | 1.6 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 2 | 0 | 78635 | 20000 | 1 | 1 | 0 | 0.0099199 | 0.0099199 | 0 | 3.54 | 3.54 | 0 | 0 | 0 | 0 | 31.7521 | 359.331 | 8.41241e-05 | 0.000285608 | 1 | 0.25434 | 300 | 0.1 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 2 | 5 | 78135 | 20000 | 1 | 1 | 0 | 0.0053949 | 0.0053949 | 1.73472e-18 | 3.46 | 3.46 | 0 | 0 | 0 | 0 | 4.34056 | 69.0268 | 8.69985e-05 | 0.000280412 | 1 | 0.255967 | 30 | 0.4 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 2 | 10 | 77635 | 20000 | 1 | 1 | 0 | 0.00492366 | 0.00492366 | 0 | 3.83 | 3.83 | 0 | 0 | 0 | 0 | 13.0502 | 137.911 | 6.96886e-05 | 0.000218854 | 1 | 0.257616 | 100 | 0.2 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 2 | 15 | 77135 | 20000 | 0.98 | 0.98 | 4 | 0.00350588 | 0.00350861 | 2.7306e-06 | 3.66 | 3.64 | -0.02 | 1.08813 | 1.0881 | -2.60042e-05 | 0.458492 | 3.30822 | 7.83847e-05 | 0.000247511 | 1 | 0.259286 | 3 | 0.4 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |
| 2 | 20 | 76635 | 20000 | 0.99 | 0.99 | 2 | 0.00289 | 0.00289191 | 1.90048e-06 | 3.48 | 3.45 | -0.03 | 1.08877 | 1.08877 | 3.3281e-06 | 0.487887 | 3.67077 | 6.98876e-05 | 0.000226688 | 1 | 0.260977 | 3 | 0.4 | cycles_0_5_10_15_20_frozen_v2_mainline_replay |

## Decision

If top100 or selected-batch overlap is below 0.95, or one-step RMSE changes materially, the shortlist approximation remains a live exactness risk and requires a production all-candidate rerun. This audit still does not replace an end-to-end full-candidate 20-cycle run.
