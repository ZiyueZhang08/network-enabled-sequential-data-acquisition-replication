# Failed Gate Reverse Investigation: paper_exact_attempt_movielens_v1

Purpose: reverse-investigate each failed exact-replication gate. This is not a tuning report.

## 1. Current mainline exact config

- Mainline variant: `schur_loglambda_tuned_full_pool_softmax_tau1_literal` with policy `paper_faithful_dunet`.
- Selection policy: `exp`; tau `1.0`.
- Entropy mode: `schur_conditional_variance`; entropy transform `none`.
- Lambda tuning grid: `[0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]`; diffusion p grid `[0.0]`.
- Sigma grid: `[0.1, 0.2, 0.4, 0.8, 1.6]`.
- Tuning frequency: every `5` cycles using `standard_mask_rmse`.
- DLS mode: scope `full`, neighbor mode `graph_edges`, top-k `20`.
- MF backend: `svdpp`, rank `20`, epochs `20`, lr `0.015`, reg `0.06`.
- Diffusion probability p is fixed to `0.0` in the mainline grid.
- Split: `movielens_u1_fallback_table1_density`, users `943`, items `1681`, initial size `1364`.

The exact-attempt variants share the same executable path (`run_policy` / `paper_faithful_dunet`) and differ by config overrides: tau1 softmax, tau0.3 softmax, deterministic top-k, and a lambda0.3 diagnostic. The Linear UCB rows share the same acquisition loop but use local proxy scoring functions. In the exact-attempt replay, `lambda0p3_diagnostic` seed0 is numerically identical to tau1 because adaptive tuning can overwrite the initial lambda override.

Within the exact-attempt seed0 final-cycle artifacts, standard RMSE, PDF-literal, and local w-RMSE come from the same run family. The 20-seed tau1 bundle is aggregate metrics only; arbitrary row-level metric formulas are currently available only for seed0 final-cycle files.

Final strict gate table:

| Gate | pass/fail | evidence |
| --- | --- | --- |
| Metric exactness | fail | v3 found no paper_wrmse_reconstructed metric; best=sqrt_mean_squared_weighted_max_rating_minus_3_eps_weighted_rmse scale_distance=0.2499914195776343 gap_distance=0.1834555770101863; schur_loglambda_tuned_full_pool_softmax_tau1_literal:1/20 final_cycle_only; linucb_concat:1/20 final_cycle_only; linucb_mf_interaction:1/20 final_cycle_only; schur_loglambda_tuned_full_pool_softmax_tau0p3:1/20 final_cycle_only; schur_tuned_topk:1/20 final_cycle_only; lambda0p3_diagnostic:1/20 final_cycle_only |
| Row-level completeness | fail | schur_loglambda_tuned_full_pool_softmax_tau1_literal:1/20 final_cycle_only; linucb_concat:1/20 final_cycle_only; linucb_mf_interaction:1/20 final_cycle_only; schur_loglambda_tuned_full_pool_softmax_tau0p3:1/20 final_cycle_only; schur_tuned_topk:1/20 final_cycle_only; lambda0p3_diagnostic:1/20 final_cycle_only |
| Split exactness | fail | exact public mask rows=0; audit rows=6; current config uses u1 fallback; public search did not recover author MovieLens initial/candidate/test masks. |
| Linear UCB exactness | fail | external installed candidates=0; local proxy rows=3; no unique paper_linear_ucb_reconstructed comparator. |
| DU-Net Eq.3.6 | pass | explicit-kernel Schur toy equivalence report shows max abs diff 2.220e-16; noise/backend caveats remain separate. |
| DLS/ELPH | fail | MovieLens exact-attempt config uses GCMC/user metadata features; no recovered author ELPH backend is installed or validated. |
| Selection | pass | tau1 final standard RMSE=1.070726; existing 20-seed gate: vs concat 20/20, vs mf_interaction 19/20; exact-attempt row-level replay currently seed0 only |
| Figure 4a | fail | standard_mask_rmse direction=yes; w_rmse direction=no; y-axis metric exactness fails. |
| Figure 4b | fail | ranking direction=yes; gap_abs_error=82.294 vs visual target |

Row-level completion:

| method | available_seed_count | available_seeds | available_cycles | test_row_count | required_seed_count | row_level_scope | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| linucb_concat | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| linucb_mf_interaction | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| schur_tuned_topk | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |
| lambda0p3_diagnostic | 1 | 0 | 20 | 20000 | 20 | final_cycle_only | partial |

## 2. Metric exactness answers

For N=20000 and standard RMSE about 1.07, sqrt(sum(err^2))/N = standard_RMSE/sqrt(N) = 1.07/sqrt(20000) = about 0.00757. Therefore a Figure 4a y-axis around 1.78-1.82 cannot be the literal PDF formula.

The paper may have plotted standard RMSE or a weighted RMSE while writing a PDF-literal formula, but the current evidence cannot prove which executable definition was used. The PDF text names w-RMSE and reward concepts, yet does not expose weights, normalization, mask, or aggregation details.

Requested metric candidate table, seed0 final-cycle where row-level formulas are available:

| metric | du_net_tau1 | linucb_concat | linucb_mf | gap_vs_concat | gap_vs_mf | scale_dist_vs_paper_concat | gap_dist_vs_paper_concat | direction_vs_concat | source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| standard_mask_rmse | 1.069631 | 1.097981 | 1.086404 | 0.02834921 | 0.01677220 | 0.71719393 | 0.00234921 | paper_like | run |
| pdf_literal_norm_rmse | 0.00756344 | 0.00776390 | 0.00768203 | 0.00020046 | 0.00011860 | 1.793336 | 0.02579954 | paper_like | run |
| current_w_rmse | 1.050984 | 0.98365176 | 1.066261 | -0.06733208 | 0.01527705 | 0.78368220 | 0.09333208 | reversed_or_tie | run |
| mild_reward_w_rmse | 1.063743 | 1.062995 | 1.080046 | -0.00074833 | 0.01630251 | 0.73763117 | 0.02674833 | reversed_or_tie | run |
| medium_reward_w_rmse | 1.060429 | 1.042877 | 1.076466 | -0.01755182 | 0.01603720 | 0.74934681 | 0.04355182 | reversed_or_tie | run |
| sigmoid_w_rmse | 0.96157232 | 0.93935214 | 0.97482427 | -0.02222018 | 0.01325195 | 0.85053777 | 0.04822018 | reversed_or_tie | run |
| reward_tail_proxy | 1.340157 | 1.243088 | 1.361387 | -0.09706828 | 0.02123015 | 0.50937752 | 0.12306828 | reversed_or_tie | run |
| rating_weighted_rmse | 0.98346243 | 0.96956383 | 0.99746383 | -0.01389861 | 0.01400140 | 0.82448687 | 0.03989861 | reversed_or_tie | row |
| rating_normalized_weighted_rmse | 0.98346243 | 0.96956383 | 0.99746383 | -0.01389861 | 0.01400140 | 0.82448687 | 0.03989861 | reversed_or_tie | row |
| centered_rating_rmse | 1.069631 | 1.097981 | 1.086404 | 0.02834921 | 0.01677220 | 0.71719393 | 0.00234921 | paper_like | row |
| unclipped_standard_rmse | 1.069641 | 1.098024 | 1.086429 | 0.02838374 | 0.01678823 | 0.71716754 | 0.00238374 | paper_like | row |
| 0_1_normalized_rmse | 0.26740787 | 0.27449517 | 0.27160092 | 0.00708730 | 0.00419305 | 1.530048 | 0.01891270 | paper_like | row |

Best candidates from the broader v3 row-level audit:

| metric | comparator | DU-Net endpoint | Linear UCB endpoint | gap | direction | scale_distance_to_paper | gap_distance_to_paper | definition_defensible_from_text | raw_match | affine_match | paper_wrmse_reconstructed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sqrt_mean_squared_weighted_max_rating_minus_3_eps_weighted_rmse | linucb_mf_interaction | 1.340157 | 1.361387 | 0.02123015 | DU-Net < Linear UCB | 0.24999142 | 0.18345558 | False | fail | not_testable_from_final_only_row_level | no |
| sqrt_mean_squared_weighted_max_rating_minus_3_eps_weighted_rmse | linucb_concat | 1.340157 | 1.243088 | -0.09706828 | not_matched | 0.28259849 | 4.733396 | False | fail | not_testable_from_final_only_row_level | no |
| inverse_item_popularity_weighted_rmse | linucb_concat | 1.115709 | 1.155912 | 0.04020323 | DU-Net < Linear UCB | 0.36939211 | 0.54627813 | False | fail | not_testable_from_final_only_row_level | no |
| inverse_item_popularity_weighted_rmse | linucb_mf_interaction | 1.115709 | 1.135082 | 0.01937333 | DU-Net < Linear UCB | 0.37513354 | 0.25487198 | False | fail | not_testable_from_final_only_row_level | no |
| raw_unclipped_rmse | linucb_concat | 1.069641 | 1.098024 | 0.02838374 | DU-Net < Linear UCB | 0.39823071 | 0.09168234 | True | fail | not_testable_from_final_only_row_level | no |

Best requested-table candidates by gap closeness to the visual 0.026 gap:

| metric | du_net_tau1 | linucb_concat | gap_vs_concat | gap_dist_vs_paper_concat | direction_vs_concat |
| --- | --- | --- | --- | --- | --- |
| standard_mask_rmse | 1.069631 | 1.097981 | 0.02834921 | 0.00234921 | paper_like |
| centered_rating_rmse | 1.069631 | 1.097981 | 0.02834921 | 0.00234921 | paper_like |
| unclipped_standard_rmse | 1.069641 | 1.098024 | 0.02838374 | 0.00238374 | paper_like |
| 0_1_normalized_rmse | 0.26740787 | 0.27449517 | 0.00708730 | 0.01891270 | paper_like |
| pdf_literal_norm_rmse | 0.00756344 | 0.00776390 | 0.00020046 | 0.02579954 | paper_like |
| mild_reward_w_rmse | 1.063743 | 1.062995 | -0.00074833 | 0.02674833 | reversed_or_tie |

Answers: no candidate simultaneously satisfies endpoint scale, gap, direction, and text-defensible definition. Standard RMSE has the right direction and a close gap versus `linucb_concat`, but its direct endpoint error is about 0.717 on the Figure 4a y-axis (the broader v3 audit's normalized scale-distance field is about 0.398). Some reward-weighted metrics can get closer to the gap, but often reverse direction or lack textual support. Exact w-RMSE is therefore unrecoverable from the paper plus current implementation.

Row-level incompleteness matters for confidence intervals and formula sweeps across all seeds, but it is unlikely to move the Figure 4a scale from about 1.07 to about 1.79 without a different metric definition or mask. A full 20-seed row-level run is still worth doing if the goal is maximum-effort evidence; without it, the metric gate must remain partial evidence.

## 3. Figure digitization answers

Figure 4a MovieLens has been digitized as a cycle-level CSV from local PDF crops. These are visual approximations, not author data.

| cycle | DU-Net | Linear UCB | paper_gap |
| --- | --- | --- | --- |
| 0 | 1.925000 | 1.925000 | 0.00000000 |
| 1 | 1.885000 | 1.910000 | 0.02500000 |
| 2 | 1.862000 | 1.895000 | 0.03300000 |
| 3 | 1.846000 | 1.880000 | 0.03400000 |
| 4 | 1.836000 | 1.866000 | 0.03000000 |
| 5 | 1.828000 | 1.855000 | 0.02700000 |
| 6 | 1.820000 | 1.847000 | 0.02700000 |
| 7 | 1.814000 | 1.841000 | 0.02700000 |
| 8 | 1.809000 | 1.836000 | 0.02700000 |
| 9 | 1.806000 | 1.832000 | 0.02600000 |
| 10 | 1.804000 | 1.828000 | 0.02400000 |
| 11 | 1.802000 | 1.825000 | 0.02300000 |
| 12 | 1.800000 | 1.822000 | 0.02200000 |
| 13 | 1.798000 | 1.820000 | 0.02200000 |
| 14 | 1.796000 | 1.818000 | 0.02200000 |
| 15 | 1.794000 | 1.817000 | 0.02300000 |
| 16 | 1.792000 | 1.816000 | 0.02400000 |
| 17 | 1.791000 | 1.815000 | 0.02400000 |
| 18 | 1.790000 | 1.814500 | 0.02450000 |
| 19 | 1.789000 | 1.814000 | 0.02500000 |
| 20 | 1.788000 | 1.814000 | 0.02600000 |

Early/mid/final Figure 4a gaps:

| cycle | DU-Net | Linear UCB | paper_gap |
| --- | --- | --- | --- |
| 0 | 1.925000 | 1.925000 | 0.00000000 |
| 1 | 1.885000 | 1.910000 | 0.02500000 |
| 5 | 1.828000 | 1.855000 | 0.02700000 |
| 10 | 1.804000 | 1.828000 | 0.02400000 |
| 20 | 1.788000 | 1.814000 | 0.02600000 |

Figure 4a endpoint and curve diagnostics:

| figure | local_metric | du_endpoint_local | linear_endpoint_local | du_endpoint_paper_visual | linear_endpoint_paper_visual | local_gap_linear_minus_du | paper_gap_linear_minus_du | gap_abs_error | direction_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | standard_mask_rmse | 1.070726 | 1.106389 | 1.788000 | 1.814000 | 0.03566297 | 0.02600000 | 0.00966297 | True |
| Figure 4a | w_rmse | 1.028977 | 0.96552449 | 1.788000 | 1.814000 | -0.06345266 | 0.02600000 | 0.08945266 | False |

| figure | method | local_metric | cycles | endpoint_local | endpoint_paper_visual | endpoint_abs_error | raw_curve_rmse | curve_correlation | affine_intercept_to_paper | affine_slope_to_paper | affine_curve_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | DU-Net | standard_mask_rmse | 21 | 1.070726 | 1.788000 | 0.71727384 | 0.73450783 | 0.89513230 | -1.964204 | 3.489467 | 0.01553237 |
| Figure 4a | Linear UCB | standard_mask_rmse | 21 | 1.106389 | 1.814000 | 0.70761086 | 0.74003075 | -0.65327755 | 17.1470 | -13.8734 | 0.02503937 |
| Figure 4a | DU-Net | w_rmse | 21 | 1.028977 | 1.788000 | 0.75902285 | 0.77087603 | 0.95759908 | -0.49848148 | 2.211802 | 0.01003799 |
| Figure 4a | Linear UCB | w_rmse | 21 | 0.96552449 | 1.814000 | 0.84847551 | 0.83044895 | 0.96506604 | 0.97015152 | 0.86189604 | 0.00866507 |

Joint affine fits using both DU-Net and Linear UCB points:

| figure | local_metric | fit_scope | slope_a | intercept_b | rmse | points |
| --- | --- | --- | --- | --- | --- | --- |
| Figure 4a | standard_mask_rmse | DU-Net + Linear UCB joint | 1.793936 | -0.13154829 | 0.02953058 | 42 |
| Figure 4a | w_rmse | DU-Net + Linear UCB joint | 0.56422791 | 1.249179 | 0.03084025 | 42 |

Separate affine fits can make individual shapes look close, especially local w-RMSE curves, but that does not recover the paper metric. A single shared affine transform is only a diagnostic and cannot close exactness without metric, mask, and comparator provenance.

Figure 4b has also been digitized as a cycle-level CSV. Endpoint target: DU-Net about 1325, Linear UCB about 1390, gap about 65.

| cycle | DU-Net | Linear UCB | paper_gap |
| --- | --- | --- | --- |
| 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| 1 | 62.0000 | 70.0000 | 8.000000 |
| 2 | 128.000 | 142.000 | 14.0000 |
| 3 | 198.000 | 215.000 | 17.0000 |
| 4 | 270.000 | 290.000 | 20.0000 |
| 5 | 345.000 | 365.000 | 20.0000 |
| 6 | 420.000 | 440.000 | 20.0000 |
| 7 | 495.000 | 515.000 | 20.0000 |
| 8 | 570.000 | 590.000 | 20.0000 |
| 9 | 645.000 | 668.000 | 23.0000 |
| 10 | 715.000 | 746.000 | 31.0000 |
| 11 | 785.000 | 824.000 | 39.0000 |
| 12 | 855.000 | 902.000 | 47.0000 |
| 13 | 925.000 | 980.000 | 55.0000 |
| 14 | 990.000 | 1058.000 | 68.0000 |
| 15 | 1055.000 | 1136.000 | 81.0000 |
| 16 | 1120.000 | 1214.000 | 94.0000 |
| 17 | 1175.000 | 1290.000 | 115.000 |
| 18 | 1225.000 | 1362.000 | 137.000 |
| 19 | 1275.000 | 1378.000 | 103.000 |
| 20 | 1325.000 | 1390.000 | 65.0000 |

| figure | local_metric | du_endpoint_local | linear_endpoint_local | du_endpoint_paper_visual | linear_endpoint_paper_visual | local_gap_linear_minus_du | paper_gap_linear_minus_du | gap_abs_error | direction_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 4b | cumulative_reward | 1246.386 | 1393.680 | 1325.000 | 1390.000 | 147.294 | 65.0000 | 82.2940 | True |

The current tau1 reward ranking is paper-like, but the local gap is about 147.294 versus the visual paper gap of about 65, so the gap is too large by about 82.294. The DU-Net local reward endpoint is about 78.614 below paper, while `linucb_concat` is only about 3.680 above paper.

## 4. Split/mask exactness answers

The current MovieLens fallback uses official `u1.base`/`u1.test`, maps users/items to zero-based ids, filters to 1681 items by dropping the final MovieLens item id after zero-basing, samples the initial observed set of 1364 rows from `u1.base` by seed, defines candidates as `u1.base` minus the initial rows, and keeps `u1.test` fixed. Seeds affect only the initial/candidate partition, not the fixed test fold.

Aligned with Table 1-style constraints: 943 users, 1681 items, 1364 initial observations, 20000 test rows, 20 cycles, batch size 100. Not aligned: exact train mask, exact initial mask, exact candidate pool, exact test mask provenance, and exact seed protocol.

Split search audit:

| search_group | query_or_source | status | evidence | local_artifact | decision |
| --- | --- | --- | --- | --- | --- |
| exact_paper_public_code | "Network-Enabled Sequential Data Acquisition" GitHub | not_found_public_web_or_local | Web search returned no author-code result for the exact title; local repo has PDF/Markdown/crops but no author code bundle. | Network-Enabled Sequential Data Acquisition (1).pdf | Do not claim exact paper masks from public code. |
| exact_paper_public_code | "Network-Enabled Sequential Data Acquisition" MovieLens split code | not_found_public_web_or_local | Exact-title and DU-Net/MovieLens searches did not recover split files or executable Figure 4 scripts. | reports/paper_url_dependency_audit.md | Keep author split/mask provenance unresolved. |
| grouplens_official_folds | MovieLens 100K u1-u5 official folds | available_not_exact | u1-u5 and ua/ub folds exist locally, but the paper does not state that Figure 4 uses one of these masks as the sequential acquisition test/candidate split. | data/raw/ml-100k/u1.base; data/raw/ml-100k/u1.test | Use only as sensitivity evidence, never as mainline by best-looking fold. |
| gcmc_loader | GCMC MovieLens official split loader | available_partial | Local GCMC provenance confirms a u1.base/u1.test loader pattern and a validation split, but not DU-Net's Figure 4 initial/candidate/test masks. | reports/gcmc_monti_split_provenance_audit.md | Provenance support only; exact mask still blocked. |
| monti_mat_files | Monti training_test_dataset.mat | available_for_flixster_douban_not_movielens | Local Monti-style .mat masks are present for Flixster and Douban, but not MovieLens. | data/raw/gcmc/flixster/training_test_dataset.mat; data/raw/gcmc/douban/training_test_dataset.mat | Do not infer MovieLens masks from other datasets. |
| local_split_distribution | official folds vs seeded random masks | distribution_similar_mask_not_exact | Prior overlap audit shows similar marginal distributions but different masks. | outputs/tables/movielens_split_overlap_audit.csv | Split alone is not the dominant scale fix, but exactness remains blocked. |

Monti/RMGCNN-style `.mat` masks were found locally for Flixster and Douban, not MovieLens. GCMC provides an official `u1.base`/`u1.test` MovieLens loader and validation split convention; it does not recover DU-Net's Figure 4 masks. It is plausible the paper used official u1 because the dimensions and public-loader conventions fit, but this remains unproven because the paper does not publish the exact initial/candidate/test masks.

Official fold sensitivity, 3 seeds per fold:

| fold | comparator | dunet_rmse | comparator_rmse | gap | reward_gap | win_rate | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| u1 | linucb_concat | 1.087699 | 1.102384 | 0.01468540 | -133.542 | 1.000000 | 3 |
| u1 | linucb_mf_interaction | 1.087699 | 1.081287 | -0.00641244 | 37.9979 | 0.33333333 | 3 |
| u2 | linucb_concat | 1.072843 | 1.083085 | 0.01024191 | -87.2519 | 1.000000 | 3 |
| u2 | linucb_mf_interaction | 1.072843 | 1.068653 | -0.00418937 | 77.3701 | 0.33333333 | 3 |
| u3 | linucb_concat | 1.070688 | 1.077344 | 0.00665593 | -35.3172 | 0.66666667 | 3 |
| u3 | linucb_mf_interaction | 1.070688 | 1.064191 | -0.00649765 | 149.882 | 0.00000000 | 3 |
| u4 | linucb_concat | 1.056182 | 1.076863 | 0.02068046 | -172.122 | 1.000000 | 3 |
| u4 | linucb_mf_interaction | 1.056182 | 1.062118 | 0.00593574 | 18.3551 | 0.66666667 | 3 |
| u5 | linucb_concat | 1.070905 | 1.088786 | 0.01788064 | -114.072 | 1.000000 | 3 |
| u5 | linucb_mf_interaction | 1.070905 | 1.065973 | -0.00493251 | 121.966 | 0.33333333 | 3 |

Fold interpretation: tau1 is stable against `linucb_concat` on most/all folds in this 3-seed diagnostic, but not stable against `linucb_mf_interaction`. This lowers pure mask-overfitting concern for the concat comparator, but it does not close exact mask provenance.

## 5. Linear UCB exactness answers

The PDF evidence audit finds Linear UCB named as a comparator and mentions the contextualbandits repository, but does not specify feature construction, alpha, regularization, reward transform, or update protocol. Local package checks found `contextualbandits` and `lrbandit` not installed at audit time.

Linear UCB protocol audit:

| candidate | implementation_status | paper_faithfulness | final_standard_rmse | final_w_rmse | final_cumulative_reward | evidence | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contextualbandits.online.LinUCB/default-disjoint | not_installed | candidate_external_protocol |  |  |  | Python package `contextualbandits` availability checked locally without adding dependencies. | Do not add dependency or promote until explicit author-code/package link is recovered. |
| lrbandit/OFUL paper-adjacent adapter | not_installed | candidate_external_protocol |  |  |  | Python package `lrbandit` availability checked locally without adding dependencies. | Do not add dependency or promote until explicit author-code/package link is recovered. |
| local linucb_concat | implemented_20seed | local_proxy_not_author_protocol | 1.106389 | 0.96552449 | 1393.680 | Included in tau1 mainline bundle summary. | Use as comparator range, not exact paper Linear UCB. |
| local linucb_mf_interaction | implemented_20seed | local_proxy_not_author_protocol | 1.087422 | 1.048264 | 1206.370 | Included in tau1 mainline bundle summary. | Use as comparator range, not exact paper Linear UCB. |
| local linucb_bias_interaction | implemented_20seed | local_proxy_not_author_protocol | 1.130332 | 0.94014922 | 1521.646 | Included in tau1 mainline bundle summary. | Use as comparator range, not exact paper Linear UCB. |

Local proxies:

- `linucb_concat`: ridge UCB on concatenated `[user_factors[u], item_factors[i]]`, target `rating - 3`.
- `linucb_mf_interaction`: MF prediction mean plus alpha times z-scored uncertainty from elementwise interaction features `user_factors[u] * item_factors[i]`.
- `linucb_bias_interaction`: ridge UCB on user factors, item factors, interaction, user bias, item bias, and intercept.

They use the same SVD++ backend, candidate set, batch size, cycle update, and selected-row removal loop as the DU-Net runs. They are fair local comparators but not recovered author implementations.

20-seed local proxy endpoints:

| claim_role | variant | policy | seed_count | pdf_literal_norm_rmse | pdf_literal_norm_rmse_ci95 | standard_mask_rmse | standard_mask_rmse_ci95 | cumulative_reward | cumulative_reward_ci95 | centered_cumulative_reward | centered_cumulative_reward_ci95 | reward_tail_proxy_rmse | reward_tail_proxy_rmse_ci95 | rmse | rmse_ci95 | w_rmse | w_rmse_ci95 | observed_size | observed_size_ci95 | candidate_size | candidate_size_ci95 | mild_reward_w_rmse | mild_reward_w_rmse_ci95 | medium_reward_w_rmse | medium_reward_w_rmse_ci95 | sigmoid_reward_w_rmse | sigmoid_reward_w_rmse_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mainline | schur_loglambda_tuned_full_pool_softmax_tau1_literal | paper_faithful_dunet | 20.0000 | 0.00757118 | 0.00002305 | 1.070726 | 0.00325958 | 1246.386 | 12.1704 | 1191.700 | 60.9962 | 1.309035 | 0.01462385 | 1.070726 | 0.00325958 | 1.028977 | 0.01030174 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |  |  |  |  |  |  |
| linear_ucb_proxy | linucb_bias_interaction | linucb_bias_interaction | 20.0000 | 0.00799265 | 0.00003120 | 1.130332 | 0.00441287 | 1521.646 | 4.576474 | 2622.550 | 23.9174 | 1.178957 | 0.01133567 | 1.130332 | 0.00441287 | 0.94014922 | 0.00751323 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |  |  |  |  |  |  |
| linear_ucb_proxy | linucb_mf_interaction | linucb_mf_interaction | 20.0000 | 0.00768923 | 0.00002452 | 1.087422 | 0.00346816 | 1206.370 | 16.6578 | 1005.650 | 84.4908 | 1.336719 | 0.00988172 | 1.087422 | 0.00346816 | 1.048264 | 0.00703642 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |  |  |  |  |  |  |
| linear_ucb_proxy | linucb_concat | linucb_concat | 20.0000 | 0.00782335 | 0.00003583 | 1.106389 | 0.00506687 | 1393.680 | 8.916846 | 1949.100 | 45.8365 | 1.216032 | 0.01524989 | 1.106389 | 0.00506687 | 0.96552449 | 0.01024673 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |  |  |  |  |  |  |

Closest Figure 4b Linear endpoint is `linucb_concat` (1393.680 versus visual 1390). Strongest reward proxy is `linucb_bias_interaction` but it overshoots reward. Strongest standard-RMSE proxy is `linucb_mf_interaction`, but it is not the closest reward comparator. The most defensible comparator for Figure 4b is `linucb_concat`; for MF-style uncertainty comparisons, `linucb_mf_interaction` remains a required sensitivity comparator. No `paper_linear_ucb_reconstructed` should be promoted.

Alpha sensitivity, 3-seed MF-interaction diagnostic:

| variant | selected_avg_rating | selected_rating_5_share | selected_rating_ge4_share | selected_unique_users | selected_unique_items | selected_avg_user_degree | selected_avg_item_popularity | selected_avg_entropy_percentile | selected_avg_reward_percentile | standard_mask_rmse | cumulative_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linucb_mf_alpha_0p005 | 4.386667 | 0.56733333 | 0.86733333 | 52.4333 | 44.6833 |  | 22.3200 |  |  | 1.126820 | 1550.131 |
| linucb_mf_alpha_0p01 | 4.399000 | 0.56933333 | 0.87450000 | 52.9500 | 44.4500 |  | 22.8255 |  |  | 1.127863 | 1555.451 |
| linucb_mf_alpha_0p02 | 4.388833 | 0.56916667 | 0.86683333 | 54.6667 | 42.8667 |  | 24.4362 |  |  | 1.129823 | 1550.888 |
| linucb_mf_alpha_0p05 | 4.369500 | 0.56366667 | 0.86200000 | 52.3333 | 45.3833 |  | 24.8275 |  |  | 1.124772 | 1543.201 |
| linucb_mf_alpha_0p1 | 4.311167 | 0.54733333 | 0.84183333 | 52.5667 | 48.5333 |  | 27.7028 |  |  | 1.120072 | 1520.364 |
| linucb_mf_alpha_0p8_current | 3.564833 | 0.30600000 | 0.60083333 | 36.3833 | 67.6000 |  | 15.8972 |  |  | 1.084941 | 1231.717 |
| schur_loglambda_tuned_standard_topk | 3.696333 | 0.28350000 | 0.61233333 | 35.1500 | 57.1833 | 33.8170 | 15.7890 | 0.92520868 | 0.58677440 | 1.088734 | 1284.706 |

Current alpha is 0.8 with regularization 1.0. It is a local choice, not recovered from the paper. Smaller alpha values increase immediate reward but worsen RMSE, which shows the comparator is sensitive to hidden protocol choices.

## 6. DU-Net exactness answers

Schur/Eq.3.6: the explicit-kernel audit used a non-orthogonal toy factorization and compared the production feature-space conditional variance with direct observed-kernel Schur complement. Maximum absolute difference was 2.220e-16. This proves implementation equivalence for the current clean-room Schur scorer, not full identity with every unpublished paper backend convention.

The implementation uses `phi = kron(q_user[u], q_item[i])`, posterior precision `(Phi_obs.T Phi_obs + lambda I)^-1`, and returns `lambda * phi.T precision phi`. This is theoretically aligned with a low-rank Schur conditional variance, but the code does not expose every paper symbol as a named object (`mu_i(U) mu_j(V)`, cross-coherence `nu`, `L_t(O_t)`, and `eta^2/sigma^2`). A formal note is useful if writing a methods appendix.

Shortlist: current Schur correction is applied to a coherence-base shortlist of 20000 candidates when the final candidate pool is about 76635. Non-shortlisted candidates receive zero entropy. This can change selected batches and has not been closed by an all-candidate diagnostic. Perfect reproduction requires at least a small all-candidate-vs-shortlist audit.

lambda_noise: current `lambda_noise=1.0` is inherited from `entropy_logdet_regularization=1.0`. There is no data-estimated eta^2/sigma^2 version in the mainline, and no source-backed sensitivity closure. A reasonable estimator would use held-out/residual variance for eta^2 and latent-score variance or rating-noise scale for sigma^2, then freeze the ratio before selection.

DLS: current DLS is full-neighborhood Gaussian smoothing when `paper_dls_scope=full`, with no residual blend unless shrinkage is configured. Sigma is dynamically selected from the configured grid every tuning interval. Its input `u_sn` is the configured network/user-feature embedding (`network_embedding_mode=user_features`, GCMC/user metadata features), not a recovered paper ELPH embedding.

ELPH: the repo has dependency-free ELPH-style proxies in other diagnostics, but the exact paper ELPH backend and parameters are not recovered or validated for MovieLens. Treat ELPH as a remaining blocker, not a branch to keep tuning blindly.

## 7. Selection exactness answers

tau1 literal softmax is implemented as full-pool `exp(score)` sampling without replacement after subtracting the max score for numerical stability. The subtraction does not change probabilities; exponent clipping only prevents extreme underflow/overflow. The score itself is the DU-Net combined score after z-scoring reward and transformed entropy before adding them.

The paper gives an exponential weighting idea, but does not specify all implementation choices: whether scores were z-normalized, whether batch sampling was without replacement, and whether numerical clipping was used. Thus selection is closed as a clean-room literal-tau gate, but not as author-source exactness. The 20-seed tau1 mainline is stable: it beats `linucb_concat` 20/20 seeds and `linucb_mf_interaction` 19/20 seeds under standard RMSE. tau1 reward is close to tau0.3 in the 20-seed bundle, so tau1 should replace tau0.3 as the formal mainline.

## 8. Figure 4b joint reproduction answers

Selected-sample behavior in the exact-attempt seed0 run:

| variant | avg_selected_rating_c0_19 | selected_ge4_share_c0_19 | avg_reward_percentile_c0_19 | avg_entropy_percentile_c0_19 | final_cumulative_reward |
| --- | --- | --- | --- | --- | --- |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | 3.520500 | 0.55250000 | 0.49176466 | 0.53802009 | 1217.356 |
| linucb_concat | 3.930000 | 0.70400000 |  |  | 1373.983 |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | 3.500500 | 0.54700000 | 0.49780546 | 0.52700568 | 1209.496 |
| schur_tuned_topk | 3.792000 | 0.63850000 | 0.61236215 | 0.89740810 | 1320.295 |

tau1 selects lower immediate-reward rows than `linucb_concat` (mean selected rating about 3.5205 versus 3.9300). Its average entropy percentile is nontrivial, so the reward gap is consistent with stronger exploration/information-value pressure. Dynamic tuning is currently driven by `standard_mask_rmse`, which naturally prioritizes Figure 4a-style error rather than Figure 4b reward.

Reward tradeoff diagnostics, 20 seeds:

| variant | policy | seed_count | pdf_literal_norm_rmse | pdf_literal_norm_rmse_ci95 | standard_mask_rmse | standard_mask_rmse_ci95 | cumulative_reward | cumulative_reward_ci95 | centered_cumulative_reward | centered_cumulative_reward_ci95 | reward_tail_proxy_rmse | reward_tail_proxy_rmse_ci95 | rmse | rmse_ci95 | w_rmse | w_rmse_ci95 | observed_size | observed_size_ci95 | candidate_size | candidate_size_ci95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linucb_concat | linucb_concat | 20.0000 | 0.00782335 | 0.00003583 | 1.106389 | 0.00506687 | 1393.680 | 8.916846 | 1949.100 | 45.8365 | 1.216032 | 0.01524989 | 1.106389 | 0.00506687 | 0.96552449 | 0.01024673 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |
| schur_loglambda_tuned_standard_topk | paper_faithful_dunet | 20.0000 | 0.00769941 | 0.00003137 | 1.088861 | 0.00443579 | 1250.557 | 33.0580 | 1219.850 | 170.956 | 1.326710 | 0.01805579 | 1.088861 | 0.00443579 | 1.040197 | 0.01297733 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |
| schur_lambda_0 | paper_faithful_dunet | 20.0000 | 0.00796783 | 0.00002789 | 1.126822 | 0.00394368 | 1547.575 | 5.375679 | 2753.300 | 28.7125 | 1.179844 | 0.01387472 | 1.126822 | 0.00394368 | 0.93859041 | 0.00932384 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |
| schur_lambda_0p3 | paper_faithful_dunet | 20.0000 | 0.00772129 | 0.00002290 | 1.091956 | 0.00323851 | 1353.194 | 5.054192 | 1742.150 | 25.5763 | 1.261191 | 0.01531903 | 1.091956 | 0.00323851 | 0.99372997 | 0.01072501 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |
| schur_lambda_1 | paper_faithful_dunet | 20.0000 | 0.00766754 | 0.00002334 | 1.084353 | 0.00330065 | 1255.557 | 3.438153 | 1237.900 | 17.4967 | 1.304373 | 0.01680602 | 1.084353 | 0.00330065 | 1.024664 | 0.01193889 | 3364.000 | 0.00000000 | 76635.000 | 0.00000000 |

`schur_lambda_0p3` shows a more paper-like Figure 4b reward endpoint/gap than the tau1 mainline, but it worsens standard RMSE relative to tau1 and is not the frozen exact-attempt mainline. `schur_lambda_0` can exceed Linear reward but badly hurts RMSE. This indicates a local objective mismatch: no single current lambda simultaneously closes Figure 4a scale/metric and Figure 4b reward gap.

## 9. Risks categorized by closability

Categories: A = can be closed by more local experiments; B = requires installation/implementation before closure; C = cannot be closed by experiments alone and needs source material.

| risk | category | next_action | estimated_effort | expected_impact |
| --- | --- | --- | --- | --- |
| 20-seed row-level metric recomputation | A | Generate final-cycle, then optionally all-cycle row-level predictions for all required methods. | high compute; hours to overnight single-process | Closes partial evidence for metric candidates, but cannot fix a wrong paper formula by itself. |
| u1-u5 and random-mask sensitivity | A | Promote existing u1-u5 3-seed evidence only if expanded to 20 seeds or paired with author mask recovery. | medium to high | Reduces mask-overfitting concern; does not recover exact masks. |
| all-candidate Schur vs 20k shortlist | A | Run a small all-candidate diagnostic and compare selected batches and endpoints. | medium to high memory/CPU | Can close shortlist approximation risk for current implementation. |
| lambda_noise sensitivity | A | Grid or estimate eta^2/sigma^2 from residual variance and rating-noise assumptions. | medium | Tests whether backend/noise convention explains reward/RMSE tradeoff. |
| alpha sensitivity for local Linear UCB | A | Broaden alpha/regularization diagnostics beyond the current 3-seed grid if comparator remains important. | medium | Clarifies comparator range; does not identify author Linear UCB. |
| Figure 4 digitization precision | A | Use WebPlotDigitizer/manual axis calibration on the paper PDF crops and regenerate target CSVs. | low to medium | Improves target precision; still not author numeric data. |
| contextualbandits/lrbandit adapters | B | Install/test minimal adapters only as external comparator candidates. | medium plus dependency risk | May widen comparator range; not exact without paper linkage. |
| ELPH backend implementation | B | Implement or vendor a documented ELPH-like graph embedding only if paper/backend evidence appears. | high | Could affect DLS inputs materially; current exactness remains blocked. |
| tiled all-candidate Schur production path | B | Optimize the all-candidate Schur computation if diagnostics show shortlist changes decisions. | high | Needed only if shortlist materially changes selection. |
| estimated noise ratio backend | B | Add a source-grounded eta^2/sigma^2 estimator and freeze it in config. | medium | Could align Eq.3.6 convention, but paper does not expose enough details. |
| author exact MovieLens masks | C | Obtain initial/candidate/test masks or generation scripts. | requires source | Required for exact replication. |
| author exact w-RMSE code or numeric Figure 4a table | C | Obtain metric code, weights, normalization, aggregation, and Figure 4 numeric data. | requires source | Required to resolve the formula/plot contradiction. |
| author Linear UCB protocol | C | Obtain feature map, alpha, regularization, reward transform, and update protocol. | requires source | Required before naming any comparator paper_linear_ucb_reconstructed. |
| author DU-Net backend/ELPH parameters | C | Obtain backend details for DLS input graph/embedding, sigma/noise/tuning, and any approximation. | requires source | Required for perfect source-faithful DU-Net reproduction. |

## 10. Final judgment and recommended next steps

If the target is perfect exact replication, the three most critical failed gates are: metric exactness, split/mask exactness, and Linear UCB protocol exactness. The next tier is DU-Net backend/ELPH/noise/shortlist exactness plus Figure 4b same-config joint reproduction.

Local experiments can close row-level completeness, fold robustness, all-candidate Schur risk, lambda_noise sensitivity, alpha sensitivity, and digitization precision. They cannot close author masks, author w-RMSE code/numeric Figure 4 data, or author Linear UCB hidden protocol.

Maximum-effort paper-only work remaining: run full 20-seed row-level final-cycle predictions for required methods; optionally run all-cycle row-level only if storage is acceptable; expand fold/mask sensitivity; run all-candidate Schur diagnostics; run lambda_noise/eta-sigma sensitivity; and add dependency-backed Linear UCB adapters only as explicitly labeled comparator candidates.

Current status: strong clean-room recovery, not near-perfect exact reproduction. It has not yet reached the strongest possible paper-only reproduction because several closable local experiments remain. Even after those are done, near-perfect replication still requires source conditions: exact metric code/numeric Figure 4 data, exact MovieLens masks/seed protocol, exact Linear UCB adapter, and DU-Net backend details. Without those, the remaining gap is source availability and paper ambiguity, not simply code quality.

Generated artifacts:

- `outputs\tables\paper_exact_attempt_movielens_v1_failed_gate_metric_candidates.csv`
- `outputs\tables\paper_exact_attempt_movielens_v1_failed_gate_joint_affine.csv`
- `outputs\tables\paper_exact_attempt_movielens_v1_failed_gate_risks.csv`