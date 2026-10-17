# Figure 4a Phase 1 Seed-15 Micro-Screen

## Result

- No Phase 1 micro-variant should be promoted to a 20-seed rerun.
- `graph_late_lambda_half_start17` is the best micro-variant on the four-seed primary mean, but its improvement is `0.002884`, just below the `0.003` gate.
- `graph_late_lambda_half_start15` repairs seed 15 most strongly, but it harms `3/3` stable seeds.
- `graph_late_half_start16_frontier95_floor25` is effectively identical to the frozen baseline and does not flip seed 15 (`seed15_linear_gap=-0.006270`).

## Four-Seed Matrix

| variant | policy | seed_count | cumulative_reward | linear_reward_norm_w_rmse | positive_reward_norm_w_rmse | w_rmse | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph_late_lambda_half_start16 | du_net | 4 | 2743.500000 | 0.993485 | 0.939086 | 0.823136 | 1.213995 |
| graph_late_lambda_half_start15 | du_net | 4 | 2749.750000 | 0.990904 | 0.936586 | 0.821749 | 1.215071 |
| graph_late_lambda_half_start17 | du_net | 4 | 2749.000000 | 0.990601 | 0.936293 | 0.821615 | 1.215624 |
| graph_late_half_start16_frontier95_floor25 | du_net | 4 | 2744.000000 | 0.993427 | 0.939029 | 0.823106 | 1.214025 |
| graph_base | du_net | 4 | 2747.000000 | 0.989356 | 0.935092 | 0.820859 | 1.215563 |
| mf_linear_ucb_a002 | mf_linear_ucb | 4 | 2735.500000 | 0.997994 | 0.943428 | 0.825731 | 1.211953 |
| myopic | myopic | 4 | 2745.250000 | 1.005574 | 0.950731 | 0.830031 | 1.208075 |

## Promotion Gates

| variant | primary_delta_vs_frozen | linear_gap_delta_vs_frozen | reward_delta_vs_frozen | seed15_primary_delta_vs_frozen | seed15_linear_gap | stable_seed_harm_count | promote_to_20seed | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph_late_lambda_half_start15 | 0.002581 | 0.002581 | 6.250000 | 0.020905 | 0.014635 | 3 | 0 | reject_borderline_seed15_tradeoff |
| graph_late_lambda_half_start17 | 0.002884 | 0.002884 | 5.500000 | 0.015771 | 0.009501 | 2 | 0 | reject_borderline_seed15_tradeoff |
| graph_late_half_start16_frontier95_floor25 | 0.000058 | 0.000058 | 0.500000 | 0.000000 | -0.006270 | 0 | 0 | reject |
| graph_base | 0.004129 | 0.004129 | 3.500000 | 0.014076 | 0.007806 | 2 | 0 | diagnostic_only |

## Seed-Level Values

| variant | seed | cumulative_reward | linear_reward_norm_w_rmse | rmse | w_rmse |
| --- | --- | --- | --- | --- | --- |
| graph_late_lambda_half_start16 | 0 | 2763.000000 | 1.004958 | 1.217832 | 0.831491 |
| graph_late_lambda_half_start16 | 1 | 2756.000000 | 1.010192 | 1.196198 | 0.831414 |
| graph_late_lambda_half_start16 | 2 | 2779.000000 | 0.971348 | 1.220573 | 0.809882 |
| graph_late_lambda_half_start16 | 15 | 2676.000000 | 0.987442 | 1.221376 | 0.819757 |
| graph_late_lambda_half_start15 | 0 | 2749.000000 | 1.010268 | 1.217140 | 0.834315 |
| graph_late_lambda_half_start15 | 1 | 2742.000000 | 1.013616 | 1.193936 | 0.833270 |
| graph_late_lambda_half_start15 | 2 | 2770.000000 | 0.973194 | 1.219244 | 0.810778 |
| graph_late_lambda_half_start15 | 15 | 2738.000000 | 0.966537 | 1.229966 | 0.808631 |
| graph_late_lambda_half_start17 | 0 | 2759.000000 | 1.000937 | 1.219399 | 0.829319 |
| graph_late_lambda_half_start17 | 1 | 2740.000000 | 1.016796 | 1.193586 | 0.834985 |
| graph_late_lambda_half_start17 | 2 | 2768.000000 | 0.972999 | 1.220554 | 0.810899 |
| graph_late_lambda_half_start17 | 15 | 2729.000000 | 0.971671 | 1.228959 | 0.811256 |
| graph_late_half_start16_frontier95_floor25 | 0 | 2765.000000 | 1.004725 | 1.217954 | 0.831372 |
| graph_late_half_start16_frontier95_floor25 | 1 | 2756.000000 | 1.010192 | 1.196198 | 0.831414 |
| graph_late_half_start16_frontier95_floor25 | 2 | 2779.000000 | 0.971348 | 1.220573 | 0.809882 |
| graph_late_half_start16_frontier95_floor25 | 15 | 2676.000000 | 0.987442 | 1.221376 | 0.819757 |
| graph_base | 0 | 2765.000000 | 0.996232 | 1.221245 | 0.826643 |
| graph_base | 1 | 2750.000000 | 1.012444 | 1.194162 | 0.832284 |
| graph_base | 2 | 2758.000000 | 0.975383 | 1.218155 | 0.811990 |
| graph_base | 15 | 2715.000000 | 0.973366 | 1.228692 | 0.812520 |
| mf_linear_ucb_a002 | 0 | 2731.000000 | 1.015829 | 1.213591 | 0.837332 |
| mf_linear_ucb_a002 | 1 | 2740.000000 | 1.015474 | 1.193515 | 0.834348 |
| mf_linear_ucb_a002 | 2 | 2736.000000 | 0.979502 | 1.216886 | 0.814743 |
| mf_linear_ucb_a002 | 15 | 2735.000000 | 0.981172 | 1.223819 | 0.816500 |
| myopic | 0 | 2762.000000 | 1.011944 | 1.210047 | 0.834805 |
| myopic | 1 | 2732.000000 | 1.024102 | 1.189322 | 0.839322 |
| myopic | 2 | 2719.000000 | 1.007802 | 1.209731 | 0.831220 |
| myopic | 15 | 2768.000000 | 0.978446 | 1.223200 | 0.814775 |

## Reflection

- Seed 15 is not random noise in the narrow sense: earlier or slightly later lambda reduction can repair it.
- The repair is not robust enough to replace the frozen schedule because the benefit concentrates on seed 15 and introduces losses on stable seeds.
- This argues against a manual schedule tweak and for either freezing the current best or testing the paper-like adaptive tuning path, where the schedule can vary from observed data instead of being hand-picked.

## Plan Adjustment

- Do not run a 20-seed promotion for Phase 1 micro-variants.
- Keep `graph_late_lambda_half_start16` as the frozen best.
- Move next to Phase 4 narrow adaptive tuning on seeds `0,1,2,15`, with a stricter gate: it must beat the frozen four-seed mean by at least `0.003` and have `stable_seed_harm_count == 0`, or else local tuning stops and final reporting/source-recovery becomes the priority.
