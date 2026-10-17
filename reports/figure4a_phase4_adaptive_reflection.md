# Figure 4a Phase 4 Narrow Adaptive Probe

## Result

- The narrow adaptive-grid probe failed the promotion gate.
- Primary delta vs frozen schedule: `-0.008614`; required `>= +0.003`.
- Reward delta vs frozen schedule: `-58.75`.
- Stable-seed harm count: `3/3`.
- Seed 15 does improve (`seed15_primary_delta_vs_frozen=0.025359`), but this repeats the Phase 1 pattern: seed-15 repair is paid for by stable-seed damage.

## Four-Seed Matrix

| variant | policy | seed_count | cumulative_reward | linear_reward_norm_w_rmse | positive_reward_norm_w_rmse | w_rmse | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph_late_lambda_half_start16 | du_net | 4 | 2743.500000 | 0.993485 | 0.939086 | 0.823136 | 1.213995 |
| graph_adaptive_grid_phase4_narrow | du_net_adaptive_grid | 4 | 2684.750000 | 1.002099 | 0.947409 | 0.827637 | 1.206448 |
| mf_linear_ucb_a002 | mf_linear_ucb | 4 | 2735.500000 | 0.997994 | 0.943428 | 0.825731 | 1.211953 |
| myopic | myopic | 4 | 2745.250000 | 1.005574 | 0.950731 | 0.830031 | 1.208075 |

## Gate

| variant | primary_delta_vs_frozen | reward_delta_vs_frozen | rmse_delta_vs_frozen | linear_gap_favoring_adaptive | myopic_gap_favoring_adaptive | seed15_primary_delta_vs_frozen | seed15_linear_gap | stable_seed_harm_count | stable_seed_delta_min | stable_seed_delta_mean | meets_primary_003_gate | strict_no_stable_harm | promote_to_20seed | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph_adaptive_grid_phase4_narrow | -0.008614 | -58.750000 | 0.007547 | -0.004105 | 0.003475 | 0.025359 | 0.019089 | 3 | -0.025776 | -0.019938 | 0 | 0 | 0 | reject_stop_local_tuning |

## Seed-Level Values

| variant | seed | cumulative_reward | linear_reward_norm_w_rmse | rmse | w_rmse |
| --- | --- | --- | --- | --- | --- |
| graph_late_lambda_half_start16 | 0 | 2763.000000 | 1.004958 | 1.217832 | 0.831491 |
| graph_late_lambda_half_start16 | 1 | 2756.000000 | 1.010192 | 1.196198 | 0.831414 |
| graph_late_lambda_half_start16 | 2 | 2779.000000 | 0.971348 | 1.220573 | 0.809882 |
| graph_late_lambda_half_start16 | 15 | 2676.000000 | 0.987442 | 1.221376 | 0.819757 |
| graph_adaptive_grid_phase4_narrow | 0 | 2616.000000 | 1.026012 | 1.203479 | 0.842808 |
| graph_adaptive_grid_phase4_narrow | 1 | 2678.000000 | 1.035968 | 1.190286 | 0.847108 |
| graph_adaptive_grid_phase4_narrow | 2 | 2707.000000 | 0.984332 | 1.207590 | 0.815888 |
| graph_adaptive_grid_phase4_narrow | 15 | 2738.000000 | 0.962083 | 1.224438 | 0.804746 |
| mf_linear_ucb_a002 | 0 | 2731.000000 | 1.015829 | 1.213591 | 0.837332 |
| mf_linear_ucb_a002 | 1 | 2740.000000 | 1.015474 | 1.193515 | 0.834348 |
| mf_linear_ucb_a002 | 2 | 2736.000000 | 0.979502 | 1.216886 | 0.814743 |
| mf_linear_ucb_a002 | 15 | 2735.000000 | 0.981172 | 1.223819 | 0.816500 |
| myopic | 0 | 2762.000000 | 1.011944 | 1.210047 | 0.834805 |
| myopic | 1 | 2732.000000 | 1.024102 | 1.189322 | 0.839322 |
| myopic | 2 | 2719.000000 | 1.007802 | 1.209731 | 0.831220 |
| myopic | 15 | 2768.000000 | 0.978446 | 1.223200 | 0.814775 |

## Adaptive Tail Parameters

| cycle | adaptive_lambda_entropy | adaptive_diffusion_prob | sigma | cumulative_reward | linear_reward_norm_w_rmse |
| --- | --- | --- | --- | --- | --- |
| 11 | 0.275000 | 0.006250 | 0.400000 | 1488.500000 | 1.066859 |
| 12 | 0.212500 | 0.006250 | 0.500000 | 1625.000000 | 1.052827 |
| 13 | 0.300000 | 0.005000 | 0.550000 | 1761.500000 | 1.043292 |
| 14 | 0.275000 | 0.002500 | 0.350000 | 1897.750000 | 1.034244 |
| 15 | 0.237500 | 0.005000 | 0.400000 | 2034.750000 | 1.023992 |
| 16 | 0.212500 | 0.005000 | 0.400000 | 2158.750000 | 1.022109 |
| 17 | 0.237500 | 0.003750 | 0.600000 | 2292.000000 | 1.017352 |
| 18 | 0.212500 | 0.003750 | 0.450000 | 2420.750000 | 1.013365 |
| 19 | 0.250000 | 0.003750 | 0.300000 | 2561.750000 | 1.004990 |
| 20 | 0.250000 | 0.005000 | 0.400000 | 2684.750000 | 1.002099 |

## Reflection

- The adaptive path is more paper-like in spirit, but under the available local proxy it chooses lower-reward batches and worsens the reward-weighted objective.
- Because Phase 1 and Phase 4 both found seed-15 fixes that do not generalize, the current evidence favors freezing `graph_late_lambda_half_start16` rather than opening more local tuning.
- The next useful work is not another schedule sweep; it is final reporting plus metric/source reconciliation.

## Plan Adjustment

- Stop local tuning unless new source information appears.
- Keep the current 20-seed best as the deliverable candidate.
- Next: finalize the report package, mark proxy caveats clearly, and optionally run Phase 3 metric formula reconciliation from saved predictions before the final write-up.
