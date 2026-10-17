# Figure 4a Latest Execution Reflection

## Source Tables

- Full-cycle metrics: `outputs/tables/movielens_100k_official_u1_head_late_half_start16_20seeds_metrics.csv`
- Full-cycle summary: `outputs/tables/movielens_100k_official_u1_head_late_half_start16_20seeds_summary.csv`
- Rating-bin diagnostics: `outputs/tables/movielens_100k_official_u1_best_candidate_rating_bins_20seeds_summary.csv`

## Current Paper Comparison

- Paper visual DU/Linear gap: `0.026000` (1.814 - 1.788).
- Local DU vs MF-centered Linear UCB proxy gap: `0.020669`.
- Local completion of paper visual gap: `79.5%`; remaining gap: `0.005331`.
- Local DU vs Myopic gap: `0.021023`.

## Execution Matrix

| variant | policy | seed_count | cumulative_reward | rmse | w_rmse | linear_reward_norm_w_rmse | positive_reward_norm_w_rmse | high_rating_rmse_ge4 | low_rating_rmse_le3 | rating5_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graph_late_lambda_half_start16 | du_net | 20.000000 | 2762.550000 | 1.210241 | 0.814739 | 0.981106 | 0.927246 | 0.641278 | 1.677584 | 0.969793 |
| myopic | myopic | 20.000000 | 2734.450000 | 1.202656 | 0.826629 | 1.002130 | 0.947546 | 0.654522 | 1.658493 | 0.991465 |
| mf_linear_ucb_a002 | mf_linear_ucb | 20.000000 | 2717.850000 | 1.202513 | 0.826414 | 1.001775 | 0.947203 | 0.654370 | 1.658312 | 0.991060 |
| graph_base | du_net | 20.000000 | 2745.350000 | 1.209617 | 0.815749 | 0.982734 | 0.928803 | 0.642603 | 1.675896 | 0.971284 |

## DU-Net Comparisons

| comparator | primary_linear_reward_norm_gap_favoring_du | primary_seed_wins | cumulative_reward_gap_favoring_du | reward_seed_wins | conventional_rmse_gap_favoring_du | high_rating_rmse_gap_favoring_du | low_rating_rmse_gap_favoring_du | rating5_rmse_gap_favoring_du |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mf_linear_ucb_a002 | 0.020669 | 19 | 44.700000 | 19 | -0.007728 | 0.013092 | -0.019273 | 0.021267 |
| myopic | 0.021023 | 18 | 28.100000 | 16 | -0.007585 | 0.013243 | -0.019091 | 0.021672 |
| graph_base | 0.001627 | 14 | 17.200000 | 17 | -0.000623 | 0.001325 | -0.001688 | 0.001490 |

## Reflection

- Phase 0/2 is now complete: Myopic is present in the same official-u1/head-init protocol.
- Myopic does not change the main conclusion. It is almost tied with the MF-centered Linear UCB proxy, and DU-Net beats both under the local paper-like reward-weighted metric.
- The remaining gap to the paper visual target did not shrink; the result is still a conditional, directionally consistent reproduction rather than an exact Figure 4a reproduction.
- Conventional RMSE still favors the comparators, so the final claim should stay scoped to reward-weighted or high-reward-tail performance.
- Seed 15 remains the useful next diagnostic: the frozen late-lambda schedule loses there to both Linear UCB proxy and Myopic on the primary metric and reward.

## Seed 15 Late-Cycle Snapshot

| variant | cycle | cumulative_reward | linear_reward_norm_w_rmse | rmse | selected_avg_reward_percentile | selected_avg_entropy_percentile | selected_high_degree_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph_late_lambda_half_start16 | 15 | 2063.000000 | 0.995779 | 1.220972 | 0.952837 | 0.887870 | 0.340000 |
| graph_late_lambda_half_start16 | 16 | 2179.000000 | 0.996217 | 1.221300 | 0.999235 | 0.743387 | 0.640000 |
| graph_late_lambda_half_start16 | 17 | 2295.000000 | 0.997133 | 1.220189 | 0.999061 | 0.619934 | 0.560000 |
| graph_late_lambda_half_start16 | 18 | 2423.000000 | 0.996956 | 1.219766 | 0.998798 | 0.900946 | 0.480000 |
| graph_late_lambda_half_start16 | 19 | 2550.000000 | 0.993522 | 1.219980 | 0.998129 | 0.787733 | 0.450000 |
| graph_late_lambda_half_start16 | 20 | 2676.000000 | 0.987442 | 1.221376 |  |  |  |
| myopic | 15 | 2127.000000 | 0.983741 | 1.227423 | 0.999363 | 0.622343 | 0.280000 |
| myopic | 16 | 2249.000000 | 0.984101 | 1.226470 | 0.999362 | 0.714477 | 0.310000 |
| myopic | 17 | 2376.000000 | 0.976071 | 1.223159 | 0.999361 | 0.774613 | 0.340000 |
| myopic | 18 | 2504.000000 | 0.976463 | 1.222657 | 0.999360 | 0.665639 | 0.200000 |
| myopic | 19 | 2647.000000 | 0.973954 | 1.224990 | 0.999360 | 0.656091 | 0.220000 |
| myopic | 20 | 2768.000000 | 0.978446 | 1.223200 |  |  |  |
| mf_linear_ucb_a002 | 15 | 2102.000000 | 0.999850 | 1.224480 |  |  |  |
| mf_linear_ucb_a002 | 16 | 2235.000000 | 0.996635 | 1.224385 |  |  |  |
| mf_linear_ucb_a002 | 17 | 2354.000000 | 0.989237 | 1.222819 |  |  |  |
| mf_linear_ucb_a002 | 18 | 2466.000000 | 0.991308 | 1.220994 |  |  |  |
| mf_linear_ucb_a002 | 19 | 2601.000000 | 0.986119 | 1.222419 |  |  |  |
| mf_linear_ucb_a002 | 20 | 2735.000000 | 0.981172 | 1.223819 |  |  |  |
| graph_base | 15 | 2063.000000 | 0.995779 | 1.220972 | 0.952837 | 0.887870 | 0.340000 |
| graph_base | 16 | 2179.000000 | 0.996217 | 1.221300 | 0.992646 | 0.818856 | 0.570000 |
| graph_base | 17 | 2301.000000 | 0.994002 | 1.221602 | 0.995775 | 0.774141 | 0.520000 |
| graph_base | 18 | 2441.000000 | 0.988254 | 1.224568 | 0.997000 | 0.948268 | 0.440000 |
| graph_base | 19 | 2576.000000 | 0.983655 | 1.225155 | 0.982520 | 0.909471 | 0.620000 |
| graph_base | 20 | 2715.000000 | 0.973366 | 1.228692 |  |  |  |

## Next Step

Run the Phase 1 micro-screen on seeds `0,1,2,15`: add/measure `late lambda half start15`, reuse `late lambda half start17`, and test the existing `frontier95_floor25` guard. Promote nothing unless it improves seed 15 without lowering the four-seed mean against the frozen baseline.
