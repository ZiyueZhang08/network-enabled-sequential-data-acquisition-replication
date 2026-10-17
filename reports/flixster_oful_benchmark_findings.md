# Flixster OFUL Benchmark Findings

Generated after `configs/flixster_svdpp_oful_3seeds.json`.

## Result

| variant | reward | w-RMSE | decision |
|---|---:|---:|---|
| `du_net` | 2988.50 | 0.6254 | Current best w-RMSE in this 3-seed screen. |
| `graph_social` | 2992.17 | 0.6274 | Best reward, close to DU-Net on w-RMSE. |
| `myopic` | 2885.83 | 0.6279 | Strong local reward-only comparator. |
| `oful_b100` | 1422.67 | 0.9432 | Best OFUL width, but far below the frontier. |
| `linear_ucb` | 1407.67 | 0.9479 | Confirms the old linear-bandit weakness. |
| `oful_b005` | 1398.67 | 0.9523 | Narrow confidence does not fix the gap. |
| `oful_b025` | 1377.67 | 0.9592 | Also weak. |

## Interpretation

OFUL is now implemented as a separate ridge-linear optimistic policy and runs
end-to-end. However, the current local adaptation does **not** reproduce the
paper's claim that OFUL is the strongest Flixster cumulative-reward benchmark.
The best OFUL setting is only slightly better than the existing `linear_ucb`
proxy and is far behind DU-Net, graph-social, and myopic on both acquisition
reward and final w-RMSE.

This is a useful negative result. It means the remaining paper benchmark gap is
not just a missing label; the bandit feature construction / reward scaling /
offline candidate-pool adaptation is still misaligned with the paper's benchmark
implementation.

## Decision

Do not promote OFUL to a 20-seed rerun yet. The next benchmark branch should
either rebuild the linear-bandit feature space around a stronger MF-centered
reward estimate or move to the remaining paper rows (`Logistic UCB`, `Logistic
Thompson Sampling`, and finite-arm bandits) with clear proxy labels.

Artifacts:

- `outputs/tables/flixster_svdpp_oful_3seeds_final_summary.csv`
- `outputs/figures/flixster_svdpp_oful_3seeds_w_rmse.png`
- `outputs/figures/flixster_svdpp_oful_3seeds_cumulative_reward.png`
