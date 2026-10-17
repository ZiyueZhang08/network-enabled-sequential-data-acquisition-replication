# Flixster Paper-Bandit Repair Findings

Generated after:

- `configs/flixster_svdpp_paper_bandit_repair_3seeds.json`
- `configs/flixster_svdpp_mf_oful_beta_zoom_3seeds.json`

## Result

| branch | variant | reward | w-RMSE | decision |
|---|---|---:|---:|---|
| Frontier | `du_net` | 2988.50 | 0.6254 | Best w-RMSE in both repair screens. |
| Frontier | `graph_social` | 2992.17 | 0.6274 | Best reward in both repair screens. |
| Reward-only | `myopic` | 2885.83 | 0.6279 | Still stronger than all OFUL repairs. |
| Raw OFUL | `oful_raw_b100` | 1422.67 | 0.9432 | Confirms the original OFUL adaptation is badly misaligned. |
| MF-centered UCB | `mf_linear_ucb_a005` | 2502.50 | 0.6704 | Better than raw OFUL, but not competitive. |
| MF-centered OFUL | `mf_oful_b0025` | 2684.50 | 0.6454 | Best in the first repair grid; large improvement over raw OFUL. |
| MF-OFUL beta zoom | `mf_oful_b0005` | 2833.00 | 0.6325 | Best repaired OFUL after zoom; closer to myopic but still behind. |
| MF-OFUL beta zoom | `mf_oful_b001` | 2813.83 | 0.6344 | Slightly weaker than the smallest bonus. |

## Interpretation

The repair confirms the failure mode: the raw linear OFUL implementation was
under-exploiting the recommender signal. Replacing its linear-regression mean
with the SVD++ predicted reward and using OFUL only as an exploration bonus
raises Flixster reward from 1422.67 to 2833.00 and improves w-RMSE from 0.9432
to 0.6325.

However, the repaired branch still does not reproduce the paper's qualitative
OFUL result. The paper reports OFUL as the best Flixster cumulative-reward
benchmark, but the best local repaired OFUL remains below myopic by 52.83 reward
points and behind graph_social by 159.17 reward points. Smaller OFUL bonuses are
monotonically better in the zoom grid, which suggests the current uncertainty
bonus is still mostly harmful rather than a useful exploration signal.

## Decision

Do not promote OFUL or MF-OFUL to a 20-seed final benchmark row. The local
feature/adaptation repair is scientifically useful because it diagnoses the
gap, but it does not close it.

The next benchmark branch should move to the remaining paper rows with strict
labels: `Logistic UCB`, `Logistic Thompson Sampling`, finite-arm `UCB`,
`Epsilon Greedy`, `Adaptive Greedy`, and `Explore-Then-Exploit`. In parallel,
the exact paper/author bandit implementation should be recovered if available
before any final claim about matching the paper's 16-benchmark suite.

Artifacts:

- `outputs/tables/flixster_svdpp_paper_bandit_repair_3seeds_final_summary.csv`
- `outputs/tables/flixster_svdpp_mf_oful_beta_zoom_3seeds_final_summary.csv`
- `outputs/figures/flixster_svdpp_paper_bandit_repair_3seeds_w_rmse.png`
- `outputs/figures/flixster_svdpp_mf_oful_beta_zoom_3seeds_w_rmse.png`
