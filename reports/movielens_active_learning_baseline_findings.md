# MovieLens Active-learning Baseline Findings

This note records a focused active-learning baseline screen after the MF-bandit
upgrade branch.

## What Was Added

The new baseline policies are intentionally separate from existing DU-Net,
bandit, and entropy policies:

- `prediction_uncertainty`: selects candidates whose predicted rating is closest
  to the rating midpoint.
- `leverage_uncertainty`: selects candidates with high interaction-feature
  leverage / UCB-style information.
- `active_learning_hybrid`: combines prediction-margin uncertainty with feature
  leverage and sparse user/item coverage.

## MovieLens 3-Seed Result

Config: `configs/movielens_100k_svdpp_active_learning_baselines_3seeds.json`

| variant | reward | w-RMSE | reading |
|---|---:|---:|---|
| DU-Net | 2717.33 | 0.8229 | Best w-RMSE in this screen. |
| graph_social | 2748.67 | 0.8246 | Highest reward. |
| MF linear UCB, alpha=0.05 | 2702.33 | 0.8306 | Best upgraded bandit baseline. |
| myopic | 2704.00 | 0.8354 | Strong reward baseline. |
| entropy | 1041.33 | 1.0536 | Pure information selection is too far from the reward frontier. |
| active hybrid, alpha=0.20 | 267.67 | 1.1637 | Best active-learning variant, still very weak. |
| prediction uncertainty | 229.33 | 1.1719 | Selects low-reward ambiguous ratings. |
| active hybrid, alpha=0.05 | 186.33 | 1.1866 | Very weak. |
| leverage uncertainty | 92.00 | 1.1868 | Very weak. |

## Interpretation

The active-learning baselines are useful for coverage, but not for promotion.
They confirm the same mechanism seen in the entropy-only row: pure information
selection moves far away from the reward frontier and does not improve final
w-RMSE enough to compensate.

The paper-level baseline gap therefore cannot be closed by naive active-learning
uncertainty baselines. Stronger baselines must either stay reward-aware or use a
more formal matrix-completion objective than these simple proxies.

## Decision

Do not promote active-learning variants to 20 seeds. Keep them as documented
negative baseline-coverage evidence.

Primary artifacts:

- `outputs/tables/movielens_100k_svdpp_active_learning_baselines_3seeds_final_summary.csv`
- `outputs/tables/movielens_100k_svdpp_active_learning_baselines_3seeds_mechanism_summary.csv`
- `reports/movielens_100k_svdpp_active_learning_baselines_3seeds_report.md`
- `reports/movielens_100k_svdpp_active_learning_baselines_3seeds_mechanism_diagnostics.md`
