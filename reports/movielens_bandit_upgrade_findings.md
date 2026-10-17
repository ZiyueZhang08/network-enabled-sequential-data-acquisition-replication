# MovieLens Bandit Upgrade Findings

This note records the first baseline-quality upgrade after filling the
MovieLens cross-dataset baseline rows.

## What Changed

The legacy bandit baselines remain available and unchanged:

- `linear_ucb`
- `low_rank_bandit`
- `thompson_sampling`

This pilot adds conservative MF-centered alternatives:

- `mf_linear_ucb`: MF prediction plus a z-scored linear-UCB uncertainty bonus.
- `mf_low_rank_ucb`: MF prediction plus a z-scored user/item count uncertainty bonus.
- `mf_thompson_sampling`: MF prediction plus count-calibrated Thompson noise.

The purpose is to avoid the legacy failure mode where the bandit score moves too
far away from the learned reward frontier.

## MovieLens 3-Seed Result

Config: `configs/movielens_100k_svdpp_bandit_upgrade_3seeds.json`

| variant | reward | w-RMSE | reading |
|---|---:|---:|---|
| DU-Net | 2717.33 | 0.8229 | Best w-RMSE in this 3-seed pilot. |
| graph_social | 2748.67 | 0.8246 | Highest reward, still slightly worse w-RMSE than DU-Net. |
| MF linear UCB, alpha=0.05 | 2702.33 | 0.8306 | Strongly improves over legacy bandits and myopic w-RMSE, but does not beat DU-Net. |
| myopic | 2704.00 | 0.8354 | Strong reward baseline. |
| MF Thompson, noise=0.08 | 2682.33 | 0.8390 | Much stronger than legacy Thompson, but still weaker than myopic/DU-Net. |
| MF low-rank UCB, alpha=0.05 | 2642.67 | 0.8424 | Improved over legacy low-rank bandit, but not enough. |
| legacy low-rank bandit | 1749.00 | 0.9978 | Confirms the old baseline was too weak. |
| legacy Thompson | 1640.67 | 1.0184 | Confirms the old baseline was too weak. |

## Interpretation

The conservative MF-centered design fixes the obvious baseline-quality problem:
the upgraded bandits no longer collapse to random-like reward. However, the
best upgraded candidate (`mf_linear_ucb_a005`) still trails DU-Net on w-RMSE and
graph_social on reward.

This is useful evidence but not yet paper-quality baseline strength. The next
branch should tune only `mf_linear_ucb` first because it is the best upgraded
bandit in this pilot. A small alpha grid is more promising than promoting the
current alpha=0.05 result directly to 20 seeds.

## Decision

Do not promote all upgraded bandits to 20 seeds yet. A focused `mf_linear_ucb`
alpha grid was run next; see below.

## Follow-up: MF Linear UCB Alpha Grid

Config: `configs/movielens_100k_svdpp_mf_linear_ucb_alpha_grid_3seeds.json`

| variant | reward | w-RMSE | reading |
|---|---:|---:|---|
| DU-Net | 2717.33 | 0.8229 | Best w-RMSE in the grid. |
| graph_social | 2748.67 | 0.8246 | Highest reward; still slightly worse w-RMSE than DU-Net. |
| MF linear UCB, alpha=0.05 | 2702.33 | 0.8306 | Best upgraded bandit, but below DU-Net. |
| MF linear UCB, alpha=0.02 | 2699.00 | 0.8349 | Near myopic. |
| myopic / alpha=0.00 | 2704.00 | 0.8354 | Alpha=0 reproduces myopic-style selection. |
| MF linear UCB, alpha=0.10 | 2559.67 | 0.8421 | Too much exploration. |
| MF linear UCB, alpha=0.20 | 2094.33 | 0.8895 | Clearly harmful. |

The alpha grid closes this branch: `alpha=0.05` is the best conservative
bandit setting, but it still does not approach the DU-Net/graph-social frontier.
Increasing alpha quickly damages both reward and w-RMSE.

## Final Decision

Do not promote `mf_linear_ucb` to 20 seeds yet. Keep the upgraded bandit
policies as improved baselines, but the next baseline-quality step should be a
stronger algorithmic definition, not more alpha tuning.

## Follow-up: Residual Bandit Pilot

Config: `configs/movielens_100k_svdpp_residual_bandit_3seeds.json`

This branch tested whether a residual correction on top of the MF prediction can
make the bandit baseline more competitive.

| variant | reward | w-RMSE | reading |
|---|---:|---:|---|
| DU-Net | 2717.33 | 0.8229 | Still best w-RMSE. |
| graph_social | 2748.67 | 0.8246 | Still best reward. |
| MF linear UCB, alpha=0.05 | 2702.33 | 0.8306 | Best upgraded bandit overall. |
| residual UCB, weight=0.25 alpha=0.02 | 2705.33 | 0.8318 | Best residual variant, but below MF-linear-UCB. |
| residual UCB, weight=0.50 alpha=0.05 | 2608.67 | 0.8344 | Residual correction is too strong. |
| residual UCB, weight=0.25 alpha=0.05 | 2650.67 | 0.8345 | Worse than the lighter-alpha residual setting. |
| residual Thompson, weight=0.25 noise=0.04 | 2699.33 | 0.8348 | Near myopic, not competitive. |
| myopic | 2704.00 | 0.8354 | Strong reward baseline. |

Residual correction improves over the legacy bandit collapse, but it does not
beat the simpler conservative `mf_linear_ucb_a005`. This suggests that residual
modeling on the already-fitted MF factors is not adding reliable signal in the
current acquisition loop.

## Updated Final Decision

Do not promote residual bandits to 20 seeds. The current baseline-quality work
has closed the obvious "too weak baseline" bug, but a genuinely stronger paper
baseline probably requires a different algorithmic family, not more local
variants of MF-plus-bonus.

Primary artifacts:

- `outputs/tables/movielens_100k_svdpp_bandit_upgrade_3seeds_final_summary.csv`
- `outputs/tables/movielens_100k_svdpp_bandit_upgrade_3seeds_mechanism_summary.csv`
- `outputs/tables/movielens_100k_svdpp_mf_linear_ucb_alpha_grid_3seeds_final_summary.csv`
- `outputs/tables/movielens_100k_svdpp_mf_linear_ucb_alpha_grid_3seeds_mechanism_summary.csv`
- `outputs/tables/movielens_100k_svdpp_residual_bandit_3seeds_final_summary.csv`
- `outputs/tables/movielens_100k_svdpp_residual_bandit_3seeds_mechanism_summary.csv`
- `reports/movielens_100k_svdpp_bandit_upgrade_3seeds_report.md`
- `reports/movielens_100k_svdpp_bandit_upgrade_3seeds_mechanism_diagnostics.md`
- `reports/movielens_100k_svdpp_mf_linear_ucb_alpha_grid_3seeds_report.md`
- `reports/movielens_100k_svdpp_mf_linear_ucb_alpha_grid_3seeds_mechanism_diagnostics.md`
- `reports/movielens_100k_svdpp_residual_bandit_3seeds_report.md`
- `reports/movielens_100k_svdpp_residual_bandit_3seeds_mechanism_diagnostics.md`
