# Douban Reward-Frontier Guard Findings

This note records the follow-up branch after weighted network entropy. The goal
was to test whether information value can help Douban only inside a high-reward
candidate frontier.

## What Was Added

`AcquisitionConfig` now supports three default-off guard parameters:

- `entropy_reward_frontier_quantile`: reward percentile threshold for allowing
  positive entropy bonuses;
- `entropy_reward_frontier_floor`: optional residual entropy-bonus multiplier
  outside the frontier;
- `entropy_reward_frontier_softness`: optional logistic smoothing around the
  threshold.

The implementation only gates positive entropy bonuses. Negative entropy
penalties are preserved, so low-information candidates are not accidentally
helped by the guard.

## Douban 3-Seed Result

Config: `configs/douban_svdpp_reward_frontier_guard_3seeds.json`

| variant | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability top-k | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| weighted guard q99 lambda=0.20 | 2950.00 | 0.6657 | 0.6467 | 1.0052 | Best guarded variant; improves weighted entropy but does not beat current netobs. |
| netobs guard q99 lambda=0.20 | 2925.67 | 0.6687 | 0.6447 | 0.9920 | More information-oriented and lower-degree, but worse w-RMSE. |
| netobs guard q99 lambda=0.10 | 2924.00 | 0.6705 | 0.6003 | 1.2283 | Only slightly above myopic. |
| weighted guard q99 lambda=0.10 | 2922.33 | 0.6706 | 0.5902 | 1.2222 | Only slightly above myopic. |
| myopic | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |
| netobs guard q95 lambda=0.20 | 2899.67 | 0.6715 | 0.6756 | 0.9803 | Strongest information shift, but over-guards reward. |

## Interpretation

The guard works mechanically: q95/q99 variants raise selected entropy percentile
from 0.5467 to 0.6003-0.6756 and reduce selected average degree from 1.2252 to
roughly 0.98-1.01 in the strongest guarded variants.

But the accuracy result is negative. The current top-k network-observational
candidate remains best by w-RMSE. Guarding helps the weighted entropy branch
relative to its previous unguarded result, but the improvement is not enough for
20-seed promotion.

## Decision

Do not promote reward-frontier guarded entropy to 20 seeds. Keep the parameters
as useful diagnostic infrastructure, but stop local entropy-gate sweeps for now.
The next reproduction work should move to either true structural representation
fidelity or paper-specific benchmark definitions.

Primary artifacts:

- `outputs/tables/douban_svdpp_reward_frontier_guard_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_reward_frontier_guard_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_reward_frontier_guard_3seeds_report.md`
- `reports/douban_svdpp_reward_frontier_guard_3seeds_mechanism_diagnostics.md`
