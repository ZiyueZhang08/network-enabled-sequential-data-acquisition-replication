# MovieLens Social-rec Baseline Findings

This note records a focused social-recommendation baseline screen after the
active-learning baseline branch.

## What Was Added

Two graph/social-aware recommendation baselines were added:

- `social_neighbor_item`: scores a candidate by already observed ratings from
  the target user's social neighbors on the same item; candidates without
  neighbor evidence fall back to the MF prediction.
- `social_mf_hybrid`: starts from the MF prediction and shifts it toward the
  same-item social-neighbor mean, damped by available neighbor evidence.

Both baselines use only the currently observed acquisition set; candidate true
ratings are not used for scoring.

## MovieLens 3-Seed Result

Config: `configs/movielens_100k_svdpp_social_rec_baselines_3seeds.json`

| variant | reward | w-RMSE | reading |
|---|---:|---:|---|
| DU-Net | 2717.33 | 0.8229 | Best w-RMSE in this screen. |
| graph_social | 2748.67 | 0.8246 | Highest reward and still the strongest reward baseline. |
| social hybrid, weight=0.50 | 2719.67 | 0.8301 | Best social-rec hybrid by w-RMSE; beats myopic reward but not DU-Net. |
| MF linear UCB, alpha=0.05 | 2702.33 | 0.8306 | Similar w-RMSE to social hybrid. |
| social hybrid, weight=0.25 | 2733.67 | 0.8306 | Higher reward than DU-Net, weaker w-RMSE. |
| myopic | 2704.00 | 0.8354 | Strong reward baseline. |
| social hybrid, weight=1.00 | 2587.33 | 0.8425 | Too much social shift hurts. |
| social neighbor item | 2362.67 | 0.8678 | Pure same-item neighbor evidence is too brittle. |

## Interpretation

The social-rec upgrade closes a baseline-definition gap but does not overturn
the current MovieLens conclusion. A moderate social hybrid is stronger than
myopic on reward and better than myopic on w-RMSE, but it remains behind DU-Net
and graph_social on the w-RMSE frontier.

The pure neighbor-item policy is a useful negative control: it selects candidates
with very high same-item neighbor evidence, but the final w-RMSE is much worse.
This suggests that social evidence is helpful only as a damped correction to the
MF reward model, not as a standalone acquisition objective.

## Decision

Do not promote the social-rec variants to 20 seeds yet. Keep them as documented
baseline-coverage evidence and use `social_hybrid_w050` only as a comparator if
a later full benchmark rerun needs a stronger social-rec row.

Primary artifacts:

- `outputs/tables/movielens_100k_svdpp_social_rec_baselines_3seeds_final_summary.csv`
- `outputs/tables/movielens_100k_svdpp_social_rec_baselines_3seeds_mechanism_summary.csv`
- `reports/movielens_100k_svdpp_social_rec_baselines_3seeds_report.md`
- `reports/movielens_100k_svdpp_social_rec_baselines_3seeds_mechanism_diagnostics.md`
