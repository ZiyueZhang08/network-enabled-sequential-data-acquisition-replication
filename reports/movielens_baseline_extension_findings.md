# MovieLens Baseline Extension Findings

This note records the 20-seed extension that fills the two cross-dataset
baseline rows previously missing from the MovieLens implemented-baseline matrix.

## What Was Added

Config: `configs/movielens_100k_svdpp_baselines_missing_20seeds.json`

Rows added under a new output prefix, without overwriting the original frozen
MovieLens baseline table:

- `graph_social`
- `thompson_sampling`

## Result

| variant | reward | w-RMSE | reading |
|---|---:|---:|---|
| DU-Net | 2714.55 | 0.8264 | Still best implemented MovieLens baseline by w-RMSE. |
| graph_social | 2751.50 | 0.8300 | Highest reward, but worse w-RMSE than DU-Net. |
| myopic | 2717.90 | 0.8348 | Slightly higher reward than DU-Net, worse w-RMSE. |
| thompson_sampling | 1657.80 | 1.0193 | Weak under the current simple Thompson proxy. |

The new evidence strengthens the MovieLens baseline coverage story: DU-Net does
not win cumulative reward against `graph_social`, but it still has the best
weighted RMSE among the implemented MovieLens baseline families.

## Decision

Keep the original `movielens_100k_svdpp_baselines_20seeds` table frozen and
track the extension through `baseline_coverage_audit.csv`.

Do not claim full paper 16-baseline coverage. This extension closes the obvious
cross-dataset row mismatch, but the bandit and social baselines remain simpler
than the paper's strongest benchmark suite.

Primary artifacts:

- `outputs/tables/movielens_100k_svdpp_baselines_missing_20seeds_final_summary.csv`
- `outputs/tables/movielens_100k_svdpp_baselines_missing_20seeds_metrics.csv`
- `reports/movielens_100k_svdpp_baselines_missing_20seeds_report.md`
