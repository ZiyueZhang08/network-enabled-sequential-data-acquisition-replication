# Douban Weighted-Logdet and Reliability-DLS Decision

This note evaluates the step after the negative 20-seed confirmation of plain `logdet + shrinkage DLS`.

## Progress Assessment Before This Step

Current confirmed state:

- MovieLens and Flixster remain successful frozen results.
- Douban current DU-Net is close to myopic but not better.
- Douban `du_net_no_dls` is the strongest confirmed 20-seed follow-up.
- Plain `logdet + adaptive shrinkage DLS` improved the information percentile but failed the 20-seed w-RMSE confirmation.

Interpretation:

- The project has moved from "does the pipeline run?" to "which mechanism explains Douban?"
- The main remaining gap is not factorization capacity. It is the mismatch between information score, graph smoothing, and the weighted error objective.

## What Was Implemented

Code changes:

- weighted log-det entropy:
  - `entropy_logdet_weighting=reward`
  - `entropy_logdet_weighting=cold`
  - `entropy_logdet_weighting=reward_cold`
- reliability-mode DLS shrinkage:
  - `dls_shrinkage_mode=reliability`
  - smoothing strength depends on graph degree, effective neighborhood, and observed-count confidence.

Configs:

- `configs/douban_svdpp_weighted_logdet_reliability_3seeds.json`
- `configs/douban_svdpp_reliability_dls_20seeds.json`

## 3-Seed Matrix

| variant | reward | w-RMSE | reward pct | entropy pct | avg DLS shrinkage | reading |
|---|---:|---:|---:|---:|---:|---|
| `reliability_dls` | 2977.33 | 0.6642 | 0.9996 | 0.5707 | 0.0500 | Best 3-seed variant, essentially tied with no-DLS. |
| `no_dls` | 2979.00 | 0.6643 | 0.9996 | 0.5727 | n/a | Strong sparse-graph baseline. |
| `weighted_reward_reliability_l005` | 2978.33 | 0.6650 | 0.9996 | 0.5949 | 0.0497 | More information-aware, close to no-DLS. |
| `weighted_reward_no_dls` | 2945.33 | 0.6686 | 0.9985 | 0.6154 | n/a | Weighted logdet helps entropy but hurts w-RMSE. |
| `current_du_net` | 2871.67 | 0.6753 | 0.9996 | 0.5321 | n/a | Baseline. |

## 20-Seed Confirmation

| policy | reward | w-RMSE | reward pct | entropy pct | avg degree | status |
|---|---:|---:|---:|---:|---:|---|
| `du_net_no_dls` | 2863.05 | 0.6777 | 0.9996 | 0.5689 | 1.8632 | Best confirmed. |
| `du_net_shrinkage_dls` reliability | 2861.65 | 0.6778 | 0.9996 | 0.5720 | 1.2998 | Essentially tied with no-DLS, lower centrality bias. |
| `du_net_logdet_shrinkage_dls` weighted reward | 2860.05 | 0.6780 | 0.9996 | 0.6073 | 1.2590 | More information-aware, but no accuracy gain. |
| `du_net` | 2838.65 | 0.6801 | 0.9996 | 0.5385 | 1.8359 | Current DU-Net. |
| `myopic` | 2844.60 | 0.6801 | 0.9996 | 0.5435 | 1.2465 | Myopic baseline. |

## Decision

Reliability DLS worked as a damage-control mechanism:

- It nearly matches no-DLS in 20-seed w-RMSE.
- It substantially lowers selected-user centrality bias.
- It prevents the worst sparse-graph smoothing damage.

But it is not a new main Douban winner:

- `du_net_no_dls` remains the best confirmed 20-seed result by a small margin.
- Weighted logdet raises entropy percentile but still does not improve weighted error.

## Basic Next Step

Freeze Douban's confirmed follow-up interpretation:

- Main confirmed Douban variant: `du_net_no_dls`.
- Secondary mechanism variant: `reliability_dls`, because it explains how to reduce centrality bias without much w-RMSE loss.
- Do not promote weighted logdet to main results yet.

The next implementation step should be mechanism diagnostics, not another entropy grid:

1. cold-start vs warm-user w-RMSE;
2. centrality-binned w-RMSE;
3. graph-degree bins for selected users;
4. DLS shrinkage distribution over cycles.

## Long-Term Next Step

To close the paper-faithfulness gap:

- replace scalar z-score mixing with batch composition: e.g. 70% reward, 30% information/diversity;
- implement information gain that is directly weighted by evaluation risk groups, not just predicted reward;
- implement real sparse Node2Vec/ELPH-style embedding;
- treat no-DLS/reliability-DLS as formal ablations in the final Douban report.
