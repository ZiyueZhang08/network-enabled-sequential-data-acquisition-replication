# Douban Network-Batch Entropy Findings

This note records the first batch-level network information selection path after the expected-network entropy pilot.

## What Changed

The new selection mode `network_batch_composition` keeps the fixed batch composition idea, but makes information/diversity slots network-aware. After selecting the reward portion of the batch, the information slots are chosen greedily with a penalty for:

- repeated users;
- repeated items;
- same-item candidates whose users are social neighbors of already selected candidates.

This is not the paper's exact acquisition optimizer, but it is closer to a batch network-observation objective than scalar z-score mixing or plain entropy top-k.

Code:

- `src/acquisition.py`: `selection="network_batch_composition"`
- `tests/test_core.py`: regression test for social-neighbor redundancy penalty

## Douban 3-Seed Check

Config: `configs/douban_svdpp_network_batch_entropy_3seeds.json`

| variant | seeds | reward | w-RMSE | reward pct | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.9996 | 0.5467 | 1.2252 | Current faithful candidate remains best. |
| expected-netobs network-batch p=0.10 | 3 | 2939.33 | 0.6661 | 0.9990 | 0.6520 | 1.1333 | More informative/lower centrality, weaker w-RMSE. |
| netobs network-batch p=0.02 | 3 | 2936.33 | 0.6661 | 0.9990 | 0.6631 | 1.0698 | Highest information orientation, lower selected degree, weaker w-RMSE. |
| expected-netobs network-batch p=0.02 | 3 | 2936.33 | 0.6661 | 0.9990 | 0.6631 | 1.0698 | Same selections as non-expected netobs under this setting. |
| myopic | 3 | 2921.33 | 0.6710 | 0.9996 | 0.5406 | 1.2343 | Strong baseline. |

## Interpretation

The batch-level network selector does change behavior in the intended direction:

- entropy percentile rises from `0.5467` to about `0.65-0.66`;
- selected average degree falls from `1.2252` to about `1.07-1.13`;
- reward percentile remains high but drops slightly from the pure top-k frontier.

The cost is worse final w-RMSE relative to the current `netobs_reliability_topk`. This is useful mechanism evidence, not a promoted main candidate. It suggests the current information signal can move selection, but the information it moves toward is not yet predictive enough to improve the learned recommender on Douban.

## Decision

Do not expand `network_batch_composition` to 20 seeds yet. The next high-value step is to improve the information representation itself:

1. stronger structural embedding, ideally true sparse Node2Vec or ELPH;
2. stochastic subset/log-det network entropy;
3. response-aware expected network entropy under alpha=0.02.

Only rerun 20 seeds after a 3-seed pilot beats or clearly matches `netobs_reliability_topk` on w-RMSE while retaining better information/centrality diagnostics.

## Artifacts

- `outputs/tables/douban_svdpp_network_batch_entropy_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_network_batch_entropy_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_network_batch_entropy_3seeds_mechanism_diagnostics.md`
- `outputs/figures/douban_svdpp_network_batch_entropy_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_network_batch_entropy_3seeds_reward_entropy_percentiles.png`
