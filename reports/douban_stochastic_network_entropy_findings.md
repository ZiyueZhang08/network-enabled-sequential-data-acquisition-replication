# Douban Stochastic Network Entropy Findings

This note records the first stochastic subset log-det implementation for network observational entropy.

## What Changed

The new entropy mode `network_observational_stochastic` treats the focal candidate as a certain observation, then samples same-item social-neighbor candidates as probabilistic network observations. Each Monte Carlo subset is scored by small-set log-det gain, and the candidate score is the deterministic per-row average across sampled subsets.

Code:

- `src/entropy.py`: `network_observational_stochastic_entropy_scores`
- `src/acquisition.py`: `entropy_mode="network_observational_stochastic"`
- `tests/test_core.py`: zero-diffusion equivalence and linked-neighbor nonnegative gain checks

## Douban 3-Seed Check

Config: `configs/douban_svdpp_stochastic_network_entropy_3seeds.json`

| variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains best. |
| stochastic netobs p=0.01 | 3 | 2981.33 | 0.6640 | 0.5653 | 1.2582 | Slightly more information-oriented, but weaker w-RMSE. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong baseline. |
| stochastic netobs p=0.05 | 3 | 2661.67 | 0.7032 | 0.3733 | 4.7875 | Over-amplifies network sampling and creates severe centrality bias. |

## Interpretation

This is a useful faithful-gap step, but not a new main result. Stochastic subset log-det is implemented and tested, yet the current Douban signal is not improved:

- At the paper-like diffusion scale (`p=0.01`), stochastic entropy raises entropy percentile only modestly and slightly worsens w-RMSE.
- At a larger diffusion scale (`p=0.05`), it over-selects high-degree users and sharply degrades final w-RMSE.

The result reinforces the current Douban diagnosis: the graph contains useful signal, but sparse-graph information value must be reliability-calibrated. Simply making network entropy more stochastic or increasing diffusion does not solve the faithful gap.

## Decision

Do not promote stochastic netobs to 20 seeds in its current form. The next useful direction is not more diffusion-probability scaling, but a better structural representation and calibrated neighbor inclusion:

1. true sparse Node2Vec or ELPH role/structure embedding;
2. degree/reliability-calibrated stochastic inclusion probability;
3. response-aware stochastic entropy under alpha=0.02;
4. only then rerun a 3-seed candidate against `netobs_reliability_topk`.

## Artifacts

- `outputs/tables/douban_svdpp_stochastic_network_entropy_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_stochastic_network_entropy_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_stochastic_network_entropy_3seeds_mechanism_diagnostics.md`
- `outputs/figures/douban_svdpp_stochastic_network_entropy_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_stochastic_network_entropy_3seeds_reward_entropy_percentiles.png`
