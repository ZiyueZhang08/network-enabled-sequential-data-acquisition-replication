# Douban Expected Network Entropy Findings

This note records the first implementation of `network_observational_expected`, a more paper-faithful entropy mode than the earlier set-max network log-det proxy.

## What Changed

The new entropy mode treats the focal candidate rating as a certain observation and same-item social-neighbor ratings as probabilistic network-enabled observations. The score is:

- single-entry log-det gain for the selected candidate;
- plus first-order expected marginal log-det gains from same-item social neighbors;
- with optional degree normalization and the same candidate weighting hooks used by existing log-det entropy.

Code:

- `src/entropy.py`: `network_observational_expected_entropy_scores`
- `src/acquisition.py`: `entropy_mode="network_observational_expected"`
- `tests/test_core.py`: regression checks for zero-diffusion equivalence and nonnegative network marginal gain

## Douban 3-Seed Check

Config: `configs/douban_svdpp_network_expected_entropy_3seeds.json`

| variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability spectral | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful Douban candidate. |
| expected-netobs reliability spectral | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Same selections/results as current netobs in this pilot. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline, weaker w-RMSE here. |

## Interpretation

The expected-network estimator is now available and tested, but it should not be promoted as a new main result yet. On this narrow Douban pilot it reproduces the same selected batches as `network_observational`, likely because the confirmed setting uses a small diffusion probability and top-k selection, so both entropy variants perturb the dominant reward frontier in nearly the same order.

This is still useful progress: the code path is closer to the paper's network-enabled observation logic and gives a clean base for the next faithful entropy step. The next version should move from first-order expected marginals to stochastic subset/log-det estimation or a batched network-observation objective that can meaningfully change rankings under sparse Douban structure.

## Artifacts

- `outputs/tables/douban_svdpp_network_expected_entropy_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_network_expected_entropy_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_network_expected_entropy_3seeds_mechanism_diagnostics.md`
- `outputs/figures/douban_svdpp_network_expected_entropy_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_network_expected_entropy_3seeds_reward_entropy_percentiles.png`
