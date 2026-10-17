# Douban Weighted Network Entropy Findings

This note records the next entropy-fidelity branch after the expected,
network-batch, stochastic, and reliability-stochastic pilots.

## What Was Added

`entropy_mode="network_observational_weighted"` implements a deterministic
probability-weighted network log-det estimator:

- the focal candidate is treated as a certain observation;
- same-item social-neighbor candidates are treated as expected observations;
- neighbor feature rows are scaled by the square root of calibrated inclusion
  probabilities before the small-set log-det is computed;
- probabilities are damped by focal/neighbor reliability, optional degree
  normalization, and the neighbor's one-step information value.

This is closer to an expected information matrix than the older set-max proxy
and avoids Monte Carlo variance from the stochastic branch.

## Douban 3-Seed Result

Config: `configs/douban_svdpp_weighted_network_entropy_3seeds.json`

| variant | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability top-k | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains best. |
| weighted netobs BFS p=0.02 | 2966.33 | 0.6662 | 0.5090 | 2.3765 | BFS embedding raises centrality and loses w-RMSE. |
| weighted netobs p=0.01 | 2956.67 | 0.6667 | 0.5707 | 1.2877 | More information-oriented than top-k, but weaker w-RMSE. |
| weighted netobs p=0.02 | 2951.00 | 0.6678 | 0.5024 | 2.2562 | Higher network probability creates centrality drift. |
| myopic | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |
| weighted netobs p=0.05 + degree norm | 2865.67 | 0.6770 | 0.4217 | 3.7360 | Degree normalization is not enough at high p. |

## Interpretation

The weighted estimator is a useful fidelity improvement in code, but not a
better Douban candidate. Low-probability weighted netobs still beats myopic, so
the network information signal is not broken. However, the current set-max
`network_observational` top-k proxy remains the strongest 3-seed result.

The important mechanism finding is that larger expected network probability
still pushes selection toward high-degree users. Even with reliability damping
and degree normalization, `p=0.05` is too aggressive for sparse Douban.

## Decision

Do not promote weighted network entropy to 20 seeds. Keep it as tested
infrastructure and negative evidence. The next entropy work should not be
another probability sweep; it should change the information target or combine
information with a stricter reward-frontier guard.

Primary artifacts:

- `outputs/tables/douban_svdpp_weighted_network_entropy_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_weighted_network_entropy_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_weighted_network_entropy_3seeds_report.md`
- `reports/douban_svdpp_weighted_network_entropy_3seeds_mechanism_diagnostics.md`
