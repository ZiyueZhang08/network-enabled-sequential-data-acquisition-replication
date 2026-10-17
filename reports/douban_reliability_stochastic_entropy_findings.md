# Douban Reliability-Calibrated Stochastic Network Entropy

This note records the follow-up after the first stochastic subset network log-det entropy pilot.

## Motivation

The previous `network_observational_stochastic` pilot showed a clear failure mode on Douban: raising network observation probability from `p=0.01` to `p=0.05` over-selected high-degree users and damaged final w-RMSE. This run therefore tested whether degree/reliability-calibrated stochastic inclusion can preserve the more faithful stochastic subset-logdet objective while reducing centrality bias.

## Implementation

New entropy mode:

- `network_observational_stochastic_reliability`

New config field:

- `network_entropy_reliability_strength`

The implementation softly damps same-item social-neighbor inclusion probabilities by focal and neighbor reliability, using observed user counts and user degree. It is deliberately separate from the existing stochastic mode so earlier results remain frozen.

Primary config:

- `configs/douban_svdpp_reliability_stochastic_entropy_3seeds.json`

## Results

| variant | seeds | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Still the strongest faithful Douban pilot. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |
| reliability stochastic p=0.05 + degree norm | 3 | 2864.33 | 0.6773 | 0.4212 | 3.8183 | Degree normalization partly controls the failure, but still lags myopic/netobs. |
| stochastic netobs p=0.05 | 3 | 2661.67 | 0.7032 | 0.3733 | 4.7875 | Severe centrality bias. |
| reliability stochastic p=0.05 | 3 | 2661.33 | 0.7039 | 0.3703 | 4.8827 | Reliability damping alone does not fix the high-degree collapse. |

## Interpretation

This is a negative but useful mechanism result.

Reliability calibration alone is too weak when stochastic network inclusion is unnormalized. The selected average user degree remains around 4.9, much higher than the netobs reliability top-k candidate at 1.23. Degree-normalized stochastic inclusion reduces selected degree to 3.82 and improves w-RMSE from about 0.704 to 0.677, which shows the root issue is network probability scaling. However, this still does not beat either myopic or the current netobs reliability top-k candidate.

The next Douban entropy work should therefore not promote stochastic entropy to 20 seeds. The practical rule is:

- stochastic network entropy must use degree normalization or a stronger structural prior;
- reliability damping is useful as a guardrail, not enough as the main fix;
- the current best faithful Douban candidate remains `netobs_reliability_spectral`.

## Artifacts

- `outputs/tables/douban_svdpp_reliability_stochastic_entropy_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_reliability_stochastic_entropy_3seeds_mechanism_summary.csv`
- `outputs/figures/douban_svdpp_reliability_stochastic_entropy_3seeds_cold_warm_w_rmse.png`
- `outputs/figures/douban_svdpp_reliability_stochastic_entropy_3seeds_centrality_binned_w_rmse.png`
- `outputs/figures/douban_svdpp_reliability_stochastic_entropy_3seeds_selected_degree_bins.png`
- `outputs/figures/douban_svdpp_reliability_stochastic_entropy_3seeds_dls_shrinkage_over_cycles.png`
- `outputs/figures/douban_svdpp_reliability_stochastic_entropy_3seeds_reward_entropy_percentiles.png`
