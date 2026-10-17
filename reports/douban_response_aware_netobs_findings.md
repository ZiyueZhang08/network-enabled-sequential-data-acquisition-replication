# Douban Response-Aware NetObs Findings

This note records the next mechanism step after the Douban heterogeneous response result showed that plain netobs reliability-DLS was weaker than reliability-DLS under alpha `0.02`.

## Progress Assessment

The replication is now in a mechanism-refinement stage:

- MovieLens and Flixster already reproduce the main DU-Net direction.
- Douban homogeneous response has a positive paper-faithful candidate: `netobs_reliability_spectral`.
- Douban heterogeneous response was the remaining weakness: plain netobs was less robust than reliability-DLS.

The next reasonable mechanism was therefore response-aware information value: expected information should be discounted when a candidate is unlikely to respond.

## Implementation Added

`AcquisitionConfig` now supports:

- `entropy_response_weighting="expected_response"`
- `entropy_response_weighting="centered_expected_response"`
- `entropy_response_weight_strength`

When `response_alpha > 0`, the entropy term is multiplied by a predicted response weight based on the model-predicted rating:

`P(response) = clip(0.8 + alpha * (predicted_rating - midpoint), 0, 1)`.

This uses predicted rating rather than true rating, so it remains a deployable acquisition rule rather than an oracle.

## 3-Seed Pilot

Config: `configs/douban_svdpp_response_aware_netobs_3seeds.json`

| variant | reward | w-RMSE | reading |
|---|---:|---:|---|
| netobs response centered strength 2 | 2455.00 | 0.6798 | Best pilot; strong enough to promote. |
| netobs response expected | 2429.67 | 0.6831 | Improves over plain netobs. |
| netobs response centered | 2429.67 | 0.6831 | Same as expected at strength 1 in this pilot. |
| netobs reliability spectral | 2389.00 | 0.6885 | Plain netobs remains weaker under response bias. |
| myopic | 2381.00 | 0.6898 | Strong baseline. |
| reliability-DLS | 2341.00 | 0.6965 | Weak in this 3-seed pilot, despite winning the earlier 20-seed heterogeneous check. |

## 20-Seed Confirmation

Config: `configs/douban_svdpp_response_aware_netobs_20seeds.json`

| variant | reward | w-RMSE | response rate | selected degree | reading |
|---|---:|---:|---:|---:|---|
| netobs response centered strength 2 | 2360.35 | 0.6934 | 0.8294 | 1.2128 | Best 20-seed heterogeneous response candidate. |
| reliability-DLS | 2353.30 | 0.6948 | 0.8303 | 1.2792 | Previous robust candidate; now slightly behind. |
| myopic | 2354.70 | 0.6953 | 0.8298 | 1.2925 | Strong reward baseline. |
| plain netobs reliability spectral | 2337.35 | 0.6969 | 0.8297 | 1.2767 | Response-aware weighting fixes this weakness. |

## Interpretation

This is a meaningful Douban improvement:

- homogeneous response: `netobs_reliability_spectral` is the faithful positive candidate;
- heterogeneous response: `netobs_response_centered_str2` becomes the best candidate;
- selected average degree falls to `1.2128`, lower than reliability-DLS and plain netobs, so the gain is not just reward chasing.

The result narrows the distance to the paper: DU-Net is now not only positive on homogeneous Douban follow-up, but also has a response-aware variant that wins the heterogeneous response matrix.

## Current Decision

Carry two Douban variants into paper-quality reruns:

- `netobs_reliability_spectral` for homogeneous response and baseline matrix;
- `netobs_response_centered_str2` for heterogeneous response and any response-bias robustness section.

Primary artifacts:

- `outputs/tables/douban_svdpp_response_aware_netobs_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_response_aware_netobs_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_response_aware_netobs_20seeds_mechanism_summary.csv`
- `outputs/figures/douban_svdpp_response_aware_netobs_20seeds_w_rmse_by_alpha.png`
- `outputs/figures/douban_svdpp_response_aware_netobs_20seeds_reward_entropy_percentiles.png`
- `reports/douban_svdpp_response_aware_netobs_20seeds_mechanism_diagnostics.md`
