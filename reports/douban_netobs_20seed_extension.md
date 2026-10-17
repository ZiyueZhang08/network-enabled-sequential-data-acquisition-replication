# Douban NetObs 20-Seed Extension

This note records the 20-seed extension requested after the formal Douban mechanism matrix: heterogeneous response, DLS/network ablation, and baseline matrix with `netobs_reliability_spectral` included.

## Heterogeneous Response

Config: `configs/douban_svdpp_robustness_netobs_20seeds.json`

Response model: `P(response)=clip(0.8 + 0.02 * (rating - 3), 0, 1)`.

| variant | reward | w-RMSE | response rate | reading |
|---|---:|---:|---:|---|
| reliability-DLS | 2353.30 | 0.6948 | 0.8303 | Best w-RMSE under response bias. |
| myopic | 2354.70 | 0.6953 | 0.8298 | Highest reward, but slightly worse w-RMSE. |
| no-DLS | 2347.20 | 0.6959 | 0.8300 | Close to reliability-DLS. |
| current DU-Net | 2342.85 | 0.6961 | 0.8304 | Close but not best. |
| netobs reliability spectral | 2337.35 | 0.6969 | 0.8297 | Does not survive heterogeneous response as the best setting. |

Interpretation: netobs reliability-DLS is positive in homogeneous response, but under response bias the simpler reliability-DLS variant is stronger. This suggests the network-observational entropy term is more sensitive to response-selection noise than the reliability-DLS smoothing itself.

## DLS / Network Ablation

Config: `configs/douban_svdpp_ablation_netobs_20seeds.json`

| variant | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---|
| no-DLS | 2863.05 | 0.6777 | 0.5689 | 1.8632 | Best numerical w-RMSE; confirms sparse-graph DLS noise. |
| reliability-DLS | 2861.65 | 0.6778 | 0.5720 | 1.2997 | Nearly ties no-DLS while reducing centrality. |
| netobs reliability spectral | 2861.80 | 0.6778 | 0.5672 | 1.2373 | Most faithful positive mechanism; ties reliability-DLS. |
| dynamic sigma | 2849.20 | 0.6791 | 0.5353 | 1.7174 | Helps but does not close the gap. |
| current DU-Net | 2838.65 | 0.6801 | 0.5385 | 1.8359 | Still centrality-biased. |
| no-network | 2838.60 | 0.6801 | 0.5306 | 1.2569 | Similar w-RMSE to current DU-Net. |
| no-entropy | 2835.00 | 0.6804 | 0.5308 | 1.8429 | Removing entropy slightly hurts. |
| no-reward | 1360.05 | 0.9533 | 0.9996 | 1.3270 | Information-only acquisition collapses reward and w-RMSE. |

Interpretation: the ablation now supports a clean hierarchy. Reward is essential; full DLS is too noisy on Douban; reliability-DLS is the right sparse-graph control; netobs reliability-DLS is the paper-faithful variant worth carrying forward even though no-DLS remains marginally best numerically.

## Baseline Matrix

Config: `configs/douban_svdpp_baselines_netobs_20seeds.json`

| variant | reward | w-RMSE | reading |
|---|---:|---:|---|
| netobs reliability spectral | 2861.80 | 0.6778 | Best baseline-matrix w-RMSE. |
| current DU-Net | 2838.65 | 0.6801 | Essentially tied with myopic. |
| myopic | 2844.60 | 0.6801 | Strong reward baseline. |
| graph-social | 2835.00 | 0.6812 | Does not beat myopic/current DU-Net. |
| Thompson sampling | 1409.65 | 0.9417 | Weak under current implementation. |
| Linear UCB | 1400.10 | 0.9424 | Weak under current implementation. |
| low-rank bandit | 1378.50 | 0.9487 | Weak under current implementation. |
| random | 1370.35 | 0.9495 | Weak baseline. |
| entropy | 1358.75 | 0.9543 | Information-only acquisition is not enough. |

Interpretation: within the current benchmark implementation, netobs reliability-DLS is now the strongest Douban method in the baseline matrix. The remaining fidelity gap is that the bandit baselines are still simple implementations, not full paper-grade competitors.

## Artifacts

- `outputs/tables/douban_svdpp_robustness_netobs_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_ablation_netobs_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_baselines_netobs_20seeds_final_summary.csv`
- `reports/douban_svdpp_ablation_netobs_20seeds_mechanism_diagnostics.md`
- `reports/douban_svdpp_baselines_netobs_20seeds_mechanism_diagnostics.md`

## Current Decision

Use two Douban candidates going forward:

- `no_dls` as the strongest numerical sparse-graph fallback;
- `netobs_reliability_spectral` as the strongest paper-faithful candidate in the baseline matrix.

Update: this decision has been superseded by response-aware netobs. `netobs_response_centered_str2` now beats reliability-DLS under alpha `0.02` in a 20-seed confirmation; see `reports/douban_response_aware_netobs_findings.md`.
