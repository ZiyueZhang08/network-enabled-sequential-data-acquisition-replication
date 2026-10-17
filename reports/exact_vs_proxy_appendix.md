# Exact vs Proxy Appendix

This appendix separates the parts of the current reproduction that closely match the paper from the parts that are deliberate, documented proxies.

## Close to the Paper

| component | current implementation | status |
|---|---|---|
| Sequential data acquisition | Initial observed set, candidate acquisition set, repeated batch acquisition, and final evaluation | Close |
| Offline candidate restriction | Policies select only user-item pairs with known offline ratings | Close |
| Acquisition schedule | 20 cycles, batch size 100, 20-seed main results | Close |
| Three datasets | MovieLens, Flixster, and Douban all run on real data | Close |
| True social graph use | Flixster and Douban use GCMC/Monti `W_users`; MovieLens uses a constructed similarity network | Close enough for current plan |
| Reward-information scoring | DU-Net-style z-scored reward plus information value | Close structurally |
| Heterogeneous response | Response probability increases with true rating under alpha=0.02 | Close |
| Ablation surface | no-DLS, no-network, no-entropy, no-reward, dynamic sigma, reliability-DLS, network-batch selection | Broadly close |

## Proxy or Partial

| paper component | current proxy | evidence / implication |
|---|---|---|
| Exact DU-Net architecture | Modular SVD++ + DLS + score-composition implementation | Captures the acquisition logic but not every neural/architectural detail. |
| ELPH structural embedding | Spectral, sparse Node2Vec-style, biased Node2Vec-style, local SGNS Node2Vec-style, and ELPH-role proxies | Spectral is currently strongest on Douban; SGNS-BFS is positive but not best; true ELPH remains a fidelity gap. |
| Observational entropy | Coverage, subspace, observational, log-det, network-observational, expected-network-observational, stochastic subset network log-det, reliability-calibrated stochastic, probability-weighted expected network log-det, and reward-frontier gated approximations | Network-observational entropy improves Douban; expected/stochastic/weighted/guarded variants are closer structurally but still not the exact paper estimator. |
| Batch information selection | Fixed reward/information composition and network-batch redundancy penalty | Captures part of batch-level network-observation behavior, but is not the paper's exact optimizer. |
| Network reward diffusion | Local same-item neighbor reward diffusion with optional normalization | Flixster DU-Net equals no-network in the tuned config, so current diffusion is not yet doing the paper's full job. |
| DLS dynamic neighborhood | Gaussian smoothing over graph-embedding neighbors plus dynamic sigma/reliability shrinkage | Reliability-DLS reduces Douban centrality bias, but exact paper DLS is not fully replicated. |
| 16 benchmarks | 14 explicitly named paper slots covered or proxy-covered, with focused MF-bandit / active-learning / social-rec pilot screens kept as diagnostics | The paper states 16 benchmarks but the available paper text, footnotes, and public footnote-code sources confirm only 14 named slots; final reporting should carry the 14+2 caveat. |

## Current Empirical Interpretation

MovieLens and Flixster reproduce the central direction: DU-Net-style acquisition improves final weighted RMSE relative to myopic, and Flixster also improves acquisition reward. Douban is now a diagnosed sparse-graph case rather than a failed reproduction. The most faithful positive Douban candidate is `netobs_reliability_spectral`, while `no_dls` remains the numerically strongest sparse-graph fallback by a very small margin.

The focused 20-seed graph-density confirmation strengthens the mechanism story:

| Douban graph condition | reward | w-RMSE | selected degree | interpretation |
|---|---:|---:|---:|---|
| original no-DLS | 2863.05 | 0.6777 | 1.8632 | Best numerical fallback, but removes DLS. |
| original netobs reliability-DLS | 2861.80 | 0.6778 | 1.2373 | Best faithful graph-aware balance. |
| densified netobs reliability-DLS | 2858.50 | 0.6781 | 2.4379 | Accuracy nearly holds, but centrality bias rises. |
| edge-dropout netobs reliability-DLS | 2841.10 | 0.6803 | 0.5292 | Removing graph edges hurts. |
| rewired netobs reliability-DLS | 2825.20 | 0.6826 | 0.9945 | Destroying graph structure hurts most. |

The first expected-network entropy implementation is now tested on Douban:

| Douban entropy variant | seeds | reward | w-RMSE | interpretation |
|---|---:|---:|---:|---|
| network-observational reliability-DLS | 3 | 2996.67 | 0.6626 | Current faithful candidate. |
| expected-network-observational reliability-DLS | 3 | 2996.67 | 0.6626 | Same selections/results in this pilot; useful fidelity infrastructure but not a new main result. |
| myopic | 3 | 2921.33 | 0.6710 | Strong baseline. |

The first network-batch selection pilot changes behavior but is not yet accuracy-positive:

| Douban batch selector | seeds | reward | w-RMSE | entropy pct | selected degree | interpretation |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| network-batch netobs | 3 | 2936.33 | 0.6661 | 0.6631 | 1.0698 | Better information/centrality diagnostics, worse w-RMSE. |

The stochastic subset log-det pilot is also a negative confirmation in its current form:

| Douban stochastic entropy | seeds | reward | w-RMSE | selected degree | interpretation |
|---|---:|---:|---:|---:|---|
| stochastic p=0.01 | 3 | 2981.33 | 0.6640 | 1.2582 | Slight information shift, weaker than current netobs. |
| stochastic p=0.05 | 3 | 2661.67 | 0.7032 | 4.7875 | Over-amplifies high-degree network observations. |

The reliability-calibrated stochastic follow-up confirms that the main issue is probability scaling:

| Douban reliability stochastic entropy | seeds | reward | w-RMSE | selected degree | interpretation |
|---|---:|---:|---:|---:|---|
| reliability stochastic p=0.05 | 3 | 2661.33 | 0.7039 | 4.8827 | Reliability damping alone does not prevent high-degree collapse. |
| reliability stochastic p=0.05 + degree norm | 3 | 2864.33 | 0.6773 | 3.8183 | Degree normalization partly fixes the collapse, but still loses to netobs top-k. |

The deterministic probability-weighted expected network log-det follow-up is also tested:

| Douban weighted network entropy | seeds | reward | w-RMSE | selected degree | interpretation |
|---|---:|---:|---:|---:|---|
| current netobs reliability top-k | 3 | 2996.67 | 0.6626 | 1.2252 | Current faithful candidate remains strongest. |
| weighted BFS p=0.02 | 3 | 2966.33 | 0.6662 | 2.3765 | Best weighted variant; beats myopic but loses to current netobs. |
| weighted spectral p=0.001 | 3 | 2956.67 | 0.6667 | 1.2877 | Low probability is stable but not promotion-worthy. |
| weighted spectral p=0.005 + degree norm | 3 | 2865.67 | 0.6770 | 3.7360 | Higher probability still drifts toward high-degree users. |
| myopic | 3 | 2921.33 | 0.6710 | 1.2343 | Strong reward baseline. |

The reward-frontier guard follow-up tests whether information value helps only
inside a high-reward candidate frontier:

| Douban reward-frontier guard | seeds | reward | w-RMSE | entropy pct | selected degree | interpretation |
|---|---:|---:|---:|---:|---:|---|
| current netobs reliability top-k | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| weighted guard q99 lambda=0.20 | 3 | 2950.00 | 0.6657 | 0.6467 | 1.0052 | Best guarded variant; mechanism-positive but weaker w-RMSE. |
| netobs guard q99 lambda=0.20 | 3 | 2925.67 | 0.6687 | 0.6447 | 0.9920 | Raises information and lowers degree, but loses accuracy. |
| netobs guard q95 lambda=0.20 | 3 | 2899.67 | 0.6715 | 0.6756 | 0.9803 | Too much guard pressure. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |

The local SGNS Node2Vec follow-up tests the strongest no-new-dependency
embedding path available in the current environment:

| Douban embedding variant | seeds | reward | w-RMSE | entropy pct | selected degree | interpretation |
|---|---:|---:|---:|---:|---:|---|
| spectral netobs reliability | 3 | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Current faithful candidate remains strongest. |
| ELPH-role proxy | 3 | 2987.00 | 0.6633 | 0.5731 | 1.3427 | Best non-spectral local structural proxy. |
| Node2Vec SGNS BFS | 3 | 2978.00 | 0.6643 | 0.5623 | 1.2702 | Positive versus myopic, but weaker than spectral/ELPH-role. |
| Node2Vec SGNS DFS | 3 | 2949.33 | 0.6676 | 0.5949 | 1.2677 | More information-oriented but weaker accuracy. |
| myopic | 3 | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline. |

The latest MovieLens baseline screens improve coverage but do not change the headline:

| MovieLens baseline branch | best new setting | seeds | reward | w-RMSE | interpretation |
|---|---|---:|---:|---:|---|
| MF-centered bandits | MF linear UCB, alpha=0.05 | 3 | 2702.33 | 0.8306 | Fixes old bandit reward collapse but does not beat DU-Net or graph_social. |
| residual MF bandits | residual UCB, weight=0.25 alpha=0.02 | 3 | 2705.33 | 0.8318 | Does not beat MF linear UCB. |
| active-learning baselines | active hybrid, alpha=0.20 | 3 | 267.67 | 1.1637 | Negative; uncertainty-only acquisition is far from the reward frontier. |
| social-rec baselines | social MF hybrid, weight=0.50 | 3 | 2719.67 | 0.8301 | Useful comparator, but still below DU-Net/graph_social on w-RMSE. |

## What Must Change Before Claiming Exact Replication

1. Do not claim exact paper-level comparison against all 16 baselines unless the two unresolved labels are recovered from the original PDF, supplement, or author code.
2. Replace the structural embedding proxy with a validated external sparse Node2Vec or ELPH implementation only if dependencies are explicitly allowed later.
3. Stop local entropy-gate and embedding-only sweeps until the paper estimator or external embedding stack changes.
4. Rerun the final paper-quality matrix only after those mechanism and benchmark upgrades.

Until then, the correct claim is:

> This is a strong, real-data, paper-style reproduction scaffold that recovers the main qualitative direction on MovieLens and Flixster and gives a mechanism-grounded partial recovery on Douban. It covers or proxy-covers the 14 benchmark policies explicitly named in the local paper text, while two paper-count benchmark labels remain unresolved. It is not yet an exact DU-Net replication.
