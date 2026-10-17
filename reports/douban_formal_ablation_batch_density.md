# Douban Formal Ablation: DLS, Batch Composition, and Graph Density

This note freezes the current Douban mechanism evidence after the no-DLS / reliability-DLS confirmation, batch-composition pilot, and graph-density sensitivity pilot.

## Status

The current Douban result should be read as a mechanism diagnosis rather than a failed tuning branch:

- Full DLS is not yet a stable win on Douban.
- No-DLS and reliability-DLS are now formal ablation evidence.
- The sparse Douban graph appears to make full smoothing noisy; reliability shrinkage reduces that damage.
- Batch composition is the first mechanism change that makes DU-Net substantially less myopic without scanning entropy weights.

## Confirmed 20-Seed Ablation

| variant | seeds | reward | w-RMSE | interpretation |
|---|---:|---:|---:|---|
| `du_net_no_dls` | 20 | 2863.05 | 0.6777 | Best confirmed 20-seed Douban setting; removing DLS avoids sparse-graph smoothing noise. |
| `du_net_shrinkage_dls` reliability | 20 | 2861.65 | 0.6778 | Nearly ties no-DLS while lowering selected-user degree from 1.8632 to 1.2997. |
| weighted logdet + reliability DLS | 20 | 2860.05 | 0.6780 | Raises information-oriented selection but does not yet improve final w-RMSE. |
| current DU-Net | 20 | 2838.65 | 0.6801 | Still too close to myopic selection. |
| myopic | 20 | 2844.60 | 0.6801 | Strong reward baseline; Douban remains the hardest dataset. |

Primary artifacts:

- `outputs/tables/douban_svdpp_reliability_dls_20seeds_final_summary.csv`
- `reports/douban_svdpp_reliability_dls_20seeds_mechanism_diagnostics.md`

## Batch Composition Pilot and 20-Seed Confirmation

Scalar z-score mixing was replaced by reward-frontier batch composition:

- 70% of each batch is selected by reward.
- 30% is selected by information value inside the reward frontier.
- The reward frontier is the top `5 x batch_size` candidates by reward score.

This avoids the failure mode of globally selecting entropy-heavy points that have poor acquisition reward.

The first 3-seed pilot was positive, but the 20-seed confirmation did not hold. This is important negative evidence: reward-frontier batch composition successfully changes selection behavior, but the current information proxy still does not improve Douban implementation error at scale.

| variant | seeds | reward | w-RMSE | reward pct | entropy pct | reading |
|---|---:|---:|---:|---:|---:|---|
| reliability-DLS frontier batch | 3 | 3012.00 | 0.6586 | 0.9991 | 0.6898 | Best current Douban pilot; information share rises without losing reward. |
| logdet + reliability frontier batch | 3 | 2929.00 | 0.6677 | 0.9991 | 0.7412 | Stronger information orientation, but lower reward than reliability batch. |
| myopic | 3 | 2921.33 | 0.6710 | 0.9996 | 0.5406 | Strong baseline but clearly less information-oriented. |
| no-DLS frontier batch | 3 | 2918.00 | 0.6713 | 0.9991 | 0.6822 | Batch composition helps exploration, but no-DLS no longer dominates. |

| variant | seeds | reward | w-RMSE | reward pct | entropy pct | reading |
|---|---:|---:|---:|---:|---:|---|
| myopic | 20 | 2844.60 | 0.6801 | 0.9996 | 0.5435 | Still the best in this 20-seed confirmation. |
| no-DLS frontier batch | 20 | 2818.00 | 0.6832 | 0.9991 | 0.6864 | Less myopic but worse final w-RMSE. |
| reliability-DLS frontier batch | 20 | 2813.95 | 0.6833 | 0.9991 | 0.6889 | Reduces centrality bias, but the pilot advantage does not survive 20 seeds. |
| logdet + reliability frontier batch | 20 | 2804.95 | 0.6843 | 0.9991 | 0.7258 | Most information-oriented, but weakest final w-RMSE among the batch variants. |

Primary artifacts:

- `configs/douban_svdpp_batch_composition_3seeds.json`
- `configs/douban_svdpp_batch_composition_20seeds.json`
- `outputs/tables/douban_svdpp_batch_composition_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_batch_composition_20seeds_final_summary.csv`
- `outputs/figures/douban_svdpp_batch_composition_3seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_batch_composition_20seeds_w_rmse.png`
- `reports/douban_svdpp_batch_composition_3seeds_report.md`
- `reports/douban_svdpp_batch_composition_20seeds_mechanism_diagnostics.md`

## Graph-Density Sensitivity Pilot

The graph-density experiment perturbs only the user graph while keeping the same Douban data split and acquisition setup.

| variant | seeds | graph edges | reward | w-RMSE | selected degree | interpretation |
|---|---:|---:|---:|---:|---:|---|
| original reliability-DLS | 3 | 1344 | 2977.33 | 0.6642 | 1.3682 | Best density-sensitivity setting; shrinkage stabilizes sparse DLS. |
| original no-DLS | 3 | 1344 | 2979.00 | 0.6643 | 1.6610 | Confirms no-DLS is a strong sparse-graph baseline. |
| densify50 reliability-DLS | 3 | 2016 | 2938.00 | 0.6696 | 2.4312 | Densification helps reliability-DLS less than expected and increases centrality. |
| densify50 full DLS | 3 | 2016 | 2884.67 | 0.6742 | 3.3412 | Densification amplifies high-degree selection under full DLS. |
| drop50 reliability-DLS | 3 | 672 | 2894.00 | 0.6748 | 0.5388 | Too few edges weaken DLS signal. |
| original full DLS | 3 | 1344 | 2871.67 | 0.6753 | 1.6682 | Full DLS remains weaker than reliability-DLS. |
| rewire50 reliability-DLS | 3 | 1344 | 2841.67 | 0.6811 | 1.0030 | Rewiring damages structural signal even with shrinkage. |
| rewire50 full DLS | 3 | 1344 | 2798.67 | 0.6851 | 1.1495 | Strong negative evidence: DLS needs meaningful network structure. |

Primary artifacts:

- `configs/douban_svdpp_graph_density_sensitivity_3seeds.json`
- `outputs/tables/douban_svdpp_graph_density_sensitivity_3seeds_final_summary.csv`
- `outputs/figures/douban_svdpp_graph_density_sensitivity_3seeds_w_rmse.png`
- `reports/douban_svdpp_graph_density_sensitivity_3seeds_report.md`

## Interpretation

The current Douban story is now sharper:

1. Full DLS is not wrong in principle, but on Douban the current proxy DLS is too sensitive to sparse/noisy graph structure.
2. Reliability-DLS is the correct ablation direction: it lowers centrality bias and almost matches no-DLS in the 20-seed confirmation.
3. Batch composition is more promising than scalar entropy mixing as a mechanism, but the 20-seed result shows that the current information proxy is not yet good enough.
4. Rewiring harms both full and reliability DLS, which supports the interpretation that the user graph contains signal, but the current smoothing mechanism does not extract it reliably enough.

## Network-Observational Update

The next mechanism layer, `network_observational` entropy with spectral embedding and reliability-DLS, has now passed a narrow 20-seed confirmation:

| variant | seeds | reward | w-RMSE | entropy pct | interpretation |
|---|---:|---:|---:|---:|---|
| netobs reliability spectral | 20 | 2861.80 | 0.6778 | 0.5672 | Modestly beats myopic while keeping network-aware entropy and reliability-DLS. |
| myopic | 20 | 2844.60 | 0.6801 | 0.5435 | Strong baseline. |

This does not erase the earlier no-DLS finding: no-DLS still has the best confirmed w-RMSE by a very small margin. The important change is that the most faithful current mechanism is no longer negative on Douban. It should be carried into the next formal 20-seed Douban comparison.

## Formal 20-Seed Mechanism Matrix

The formal matrix was recovered by synthesizing two matching 20-seed runs after the full 6-variant unified rerun exceeded the local command timeout. The merged runs use the same Douban predefined split, seeds, SVD++ settings, and acquisition baseline.

| variant | reward | w-RMSE | entropy pct | selected degree | interpretation |
|---|---:|---:|---:|---:|---|
| no-DLS | 2863.05 | 0.6777 | 0.5689 | 1.8632 | Best numerical w-RMSE; confirms DLS noise is the key Douban issue. |
| reliability-DLS | 2861.65 | 0.6778 | 0.5720 | 1.2997 | Nearly ties no-DLS with much lower selected-user centrality. |
| netobs reliability spectral | 2861.80 | 0.6778 | 0.5672 | 1.2373 | Most faithful positive candidate: network-observational entropy plus reliability-DLS. |
| weighted logdet reliability | 2860.05 | 0.6780 | 0.6073 | 1.2590 | Moves selection most toward information, but does not win on w-RMSE. |
| current DU-Net | 2838.65 | 0.6801 | 0.5385 | 1.8359 | Over-smoothed and centrality-biased. |
| myopic | 2844.60 | 0.6801 | 0.5435 | 1.2465 | Strong baseline. |

Primary artifacts:

- `configs/douban_svdpp_formal_mechanism_20seeds.json`
- `outputs/tables/douban_svdpp_formal_mechanism_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_formal_mechanism_20seeds_mechanism_summary.csv`
- `reports/douban_svdpp_formal_mechanism_20seeds_mechanism_diagnostics.md`

## NetObs 20-Seed Extension

The next extension adds three formal 20-seed checks:

| check | best variant | reward | w-RMSE | interpretation |
|---|---|---:|---:|---|
| heterogeneous response alpha=0.02 | response-aware netobs | 2360.35 | 0.6934 | Supersedes reliability-DLS after adding response-aware entropy. |
| DLS/network ablation | no-DLS | 2863.05 | 0.6777 | No-DLS remains the numerical sparse-graph fallback. |
| baseline matrix | netobs reliability spectral | 2861.80 | 0.6778 | Netobs reliability-DLS is strongest among the current baseline implementations. |

See `reports/douban_netobs_20seed_extension.md` for the detailed tables.

## Next Step

The next paper-quality Douban candidate should not be another scalar lambda or batch-share sweep. The 20-seed confirmations say that no-DLS/reliability-DLS stabilize the sparse graph and that network-observational entropy can now beat myopic modestly.

The next implementation layer should therefore be:

1. improve sparse Node2Vec / stronger ELPH-style structural embedding;
2. run heterogeneous response and network/DLS ablations for the winning Douban candidate;
3. then do a paper-quality rerun with the final Douban candidate and the full baseline matrix.
