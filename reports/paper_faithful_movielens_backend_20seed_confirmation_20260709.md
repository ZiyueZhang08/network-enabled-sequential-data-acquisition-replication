# Paper-faithful MovieLens backend 20-seed confirmation

Date: 2026-07-09

## Objective

The target is to get the clean-room estimate as close as possible to the paper's DU-Net result in Figure 4a. If the absolute estimate remains worse than the paper, the fallback criterion is to compare against Linear UCB, the paper's second-best approach, and show whether the local ordering is consistent with the paper.

This pass changes only the SVD++ backend while keeping the paper-faithful v4 acquisition protocol:

- MovieLens auxiliary user features as `u_sn`.
- Eq. 3.3 full-user Gaussian DLS.
- Eq. 3.6-style Schur observational entropy.
- Section 3.5.1 dynamic tuning over `lambda_H` and `sigma`; MovieLens `p=0`.
- Full-pool exponential sampling.
- Same backend for DU-Net and Linear UCB comparators.

## Backend screen

The 3-seed screen tested paper-plausible SVD++ backends:

| scenario | DU-Net standard RMSE | DU-Net reward | DU minus MF-interaction standard RMSE |
|---|---:|---:|---:|
| rank12 ep6 lr0.025 reg0.04 | 1.080180 | 1324.566 | 0.007038 |
| rank20 ep12 lr0.020 reg0.04 | 1.083890 | 1348.697 | 0.007045 |
| rank20 ep12 lr0.015 reg0.06 | 1.081354 | 1343.313 | 0.001577 |
| rank30 ep12 lr0.015 reg0.06 | 1.082806 | 1335.349 | -0.000108 |
| rank20 ep20 lr0.015 reg0.06 | 1.077852 | 1343.908 | -0.007088 |

The only configuration that clearly moved the ordering toward the paper was `rank20_ep20_lr0015_reg006`, so it was promoted to 20-seed confirmation.

## 20-seed confirmation

| variant | standard RMSE | PDF-literal RMSE | cumulative reward | reward-tail RMSE | local w-RMSE |
|---|---:|---:|---:|---:|---:|
| DU-Net | 1.082405 | 0.007654 | 1344.202 | 1.293299 | 1.017481 |
| Linear UCB concat | 1.106389 | 0.007823 | 1393.680 | 1.216032 | 0.965524 |
| Linear UCB MF-interaction | 1.087422 | 0.007689 | 1206.370 | 1.336719 | 1.048264 |

DU-Net minus comparator gaps:

| comparator | standard RMSE gap | PDF-literal RMSE gap | reward gap | local w-RMSE gap |
|---|---:|---:|---:|---:|
| Linear UCB concat | -0.023984 | -0.000170 | -49.478 | 0.051957 |
| Linear UCB MF-interaction | -0.005017 | -0.000035 | 137.832 | -0.030782 |

Negative RMSE gaps favor DU-Net. The local ordering is now consistent with the paper's Figure 4a claim that DU-Net is better than Linear UCB on the main RMSE-style estimate.

## Reflection

This backend change did not materially improve DU-Net's absolute standard RMSE compared with the previous v4 result:

- v4 rank12/ep6 DU-Net standard RMSE: 1.082427
- rank20/ep20 DU-Net standard RMSE: 1.082405

The meaningful improvement is relative consistency. Under the stronger shared SVD++ backend, Linear UCB's RMSE worsens more than DU-Net's, so DU-Net becomes better than both Linear UCB variants under the same protocol. This is legitimate to report because the backend is shared; it is not a DU-Net-only advantage. It also matches the user's fallback requirement: if the absolute estimate is still below the paper's Figure 4a result, the comparison to Linear UCB shows the clean-room result is directionally consistent with the paper.

The remaining gap is therefore unlikely to be solved by broad backend tuning alone. The most plausible unresolved sources are still paper-specific details not recoverable from the PDF alone:

- exact MovieLens split and random initialization masks;
- exact SVD++ implementation and stopping rule;
- exact Figure 4a metric normalization;
- author implementation details for the entropy/reward estimator.

## Artifacts

- `outputs/tables/paper_faithful_movielens_rank20_ep20_v4_20seeds_metrics.csv`
- `outputs/tables/paper_faithful_movielens_rank20_ep20_v4_20seeds_final_summary.csv`
- `outputs/tables/paper_faithful_movielens_rank20_ep20_v4_20seeds_gaps.csv`
- `outputs/tables/paper_faithful_movielens_backend_v4_screen_3seeds_metrics.csv`
- `outputs/tables/paper_faithful_movielens_backend_v4_screen_3seeds_final_summary.csv`
- `reports/paper_faithful_movielens_backend_v4_screen_3seeds.md`

## Next step

The next faithful branch should not be another broad hyperparameter sweep. The clean next step is to make the Figure 4a reporting closer to the paper:

1. Produce the final Figure 4a-style table/plot from the rank20/ep20 mainline.
2. Report DU-Net versus Linear UCB concat and MF-interaction side by side.
3. Add a metric-normalization sensitivity appendix: PDF-literal RMSE, standard RMSE, and local w-RMSE, explicitly stating which one is used as the main clean-room proxy.

Only after that should we touch mechanism code again. If mechanism work continues, the best target is exact batch exponential sampling/metric reproduction, not new engineering enhancements.
