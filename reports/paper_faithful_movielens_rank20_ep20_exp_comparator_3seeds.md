# Paper-faithful MovieLens v4 backend screen

This screen changes only the SVD++ backend while keeping the v4 paper-faithful acquisition protocol:
user-feature full DLS, Eq. 3.6-style Schur entropy, full-pool exponential sampling, and MovieLens `p=0`.
The same backend is used for DU-Net and all Linear UCB comparators.

## DU-Net summary

| scenario | seeds | rank | epochs | lr | reg | PDF RMSE | standard RMSE | reward | tail RMSE | local w-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank20_ep20_lr0015_reg006 | 20 | 20 | 20 | 0.0150 | 0.0600 | 0.007654 | 1.082405 | 1344.202 | 1.293299 | 1.017481 |

## DU-Net minus comparator gaps

Negative RMSE gaps favor DU-Net; positive reward gaps favor DU-Net.

| scenario | comparator | standard RMSE gap | PDF RMSE gap | reward gap | local w-RMSE gap |
|---|---|---:|---:|---:|---:|
| rank20_ep20_lr0015_reg006 | linucb_concat | 0.010612 | 0.000075 | 117.893 | -0.025203 |
| rank20_ep20_lr0015_reg006 | linucb_mf_interaction | -0.001161 | -0.000008 | 63.828 | -0.000425 |

## Interim interpretation

- Best DU-Net standard RMSE in this screen: `rank20_ep20_lr0015_reg006` at 1.082405.
- Backend changes are promotion candidates only if the same-backend comparator gaps also stay consistent with the paper direction.
- If stronger backends improve Linear UCB as much as DU-Net, the remaining Figure 4a gap is more likely exact split/source/backend-detail uncertainty than a tunable DU-Net mechanism.
