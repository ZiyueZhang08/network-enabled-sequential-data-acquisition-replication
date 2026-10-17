# Paper-faithful MovieLens v4 backend screen

This screen changes only the SVD++ backend while keeping the v4 paper-faithful acquisition protocol:
user-feature full DLS, Eq. 3.6-style Schur entropy, full-pool exponential sampling, and MovieLens `p=0`.
The same backend is used for DU-Net and all Linear UCB comparators.

## DU-Net summary

| scenario | seeds | rank | epochs | lr | reg | PDF RMSE | standard RMSE | reward | tail RMSE | local w-RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rank20_ep20_lr0015_reg006 | 20 | 20 | 20 | 0.0150 | 0.0600 | 0.007654 | 1.082405 | 1344.202 | 1.293299 | 1.017481 |
| v4_rank12_ep6_lr0025_reg004 | 3 | 12 | 6 | 0.0250 | 0.0400 | 0.007638 | 1.080180 | 1324.566 | 1.269315 | 0.995992 |
| rank30_ep12_lr0015_reg006 | 3 | 30 | 12 | 0.0150 | 0.0600 | 0.007657 | 1.082806 | 1335.349 | 1.283271 | 1.007119 |
| rank20_ep12_lr002_reg004 | 3 | 20 | 12 | 0.0200 | 0.0400 | 0.007664 | 1.083890 | 1348.697 | 1.303936 | 1.023339 |
| rank20_ep12_lr0015_reg006 | 3 | 20 | 12 | 0.0150 | 0.0600 | 0.007646 | 1.081354 | 1343.313 | 1.285577 | 1.008605 |

## DU-Net minus comparator gaps

Negative RMSE gaps favor DU-Net; positive reward gaps favor DU-Net.

| scenario | comparator | standard RMSE gap | PDF RMSE gap | reward gap | local w-RMSE gap |
|---|---|---:|---:|---:|---:|
| rank20_ep20_lr0015_reg006 | linucb_concat | -0.023984 | -0.000170 | -49.478 | 0.051957 |
| rank20_ep20_lr0015_reg006 | linucb_mf_interaction | -0.005017 | -0.000035 | 137.832 | -0.030782 |
| v4_rank12_ep6_lr0025_reg004 | linucb_concat | -0.015898 | -0.000112 | -55.406 | 0.042650 |
| v4_rank12_ep6_lr0025_reg004 | linucb_mf_interaction | 0.007038 | 0.000050 | 129.586 | -0.070431 |
| rank30_ep12_lr0015_reg006 | linucb_concat | -0.014938 | -0.000106 | -70.214 | 0.059623 |
| rank30_ep12_lr0015_reg006 | linucb_mf_interaction | -0.000108 | -0.000001 | 170.709 | -0.083051 |
| rank20_ep12_lr002_reg004 | linucb_concat | -0.014111 | -0.000100 | -48.060 | 0.059438 |
| rank20_ep12_lr002_reg004 | linucb_mf_interaction | 0.007045 | 0.000050 | 186.076 | -0.056340 |
| rank20_ep12_lr0015_reg006 | linucb_concat | -0.006875 | -0.000049 | -33.203 | 0.043291 |
| rank20_ep12_lr0015_reg006 | linucb_mf_interaction | 0.001577 | 0.000011 | 253.135 | -0.123347 |

## Interim interpretation

- Best DU-Net standard RMSE in this screen: `v4_rank12_ep6_lr0025_reg004` at 1.080180.
- Backend changes are promotion candidates only if the same-backend comparator gaps also stay consistent with the paper direction.
- If stronger backends improve Linear UCB as much as DU-Net, the remaining Figure 4a gap is more likely exact split/source/backend-detail uncertainty than a tunable DU-Net mechanism.
