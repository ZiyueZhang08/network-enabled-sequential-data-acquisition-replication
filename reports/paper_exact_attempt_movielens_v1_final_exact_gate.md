# Paper Exact Attempt MovieLens v1: Final Gate

Status: **strong clean-room recovery; exact replication blocked**.

This table uses strict pass/fail semantics. Partial evidence is reported as `fail` until the gate's exact condition is satisfied.

- Passed gates: 2
- Failed gates: 7

| Gate | pass/fail | evidence |
| --- | --- | --- |
| Metric exactness | fail | v3 found no paper_wrmse_reconstructed metric; best=sqrt_mean_squared_weighted_max_rating_minus_3_eps_weighted_rmse scale_distance=0.2499914195776343 gap_distance=0.1834555770101863; schur_loglambda_tuned_full_pool_softmax_tau1_literal:1/20 final_cycle_only; linucb_concat:1/20 final_cycle_only; linucb_mf_interaction:1/20 final_cycle_only; schur_loglambda_tuned_full_pool_softmax_tau0p3:1/20 final_cycle_only; schur_tuned_topk:1/20 final_cycle_only; lambda0p3_diagnostic:1/20 final_cycle_only |
| Row-level completeness | fail | schur_loglambda_tuned_full_pool_softmax_tau1_literal:1/20 final_cycle_only; linucb_concat:1/20 final_cycle_only; linucb_mf_interaction:1/20 final_cycle_only; schur_loglambda_tuned_full_pool_softmax_tau0p3:1/20 final_cycle_only; schur_tuned_topk:1/20 final_cycle_only; lambda0p3_diagnostic:1/20 final_cycle_only |
| Split exactness | fail | exact public mask rows=0; audit rows=6; current config uses u1 fallback; public search did not recover author MovieLens initial/candidate/test masks. |
| Linear UCB exactness | fail | external installed candidates=0; local proxy rows=3; no unique paper_linear_ucb_reconstructed comparator. |
| DU-Net Eq.3.6 | pass | explicit-kernel Schur toy equivalence report shows max abs diff 2.220e-16; noise/backend caveats remain separate. |
| DLS/ELPH | fail | MovieLens exact-attempt config uses GCMC/user metadata features; no recovered author ELPH backend is installed or validated. |
| Selection | pass | tau1 final standard RMSE=1.070726; existing 20-seed gate: vs concat 20/20, vs mf_interaction 19/20; exact-attempt row-level replay currently seed0 only |
| Figure 4a | fail | standard_mask_rmse direction=yes; w_rmse direction=no; y-axis metric exactness fails. |
| Figure 4b | fail | ranking direction=yes; gap_abs_error=82.294 vs visual target |

## Public Source Notes

- MovieLens 100K official folds are useful sensitivity inputs, but are not the recovered DU-Net Figure 4 masks.
- Monti/RMGCNN and GCMC papers establish related graph matrix-completion data conventions, but the exact MovieLens sequential acquisition mask remains unrecovered.

## Decision

The current exact attempt cannot be called near-perfect. The defensible claim remains: strong clean-room recovery, exact replication blocked by unavailable or internally inconsistent paper details.
