# paper_exact_attempt_movielens_v2 Tuning Frequency/Objective Audit

This Phase 5 artifact audits the tuning protocol as an exact-replication gate. It does not promote a new branch by performance.

## Current v2 Protocol

- Retuning frequency: every `5` cycles
- Inner objective: `standard_mask_rmse`
- Lambda grid: `[0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]`
- Sigma grid: `[0.1, 0.2, 0.4, 0.8, 1.6]`
- Diffusion p grid: `[0.0]`

## Audit Table

| audit_item | current_protocol | paper_required_for_exact | status | evidence | exactness_risk |
| --- | --- | --- | --- | --- | --- |
| retuning_frequency | every 5 cycles | author-equivalent retuning cadence for lambda_H, sigma_t, and p_t | source_blocked_not_closed | prior 5-seed every-cycle vs every-5 diagnostic: standard_mask_rmse_delta=0.00056484; w_rmse_delta=-0.00854367; reward_tail_proxy_rmse_delta=-0.0129611; cumulative_reward_delta=-3.32959; source=outputs/tables/paper_faithful_tune_every_cycle_5seed_diagnostic_20260709.csv | Paper text says hyperparameters are tuned during acquisition, but does not publish a cadence sufficient to choose every-cycle vs every-5-cycle exactly. |
| inner_objective | standard_mask_rmse | author-equivalent inner train/acquisition/test objective and held-out split | source_blocked_not_closed | mainline seed_count=5; standard_mask_rmse=1.069519; w_rmse=1.026202; reward_tail_proxy_rmse=1.304808; cumulative_reward=1241.076444; vs linucb_concat: standard_gap_linear_minus_du=0.032862, reward_gap_du_minus_linear=-148.336, reward_tail_gap_linear_minus_du=-0.0903332; vs linucb_mf_interaction: standard_gap_linear_minus_du=0.0188889, reward_gap_du_minus_linear=29.9259, reward_tail_gap_linear_minus_du=0.028623; vs linucb_bias_interaction: standard_gap_linear_minus_du=0.0583272, reward_gap_du_minus_linear=-276.25, reward_tail_gap_linear_minus_du=-0.123278 | The current objective optimizes Figure-4a-like RMSE behavior, while Figure 4b reward can move differently under the same branch. |
| lambda_sigma_p_grid | lambda=0.0 0.1 1.0 3.0 10.0 30.0 100.0 300.0; sigma=0.1 0.2 0.4 0.8 1.6; p=0.0 | author grid or continuous optimizer, tie-breaks, and any staged schedule | source_blocked_not_closed | v2 config grid is explicit and reproducible, but not source-verified. | Grid endpoints and tie-breaks can change selected batches even when the acquisition formula is otherwise close. |
| metric_gate_interaction | tuning evidence must be judged under metric v4 reconstruction | same metric definition used in paper Figure 4a | blocked_by_metric_gate | best_metric=reward_tail_proxy; comparator=linucb_mf_interaction; row_level_seed_count=5; scale_distance=0.267582; gap_distance=0.100883; paper_wrmse_reconstructed=no | No tuning protocol can be called exact while the target w-RMSE formula and y-axis scale remain unreconstructed. |
| joint_figure4_objective | standard_mask_rmse inner objective with reward tracked as an outcome | single same-config protocol matching Figure 4a and Figure 4b | not_jointly_closed | mainline seed_count=5; standard_mask_rmse=1.069519; w_rmse=1.026202; reward_tail_proxy_rmse=1.304808; cumulative_reward=1241.076444; vs linucb_concat: standard_gap_linear_minus_du=0.032862, reward_gap_du_minus_linear=-148.336, reward_tail_gap_linear_minus_du=-0.0903332; vs linucb_mf_interaction: standard_gap_linear_minus_du=0.0188889, reward_gap_du_minus_linear=29.9259, reward_tail_gap_linear_minus_du=0.028623; vs linucb_bias_interaction: standard_gap_linear_minus_du=0.0583272, reward_gap_du_minus_linear=-276.25, reward_tail_gap_linear_minus_du=-0.123278 | A branch that improves RMSE can reduce immediate reward, so Figure 4b cannot be inferred from Figure 4a tuning success. |

## Current Same-Config Outcome Snapshot

| variant | seed_count | standard_mask_rmse | w_rmse | reward_tail_proxy_rmse | cumulative_reward | centered_cumulative_reward |
| --- | --- | --- | --- | --- | --- | --- |
| linucb_bias_interaction | 5 | 1.12785 | 0.941443 | 1.18153 | 1517.33 | 2600.6 |
| linucb_concat | 5 | 1.10238 | 0.963571 | 1.21447 | 1389.41 | 1927.2 |
| linucb_mf_interaction | 5 | 1.08841 | 1.04539 | 1.33343 | 1211.15 | 1027.8 |
| schur_loglambda_tuned_full_pool_softmax_tau1_literal | 5 | 1.06952 | 1.0262 | 1.30481 | 1241.08 | 1165.2 |
| lambda0p3_diagnostic | 1 | 1.06963 | 1.05098 | 1.34016 | 1217.36 | 1041 |
| schur_loglambda_tuned_full_pool_softmax_tau0p3 | 1 | 1.0667 | 1.07211 | 1.36863 | 1209.5 | 1001 |
| schur_tuned_topk | 1 | 1.09789 | 1.0278 | 1.30814 | 1320.29 | 1584 |

## Decision

- Source-blocked tuning items: `3`
- The v2 tuning protocol is reproducible and fixed, but it is not author-source exact.
- Phase 5 remains open until the author-equivalent retuning cadence, inner objective, held-out construction, grid/optimizer, and tie-break rules are recovered.
- Because metric v4 still fails, running broader local tuning would not close exact replication; it would only create additional proxy branches.
