# inverse_reconstruction_v2 Linear UCB protocol audit

Scope: confirm the most relevant clean-room Linear UCB comparator candidates on the current u1 fallback mask.

- Confirm table: `outputs/tables/inverse_reconstruction_v2_linucb_candidate_confirm.csv`
- Custom metrics: `outputs/tables/inverse_reconstruction_v2_linucb_candidate_confirm_metrics.csv`
- Paper Linear UCB reconstructed candidate: `no`

## Candidate Ranking

| rank | candidate | policy | selection_source | feature_map | ucb_alpha | ucb_regularization | seed_count | cycle_count | figure4a_linear_curve_error | figure4a_linear_endpoint_error | figure4b_linear_reward_error | figure4b_linear_reward_error_norm | figure4b_linear_endpoint_error | figure4b_linear_endpoint_error_norm | linear_ucb_joint_score | protocol_status | paper_linear_ucb_reconstructed_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | linucb_concat | linucb_concat | v2_20seed_base | concat |  |  | 20 | 21 | 0.740031 | 0.707611 | 61.5131 | 0.044254 | 3.68025 | 0.00264766 | 1.49454 | clean_room_proxy_unverified_protocol | no |
| 2 | linucb_mf_alpha_0p1_reg_10 | linucb_mf_interaction | alpha_reg_sensitivity_v2 | mf_interaction | 0.1 | 10 | 20 | 21 | 0.724449 | 0.683684 | 35.1791 | 0.0253087 | 128.976 | 0.0927883 | 1.52623 | clean_room_proxy_unverified_protocol | no |
| 3 | linucb_bias_interaction | linucb_bias_interaction | v2_20seed_base | bias_interaction |  |  | 20 | 21 | 0.725023 | 0.683668 | 35.6704 | 0.0256621 | 131.646 | 0.0947093 | 1.52906 | clean_room_proxy_unverified_protocol | no |
| 4 | linucb_mf_alpha_0p02_reg_1 | linucb_mf_interaction | alpha_reg_sensitivity_v2 | mf_interaction | 0.02 | 1 | 20 | 21 | 0.72283 | 0.680148 | 48.1916 | 0.0346702 | 159.851 | 0.115001 | 1.55265 | clean_room_proxy_unverified_protocol | no |
| 5 | linucb_mf_interaction | linucb_mf_interaction | v2_20seed_base | mf_interaction |  |  | 20 | 21 | 0.748427 | 0.726578 | 161.061 | 0.115871 | 183.63 | 0.132108 | 1.72298 | clean_room_proxy_unverified_protocol | no |

## Claim Boundary

The paper does not expose Linear UCB feature construction, alpha, regularization, reward transform, or update protocol. A close empirical row remains an unverified clean-room protocol candidate, not an exact comparator.
