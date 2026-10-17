# MovieLens Code-Improvement Gate

Generated from `outputs\tables\movielens_100k_code_improvement_gate_3seeds_summary.csv`.

## Protocol

- Split: official MovieLens `u1.base` / `u1.test`.
- Initial size: `1364`, aligned with the paper's reported MovieLens initial matrix density.
- Seeds: `0, 1, 2`.
- Current comparator DU-Net-style row: `graph_late_lambda_half_start16`.
- Linear comparator: `mf_linear_ucb_a002`.

## Fixed Gate

A variant passes only if its primary Linear-minus-DU gap is greater than the current `graph_late_lambda_half_start16` gap (`0.0075`), conventional RMSE gap is non-negative, mild and medium reward-weighted gaps are non-negative, and cumulative reward is not more than `30` below current DU-Net.

## Decision Table

| variant | primary gap | delta vs current | RMSE gap | mild gap | medium gap | reward delta vs current | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `backend_rank20_ep12` | -0.0037 | -0.0113 | 0.0053 | 0.0033 | 0.0020 | -35.33 | do not promote |
| `backend_rank30_ep15` | 0.0222 | 0.0146 | -0.0011 | 0.0033 | 0.0060 | 0.00 | do not promote |
| `dynamic_tune_lam_sig` | -0.0158 | -0.0234 | 0.0140 | 0.0079 | 0.0042 | -180.33 | do not promote |
| `residual_dls_alpha_schedule` | 0.0008 | -0.0067 | 0.0124 | 0.0096 | 0.0079 | -141.67 | do not promote |
| `entropy_rho010` | 0.0177 | 0.0102 | 0.0001 | 0.0031 | 0.0050 | 9.00 | promote to 20-seed confirmation |
| `divcap_user2_item5` | 0.0007 | -0.0069 | 0.0014 | 0.0011 | 0.0010 | -16.00 | do not promote |

## Verdict

Promote `entropy_rho010` to 20-seed confirmation.

Gap convention: positive error-metric gaps mean Linear UCB has higher error than the variant. Positive reward delta means the variant has higher cumulative reward than current DU-Net.

## Artifacts

- `outputs\tables\movielens_100k_code_improvement_gate_3seeds_decision.csv`
- `reports\movielens_code_improvement_gate_3seeds.md`
