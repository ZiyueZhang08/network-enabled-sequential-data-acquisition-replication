# MovieLens Entropy-Rho 20-Seed Confirmation

Generated from `outputs\tables\movielens_100k_entropy_rho010_confirm_20seeds_metrics.csv` and reused the unchanged 20-seed `graph_late_lambda_half_start16` / `mf_linear_ucb_a002` baselines from `outputs\tables\movielens_100k_source_faithful_confirm_20seeds_metrics.csv`.

## Fixed Confirmation Rule

Promote `entropy_rho010` only if it beats the current `graph_late_lambda_half_start16` primary gap, wins the primary metric against Linear in at least `14/20` paired seeds, has non-negative conventional RMSE, mild, and medium gaps versus Linear, and does not lose more than `30` cumulative reward points versus current DU-Net.

## Decision

| variant | primary gap | win vs Linear | current primary gap | delta vs current | win vs current | RMSE gap | mild gap | medium gap | reward delta vs current | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `entropy_rho010` | 0.0095 | 16/20 | 0.0062 | 0.0033 | 14/20 | -0.0018 | 0.0002 | 0.0015 | -4.50 | do not promote; keep as primary-tail diagnostic |

## Interpretation

`entropy_rho010` improves the primary reward-tail gap over current DU-Net, but the fixed confirmation gate requires conventional RMSE not to reverse. If the RMSE gap is negative, this row should be reported as a useful primary-tail/coherence diagnostic rather than promoted as the new main code-improved reproduction.

## Artifacts

- `outputs\tables\movielens_100k_entropy_rho010_confirm_20seeds_decision.csv`
- `outputs\tables\movielens_100k_entropy_rho010_confirm_20seeds_paired_gaps.csv`
- `reports\movielens_entropy_rho010_confirm_20seeds.md`
