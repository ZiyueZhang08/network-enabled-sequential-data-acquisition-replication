# MovieLens Source-Faithful 20-Seed Confirmation

Generated from `outputs\tables\movielens_100k_source_faithful_confirm_20seeds_metrics.csv`.

## Protocol

- Split: official MovieLens `u1.base` / `u1.test`.
- Initial size: `1364` ratings, aligned to the paper's reported MovieLens initial matrix density of approximately `0.086%`; under `u1.base`, this is `1.705%` of the 80,000 training ratings.
- Seeds: target `0..19`.
- Comparator: MF-centered Linear UCB proxy, `ucb_alpha=0.02`.
- Candidate set: `p=0`, `p=0.001`, and `p=0.005` DU-Net-style graph-late variants only.
- Status: `complete`.

## Fixed Promotion Rule

Promotion rule, fixed before reading the 20-seed result: primary gap mean > 0; primary paired-seed wins >= 14/20; conventional RMSE gap >= `-0.001`; mild and medium reward-weighted RMSE gaps >= 0; mean cumulative reward gap >= `-150`.

Hierarchy: if `p=0` passes, it is the main source-faithful result; if `p=0` fails and `p=0.001` passes, `p=0.001` is the main fidelity/performance compromise; if only `p=0.005` passes, it is reported as a network-amplified DU-Net-style proxy, not the most source-faithful setting.

## Decision Table

| variant | primary gap | win rate | RMSE gap | mild gap | medium gap | reward gap | decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `graph_late_lambda_half_start16_p000` | 0.0012 | 10/20 | 0.0009 | 0.0010 | 0.0011 | -13.60 | most faithful diagnostic; gate failed |
| `graph_late_lambda_half_start16_p001` | 0.0019 | 14/20 | -0.0004 | 0.0001 | 0.0004 | -8.60 | promote as fidelity/performance compromise |
| `graph_late_lambda_half_start16` | 0.0062 | 15/20 | -0.0014 | 0.0000 | 0.0009 | -19.25 | not promoted |

Gap convention: positive error-metric gaps mean Linear UCB has higher error than the DU-Net variant; positive reward gap means the DU-Net variant has higher cumulative reward.

## Reporting Structure

- Result A, exploratory best proxy: initial size `800`, primary gap `+0.0207`, `19/20` seed wins, and `79.5%` of the paper visual gap, with a conventional-RMSE caveat.
- Result B, source-faithful improvement gate: initial size `1364`; this confirmation table decides whether the more paper-like MovieLens protocol supports `p=0`, `p=0.001`, or only the network-amplified `p=0.005` proxy.

## Artifacts

- `outputs\tables\movielens_100k_source_faithful_confirm_20seeds_decision.csv`
- `outputs\tables\movielens_100k_source_faithful_confirm_20seeds_paired_seed_gaps.csv`
- `reports\movielens_source_faithful_confirm_20seeds.md`
