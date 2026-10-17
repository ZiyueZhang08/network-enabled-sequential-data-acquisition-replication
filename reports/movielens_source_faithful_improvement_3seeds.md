# MovieLens Source-Faithful Improvement Gate

Generated from `experiments/screen_movielens_protocol_sensitivity.py`.

## Protocol

- Split: official MovieLens `u1.base` / `u1.test`.
- Initial size: `1364` ratings, chosen to align with the paper's reported MovieLens initial matrix density of approximately `0.086%` over the user-item matrix. Under the official `u1.base` fold, this corresponds to `1364 / 80000 = 1.705%` of the training ratings.
- Seeds: `0, 1, 2`.
- Cycles / batch: 20 acquisition cycles, batch size 100.
- Comparator: MF-centered Linear UCB proxy with `ucb_alpha=0.02`.

## Three-Seed Result

| variant | reward | RMSE | mild w-RMSE | medium w-RMSE | primary linear-norm w-RMSE | primary gap vs Linear |
|---|---:|---:|---:|---:|---:|---:|
| `graph_late_lambda_half_start16` | 2800.67 | 1.1491 | 1.0723 | 1.0267 | 1.0991 | 0.0075 |
| `graph_late_lambda_half_start16_p001` | 2801.67 | 1.1490 | 1.0727 | 1.0275 | 1.1027 | 0.0039 |
| `graph_late_lambda_half_start16_p000` | 2798.00 | 1.1461 | 1.0709 | 1.0264 | 1.1053 | 0.0013 |
| `graph_late_lambda_half_start16_topm_exp_tau010` | 2777.00 | 1.1475 | 1.0715 | 1.0265 | 1.1022 | 0.0043 |
| `coherence_lite_p000` | 2681.00 | 1.1437 | 1.0714 | 1.0287 | 1.1178 | -0.0113 |
| `mf_linear_ucb_a002` | 2833.67 | 1.1509 | 1.0749 | 1.0299 | 1.1066 | 0.0000 |
| `myopic` | 2823.33 | 1.1502 | 1.0749 | 1.0303 | 1.1099 | n/a |

Gap convention: positive primary gap means Linear UCB has higher error than the DU-Net variant.

## Decision

The source-faithful protocol change is useful, but it is not a promotion over the frozen exploratory best. It improves fidelity and largely removes the conventional-RMSE caveat, while shrinking the primary reward-tail margin:

- `p=0` is the most paper-faithful MovieLens diffusion setting and still keeps the DU-Net direction on the primary proxy, but the margin falls to `0.0013`.
- `p=0.001` is the better fidelity/performance compromise among diffusion controls, with primary gap `0.0039`.
- `top_m_exp_tau010` recovers stochastic batch selection without the full-pool reward collapse seen at `tau=1`; it keeps primary gap `0.0043`, but reward is lower than deterministic top-k.
- `coherence_lite_p000` improves conventional RMSE and mild/medium weighted RMSE versus Linear, but fails the primary reward-tail proxy and loses too much reward, so it should not replace the current entropy backend.
- The incomplete one-seed adaptive rows are diagnostic only and are not promotion evidence.

This should be reported as a two-track result rather than merged into the earlier exploratory result:

- Result A, exploratory best proxy: initial size `800`, primary gap `+0.0207`, `19/20` seed wins, and `79.5%` of the paper visual gap, with a conventional-RMSE caveat.
- Result B, source-faithful improvement gate: initial size `1364`, the `p=0` and `p=0.001` variants keep the direction over three seeds and the RMSE caveat mostly disappears, but the primary gap is small and needs 20-seed confirmation.

The confirmatory run should be restricted to `graph_late_lambda_half_start16_p000`, `graph_late_lambda_half_start16_p001`, and `graph_late_lambda_half_start16` plus the Linear UCB comparator. `coherence_lite`, `top_m_exp`, and adaptive-grid rows remain diagnostics for now.

## Artifacts

- `outputs/tables/movielens_100k_source_faithful_improvement_3seeds_metrics.csv`
- `outputs/tables/movielens_100k_source_faithful_improvement_3seeds_summary.csv`
- `outputs/tables/movielens_100k_source_faithful_improvement_3seeds_gaps.csv`
