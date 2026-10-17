# Figure 4a Selection Diagnostic: 3 Seeds

Generated after the paper/code alignment audit.

## Question

Does switching the current MovieLens Figure 4a configuration from deterministic
top-k selection to the paper's exponential-weighted batch sampling close the
replication gap?

## Paper-Grounded Trigger

- The paper states that DU-Net z-normalizes reward and information value.
- The paper then samples a batch of 100 from the acquisition set using
  exponential weighting.
- The current 20-seed MovieLens Figure 4a configuration used deterministic
  `selection: "topk"`.
- The code already supports `selection: "exp"` with a temperature parameter, so
  this is a configuration-level diagnostic rather than an algorithm rewrite.

## Command

```powershell
python experiments\run_policy_variants.py --config configs\movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds.json
```

## Artifacts

- Config: `configs/movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds.json`
- Metrics: `outputs/tables/movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds_metrics.csv`
- Summary: `outputs/tables/movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds_final_summary.csv`
- Auto report: `reports/movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds_report.md`
- Figures:
  - `outputs/figures/movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds_w_rmse.png`
  - `outputs/figures/movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds_cumulative_reward.png`

## Verification

```powershell
python -m pytest tests\test_core.py -q
```

Result: 15 passed.

## Final-Cycle Results

| variant | cumulative reward | conventional RMSE | local reward-weighted RMSE | sigmoid reward-weighted RMSE | reward percentile | entropy percentile |
|---|---:|---:|---:|---:|---:|---:|
| du_net_topk_current | 2717.3333 | 1.1666 | 0.8229 | 0.8987 | 0.9812 | 0.8582 |
| mf_linear_ucb_a002_topk | 2699.0000 | 1.1597 | 0.8349 | 0.8989 | n/a | n/a |
| myopic_topk | 2704.0000 | 1.1617 | 0.8354 | 0.9000 | 0.9994 | 0.5641 |
| du_net_exp_tau05 | 2321.3333 | 1.1174 | 0.8870 | 0.8966 | 0.9313 | 0.7011 |
| du_net_dynamic_sigma_exp_tau1 | 1783.6667 | 1.0743 | 0.9601 | 0.9089 | 0.7707 | 0.5909 |
| du_net_exp_tau1 | 1761.3333 | 1.0754 | 0.9640 | 0.9111 | 0.7717 | 0.5916 |
| myopic_exp_tau1 | 1312.3333 | 1.0536 | 1.0044 | 0.9245 | 0.5919 | 0.4980 |
| mf_linear_ucb_a002_exp_tau1 | 1214.3333 | 1.0492 | 1.0213 | 0.9321 | n/a | n/a |

## Key Gaps

Positive error gaps use `Linear UCB - DU-Net`; positive means lower error for
DU-Net.

| comparison | conventional RMSE gap | local reward-weighted RMSE gap | sigmoid reward-weighted RMSE gap | DU reward advantage |
|---|---:|---:|---:|---:|
| topk Linear-DU | -0.0068 | 0.0120 | 0.0002 | 18.3333 |
| exp tau1 Linear-DU | -0.0262 | 0.0573 | 0.0210 | 547.0000 |
| du tau0.5 vs topk Linear | 0.0423 | -0.0521 | 0.0023 | -377.6667 |

## Interpretation

1. The top-k versus exponential-sampling mismatch is real and source-grounded,
   but switching to exponential sampling does not move the local reward-weighted
   Figure 4a proxy closer to the paper. It moves the policy away from
   high-reward acquisitions.

2. Exponential sampling strongly improves conventional RMSE because it diversifies
   the selected entries. However, it substantially hurts cumulative reward and
   the local reward-weighted RMSE proxy. This reproduces the already-observed
   metric tension: DU-Net looks best in high-reward/reward-weighted regions, while
   broader sampling can improve global RMSE.

3. The source-faithful `exp(tau=1)` run should not be promoted directly to a
   20-seed Figure 4a rerun. It is fidelity-relevant, but the 3-seed result says it
   is not a standalone performance fix for the user's Figure 4a target.

4. The next highest-value work is no longer "try more selection variants." The
   remaining large gap is more likely in the exact Figure 4a w-RMSE definition and
   the observational entropy implementation. The current main configuration uses
   a hybrid/subspace entropy proxy, while the paper emphasizes observational
   entropy through DLS and per-step tuning of lambda, sigma, and diffusion.

## Next Step

Run a 3-seed observational-entropy diagnostic on MovieLens with the same SVD++
backend and comparator:

- current `du_net` with `hybrid` entropy and `topk`
- `du_net` with `observational_logdet` entropy and `topk`
- `du_net` with `observational_logdet` entropy and `exp(tau=0.5)`
- `du_net_dynamic_sigma` with `observational_logdet`
- `mf_linear_ucb_a002_topk`

Promotion criterion: only promote to 20 seeds if the observational entropy
variant improves the local reward-weighted metric without worsening conventional
RMSE relative to current DU-Net, or if it gives a clearer DU-Net versus Linear UCB
advantage under the metric family that best matches the paper.
