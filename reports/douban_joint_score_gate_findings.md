# Douban Joint Score Gate Findings

This note records a focused DU-Net acquisition-scoring pilot. The change adds
an optional joint reward-information interaction to the existing linear score:

`z(reward) + lambda * z(entropy)`

becomes, only when explicitly enabled:

`z(reward) + lambda * z(entropy) + weight * max(z(reward), 0) * max(z(entropy), 0)`

The default `score_interaction_weight=0.0` preserves all existing configs.

## Why This Was Tested

The current best Douban homogeneous candidate is still
`network_observational + reliability-DLS + spectral`. Recent embedding and
expected-entropy pilots did not beat it. This pilot checks whether a closer
reward-information acquisition gate can improve selection without changing the
embedding or entropy implementation.

## Douban 3-Seed Result

Config: `configs/douban_svdpp_joint_score_gate_3seeds.json`

| variant | reward | w-RMSE | reward pct | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---:|---|
| spectral netobs linear | 2996.67 | 0.6626 | 0.9996 | 0.5467 | 1.2252 | Current best 3-seed baseline. |
| joint gate 0.05 | 2946.67 | 0.6660 | 0.9992 | 0.6629 | 0.9598 | More information-oriented, but worse w-RMSE. |
| myopic | 2921.33 | 0.6710 | 0.9996 | 0.5406 | 1.2343 | Strong reward baseline. |
| joint gate 0.10 | 2848.33 | 0.6780 | 0.9982 | 0.7352 | 0.7610 | Too much information pressure. |
| joint gate 0.20 | 2721.00 | 0.6945 | 0.9960 | 0.8643 | 0.4025 | Clearly harmful. |
| joint gate 0.40 | 2549.67 | 0.7192 | 0.9930 | 0.9419 | 0.2742 | Strongly harmful. |

## Interpretation

The gate works mechanically: higher weights push the policy toward higher
entropy percentile and lower-degree users. But the prediction outcome worsens
monotonically. Even `0.05` loses to the linear spectral-netobs candidate on both
reward and w-RMSE.

This means the current Douban gap is not solved by simply forcing selected
batches to be jointly high reward and high entropy. The evidence points back to
model-fidelity gaps: exact DU-Net representation learning and the full paper
baseline suite, rather than another lightweight scoring tweak.

## Decision

Do not promote `score_interaction_weight` variants to 20 seeds. Keep the option
as an experimental knob with default `0.0`, because it is useful diagnostically
and does not alter existing configs.

Next recommended branch:

1. Freeze `network_observational + reliability-DLS + spectral` as the current
   homogeneous Douban candidate.
2. Build the missing baseline-coverage matrix across all three datasets, or
   implement a more faithful DU-Net representation/acquisition module before
   further tuning.

Primary artifacts:

- `outputs/tables/douban_svdpp_joint_score_gate_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_joint_score_gate_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_joint_score_gate_3seeds_report.md`
- `reports/douban_svdpp_joint_score_gate_3seeds_mechanism_diagnostics.md`
- `outputs/figures/douban_svdpp_joint_score_gate_3seeds_w_rmse.png`
