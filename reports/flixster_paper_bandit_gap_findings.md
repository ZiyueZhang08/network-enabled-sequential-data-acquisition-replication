# Flixster Paper Bandit Gap Findings

This note records the follow-up after the OFUL and MF-OFUL repair screens. The
goal was to fill the paper-specific Logistic UCB/Thompson and finite-arm
benchmark labels without pretending these local rows are exact author-code
implementations.

## What Changed

- Added proxy policies for `logistic_ucb`, `logistic_thompson_sampling`,
  `finite_ucb`, `epsilon_greedy`, `adaptive_greedy`, and
  `explore_then_exploit`.
- Kept all new paper-specific rows labeled as `proxy` in
  `src/baselines.py`.
- Ran two Flixster 3-seed screens:
  - `configs/flixster_svdpp_paper_bandit_gap_3seeds.json`
  - `configs/flixster_svdpp_logistic_bandit_scale_3seeds.json`

## Key Results

| branch | variant | reward | w-RMSE | decision |
|---|---|---:|---:|---|
| Frontier anchor | `du_net` | 2988.50 | 0.6254 | Best w-RMSE anchor. |
| Frontier anchor | `graph_social` | 2992.17 | 0.6274 | Best reward anchor. |
| Frontier anchor | `myopic` | 2885.83 | 0.6279 | Strong non-network exploitation baseline. |
| Prior repair | `mf_oful_b0005` | 2833.00 | 0.6325 | Still below myopic/frontier. |
| Finite-arm proxy | `epsilon_greedy_e010` | 2782.83 | 0.6335 | Best finite-arm stress row in the first screen. |
| Finite-arm proxy | `adaptive_greedy_e010` | 2806.17 | 0.6351 | Slightly higher reward than epsilon greedy, weaker w-RMSE. |
| Finite-arm proxy | `finite_ucb_a005` | 2722.00 | 0.6408 | Count-UCB exploration is useful but not frontier-level. |
| Finite-arm proxy | `explore_then_exploit_q025` | 2675.33 | 0.6436 | Initial random exploration costs too much. |
| Logistic scale zoom | `logistic_ucb_a0005` | 2868.00 | 0.6295 | Best logistic UCB; close to myopic but not better. |
| Logistic scale zoom | `logistic_ts_n002` | 2845.17 | 0.6313 | Best logistic Thompson; close to myopic but not better. |
| Logistic uncalibrated | `logistic_ucb_a005` | 1949.17 | 0.7880 | Probability-scale mean cannot absorb large UCB bonuses. |
| Logistic uncalibrated | `logistic_ts_n008` | 1755.67 | 0.8382 | Large TS noise is destructive. |

## Interpretation

The catalog gap is materially smaller: the 14 explicitly named paper benchmark
slots now have implemented or proxy-covered local rows. The remaining benchmark
count gap is the two unresolved slots implied by the paper's 16-benchmark claim.

The result gap is still not closed. Finite-arm rows are useful stress baselines,
but the best local rows remain below DU-Net, graph_social, and myopic on
Flixster. Logistic rows only become reasonable when the exploration scale is
shrunk to the probability scale; at that point they behave like lightly
perturbed myopic acquisition rather than a strong paper comparator.

## Decision

Do not promote these rows to a new 20-seed paper-quality run yet. Keep them as
3-seed proxy evidence in the benchmark audit. The next high-value step is to
recover the two unresolved paper benchmark names and compare the local bandit
definitions against the author implementation or supplement.
