# Douban Network Entropy and Embedding Findings

This note records the first mechanism pass after the 20-seed negative confirmation for reward-frontier batch composition.

## Implementation Added

- `graph_embedding_method=sparse_node2vec`: sparse random-walk co-occurrence, PPMI, and sparse SVD.
- `graph_embedding_method=elph_struct`: stronger structural features plus sparse random-walk context.
- `graph_embedding_method=sparse_node2vec_biased`: second-order Node2Vec-style biased sparse walks.
- `graph_embedding_method=elph_role`: role-oriented ELPH-style structural features plus biased-walk context.
- `entropy_mode=network_observational`: set-level log-det gain for a candidate plus same-item social-neighbor candidates.

These are still lightweight approximations, but they are closer to the paper mechanism than the earlier spectral / hybrid proxy.

## Douban 3-Seed Result

| variant | reward | w-RMSE | reward pct | entropy pct | reading |
|---|---:|---:|---:|---:|---|
| reliability batch hybrid spectral | 3012.00 | 0.6586 | 0.9991 | 0.6898 | Strong 3-seed pilot, but already failed 20-seed confirmation. |
| netobs reliability spectral | 2996.67 | 0.6626 | 0.9996 | 0.5467 | Best new network-observational setting; beats myopic in 3 seeds. |
| netobs reliability sparse Node2Vec | 2949.00 | 0.6672 | 0.9996 | 0.5735 | Beats myopic, but not spectral netobs. |
| netobs reliability ELPH-struct | 2946.67 | 0.6681 | 0.9996 | 0.5515 | Beats myopic, but not spectral netobs. |
| netobs ELPH-struct frontier batch | 2902.33 | 0.6706 | 0.9991 | 0.6889 | Less myopic but close to myopic; batch composition is still not enough. |
| myopic | 2921.33 | 0.6710 | 0.9996 | 0.5406 | Strong baseline. |

## Compact Confirmation

After adding graph-embedding caching in the variant runner, I reran the narrowed matrix:

| variant | reward | w-RMSE | reward pct | entropy pct | reading |
|---|---:|---:|---:|---:|---|
| netobs reliability spectral | 2996.67 | 0.6626 | 0.9996 | 0.5467 | Stable best network-observational candidate. |
| netobs reliability sparse Node2Vec | 2949.00 | 0.6672 | 0.9996 | 0.5735 | Stable positive candidate, but still below spectral. |
| myopic | 2921.33 | 0.6710 | 0.9996 | 0.5406 | Strong baseline. |

Primary compact artifacts:

- `configs/douban_svdpp_network_entropy_compact_3seeds.json`
- `outputs/tables/douban_svdpp_network_entropy_compact_3seeds_final_summary.csv`
- `outputs/figures/douban_svdpp_network_entropy_compact_3seeds_w_rmse.png`
- `reports/douban_svdpp_network_entropy_compact_3seeds_mechanism_diagnostics.md`

## 20-Seed Spectral Confirmation

The narrowed 20-seed confirmation promoted only the best stable candidate: `network_observational` entropy with spectral embedding and reliability-DLS.

| variant | seeds | reward | w-RMSE | reward pct | entropy pct | reading |
|---|---:|---:|---:|---:|---:|---|
| netobs reliability spectral | 20 | 2861.80 | 0.6778 | 0.9996 | 0.5672 | First 20-seed Douban mechanism variant that beats myopic while using a more faithful network-observational entropy. |
| myopic | 20 | 2844.60 | 0.6801 | 0.9996 | 0.5435 | Strong reward baseline. |

This is a small but important positive result. It does not produce a dramatic paper-level separation, and it essentially ties the earlier no-DLS / reliability-DLS best follow-ups. Its value is mechanism fidelity: unlike the no-DLS result, it improves over myopic while retaining network-aware entropy and reliability-controlled DLS.

20-seed artifacts:

- `configs/douban_svdpp_network_entropy_spectral_20seeds.json`
- `outputs/tables/douban_svdpp_network_entropy_spectral_20seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_network_entropy_spectral_20seeds_mechanism_summary.csv`
- `outputs/figures/douban_svdpp_network_entropy_spectral_20seeds_w_rmse.png`
- `outputs/figures/douban_svdpp_network_entropy_spectral_20seeds_reward_entropy_percentiles.png`
- `reports/douban_svdpp_network_entropy_spectral_20seeds_mechanism_diagnostics.md`

Primary artifacts:

- `configs/douban_svdpp_network_entropy_embedding_3seeds.json`
- `outputs/tables/douban_svdpp_network_entropy_embedding_3seeds_final_summary.csv`
- `outputs/figures/douban_svdpp_network_entropy_embedding_3seeds_w_rmse.png`
- `reports/douban_svdpp_network_entropy_embedding_3seeds_report.md`
- `reports/douban_svdpp_network_entropy_embedding_3seeds_mechanism_diagnostics.md`

## Interpretation

The new network-observational entropy has signal: all three top-k netobs variants beat myopic in this 3-seed run. However:

- sparse Node2Vec and ELPH-struct do not yet beat spectral embedding;
- frontier batch composition again increases information-oriented selection, but does not improve w-RMSE enough;
- the new entropy calculation is much slower than the previous proxy and should be optimized before any 20-seed run.

The narrow 20-seed confirmation is now complete and positive for `netobs_reliability_spectral`.

## Current Decision

Promote `netobs_reliability_spectral` into the formal Douban follow-up set. Do not yet promote sparse Node2Vec / ELPH-struct; they need embedding-quality work before a paper-quality run.

After the embedding upgrade pass, this decision still holds. `elph_role` and `sparse_node2vec_biased` both improve over the earlier proxy variants, but neither beats spectral in the 3-seed netobs reliability-DLS matrix.

Upgrade artifacts:

- `configs/douban_svdpp_embedding_upgrade_3seeds.json`
- `outputs/tables/douban_svdpp_embedding_upgrade_3seeds_final_summary.csv`
- `reports/douban_embedding_upgrade_findings.md`
