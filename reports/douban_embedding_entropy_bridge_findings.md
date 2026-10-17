# Douban Embedding x Entropy Bridge Findings

This note records the focused follow-up after the Node2Vec p/q grid. The goal
was to test whether the best local structural embeddings improve when paired
with first-order expected network-observational entropy.

## Status Before This Run

The current strongest faithful Douban homogeneous candidate remains
network-observational entropy with reliability-DLS and spectral graph
embedding. In 20-seed confirmation it beats myopic modestly on both reward and
w-RMSE, but it is still not a full paper-level reproduction of DU-Net because
the implementation does not yet include the exact published DU-Net network
learning stack or the full 16-baseline matrix.

The previous 3-seed Node2Vec p/q grid showed that BFS-like local walks are the
closest Node2Vec setting, while outward-biased walks underperform. This run
therefore tests only the top local structural candidates:

- `spectral`
- `sparse_node2vec_bfs`
- `elph_role`

Each embedding is paired with:

- `network_observational`
- `network_observational_expected`

## Douban 3-Seed Result

Config: `configs/douban_svdpp_embedding_entropy_bridge_3seeds.json`

| variant | reward | w-RMSE | entropy pct | selected degree | reading |
|---|---:|---:|---:|---:|---|
| spectral netobs | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Best current homogeneous Douban pilot result. |
| spectral expected | 2996.67 | 0.6626 | 0.5467 | 1.2252 | Same aggregate result as current netobs. |
| Node2Vec BFS expected | 2983.33 | 0.6633 | 0.5717 | 1.4160 | More information-oriented, but slightly worse w-RMSE. |
| Node2Vec BFS netobs | 2983.33 | 0.6633 | 0.5717 | 1.4160 | Same aggregate result as expected. |
| ELPH-role expected | 2987.00 | 0.6633 | 0.5731 | 1.3427 | Competitive, but still behind spectral. |
| ELPH-role netobs | 2987.00 | 0.6633 | 0.5731 | 1.3427 | Same aggregate result as expected. |
| myopic | 2921.33 | 0.6710 | 0.5406 | 1.2343 | Strong reward baseline, but clearly worse w-RMSE. |

## Mechanism Reading

Expected network-observational entropy does not currently change the selected
batches relative to `network_observational` under this top-k/reliability-DLS
setting. That means it should not be promoted to a 20-seed rerun yet.

Spectral remains the best embedding choice. Node2Vec BFS and ELPH-role raise the
average selected entropy percentile, but that extra information-seeking does
not improve final w-RMSE. The likely failure mode is that the policy is already
operating near the reward frontier, so the extra structural signal nudges
selection toward higher-degree users without enough downstream prediction gain.

The diagnostic report shows the current spectral candidate improves over myopic
in cold/warm and low/high-degree segments, so the positive result is not just a
single subgroup artifact.

## Decision

Do not promote expected-netobs or Node2Vec/ELPH-role bridge variants to 20
seeds. Keep `network_observational + reliability-DLS + spectral` as the current
best homogeneous Douban candidate.

The next useful step is not another embedding-only sweep. The next step should
target remaining fidelity gaps:

1. Add a compact missing-baseline family for Douban/MovieLens/Flixster if the
   goal is paper-table coverage.
2. Or, if the goal is mechanism fidelity, implement a closer DU-Net acquisition
   approximation that changes the reward/information tradeoff before selection,
   then test it against the frozen spectral-netobs candidate.

Primary artifacts:

- `outputs/tables/douban_svdpp_embedding_entropy_bridge_3seeds_final_summary.csv`
- `outputs/tables/douban_svdpp_embedding_entropy_bridge_3seeds_mechanism_summary.csv`
- `reports/douban_svdpp_embedding_entropy_bridge_3seeds_report.md`
- `reports/douban_svdpp_embedding_entropy_bridge_3seeds_mechanism_diagnostics.md`
- `outputs/figures/douban_svdpp_embedding_entropy_bridge_3seeds_w_rmse.png`
