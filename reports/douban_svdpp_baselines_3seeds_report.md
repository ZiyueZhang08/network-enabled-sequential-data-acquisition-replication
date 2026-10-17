# douban Reproduction Report

Dataset: `douban` loaded from `data/raw/gcmc/douban`.

## Configuration

- Seeds: [0, 1, 2]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: random, myopic, entropy, linear_ucb, low_rank_bandit, thompson_sampling, graph_social, du_net
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2871.6667 | 0.6753 | 0.0136 | 1.0027 | 3232 |
| entropy | 1391.3333 | 0.9455 | 0.0135 | 0.8935 | 3232 |
| graph_social | 2896.6667 | 0.6736 | 0.0264 | 1.0071 | 3232 |
| linear_ucb | 1422.0000 | 0.9374 | 0.0134 | 0.8929 | 3232 |
| low_rank_bandit | 1388.0000 | 0.9462 | 0.0168 | 0.8928 | 3232 |
| myopic | 2921.3333 | 0.6710 | 0.0279 | 1.0104 | 3232 |
| random | 1385.3333 | 0.9457 | 0.0141 | 0.8927 | 3232 |
| thompson_sampling | 1418.6667 | 0.9391 | 0.0089 | 0.8932 | 3232 |

## Notes

- This is a 3-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
