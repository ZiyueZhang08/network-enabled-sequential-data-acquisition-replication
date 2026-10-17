# douban Reproduction Report

Dataset: `douban` loaded from `data/raw/gcmc/douban`.

## Configuration

- Seeds: [0, 1, 2]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: random, myopic, entropy, du_net
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2913.3333 | 0.6702 | 0.0225 | 1.0086 | 3232 |
| entropy | 1391.3333 | 0.9455 | 0.0135 | 0.8935 | 3232 |
| myopic | 2921.3333 | 0.6710 | 0.0279 | 1.0104 | 3232 |
| random | 1385.3333 | 0.9457 | 0.0141 | 0.8927 | 3232 |

## Notes

- This is a 3-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
