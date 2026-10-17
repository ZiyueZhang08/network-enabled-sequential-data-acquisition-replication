# douban Reproduction Report

Dataset: `douban` loaded from `data/raw/gcmc/douban`.

## Configuration

- Seeds: [0]
- Initial fraction of train: 0.01
- Cycles: 2
- Batch size: 100
- Policies: random, myopic, entropy, du_net
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 274.0000 | 0.9141 | 0.0000 | 0.9039 | 1432 |
| entropy | 141.0000 | 0.9842 | 0.0000 | 0.9034 | 1432 |
| myopic | 264.0000 | 0.9194 | 0.0000 | 0.9042 | 1432 |
| random | 125.0000 | 0.9921 | 0.0000 | 0.9038 | 1432 |

## Notes

- This is a 1-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
