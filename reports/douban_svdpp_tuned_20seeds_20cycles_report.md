# douban Reproduction Report

Dataset: `douban` loaded from `data/raw/gcmc/douban`.

## Configuration

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: random, myopic, entropy, du_net
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2836.9000 | 0.6805 | 0.0083 | 0.9989 | 3232 |
| entropy | 1358.7500 | 0.9543 | 0.0059 | 0.8943 | 3232 |
| myopic | 2844.6000 | 0.6801 | 0.0083 | 1.0002 | 3232 |
| random | 1370.3500 | 0.9495 | 0.0052 | 0.8928 | 3232 |

## Notes

- This is a 20-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
