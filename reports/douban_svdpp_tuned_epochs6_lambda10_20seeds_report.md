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
| du_net | 2987.0500 | 0.6732 | 0.0054 | 0.9937 | 3232 |
| entropy | 1205.9000 | 0.9684 | 0.0084 | 0.8812 | 3232 |
| myopic | 3023.8000 | 0.6701 | 0.0046 | 0.9986 | 3232 |
| random | 1370.3500 | 0.9281 | 0.0051 | 0.8759 | 3232 |

## Notes

- This is a 20-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
