# flixster Reproduction Report

Dataset: `flixster` loaded from `data/raw/gcmc/flixster`.

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
| du_net | 2919.6500 | 0.6309 | 0.0023 | 1.2389 | 2236 |
| entropy | 1284.2250 | 0.9984 | 0.0148 | 1.0417 | 2236 |
| myopic | 2845.3750 | 0.6338 | 0.0023 | 1.2181 | 2236 |
| random | 1520.8250 | 0.9178 | 0.0068 | 1.0289 | 2236 |

## Notes

- This is a 20-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
