# flixster Reproduction Report

Dataset: `flixster` loaded from `data/raw/gcmc/flixster`.

## Configuration

- Seeds: [0]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: random, myopic, entropy, du_net
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2978.0000 | 0.6250 | 0.0000 | 1.2500 | 2236 |
| entropy | 1274.0000 | 1.0028 | 0.0000 | 1.0406 | 2236 |
| myopic | 2953.0000 | 0.6240 | 0.0000 | 1.2384 | 2236 |
| random | 1557.5000 | 0.9035 | 0.0000 | 1.0299 | 2236 |

## Notes

- This is a 1-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
