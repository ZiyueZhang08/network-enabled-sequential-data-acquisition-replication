# flixster Reproduction Report

Dataset: `flixster` loaded from `data/raw/gcmc/flixster`.

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
| du_net | 263.5000 | 0.7703 | 0.0000 | 1.1039 | 436 |
| entropy | 148.5000 | 0.9453 | 0.0000 | 1.0659 | 436 |
| myopic | 275.0000 | 0.7548 | 0.0000 | 1.1101 | 436 |
| random | 176.0000 | 0.9003 | 0.0000 | 1.0676 | 436 |

## Notes

- This is a 1-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
