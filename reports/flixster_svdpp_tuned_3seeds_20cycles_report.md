# flixster Reproduction Report

Dataset: `flixster` loaded from `data/raw/gcmc/flixster`.

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
| du_net | 2988.5000 | 0.6254 | 0.0027 | 1.2585 | 2236 |
| entropy | 1263.6667 | 0.9993 | 0.0305 | 1.0412 | 2236 |
| myopic | 2885.8333 | 0.6279 | 0.0048 | 1.2297 | 2236 |
| random | 1537.5000 | 0.9055 | 0.0068 | 1.0292 | 2236 |

## Notes

- This is a 3-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
