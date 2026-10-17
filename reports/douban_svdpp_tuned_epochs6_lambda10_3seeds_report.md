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
| du_net | 3062.0000 | 0.6647 | 0.0176 | 1.0030 | 3232 |
| entropy | 1247.0000 | 0.9583 | 0.0013 | 0.8790 | 3232 |
| myopic | 3036.3333 | 0.6687 | 0.0149 | 0.9992 | 3232 |
| random | 1385.3333 | 0.9235 | 0.0156 | 0.8751 | 3232 |

## Notes

- This is a 3-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
