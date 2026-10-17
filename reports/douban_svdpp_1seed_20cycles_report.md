# douban Reproduction Report

Dataset: `douban` loaded from `data/raw/gcmc/douban`.

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
| du_net | 2667.0000 | 0.7131 | 0.0000 | 0.9693 | 3232 |
| entropy | 1394.0000 | 0.9591 | 0.0000 | 0.8933 | 3232 |
| myopic | 2765.0000 | 0.6986 | 0.0000 | 0.9815 | 3232 |
| random | 1400.0000 | 0.9553 | 0.0000 | 0.8921 | 3232 |

## Notes

- This is a 1-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
