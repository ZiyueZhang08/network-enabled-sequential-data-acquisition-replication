# MovieLens 100K Reproduction Report

Dataset: `movielens-100k` loaded from `data/raw/ml-100k`.

## Configuration

- Seeds: [0]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: random, myopic, entropy, du_net
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | rmse | w_rmse | observed_size |
|---|---:|---:|---:|---:|
| du_net | 2671.0000 | 1.1638 | 0.8338 | 2800 |
| entropy | 735.0000 | 1.0668 | 1.1164 | 2800 |
| myopic | 2752.0000 | 1.1669 | 0.8317 | 2800 |
| random | 1053.0000 | 1.0475 | 1.0463 | 2800 |

## Notes

- MovieLens has no explicit social graph in the 100K release; this run uses a user similarity network from demographic metadata, falling back to co-rating similarity if metadata is unavailable.
- This is a one-seed real-data reproduction run. Paper-quality reproduction still requires 10-20 seeds and the Flixster/Douban social-network datasets.
