# MovieLens 100K Reproduction Report

Dataset: `movielens-100k` loaded from `data/raw/ml-100k`.

## Configuration

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: random, myopic, entropy, du_net
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2712.8000 | 0.8251 | 0.0049 | 1.1637 | 2800 |
| entropy | 1208.0000 | 1.0256 | 0.0281 | 1.0704 | 2800 |
| myopic | 2727.1000 | 0.8326 | 0.0048 | 1.1602 | 2800 |
| random | 1051.9000 | 1.0410 | 0.0100 | 1.0474 | 2800 |

## Notes

- MovieLens has no explicit social graph in the 100K release; this run uses a user similarity network from demographic metadata, falling back to co-rating similarity if metadata is unavailable.
- This is a 10-seed real-data reproduction run. Paper-quality reproduction still requires 10-20 seeds and the Flixster/Douban social-network datasets.
