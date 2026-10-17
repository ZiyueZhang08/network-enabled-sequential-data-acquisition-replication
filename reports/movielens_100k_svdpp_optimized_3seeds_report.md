# MovieLens 100K Reproduction Report

Dataset: `movielens-100k` loaded from `data/raw/ml-100k`.

## Configuration

- Seeds: [0, 1, 2]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: random, myopic, entropy, du_net
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | rmse | w_rmse | observed_size |
|---|---:|---:|---:|---:|
| du_net | 2704.0000 | 1.1659 | 0.8242 | 2800 |
| entropy | 1041.3333 | 1.0706 | 1.0536 | 2800 |
| myopic | 2704.0000 | 1.1617 | 0.8354 | 2800 |
| random | 1012.3333 | 1.0454 | 1.0499 | 2800 |

## Notes

- MovieLens has no explicit social graph in the 100K release; this run uses a user similarity network from demographic metadata, falling back to co-rating similarity if metadata is unavailable.
- This is a 3-seed real-data reproduction run. Paper-quality reproduction still requires 10-20 seeds and the Flixster/Douban social-network datasets.
