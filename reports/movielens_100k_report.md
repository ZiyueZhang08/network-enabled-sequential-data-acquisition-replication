# MovieLens 100K Reproduction Report

Dataset: `movielens-100k` loaded from `data/raw/ml-100k`.

## Configuration

- Seeds: [0]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100

## Final-Cycle Summary

| policy | cumulative_reward | rmse | w_rmse | observed_size |
|---|---:|---:|---:|---:|
| du_net | 963.0000 | 1.0808 | 1.1398 | 2800 |
| entropy | 451.0000 | 1.0887 | 1.1940 | 2800 |
| myopic | 2711.0000 | 1.1339 | 0.9214 | 2800 |
| random | 1053.0000 | 1.0537 | 1.0473 | 2800 |

## Notes

- MovieLens has no explicit social graph in the 100K release; this run uses a user similarity network from demographic metadata, falling back to co-rating similarity if metadata is unavailable.
- This is a one-seed real-data reproduction run. Paper-quality reproduction still requires 10-20 seeds and the Flixster/Douban social-network datasets.
