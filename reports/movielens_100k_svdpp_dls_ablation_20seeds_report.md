# MovieLens 100K Reproduction Report

Dataset: `movielens-100k` loaded from `data/raw/ml-100k`.

## Configuration

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: du_net, du_net_no_dls, du_net_static_sigma, du_net_dynamic_sigma, du_net_no_network
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2714.5500 | 0.8264 | 0.0050 | 1.1606 | 2800 |
| du_net_dynamic_sigma | 2719.5500 | 0.8235 | 0.0063 | 1.1610 | 2800 |
| du_net_no_dls | 2702.3000 | 0.8282 | 0.0050 | 1.1599 | 2800 |
| du_net_no_network | 2688.5000 | 0.8333 | 0.0064 | 1.1552 | 2800 |
| du_net_static_sigma | 2714.5500 | 0.8264 | 0.0050 | 1.1606 | 2800 |

## Notes

- MovieLens has no explicit social graph in the 100K release; this run uses a user similarity network from demographic metadata, falling back to co-rating similarity if metadata is unavailable.
- This is a 20-seed real-data reproduction run. Paper-quality reproduction still requires 10-20 seeds and the Flixster/Douban social-network datasets.
