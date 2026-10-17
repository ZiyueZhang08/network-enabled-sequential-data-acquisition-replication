# MovieLens 100K Reproduction Report

Dataset: `movielens-100k` loaded from `data/raw/ml-100k`.

## Configuration

- Seeds: [0]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: myopic, du_net, du_net_no_dls, du_net_no_network, du_net_dynamic_sigma
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | rmse | w_rmse | observed_size |
|---|---:|---:|---:|---:|
| du_net | 2545.0000 | 1.1420 | 0.8888 | 2800 |
| du_net_dynamic_sigma | 2584.0000 | 1.1445 | 0.8913 | 2800 |
| du_net_no_dls | 2505.0000 | 1.1421 | 0.8947 | 2800 |
| du_net_no_network | 2488.0000 | 1.1316 | 0.8943 | 2800 |
| myopic | 2766.0000 | 1.1478 | 0.8766 | 2800 |

## Notes

- MovieLens has no explicit social graph in the 100K release; this run uses a user similarity network from demographic metadata, falling back to co-rating similarity if metadata is unavailable.
- This is a one-seed real-data reproduction run. Paper-quality reproduction still requires 10-20 seeds and the Flixster/Douban social-network datasets.
