# douban Reproduction Report

Dataset: `douban` loaded from `data/raw/gcmc/douban`.

## Configuration

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: myopic, du_net, du_net_no_dls, du_net_logdet_shrinkage_dls
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2838.6500 | 0.6801 | 0.0075 | 0.9990 | 3232 |
| du_net_logdet_shrinkage_dls | 2821.1500 | 0.6824 | 0.0075 | 0.9963 | 3232 |
| du_net_no_dls | 2863.0500 | 0.6777 | 0.0076 | 1.0027 | 3232 |
| myopic | 2844.6000 | 0.6801 | 0.0083 | 1.0002 | 3232 |

## Notes

- This is a 20-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
