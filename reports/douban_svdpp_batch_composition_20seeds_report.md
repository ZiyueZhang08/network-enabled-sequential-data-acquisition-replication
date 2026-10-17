# douban_svdpp_batch_composition_20seeds Policy-Variant Grid

Dataset: `douban` from `data/raw/gcmc/douban`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| myopic | myopic | 2844.6000 | 0.6801 | 0.9996 | 0.5435 |
| no_dls_frontier_batch_70r_30i | du_net_no_dls | 2818.0000 | 0.6832 | 0.9991 | 0.6864 |
| reliability_dls_frontier_batch_70r_30i | du_net_shrinkage_dls | 2813.9500 | 0.6833 | 0.9991 | 0.6889 |
| logdet_reliability_frontier_batch_70r_30i | du_net_logdet_shrinkage_dls | 2804.9500 | 0.6843 | 0.9991 | 0.7258 |
