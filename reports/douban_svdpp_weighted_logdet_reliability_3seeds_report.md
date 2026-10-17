# douban_svdpp_weighted_logdet_reliability_3seeds Policy-Variant Grid

Dataset: `douban` from `data/raw/gcmc/douban`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| reliability_dls | du_net_shrinkage_dls | 2977.3333 | 0.6642 | 0.9996 | 0.5707 |
| no_dls | du_net_no_dls | 2979.0000 | 0.6643 | 0.9996 | 0.5727 |
| weighted_reward_reliability_l005 | du_net_logdet_shrinkage_dls | 2978.3333 | 0.6649 | 0.9996 | 0.5949 |
| weighted_reward_no_dls | du_net_no_dls | 2945.3333 | 0.6686 | 0.9985 | 0.6154 |
| weighted_cold_reliability | du_net_logdet_shrinkage_dls | 2921.6667 | 0.6711 | 0.9995 | 0.5410 |
| weighted_reward_reliability | du_net_logdet_shrinkage_dls | 2909.3333 | 0.6726 | 0.9995 | 0.6407 |
| weighted_reward_cold_reliability | du_net_logdet_shrinkage_dls | 2904.6667 | 0.6736 | 0.9995 | 0.5719 |
| current_du_net | du_net | 2871.6667 | 0.6753 | 0.9996 | 0.5321 |
