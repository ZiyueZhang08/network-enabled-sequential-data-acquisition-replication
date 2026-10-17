# douban_svdpp_ablation_netobs_20seeds Policy-Variant Grid

Dataset: `douban` from `data/raw/gcmc/douban`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| no_dls | du_net_no_dls | 2863.0500 | 0.6777 | 0.9996 | 0.5689 |
| reliability_dls | du_net_shrinkage_dls | 2861.6500 | 0.6778 | 0.9996 | 0.5720 |
| netobs_reliability_spectral | du_net_shrinkage_dls | 2861.8000 | 0.6778 | 0.9996 | 0.5672 |
| dynamic_sigma | du_net_dynamic_sigma | 2849.2000 | 0.6791 | 0.9996 | 0.5353 |
| current_du_net | du_net | 2838.6500 | 0.6801 | 0.9996 | 0.5385 |
| no_network | du_net_no_network | 2838.6000 | 0.6801 | 0.9996 | 0.5306 |
| no_entropy | du_net_no_entropy | 2835.0000 | 0.6804 | 0.9996 | 0.5308 |
| no_reward | du_net_no_reward | 1360.0500 | 0.9533 | 0.5023 | 0.9996 |
