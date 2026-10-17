# douban_svdpp_baselines_netobs_20seeds Policy-Variant Grid

Dataset: `douban` from `data/raw/gcmc/douban`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| netobs_reliability_spectral | du_net_shrinkage_dls | 2861.8000 | 0.6778 | 0.9996 | 0.5672 |
| current_du_net | du_net | 2838.6500 | 0.6801 | 0.9996 | 0.5385 |
| myopic | myopic | 2844.6000 | 0.6801 | 0.9996 | 0.5435 |
| graph_social | graph_social | 2835.0000 | 0.6812 | 0.9996 | 0.5308 |
| thompson_sampling | thompson_sampling | 1409.6500 | 0.9417 | nan | nan |
| linear_ucb | linear_ucb | 1400.1000 | 0.9424 | nan | nan |
| low_rank_bandit | low_rank_bandit | 1378.5000 | 0.9487 | nan | nan |
| random | random | 1370.3500 | 0.9495 | 0.4989 | 0.5007 |
| entropy | entropy | 1358.7500 | 0.9543 | 0.4957 | 0.9996 |
