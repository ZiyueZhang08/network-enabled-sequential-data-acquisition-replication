# flixster_svdpp_ablation_20seeds Policy-Variant Grid

Dataset: `flixster` from `data/raw/gcmc/flixster`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| no_network | du_net_no_network | 2919.6500 | 0.6309 | 0.9977 | 0.5499 |
| current_du_net | du_net | 2919.6500 | 0.6309 | 0.9977 | 0.5499 |
| no_entropy | du_net_no_entropy | 2911.6750 | 0.6314 | 0.9978 | 0.5194 |
| no_dls | du_net_no_dls | 2831.2000 | 0.6347 | 0.9978 | 0.6116 |
| dynamic_sigma | du_net_dynamic_sigma | 2770.0500 | 0.6406 | 0.9977 | 0.7219 |
| no_reward | du_net_no_reward | 1284.2250 | 0.9979 | 0.4761 | 0.9978 |
