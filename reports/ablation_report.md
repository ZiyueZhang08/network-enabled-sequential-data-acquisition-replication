# Ablation Report

Synthetic reproduction of the DU-Net ablation plan.

| policy | cumulative_reward | rmse | w_rmse | sigma | avg_effective_neighborhood |
|---|---:|---:|---:|---:|---:|
| du_net_no_dls | 48.9420 | 0.8635 | 1.0638 | 0.4000 | 19.8030 |
| du_net_dynamic_sigma | 43.6297 | 0.8673 | 1.0704 | 0.1000 | 3.4598 |
| du_net_no_reward | -35.6542 | 0.8782 | 1.0947 | 0.4000 | 19.8030 |
| du_net_no_network | 46.7140 | 0.8856 | 1.1026 | 0.4000 | 19.8030 |
| du_net_static_sigma | 21.2998 | 0.8917 | 1.1287 | 0.4000 | 19.8030 |
| du_net | 21.2998 | 0.8917 | 1.1287 | 0.4000 | 19.8030 |
| du_net_no_entropy | 20.1576 | 0.9310 | 1.2018 | 0.4000 | 19.8030 |

Notes:
- `du_net_no_dls` removes DLS-smoothed user factors.
- `du_net_no_entropy` uses reward only.
- `du_net_no_reward` uses information value only.
- `du_net_no_network` removes neighbor reward and network entropy amplification.
- `du_net_dynamic_sigma` chooses sigma from the configured grid each cycle using observed reconstruction error.
