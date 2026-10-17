# flixster Reproduction Report

Dataset: `flixster` loaded from `data/raw/gcmc/flixster`.

## Configuration

- Seeds: [0, 1, 2]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: du_net, du_net_no_dls, du_net_no_network, du_net_dynamic_sigma, du_net_no_entropy, du_net_no_reward
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2988.5000 | 0.6254 | 0.0027 | 1.2585 | 2236 |
| du_net_dynamic_sigma | 2803.8333 | 0.6364 | 0.0055 | 1.2155 | 2236 |
| du_net_no_dls | 2864.0000 | 0.6302 | 0.0027 | 1.2245 | 2236 |
| du_net_no_entropy | 2992.1667 | 0.6264 | 0.0018 | 1.2605 | 2236 |
| du_net_no_network | 2988.5000 | 0.6254 | 0.0027 | 1.2585 | 2236 |
| du_net_no_reward | 1263.6667 | 0.9987 | 0.0306 | 1.0411 | 2236 |

## Notes

- This is a 3-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
