# douban Reproduction Report

Dataset: `douban` loaded from `data/raw/gcmc/douban`.

## Configuration

- Seeds: [0, 1, 2]
- Initial fraction of train: 0.01
- Cycles: 20
- Batch size: 100
- Policies: du_net, du_net_no_dls, du_net_shrinkage_dls, du_net_logdet_entropy, du_net_logdet_shrinkage_dls, du_net_logdet_shrinkage_exp_tau05
- Entropy mode: hybrid

## Final-Cycle Summary

| policy | cumulative_reward | w_rmse | w_rmse_ci95 | rmse | observed_size |
|---|---:|---:|---:|---:|---:|
| du_net | 2871.6667 | 0.6753 | 0.0136 | 1.0027 | 3232 |
| du_net_logdet_entropy | 2792.3333 | 0.6861 | 0.0293 | 0.9916 | 3232 |
| du_net_logdet_shrinkage_dls | 2901.0000 | 0.6715 | 0.0058 | 1.0054 | 3232 |
| du_net_logdet_shrinkage_exp_tau05 | 2822.0000 | 0.6820 | 0.0223 | 0.9952 | 3232 |
| du_net_no_dls | 2979.0000 | 0.6643 | 0.0286 | 1.0188 | 3232 |
| du_net_shrinkage_dls | 2955.3333 | 0.6666 | 0.0261 | 1.0152 | 3232 |

## Notes

- This is a 3-seed real-data reproduction run.
- GCMC/Monti datasets use the provided train/test masks; MovieLens uses the configured random train/test split and a metadata-derived user similarity network.
