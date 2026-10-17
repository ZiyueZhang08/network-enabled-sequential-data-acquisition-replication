# movielens_100k_svdpp_active_learning_baselines_3seeds Policy-Variant Grid

Dataset: `movielens-100k` from `data/raw/ml-100k`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| du_net | du_net | 2717.3333 | 0.8229 | 0.9812 | 0.8582 |
| graph_social | graph_social | 2748.6667 | 0.8246 | 0.9994 | 0.5184 |
| mf_linear_ucb_a005 | mf_linear_ucb | 2702.3333 | 0.8306 | nan | nan |
| myopic | myopic | 2704.0000 | 0.8354 | 0.9994 | 0.5641 |
| entropy | entropy | 1041.3333 | 1.0536 | 0.4450 | 0.9994 |
| active_hybrid_a020 | active_learning_hybrid | 267.6667 | 1.1637 | nan | nan |
| prediction_uncertainty | prediction_uncertainty | 229.3333 | 1.1719 | nan | nan |
| active_hybrid_a005 | active_learning_hybrid | 186.3333 | 1.1866 | nan | nan |
| leverage_uncertainty | leverage_uncertainty | 92.0000 | 1.1868 | nan | nan |
