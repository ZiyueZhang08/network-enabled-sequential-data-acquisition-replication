# movielens_100k_svdpp_residual_bandit_3seeds Policy-Variant Grid

Dataset: `movielens-100k` from `data/raw/ml-100k`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| du_net | du_net | 2717.3333 | 0.8229 | 0.9812 | 0.8582 |
| graph_social | graph_social | 2748.6667 | 0.8246 | 0.9994 | 0.5184 |
| mf_linear_ucb_a005 | mf_linear_ucb | 2702.3333 | 0.8306 | nan | nan |
| residual_ucb_w025_a002 | mf_residual_ucb | 2705.3333 | 0.8318 | nan | nan |
| residual_ucb_w050_a005 | mf_residual_ucb | 2608.6667 | 0.8344 | nan | nan |
| residual_ucb_w025_a005 | mf_residual_ucb | 2650.6667 | 0.8345 | nan | nan |
| residual_thompson_w025_n004 | mf_residual_thompson_sampling | 2699.3333 | 0.8348 | nan | nan |
| myopic | myopic | 2704.0000 | 0.8354 | 0.9994 | 0.5641 |
