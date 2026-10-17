# movielens_100k_svdpp_social_rec_baselines_3seeds Policy-Variant Grid

Dataset: `movielens-100k` from `data/raw/ml-100k`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| du_net | du_net | 2717.3333 | 0.8229 | 0.9812 | 0.8582 |
| graph_social | graph_social | 2748.6667 | 0.8246 | 0.9994 | 0.5184 |
| social_hybrid_w050 | social_mf_hybrid | 2719.6667 | 0.8301 | 0.9994 | 0.7238 |
| mf_linear_ucb_a005 | mf_linear_ucb | 2702.3333 | 0.8306 | nan | nan |
| social_hybrid_w025 | social_mf_hybrid | 2733.6667 | 0.8306 | 0.9994 | 0.6567 |
| myopic | myopic | 2704.0000 | 0.8354 | 0.9994 | 0.5641 |
| social_hybrid_w100 | social_mf_hybrid | 2587.3333 | 0.8425 | 0.9994 | 0.8586 |
| social_neighbor_item | social_neighbor_item | 2362.6667 | 0.8678 | 0.9994 | 0.9112 |
