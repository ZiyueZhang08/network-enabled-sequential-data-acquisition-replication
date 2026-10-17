# movielens_100k_svdpp_figure4a_observational_entropy_3seeds Policy-Variant Grid

Dataset: `movielens-100k` from `data/raw/ml-100k`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| du_net_hybrid_topk_current | du_net | 2717.3333 | 0.8229 | 0.9812 | 0.8582 |
| mf_linear_ucb_a002_topk | mf_linear_ucb | 2699.0000 | 0.8349 | nan | nan |
| myopic_topk | myopic | 2704.0000 | 0.8354 | 0.9994 | 0.5641 |
| du_net_network_observational_weighted_topk | du_net | 2497.6667 | 0.8413 | 0.9323 | 0.9287 |
| du_net_dynamic_sigma_observational_logdet_topk | du_net_dynamic_sigma | 2525.6667 | 0.8417 | 0.9437 | 0.9351 |
| du_net_observational_logdet_topk | du_net | 2491.6667 | 0.8431 | 0.9426 | 0.9315 |
| du_net_observational_logdet_exp_tau05 | du_net | 2307.3333 | 0.8909 | 0.9281 | 0.7604 |
