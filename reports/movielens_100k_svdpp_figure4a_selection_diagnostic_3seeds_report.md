# movielens_100k_svdpp_figure4a_selection_diagnostic_3seeds Policy-Variant Grid

Dataset: `movielens-100k` from `data/raw/ml-100k`.

## Top Variants by w-RMSE

| variant | policy | reward | w-RMSE | reward pct | entropy pct |
|---|---|---:|---:|---:|---:|
| du_net_topk_current | du_net | 2717.3333 | 0.8229 | 0.9812 | 0.8582 |
| mf_linear_ucb_a002_topk | mf_linear_ucb | 2699.0000 | 0.8349 | nan | nan |
| myopic_topk | myopic | 2704.0000 | 0.8354 | 0.9994 | 0.5641 |
| du_net_exp_tau05 | du_net | 2321.3333 | 0.8870 | 0.9313 | 0.7011 |
| du_net_dynamic_sigma_exp_tau1 | du_net_dynamic_sigma | 1783.6667 | 0.9601 | 0.7707 | 0.5909 |
| du_net_exp_tau1 | du_net | 1761.3333 | 0.9640 | 0.7717 | 0.5916 |
| myopic_exp_tau1 | myopic | 1312.3333 | 1.0044 | 0.5919 | 0.4980 |
| mf_linear_ucb_a002_exp_tau1 | mf_linear_ucb | 1214.3333 | 1.0213 | nan | nan |
