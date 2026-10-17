# douban Heterogeneous Response Robustness

Response model: `P(response)=clip(0.8 + alpha * (rating - 3), 0, 1)`.

## Configuration

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- Response alphas: [0.02]
- Policies: myopic, du_net_shrinkage_dls, du_net_shrinkage_dls, du_net_shrinkage_dls
- Factorization: svdpp

## Final Summary

| alpha | variant | policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate | observed_size |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.0200 | myopic | myopic | 2354.7000 | 55.6638 | 0.6953 | 0.0097 | 0.8298 | 2892 |
| 0.0200 | netobs_reliability_spectral | du_net_shrinkage_dls | 2337.3500 | 40.9720 | 0.6969 | 0.0072 | 0.8297 | 2891 |
| 0.0200 | netobs_response_centered_str2 | du_net_shrinkage_dls | 2360.3500 | 43.6844 | 0.6934 | 0.0077 | 0.8294 | 2891 |
| 0.0200 | reliability_dls | du_net_shrinkage_dls | 2353.3000 | 41.4452 | 0.6948 | 0.0069 | 0.8303 | 2893 |
