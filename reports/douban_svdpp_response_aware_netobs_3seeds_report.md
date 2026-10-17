# douban Heterogeneous Response Robustness

Response model: `P(response)=clip(0.8 + alpha * (rating - 3), 0, 1)`.

## Configuration

- Seeds: [0, 1, 2]
- Response alphas: [0.02]
- Policies: myopic, du_net_shrinkage_dls, du_net_shrinkage_dls, du_net_shrinkage_dls, du_net_shrinkage_dls, du_net_shrinkage_dls
- Factorization: svdpp

## Final Summary

| alpha | variant | policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate | observed_size |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.0200 | myopic | myopic | 2381.0000 | 127.4653 | 0.6898 | 0.0291 | 0.8255 | 2883 |
| 0.0200 | netobs_reliability_spectral | du_net_shrinkage_dls | 2389.0000 | 148.1843 | 0.6885 | 0.0285 | 0.8257 | 2883 |
| 0.0200 | netobs_response_centered | du_net_shrinkage_dls | 2429.6667 | 176.6986 | 0.6831 | 0.0348 | 0.8275 | 2887 |
| 0.0200 | netobs_response_centered_str2 | du_net_shrinkage_dls | 2455.0000 | 168.1759 | 0.6798 | 0.0303 | 0.8273 | 2887 |
| 0.0200 | netobs_response_expected | du_net_shrinkage_dls | 2429.6667 | 176.6986 | 0.6831 | 0.0348 | 0.8275 | 2887 |
| 0.0200 | reliability_dls | du_net_shrinkage_dls | 2341.0000 | 54.3171 | 0.6965 | 0.0159 | 0.8275 | 2887 |
