# flixster Heterogeneous Response Robustness

Response model: `P(response)=clip(0.8 + alpha * (rating - 3), 0, 1)`.

## Configuration

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- Response alphas: [0.02]
- Policies: random, myopic, entropy, du_net
- Factorization: svdpp

## Final Summary

| alpha | policy | policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate | observed_size |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.0200 | du_net | du_net | 2407.1000 | 22.7544 | 0.6334 | 0.0021 | 0.8302 | 1896 |
| 0.0200 | entropy | entropy | 1099.2250 | 37.8040 | 0.9778 | 0.0150 | 0.8140 | 1864 |
| 0.0200 | myopic | myopic | 2352.4750 | 20.9626 | 0.6375 | 0.0024 | 0.8306 | 1897 |
| 0.0200 | random | random | 1297.7000 | 16.7807 | 0.8989 | 0.0066 | 0.8171 | 1870 |
