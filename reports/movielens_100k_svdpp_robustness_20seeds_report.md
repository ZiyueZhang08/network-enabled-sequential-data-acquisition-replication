# MovieLens Heterogeneous Response Robustness

Response model: `P(response)=clip(0.8 + alpha * (rating - 3), 0, 1)`.

## Configuration

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- Response alphas: [0.02]
- Policies: random, myopic, entropy, du_net
- Factorization: svdpp

## Final Summary

| alpha | policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate | observed_size |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0200 | du_net | 2268.0500 | 24.7394 | 0.8328 | 0.0057 | 0.8295 | 2459 |
| 0.0200 | entropy | 990.9500 | 65.0291 | 1.0285 | 0.0143 | 0.8129 | 2426 |
| 0.0200 | myopic | 2290.0500 | 22.5781 | 0.8364 | 0.0048 | 0.8288 | 2458 |
| 0.0200 | random | 903.3500 | 13.6174 | 1.0381 | 0.0054 | 0.8134 | 2427 |
