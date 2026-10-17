# MovieLens Heterogeneous Response Robustness

Response model: `P(response)=clip(0.8 + alpha * (rating - 3), 0, 1)`.

## Configuration

- Seeds: [0, 1, 2]
- Response alphas: [0.02]
- Policies: random, myopic, entropy, du_net
- Factorization: svdpp

## Final Summary

| alpha | policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate | observed_size |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0200 | du_net | 2449.5000 | 58.0135 | 0.6279 | 0.0016 | 0.8277 | 1891 |
| 0.0200 | entropy | 1124.1667 | 58.8580 | 0.9570 | 0.0172 | 0.8110 | 1858 |
| 0.0200 | myopic | 2377.0000 | 29.6602 | 0.6326 | 0.0014 | 0.8297 | 1895 |
| 0.0200 | random | 1316.1667 | 16.6824 | 0.8848 | 0.0055 | 0.8143 | 1865 |
