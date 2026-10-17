# MovieLens Heterogeneous Response Robustness

Response model: `P(response)=clip(0.8 + alpha * (rating - 3), 0, 1)`.

## Configuration

- Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- Response alphas: [0.02]
- Policies: random, myopic, entropy, du_net
- Factorization: svdpp

## Final Summary

| alpha | policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate | observed_size |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.0200 | du_net | 2269.0000 | 40.8268 | 0.8305 | 0.0055 | 0.8301 | 2460 |
| 0.0200 | entropy | 991.8000 | 111.6546 | 1.0267 | 0.0248 | 0.8136 | 2427 |
| 0.0200 | myopic | 2309.4000 | 36.4572 | 0.8318 | 0.0058 | 0.8295 | 2459 |
| 0.0200 | random | 903.8000 | 18.4416 | 1.0347 | 0.0081 | 0.8143 | 2429 |
