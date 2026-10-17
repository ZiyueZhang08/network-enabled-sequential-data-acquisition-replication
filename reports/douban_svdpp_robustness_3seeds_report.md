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
| 0.0200 | du_net | 2454.3333 | 98.5494 | 0.6770 | 0.0214 | 0.8273 | 2887 |
| 0.0200 | entropy | 1193.0000 | 38.6738 | 0.9307 | 0.0235 | 0.8103 | 2853 |
| 0.0200 | myopic | 2381.0000 | 127.4653 | 0.6898 | 0.0291 | 0.8255 | 2883 |
| 0.0200 | random | 1152.3333 | 57.6712 | 0.9419 | 0.0242 | 0.8155 | 2863 |
