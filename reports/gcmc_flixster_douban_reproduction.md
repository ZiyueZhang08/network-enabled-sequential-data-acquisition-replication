# Flixster And Douban GCMC Reproduction

本章记录 Flixster/Douban 从数据寻找进入真实数据运行的进展。当前完成的是 GCMC/Monti `.mat` 解析、官方 train/test mask 接入、smoke test、以及 20 acquisition cycles 的单 seed 曲线。它还不是 paper-quality 多 seed 结果，但已经证明真实社交图数据可以进入 DU-Net acquisition loop。

## Data Loader

数据来源为 GCMC 官方仓库：

- `data/raw/gcmc/flixster/training_test_dataset.mat`
- `data/raw/gcmc/douban/training_test_dataset.mat`

两个文件均为 MATLAB v7.3/HDF5 格式，已用 `h5py` 接入。loader 沿用 GCMC 参考脚本规则：dense matrix 读入后转置，以处理 MATLAB/Python 行列顺序差异。之后去掉没有任何 rating 的全零用户/物品，并使用：

- `M`: rating matrix。
- `Otraining`: official train mask。
- `Otest`: official test mask。
- `W_users`: user network adjacency。

| dataset | active users | active items | ratings | train ratings | test ratings | user-network directed edges after symmetrization count |
|---|---:|---:|---:|---:|---:|---:|
| Flixster | 2341 | 2956 | 26173 | 23556 | 2617 | 30558 |
| Douban | 2999 | 3000 | 136891 | 123202 | 13689 | 2688 |

## Smoke Tests

Smoke config:

- 1 seed。
- 2 acquisition cycles。
- batch size 100。
- initial set = 1% of official training ratings。
- policies = random, myopic, entropy, DU-Net。
- SVD++ rank 8, epochs 2。

### Flixster Smoke

| policy | cumulative reward | w-RMSE | RMSE | observed size |
|---|---:|---:|---:|---:|
| Myopic | 275.0 | 0.7548 | 1.1101 | 436 |
| DU-Net | 263.5 | 0.7703 | 1.1039 | 436 |
| Random | 176.0 | 0.9003 | 1.0676 | 436 |
| Entropy | 148.5 | 0.9453 | 1.0659 | 436 |

### Douban Smoke

| policy | cumulative reward | w-RMSE | RMSE | observed size |
|---|---:|---:|---:|---:|
| DU-Net | 274.0 | 0.9141 | 0.9039 | 1432 |
| Myopic | 264.0 | 0.9194 | 0.9042 | 1432 |
| Random | 125.0 | 0.9921 | 0.9038 | 1432 |
| Entropy | 141.0 | 0.9842 | 0.9034 | 1432 |

Smoke 结论：真实数据 loader 和 acquisition loop 已经端到端跑通。Douban smoke 上 DU-Net 初步优于 myopic；Flixster smoke 上 myopic 略强。

## 20-Cycle Single-Seed Runs

这一步把 acquisition cycles 扩为复现计划要求的 20，但仍只跑 seed 0，目的是先观察曲线 shape 和参数迁移情况。

### Flixster 20 Cycles

![Flixster 20-cycle w-RMSE](<../outputs/figures/flixster_svdpp_1seed_20cycles_w_rmse.png>)

![Flixster 20-cycle reward](<../outputs/figures/flixster_svdpp_1seed_20cycles_cumulative_reward.png>)

| policy | cumulative reward | w-RMSE | RMSE | observed size |
|---|---:|---:|---:|---:|
| Myopic | 2953.0 | 0.6240 | 1.2384 | 2236 |
| DU-Net | 2801.5 | 0.6368 | 1.2067 | 2236 |
| Random | 1557.5 | 0.9035 | 1.0299 | 2236 |
| Entropy | 1274.0 | 1.0028 | 1.0406 | 2236 |

### Douban 20 Cycles

![Douban 20-cycle w-RMSE](<../outputs/figures/douban_svdpp_1seed_20cycles_w_rmse.png>)

![Douban 20-cycle reward](<../outputs/figures/douban_svdpp_1seed_20cycles_cumulative_reward.png>)

| policy | cumulative reward | w-RMSE | RMSE | observed size |
|---|---:|---:|---:|---:|
| Myopic | 2765.0 | 0.6986 | 0.9815 | 3232 |
| DU-Net | 2667.0 | 0.7131 | 0.9693 | 3232 |
| Random | 1400.0 | 0.9553 | 0.8921 | 3232 |
| Entropy | 1394.0 | 0.9591 | 0.8933 | 3232 |

## Assessment

MovieLens 上调好的参数不能直接迁移到 Flixster/Douban。两个 20-cycle 单 seed 实验中，DU-Net 仍明显优于 random/entropy，但 myopic 更强，尤其在 w-RMSE 上。这不推翻 DU-Net 方向，因为当前还没有针对真实社交图数据单独调：

- `lambda_entropy`
- `diffusion_prob`
- `entropy_coverage_weight`
- `sigma`
- `dls_top_k`
- `selection=topk` vs `selection=exp`

从数据结构看，Douban 的 user graph 很稀疏，Flixster graph 更密；同一个 diffusion probability 在两个数据集上的含义不同。下一步应先做小网格搜索，再上 10/20 seeds。

## Dataset-Specific Tuning

我先做了两个单 seed 小网格：

- `lambda_entropy=[0, 0.05, 0.15, 0.25, 0.4]`，固定 `diffusion_prob=0.005`。
- `diffusion_prob=[0, 0.001, 0.003, 0.005, 0.01]`，固定 `lambda_entropy=0.05`。

### Lambda Grid

| dataset | best lambda | best cumulative reward | best w-RMSE | note |
|---|---:|---:|---:|---|
| Flixster | 0.05 | 2907.5 | 0.6280 | MovieLens 的 0.25 过强 |
| Douban | 0.05 | 2741.0 | 0.7010 | 0.25/0.40 明显伤害 w-RMSE |

### Diffusion Grid

| dataset | best diffusion p | best cumulative reward | best w-RMSE | interpretation |
|---|---:|---:|---:|---|
| Flixster | 0.000 | 2978.0 | 0.6250 | 图更密，当前 network reward diffusion 容易过扩散；DLS 仍在用网络 |
| Douban | 0.010 | 2795.0 | 0.6931 | 图更稀疏，需要更强扩散才显现 network reward |

### Tuned 20-Cycle Single-Seed Comparison

Flixster tuned setting: `lambda_entropy=0.05`, `diffusion_prob=0.0`。

| policy | cumulative reward | w-RMSE | RMSE |
|---|---:|---:|---:|
| DU-Net | 2978.0 | 0.6250 | 1.2500 |
| Myopic | 2953.0 | 0.6240 | 1.2384 |
| Random | 1557.5 | 0.9035 | 1.0299 |
| Entropy | 1274.0 | 1.0028 | 1.0406 |

Douban tuned setting: `lambda_entropy=0.05`, `diffusion_prob=0.01`。

| policy | cumulative reward | w-RMSE | RMSE |
|---|---:|---:|---:|
| DU-Net | 2795.0 | 0.6931 | 0.9846 |
| Myopic | 2765.0 | 0.6986 | 0.9815 |
| Random | 1400.0 | 0.9553 | 0.8921 |
| Entropy | 1394.0 | 0.9591 | 0.8933 |

调参后，Douban 上 DU-Net 在单 seed 上同时超过 myopic 的 cumulative reward 和 w-RMSE；Flixster 上 DU-Net reward 超过 myopic，w-RMSE 基本打平。下一步应把 tuned settings 扩到 3 seeds，再决定是否上 20 seeds。

## Tuned 20-Seed Results

在 tuned settings 通过 3 seeds 检查后，我扩展到 20 seeds。配置仍然保持 lightweight SVD++ rank 8、epochs 2，目的是先得到三数据集 paper-style 稳定曲线；后续可再提高 epochs/rank。

### Flixster Tuned 20 Seeds

Setting: `lambda_entropy=0.05`, `diffusion_prob=0.0`。

![Flixster tuned 20-seed w-RMSE](<../outputs/figures/flixster_svdpp_tuned_20seeds_20cycles_w_rmse.png>)

![Flixster tuned 20-seed reward](<../outputs/figures/flixster_svdpp_tuned_20seeds_20cycles_cumulative_reward.png>)

| policy | cumulative reward | reward CI95 | w-RMSE | w-RMSE CI95 | RMSE |
|---|---:|---:|---:|---:|---:|
| DU-Net | 2919.65 | 31.17 | 0.6309 | 0.0023 | 1.2389 |
| Myopic | 2845.38 | 23.50 | 0.6338 | 0.0023 | 1.2181 |
| Random | 1520.83 | 19.27 | 0.9178 | 0.0068 | 1.0289 |
| Entropy | 1284.23 | 40.79 | 0.9984 | 0.0148 | 1.0417 |

Flixster 结论：DU-Net 在 20 seeds 上同时超过 myopic 的 cumulative reward 和 w-RMSE，且显著优于 random/entropy。这是目前最接近论文主结论的真实社交图结果。

### Douban Tuned 20 Seeds

Setting: `lambda_entropy=0.05`, `diffusion_prob=0.01`。

![Douban tuned 20-seed w-RMSE](<../outputs/figures/douban_svdpp_tuned_20seeds_20cycles_w_rmse.png>)

![Douban tuned 20-seed reward](<../outputs/figures/douban_svdpp_tuned_20seeds_20cycles_cumulative_reward.png>)

| policy | cumulative reward | reward CI95 | w-RMSE | w-RMSE CI95 | RMSE |
|---|---:|---:|---:|---:|---:|
| DU-Net | 2836.90 | 52.60 | 0.6805 | 0.0083 | 0.9989 |
| Myopic | 2844.60 | 57.83 | 0.6801 | 0.0083 | 1.0002 |
| Random | 1370.35 | 16.24 | 0.9495 | 0.0052 | 0.8928 |
| Entropy | 1358.75 | 21.37 | 0.9543 | 0.0059 | 0.8943 |

Douban 结论：DU-Net 与 myopic 在 20 seeds 上几乎打平，二者都明显优于 random/entropy。DU-Net 的 RMSE 略低，但 w-RMSE 和 reward 均没有稳定超过 myopic；下一步应继续调 `selection=exp`、sigma、DLS top-k，或者提高 SVD++ epochs/rank。

## Follow-Up Matrix

### Douban Tuning

The follow-up Douban grids tested:

- `selection=topk/exp`: top-k remained better; current exp sampling is too noisy.
- `diffusion_normalization=none/degree/sqrt_degree`: raw diffusion remained best.
- `sigma=[0.1,0.2,0.4,0.8]`: sigma 0.1 helped in the lightweight setting.
- `dls_top_k=[10,20,40,80]`: top-k 20 remained best.
- `entropy_mode=hybrid/observational/observational_hybrid`: observational was slightly better than hybrid, but the gain was small.
- `graph_embedding_method=spectral/elph`: current ELPH proxy underperformed spectral.
- `rank=[8,12,16]`: rank 8 remained best.
- `epochs=[2,4,6]`: epochs 6 improved DU-Net absolute w-RMSE.

Best 3-seed Douban setting found during this pass was `epochs=6`, `lambda_entropy=0.10`, with DU-Net w-RMSE 0.6647 versus myopic 0.6687. However, the 20-seed check did not preserve the win: DU-Net w-RMSE was 0.6732 versus myopic 0.6701. This makes Douban the key remaining hard case.

### Ablation

Flixster 3-seed ablation shows that DLS and reward are important. Removing reward collapses performance; removing DLS hurts reward and w-RMSE.

Douban 3-seed ablation shows a different pattern: dynamic sigma and no-DLS variants can beat the current full DU-Net. This suggests the sparse Douban graph makes DLS smoothing fragile, and sigma should be adapted rather than fixed.

### Heterogeneous Response

Under alpha=0.02 heterogeneous response:

| dataset | policy | reward | w-RMSE | response rate |
|---|---|---:|---:|---:|
| Flixster | DU-Net | 2449.50 | 0.6279 | 0.8277 |
| Flixster | myopic | 2377.00 | 0.6326 | 0.8297 |
| Douban | DU-Net | 2454.33 | 0.6770 | 0.8273 |
| Douban | myopic | 2381.00 | 0.6898 | 0.8255 |

This is encouraging: under response bias, DU-Net beats myopic on both true social datasets in the 3-seed pilot.

### Extended Baselines

Flixster 3-seed baseline matrix: DU-Net leads myopic, graph-social, low-rank bandit, Thompson sampling, Linear UCB, random, and entropy.

Douban 3-seed baseline matrix: myopic and graph-social remain very competitive; DU-Net is close, while bandit/random/entropy baselines are much weaker.

### Mechanism Diagnostics

The diagnostics show DU-Net still selects very high reward-percentile pairs:

| dataset | reward percentile | entropy percentile | avg selected degree | cold-user share |
|---|---:|---:|---:|---:|
| Flixster | 0.9977 | 0.5425 | 8.3952 | 0.1973 |
| Douban | 0.9996 | 0.5321 | 1.6682 | 0.1483 |

This means the current policy is closer to myopic exploitation than the paper's intended dynamic reward-information tradeoff. Future work should improve entropy and stochastic sampling rather than only tuning reward-side parameters.

## Artifacts

- Loader: `src/data_loader.py`
- Configs:
  - `configs/flixster_svdpp_smoke.json`
  - `configs/douban_svdpp_smoke.json`
  - `configs/flixster_svdpp_1seed_20cycles.json`
  - `configs/douban_svdpp_1seed_20cycles.json`
  - `configs/flixster_svdpp_lambda_grid_1seed.json`
  - `configs/douban_svdpp_lambda_grid_1seed.json`
  - `configs/flixster_svdpp_diffusion_grid_1seed.json`
  - `configs/douban_svdpp_diffusion_grid_1seed.json`
  - `configs/flixster_svdpp_tuned_1seed_20cycles.json`
  - `configs/douban_svdpp_tuned_1seed_20cycles.json`
  - `configs/flixster_svdpp_tuned_3seeds_20cycles.json`
  - `configs/douban_svdpp_tuned_3seeds_20cycles.json`
  - `configs/flixster_svdpp_tuned_20seeds_20cycles.json`
  - `configs/douban_svdpp_tuned_20seeds_20cycles.json`
- Tables:
  - `outputs/tables/flixster_svdpp_smoke_final_summary.csv`
  - `outputs/tables/douban_svdpp_smoke_final_summary.csv`
  - `outputs/tables/flixster_svdpp_1seed_20cycles_final_summary.csv`
  - `outputs/tables/douban_svdpp_1seed_20cycles_final_summary.csv`
  - `outputs/tables/flixster_svdpp_tuned_1seed_20cycles_final_summary.csv`
  - `outputs/tables/douban_svdpp_tuned_1seed_20cycles_final_summary.csv`
  - `outputs/tables/flixster_svdpp_tuned_20seeds_20cycles_final_summary.csv`
  - `outputs/tables/douban_svdpp_tuned_20seeds_20cycles_final_summary.csv`
- Figures:
  - `outputs/figures/flixster_svdpp_1seed_20cycles_w_rmse.png`
  - `outputs/figures/flixster_svdpp_1seed_20cycles_cumulative_reward.png`
  - `outputs/figures/douban_svdpp_1seed_20cycles_w_rmse.png`
  - `outputs/figures/douban_svdpp_1seed_20cycles_cumulative_reward.png`
  - `outputs/figures/flixster_svdpp_tuned_20seeds_20cycles_w_rmse.png`
  - `outputs/figures/flixster_svdpp_tuned_20seeds_20cycles_cumulative_reward.png`
  - `outputs/figures/douban_svdpp_tuned_20seeds_20cycles_w_rmse.png`
  - `outputs/figures/douban_svdpp_tuned_20seeds_20cycles_cumulative_reward.png`

## Next Step

Run dataset-specific tuning on Flixster and Douban:

1. Test `selection=exp` because the paper uses exponential weighting rather than deterministic top-k。
2. Tune `sigma` and `dls_top_k` separately for the sparse Douban graph and denser Flixster graph。
3. Increase SVD++ rank/epochs for Flixster/Douban after the policy-level signal is stable。
4. Add DLS/network ablations on Flixster and Douban, mirroring the MovieLens ablation table。
