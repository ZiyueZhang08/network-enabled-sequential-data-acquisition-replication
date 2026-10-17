# MovieLens Main Reproduction

本章整理 MovieLens 100K 上已经完成的 paper-style 复现实验。当前版本对应复现计划的 Level 2 到 Level 3 过渡：真实 MovieLens 数据、1% initial train、20 acquisition cycles、batch size 100、20 random seeds、SVD++、DLS、network reward、network information value、95% CI。

## Experiment Setup

- Dataset: MovieLens 100K, local path `data/raw/ml-100k/`.
- User network: MovieLens 100K 没有真实社交图，因此按照论文说明使用用户辅助信息构造 demographic similarity network。
- Split: observed ratings 随机拆分为 train/test；train 中 1% 作为 initial set，其余作为 candidate acquisition set。
- Acquisition: 20 cycles，每轮 acquisition batch size = 100。
- Factorization: lightweight SVD++，rank 12，epochs 6。
- DU-Net score: z-normalized network reward + `0.25 *` z-normalized network entropy。
- Entropy proxy: DLS user subspace + item subspace 的 QR leverage/coherence proxy；当前最优配置是 hybrid entropy 且 coverage weight = 0。
- Repetitions: 20 seeds，报告均值和 95% confidence interval。

## Figure 4 Style: Homogeneous Response

![Figure 4 style MovieLens main 20 seeds](<../outputs/figures/figure4_movielens_main_20seeds.png>)

| policy | cumulative reward | reward CI95 | w-RMSE | w-RMSE CI95 | RMSE |
|---|---:|---:|---:|---:|---:|
| DU-Net | 2714.55 | 20.30 | 0.8264 | 0.0050 | 1.1606 |
| Myopic | 2717.90 | 26.01 | 0.8348 | 0.0059 | 1.1568 |
| Entropy | 1186.10 | 94.88 | 1.0305 | 0.0173 | 1.0678 |
| Random | 1046.90 | 23.13 | 1.0437 | 0.0062 | 1.0480 |

主结论：DU-Net 的 acquisition reward 几乎贴近 myopic，但 final w-RMSE 更低。这正是论文 Figure 4 想证明的核心 tradeoff：不是只追逐短期 reward，而是用 information value 改善 implementation phase 的矩阵估计。

## Figure 5 Style: Heterogeneous Response

异质响应模型为 `P(response)=clip(0.8 + 0.02 * (rating - 3), 0, 1)`；未响应的 selected pairs 不加入 observed set。

![Figure 5 style MovieLens heterogeneous response 20 seeds](<../outputs/figures/figure5_movielens_heterogeneous_20seeds.png>)

| policy | cumulative reward | reward CI95 | w-RMSE | w-RMSE CI95 | response rate | observed size |
|---|---:|---:|---:|---:|---:|---:|
| DU-Net | 2268.05 | 24.74 | 0.8328 | 0.0057 | 0.8295 | 2459.1 |
| Myopic | 2290.05 | 22.58 | 0.8364 | 0.0048 | 0.8288 | 2457.7 |
| Entropy | 990.95 | 65.03 | 1.0285 | 0.0143 | 0.8129 | 2425.7 |
| Random | 903.35 | 13.62 | 1.0381 | 0.0054 | 0.8134 | 2426.7 |

主结论：在 response 与 true rating 正相关时，myopic 仍然拿到略高 acquisition reward，但 DU-Net 的 w-RMSE 更低。这个结果方向与论文的 robustness 叙事一致：纯 reward/bandit 型策略更容易偏向高 rating 区域，DU-Net 保留信息采集能力。

## Bandit Baselines

![MovieLens bandit baselines 20 seeds](<../outputs/figures/movielens_bandit_baselines_20seeds.png>)

| policy | cumulative reward | reward CI95 | w-RMSE | w-RMSE CI95 |
|---|---:|---:|---:|---:|
| DU-Net | 2714.55 | 20.30 | 0.8264 | 0.0050 |
| Myopic | 2717.90 | 26.01 | 0.8348 | 0.0059 |
| Low-rank bandit | 1790.80 | 24.89 | 0.9935 | 0.0058 |
| Random | 1046.90 | 23.13 | 1.0437 | 0.0062 |
| Entropy | 1186.10 | 94.88 | 1.0305 | 0.0173 |
| Linear UCB | 72.35 | 42.69 | 1.1999 | 0.0113 |

实现说明：当前 `linear_ucb` 是 SVD++ interaction features 上的 ridge-UCB，`low_rank_bandit` 是低秩预测均值加 user/item count uncertainty bonus。它们是复现计划中的轻量 baseline，不等价于论文全部 16 个 benchmark 的精确实现。结果显示低秩 bandit 能拿到高于 random/entropy 的 reward，但 w-RMSE 与 DU-Net 仍有明显差距；linear UCB 当前版本表现很弱，后续若要贴近论文，应调 context construction、bonus scale 和 stochastic selection。

## DLS And Network Ablation

![MovieLens DLS ablation 20 seeds](<../outputs/figures/movielens_dls_ablation_20seeds.png>)

| policy | cumulative reward | reward CI95 | w-RMSE | w-RMSE CI95 |
|---|---:|---:|---:|---:|
| Dynamic sigma | 2719.55 | 28.67 | 0.8235 | 0.0063 |
| Full DU-Net | 2714.55 | 20.30 | 0.8264 | 0.0050 |
| Static sigma | 2714.55 | 20.30 | 0.8264 | 0.0050 |
| w/o DLS | 2702.30 | 24.41 | 0.8282 | 0.0050 |
| w/o network | 2688.50 | 28.37 | 0.8333 | 0.0064 |

主结论：DLS 和 network diffusion 都有正贡献。dynamic sigma 在 20-seed 平均上拿到最低 w-RMSE 和最高 reward，说明复现计划中“sigma_t 随 acquisition 阶段动态变化”的方向值得继续加深。`no network` 退化最明显，支持 network-enabled acquisition 不是普通 active learning 的说法。

## Flixster And Douban Data Status

已找到并下载 GCMC/Monti 格式真实数据：

- `data/raw/gcmc/flixster/training_test_dataset.mat`, 2,776,221 bytes。
- `data/raw/gcmc/douban/training_test_dataset.mat`, 2,778,554 bytes。
- Source: `https://github.com/riannevdberg/gc-mc/tree/master/gcmc/data`.

这两个 `.mat` 文件是 MATLAB v7.3/HDF5 格式。当前本机 Python 环境缺少 `h5py`/`tables`，因此已完成下载和文件级验证，但还没有接入 loader。下一步接入 Flixster/Douban 时，需要加 optional HDF5 reader，读取 `M`、`Otraining`、`Otest`、`W_users`，其中 Flixster 还可能包含 `W_movies`。

## Artifacts

- Configs:
  - `configs/movielens_100k_svdpp_optimized_20seeds.json`
  - `configs/movielens_100k_svdpp_robustness_20seeds.json`
  - `configs/movielens_100k_svdpp_baselines_20seeds.json`
  - `configs/movielens_100k_svdpp_dls_ablation_20seeds.json`
- Tables:
  - `outputs/tables/movielens_100k_svdpp_optimized_20seeds_final_summary.csv`
  - `outputs/tables/movielens_100k_svdpp_robustness_20seeds_final_summary.csv`
  - `outputs/tables/movielens_100k_svdpp_baselines_20seeds_final_summary.csv`
  - `outputs/tables/movielens_100k_svdpp_dls_ablation_20seeds_final_summary.csv`
- Figures:
  - `outputs/figures/figure4_movielens_main_20seeds.png`
  - `outputs/figures/figure5_movielens_heterogeneous_20seeds.png`
  - `outputs/figures/movielens_bandit_baselines_20seeds.png`
  - `outputs/figures/movielens_dls_ablation_20seeds.png`

## Current Assessment

MovieLens 主复现已经达到较强的 Level 3 局部标准：20 seeds、Figure 4/5 风格图、homogeneous/heterogeneous 两条主线、bandit baseline、DLS/network ablation 都已完成。和论文仍有差距的地方主要是：MovieLens 用户网络为辅助信息构造而非真实社交网络；observational entropy 仍是 coherence proxy；ELPH 还没有替换 spectral embedding；Flixster/Douban 真实图数据已下载但未解析运行。
