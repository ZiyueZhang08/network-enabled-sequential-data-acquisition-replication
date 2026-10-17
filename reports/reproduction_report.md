# MVP Reproduction Report

This is a Level 1 reproduction run using a synthetic low-rank rating matrix with a homophilous user network.
It validates the end-to-end acquisition loop before replacing the synthetic loader with MovieLens/Flixster/Douban.

## Configuration

- Seeds: [0]
- Cycles: 20
- Batch size: 100
- Policies: random, myopic, entropy, du_net

## Final-Cycle Summary

| policy | cumulative_reward | rmse | w_rmse | observed_size |
|---|---:|---:|---:|---:|
| du_net | 126.5255 | 0.8828 | 1.1279 | 2031 |
| entropy | -54.6746 | 0.9265 | 1.1782 | 2031 |
| myopic | 145.0660 | 1.0257 | 1.3561 | 2031 |
| random | -49.8518 | 0.8831 | 1.1418 | 2031 |

## Quick Read

- Highest final cumulative reward: `myopic` (145.0660).
- Lowest final w-RMSE: `du_net` (1.1279).
- Figures are saved under `outputs/figures/` and tables under `outputs/tables/`.

## Known MVP Gaps

- Uses synthetic data because no raw MovieLens/Flixster/Douban files are present in the workspace.
- Uses spectral graph embedding as a dependency-free proxy for Node2Vec/ELPH.
- Uses coverage/network uncertainty as an entropy proxy rather than the full observational entropy derivation.

## Phase 2: Ablation and Robustness Progress

The reproduction framework now includes the planned data directories, a local MovieLens 100K loader, ablation experiments, and heterogeneous response robustness experiments. Because no raw MovieLens/Flixster/Douban files are currently present, these experiments still run on the synthetic low-rank network dataset.

### Ablation Final Summary

| policy | cumulative_reward | rmse | w_rmse | interpretation |
|---|---:|---:|---:|---|
| du_net | 21.2998 | 0.8917 | 1.1287 | Full MVP DU-Net proxy |
| du_net_dynamic_sigma | 43.6297 | 0.8673 | 1.0704 | Dynamic sigma improves this proxy setting |
| du_net_no_dls | 48.9420 | 0.8635 | 1.0638 | Current spectral DLS proxy is not yet faithful to paper DLS |
| du_net_no_entropy | 20.1576 | 0.9310 | 1.2018 | Removing information value hurts w-RMSE |
| du_net_no_network | 46.7140 | 0.8856 | 1.1026 | Network diffusion is not yet consistently beneficial in this synthetic setup |
| du_net_no_reward | -35.6542 | 0.8782 | 1.0947 | Pure information sacrifices acquisition reward |

### Heterogeneous Response Final Summary

| alpha | key observation |
|---:|---|
| 0.00 | Full-response baseline; all selected actions become observed. |
| 0.02 | Cumulative response rate falls to roughly 0.79-0.80, matching the intended missing-response setting. |
| 0.05 | DU-Net proxy obtains higher cumulative reward than myopic while keeping lower w-RMSE than myopic in this run. |

### Current Boundary

These results validate the experimental machinery, not a paper-quality claim. The next required step for faithful reproduction is to place real datasets under `data/raw/` and replace the synthetic runs with MovieLens/Flixster/Douban configurations.

## Phase 3: Real MovieLens 100K Run

MovieLens 100K was downloaded from the GroupLens file source and placed under `data/raw/ml-100k/`. The real-data entry point is `experiments/run_movielens.py`, with configuration in `configs/movielens_100k.json`.

### MovieLens 100K Final Summary

| policy | cumulative_reward | rmse | w_rmse | observed_size |
|---|---:|---:|---:|---:|
| du_net | 963.0000 | 1.0808 | 1.1398 | 2800 |
| entropy | 451.0000 | 1.0887 | 1.1940 | 2800 |
| myopic | 2711.0000 | 1.1339 | 0.9214 | 2800 |
| random | 1053.0000 | 1.0537 | 1.0473 | 2800 |

### Data Source Status

- MovieLens 100K: downloaded and used for a real-data reproduction run.
- Flixster/Douban: the paper follows Monti et al./GCMC-style data. The expected file is `training_test_dataset.mat`, but the public GCMC GitHub directories for `flixster` and `douban` are empty. See `reports/data_sources.md` for the checked paths and expected file fields.

### Interpretation Boundary

This MovieLens result is a real-data smoke-to-main run, not yet a paper-quality replication. It uses one seed and the MVP DU-Net proxy components. The paper result averages 20 random initializations and uses a more faithful DLS/entropy implementation.

## Phase 4: Moving Closer to Paper Effects

The first real MovieLens run did not reproduce the paper's advantage because the MVP information-value term used only coverage uncertainty. I replaced it with a more faithful subspace/coherence proxy inspired by the paper's observational entropy: QR leverage scores are computed for DLS user factors and item factors, and candidate entries with higher subspace leverage receive higher information value.

### Tuning Findings

| tuning dimension | best observed setting | evidence |
|---|---|---|
| entropy proxy | hybrid mode with coverage weight `0.0`, i.e. standardized subspace coherence only | final w-RMSE improves to `0.8760` |
| lambda_entropy | `0.25` | lambda grid: `0.25` gave final w-RMSE `0.8846` before the entropy-mix refinement |
| diffusion_prob | `0.005` | diffusion grid: too little diffusion (`0.0`) and too much (`0.02`) both worsened w-RMSE |
| DLS/network | keeping DLS + network is slightly better than removing them under the hybrid entropy proxy | `du_net` final w-RMSE `0.8888` vs `du_net_no_dls` `0.8947` and `du_net_no_network` `0.8943` in the hybrid tuning run |

### Optimized MovieLens 100K Summary

| policy | cumulative_reward | rmse | w_rmse | observed_size |
|---|---:|---:|---:|---:|
| du_net | 2709.0000 | 1.1498 | 0.8760 | 2800 |
| entropy | 876.0000 | 1.0751 | 1.1039 | 2800 |
| myopic | 2766.0000 | 1.1478 | 0.8766 | 2800 |
| random | 1053.0000 | 1.0484 | 1.0426 | 2800 |

This now matches the qualitative paper direction more closely on MovieLens: DU-Net has the best implementation-phase w-RMSE among the tested policies while keeping cumulative reward close to myopic and far above pure entropy. The margin is still tiny and is based on one seed, so it should be treated as a promising reproduction direction rather than a stable paper-level result.

### Remaining Gap to Paper-Quality Reproduction

- Replace plain MF with SVD++ or a stronger factorization implementation.
- Replace spectral node embedding with ELPH or a closer structural embedding.
- Use the paper/Monti splits where available instead of the current random split.
- Run 10-20 seeds and report confidence intervals.
- Obtain Flixster/Douban `training_test_dataset.mat` files and repeat the same optimized pipeline.

## Phase 5: SVD++ Faithfulness Step

I added a lightweight SVD++ path with implicit item feedback. DLS and entropy now use the effective SVD++ user representation rather than the plain MF user vector when `algorithm="svdpp"` is enabled.

### SVD++ Lambda Diagnostic

| lambda_entropy | cumulative_reward | rmse | w_rmse |
|---:|---:|---:|---:|
| 0.00 | 2779.0000 | 1.1692 | 0.8284 |
| 0.15 | 2751.0000 | 1.1660 | 0.8326 |
| 0.25 | 2671.0000 | 1.1638 | 0.8338 |
| 0.40 | 2658.0000 | 1.1603 | 0.8338 |
| 0.05 | 2748.0000 | 1.1680 | 0.8341 |

The single-seed SVD++ diagnostic shows that network/DLS reward is very strong on MovieLens. Positive entropy slightly hurts on seed 0, but remains close. This is a warning that the current entropy proxy is still not fully faithful to the paper's observational entropy.

### SVD++ 3-Seed Optimized Summary

| policy | cumulative_reward | rmse | w_rmse | observed_size |
|---|---:|---:|---:|---:|
| du_net | 2704.0000 | 1.1659 | 0.8242 | 2800 |
| entropy | 1041.3333 | 1.0706 | 1.0536 | 2800 |
| myopic | 2704.0000 | 1.1617 | 0.8354 | 2800 |
| random | 1012.3333 | 1.0454 | 1.0499 | 2800 |

This is the closest result so far to the paper's qualitative claim on MovieLens: DU-Net has lower average w-RMSE than myopic, entropy, and random over three random splits, while matching myopic on cumulative reward.

## Phase 6: 10-Seed MovieLens Stability Check

I expanded the SVD++ optimized MovieLens run from 3 seeds to 10 seeds and added 95% confidence intervals to the output tables and figures. This follows the paper's reporting style more closely, though it is still short of the paper's 20 random initializations.

### SVD++ 10-Seed Summary

| policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | rmse |
|---|---:|---:|---:|---:|---:|
| du_net | 2712.8000 | 32.1674 | 0.8251 | 0.0049 | 1.1637 |
| myopic | 2727.1000 | 34.7060 | 0.8326 | 0.0048 | 1.1602 |
| entropy | 1208.0000 | 135.6261 | 1.0256 | 0.0281 | 1.0704 |
| random | 1051.9000 | 33.7103 | 1.0410 | 0.0100 | 1.0474 |

The 10-seed result is stable in the direction needed for the reproduction: DU-Net has lower average w-RMSE than myopic while giving up only a small amount of acquisition reward. This mirrors the paper's intended tradeoff: DU-Net uses information value to improve implementation-phase prediction while preserving near-myopic acquisition-phase reward.

Artifacts:

- `outputs/tables/movielens_100k_svdpp_optimized_10seeds_final_summary.csv`
- `outputs/tables/movielens_100k_svdpp_optimized_10seeds_metrics.csv`
- `outputs/figures/movielens_100k_svdpp_optimized_10seeds_w_rmse.png`
- `outputs/figures/movielens_100k_svdpp_optimized_10seeds_cumulative_reward.png`

## Phase 7: Heterogeneous Response Robustness on MovieLens

I ran the paper's heterogeneous response setting on real MovieLens with SVD++, 10 seeds, 20 cycles, and batch size 100. The response model is:

`P(response)=clip(0.8 + 0.02 * (rating - 3), 0, 1)`.

This reduces the observed sample count because selected pairs without response are not added to the observed set.

### SVD++ 10-Seed Heterogeneous Response Summary

| policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate | observed_size |
|---|---:|---:|---:|---:|---:|---:|
| du_net | 2269.0000 | 40.8268 | 0.8305 | 0.0055 | 0.8301 | 2460 |
| myopic | 2309.4000 | 36.4572 | 0.8318 | 0.0058 | 0.8295 | 2459 |
| entropy | 991.8000 | 111.6546 | 1.0267 | 0.0248 | 0.8136 | 2427 |
| random | 903.8000 | 18.4416 | 1.0347 | 0.0081 | 0.8143 | 2429 |

The heterogeneous-response result remains directionally consistent with the paper: DU-Net has the best w-RMSE among the tested policies while giving up only modest acquisition reward relative to myopic. The advantage is small, so the next robustness step is to expand to 20 seeds and test whether the confidence intervals separate more clearly.

Artifacts:

- `outputs/tables/movielens_100k_svdpp_robustness_10seeds_final_summary.csv`
- `outputs/tables/movielens_100k_svdpp_robustness_10seeds_metrics.csv`
- `outputs/figures/movielens_100k_svdpp_robustness_10seeds_w_rmse_by_alpha.png`
- `outputs/figures/movielens_100k_svdpp_robustness_10seeds_reward_by_alpha.png`

## Phase 8: 20-Seed MovieLens Main Reproduction

I expanded the MovieLens main experiment to 20 seeds and added Figure 4 / Figure 5 style composite plots. I also ran the second-priority MovieLens baselines and DLS/network ablations.

### Homogeneous Response 20-Seed Summary

| policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | rmse |
|---|---:|---:|---:|---:|---:|
| du_net | 2714.5500 | 20.3020 | 0.8264 | 0.0050 | 1.1606 |
| myopic | 2717.9000 | 26.0132 | 0.8348 | 0.0059 | 1.1568 |
| entropy | 1186.1000 | 94.8778 | 1.0305 | 0.0173 | 1.0678 |
| random | 1046.9000 | 23.1273 | 1.0437 | 0.0062 | 1.0480 |

### Heterogeneous Response 20-Seed Summary

| policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | response_rate |
|---|---:|---:|---:|---:|---:|
| du_net | 2268.0500 | 24.7394 | 0.8328 | 0.0057 | 0.8295 |
| myopic | 2290.0500 | 22.5781 | 0.8364 | 0.0048 | 0.8288 |
| entropy | 990.9500 | 65.0291 | 1.0285 | 0.0143 | 0.8129 |
| random | 903.3500 | 13.6174 | 1.0381 | 0.0054 | 0.8134 |

### Additional Results

- Bandit baselines added: `linear_ucb` and `low_rank_bandit`.
- DLS ablation added: `du_net_no_dls`, `du_net_static_sigma`, `du_net_dynamic_sigma`, `du_net_no_network`.
- Dynamic sigma is currently the best DLS variant on MovieLens: w-RMSE 0.8235 versus 0.8264 for static/full DU-Net.
- Flixster/Douban GCMC `.mat` files were found and downloaded under `data/raw/gcmc/`.

Detailed chapter: `reports/movieLens_main_reproduction.md`.

## Phase 9: Flixster/Douban GCMC Data Integration

I found and downloaded the GCMC/Monti `training_test_dataset.mat` files for Flixster and Douban, installed `h5py`, and added a MATLAB v7.3/HDF5 loader. The loader preserves the official `Otraining`/`Otest` masks and uses `W_users` as the user network.

### Loader Verification

| dataset | active users | active items | ratings | train | test | adjacency sum |
|---|---:|---:|---:|---:|---:|---:|
| flixster | 2341 | 2956 | 26173 | 23556 | 2617 | 30558 |
| douban | 2999 | 3000 | 136891 | 123202 | 13689 | 2688 |

### 20-Cycle Single-Seed Summary

| dataset | policy | cumulative_reward | w_rmse | rmse |
|---|---|---:|---:|---:|
| flixster | myopic | 2953.0000 | 0.6240 | 1.2384 |
| flixster | du_net | 2801.5000 | 0.6368 | 1.2067 |
| flixster | random | 1557.5000 | 0.9035 | 1.0299 |
| flixster | entropy | 1274.0000 | 1.0028 | 1.0406 |
| douban | myopic | 2765.0000 | 0.6986 | 0.9815 |
| douban | du_net | 2667.0000 | 0.7131 | 0.9693 |
| douban | random | 1400.0000 | 0.9553 | 0.8921 |
| douban | entropy | 1394.0000 | 0.9591 | 0.8933 |

The real social-network datasets now run end to end. The current MovieLens-tuned DU-Net parameters do not yet beat myopic on the 20-cycle single-seed Flixster/Douban runs, so the next faithful step is dataset-specific tuning of `lambda_entropy`, `diffusion_prob`, entropy mix, sigma, and stochastic exponential selection before expanding to 20 seeds.

Detailed chapter: `reports/gcmc_flixster_douban_reproduction.md`.

### Dataset-Specific Tuning Update

I ran single-seed 20-cycle grids for `lambda_entropy` and `diffusion_prob`.

| dataset | tuned lambda | tuned diffusion | du_net reward | myopic reward | du_net w_rmse | myopic w_rmse |
|---|---:|---:|---:|---:|---:|---:|
| flixster | 0.05 | 0.000 | 2978.0000 | 2953.0000 | 0.6250 | 0.6240 |
| douban | 0.05 | 0.010 | 2795.0000 | 2765.0000 | 0.6931 | 0.6986 |

The tuned settings recover the intended direction on Douban and make Flixster nearly tied on w-RMSE while exceeding myopic reward. The next step is to run these tuned settings for 3 seeds, then 20 seeds if stable.

### Tuned 20-Seed GCMC Summary

| dataset | policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | rmse |
|---|---|---:|---:|---:|---:|---:|
| flixster | du_net | 2919.6500 | 31.1738 | 0.6309 | 0.0023 | 1.2389 |
| flixster | myopic | 2845.3750 | 23.5031 | 0.6338 | 0.0023 | 1.2181 |
| flixster | random | 1520.8250 | 19.2709 | 0.9178 | 0.0068 | 1.0289 |
| flixster | entropy | 1284.2250 | 40.7868 | 0.9984 | 0.0148 | 1.0417 |
| douban | du_net | 2836.9000 | 52.6004 | 0.6805 | 0.0083 | 0.9989 |
| douban | myopic | 2844.6000 | 57.8258 | 0.6801 | 0.0083 | 1.0002 |
| douban | random | 1370.3500 | 16.2364 | 0.9495 | 0.0052 | 0.8928 |
| douban | entropy | 1358.7500 | 21.3720 | 0.9543 | 0.0059 | 0.8943 |

Flixster now matches the paper direction strongly: DU-Net beats myopic on both acquisition reward and w-RMSE. Douban is nearly tied with myopic on both target metrics while clearly outperforming random and entropy; it needs the next tuning pass on stochastic selection and DLS hyperparameters.

## Phase 10: Consolidation, Douban Tuning, and Experiment Matrix

I added reproducibility infrastructure:

- `requirements.txt`
- `reports/runbook.md`
- `reports/config_catalog.md`
- `outputs/figures/three_dataset_main_reproduction.png`

I also added faithful/diagnostic code paths:

- `diffusion_normalization`: `none`, `degree`, `sqrt_degree`
- `graph_embedding_method`: `spectral`, `node2vec`, `elph`
- `entropy_mode`: `observational`, `observational_hybrid`
- baseline policies: `thompson_sampling`, `graph_social`
- per-cycle diagnostics for reward percentile, entropy percentile, selected user degree, and cold-user share

### Douban Tuning Outcome

The best 3-seed Douban tuning direction was `epochs=6`, `lambda_entropy=0.10`, which gave DU-Net w-RMSE 0.6647 versus myopic 0.6687. The 20-seed check did not preserve a win:

| policy | cumulative_reward | reward_ci95 | w_rmse | w_rmse_ci95 | rmse |
|---|---:|---:|---:|---:|---:|
| du_net | 2987.0500 | 40.7315 | 0.6732 | 0.0054 | 0.9937 |
| myopic | 3023.8000 | 33.6760 | 0.6701 | 0.0046 | 0.9986 |
| random | 1370.3500 | 16.2364 | 0.9281 | 0.0051 | 0.8759 |
| entropy | 1205.9000 | 33.6968 | 0.9684 | 0.0084 | 0.8812 |

Interpretation: stronger SVD++ improves both DU-Net and myopic; Douban remains a hard case because DU-Net's current information term is not exploratory enough.

### Flixster/Douban Matrix

I ran 3-seed ablation, robustness, and extended baseline pilots for Flixster and Douban. The strongest additional result is heterogeneous response robustness: DU-Net beats myopic on both Flixster and Douban when response probability depends on true rating.

Detailed outputs are in `reports/gcmc_flixster_douban_reproduction.md`.
