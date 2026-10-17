# Network-Enabled Sequential Data Acquisition 复现工作区

本文件夹是对论文 **Network-Enabled Sequential Data Acquisition** 的复现、诊断和交付工作区。核心目标是围绕论文提出的 DU-Net / network-enabled sequential data acquisition 问题，搭建一个可运行、可审计、可继续推进的 clean-room 复现管线：从最小可行实验、MovieLens / Flixster / Douban 多数据集基线，到 Figure 4a/4b 曲线对齐、精确复现门槛审计、作者源码/掩码请求包。

当前状态：截至本目录最新产物时间 2026-07-12，项目已经完成了大量本地可执行复现和诊断，但**不能宣称 exact replication**。最新严格结论见 `reports/paper_exact_attempt_movielens_v2_final_exact_gate.md`：本地已关闭或通过了部分 clean-room gate，例如 Schur 条件熵、tau=1 full-pool softmax 选择、row-level 预测完整性；但 exact w-RMSE 公式、作者 split/mask、Linear UCB 具体协议、DLS/ELPH 后端、调参协议以及 Figure 4 endpoint/gap 仍有 source-blocked 或 fail gate。

## 目录总览

当前目录已经按复现项目的主要工作流组织好，整理原则是**保留既有脚本路径和产物路径**，避免移动 `outputs/`、`reports/`、`configs/` 这类已被大量脚本引用的目录。

| 路径 | 内容 | 当前用途 |
| --- | --- | --- |
| `Network-Enabled Sequential Data Acquisition (1).pdf` | 原论文 PDF | 论文原文证据源 |
| `Network-Enabled Sequential Data Acquisition.md` | 从论文抽出的 Markdown 文本 | 便于检索论文内容 |
| `reproduction_plan.md` | 早期复现计划 | 记录最初的 Level 1/2/3 复现路线 |
| `requirements.txt` | Python 依赖 | `numpy`、`pandas`、`matplotlib`、`scipy`、`h5py` |
| `src/` | 核心实现 | 数据加载、矩阵分解、DLS、熵、奖励、采样策略、指标、调参 |
| `experiments/` | 实验入口和报告生成脚本 | 运行主实验、敏感性实验、Figure 4 对齐、exact gate、交付包构建 |
| `configs/` | 实验配置 JSON | MovieLens / Flixster / Douban / paper-faithful / exact-attempt 配置 |
| `data/` | 原始和处理中间数据 | MovieLens 100K、GCMC Flixster/Douban 数据、processed split/mask |
| `outputs/` | 机器生成产物 | CSV 表、PNG 图、row-level predictions、mask、交付 bundle、日志 |
| `reports/` | 人可读报告 | 复现报告、诊断报告、gate 审计、交付说明、作者源码请求材料 |
| `scripts/` | 辅助脚本 | MovieLens / Monti split 检索和 fallback 准备 |
| `tests/` | 回归测试 | 核心行为、交付包、source trigger、inverse reconstruction 等测试 |
| `tmp/` | 临时运行状态 | 后台任务 pid/out/err、临时 PDF 裁剪图等，可按需清理 |

粗略规模：除 `.git/.omx/cache` 类目录外，当前有约 176 个 Python 脚本、898 个 CSV、405 张 PNG、388 个 Markdown 报告/文档、143 个 JSON 配置/清单。`outputs/` 是最大目录，约 2.1 GB，主要来自 row-level prediction 和多轮实验结果。

## 已完成的主要工作

### 1. 论文问题和复现路线拆解

- 整理了论文文本和初始复现计划。
- 将复现目标拆成三个层级：
  - Minimal viable reproduction：跑通 sequential acquisition loop。
  - Faithful reproduction：接近论文协议，使用 MovieLens / Flixster / Douban、1% initial set、100 batch、20 cycles、SVD++、DLS、network reward、observational entropy。
  - Paper-quality / exact-attempt reproduction：对齐 Figure 4/5、20 seeds、ablation、robustness、严格 gate。
- 形成了模块化项目结构，避免把 acquisition loop、矩阵分解、DLS、entropy、metric 全写在单文件里。

### 2. 核心复现代码

`src/` 已经实现了复现实验所需的主要组件：

- `src/data_loader.py`
  - 合成数据生成。
  - MovieLens 100K 加载。
  - GCMC `.mat` 数据加载。
  - rating split、initial/candidate/test 划分。
  - MovieLens user feature 与相似网络构造。
- `src/matrix_factorization.py`
  - `MatrixFactorization` 和 `MFConfig`。
  - 支持 SVD++ 风格实验配置所需的低秩评分预测后端。
- `src/acquisition.py`
  - 项目最大核心文件，包含 `AcquisitionConfig` 和 `run_policy`。
  - 实现 DU-Net 风格 score、reward/entropy 组合、batch selection、softmax/top-k 选择、DLS shrinkage、adaptive grid、late guard。
  - 包含多种 comparator/proxy：Linear UCB、MF Linear UCB、OFUL、Thompson、finite-arm、active-learning hybrid、social-rec baseline 等。
  - 输出 cycle-level metrics、batch diagnostics、row-level predictions。
- `src/entropy.py`、`src/entropy_paper.py`、`src/entropy_schur.py`
  - coverage / subspace / coherence / observational / network observational entropy。
  - Schur conditional variance entropy，用于 paper-faithful / exact-attempt 主线。
- `src/dls.py`、`src/dls_paper.py`
  - DLS neighbor 计算。
  - Gaussian DLS、full/scoped DLS user factor 平滑。
- `src/graph_embedding.py`、`src/graph_transforms.py`
  - spectral、node2vec proxy、sparse node2vec、SGNS、ELPH proxy/role embedding。
  - edge dropout、rewiring、two-hop densification 等网络变换。
- `src/reward.py`
  - predicted reward、network reward、true centered reward。
- `src/metrics.py`
  - RMSE、standard mask RMSE、PDF literal norm RMSE、weighted/reward-weighted RMSE、tail proxy RMSE、cumulative reward。
- `src/tuning.py`
  - paper-style DU-Net 参数调优、inner split、softmax sampling、metric loss。
- `src/baselines.py`
  - 论文 benchmark slot 追踪和未解决 benchmark 显式清单。

### 3. 数据和 split/mask 准备

`data/` 已经包含：

- `data/raw/ml-100k.zip` 和解压后的 `data/raw/ml-100k/`。
- MovieLens 官方 `u1` 到 `u5` folds，以及 `ua/ub` folds。
- GCMC 相关 Douban / Flixster 原始数据和预处理脚本。
- `data/processed/` 下的 MovieLens Monti scan 和 `u1_fallback` split/mask 产物。
- `outputs/masks/inverse_reconstruction_v1/table1_random_masks_v1.npz`，用于 Table-1-compatible random mask screening。

辅助脚本：

- `scripts/find_or_prepare_movielens_monti_split.py`
  - 扫描本地 MovieLens / GCMC / Monti 相关 artifact。
  - 检查是否存在精确 `943 x 1681` mask。
  - 在缺少作者 mask 时准备 `u1` fallback split。

### 4. 基线、消融和多数据集实验

`configs/` 和早期 `experiments/run_*.py` 覆盖了多类实验：

- Synthetic MVP：`configs/mvp_synthetic.json`、`configs/ablation_synthetic.json`、`configs/robustness_synthetic.json`。
- MovieLens 100K：
  - `movielens_100k*.json`
  - SVD++ baseline、DLS ablation、lambda grid、entropy mix、hybrid tuning、optimized runs。
- Flixster：
  - `flixster_svdpp_*`
  - baseline、ablation、robustness、OFUL、MF-OFUL、paper bandit gap/repair、logistic bandit scale。
- Douban：
  - `douban_svdpp_*`
  - baseline、ablation、robustness、network entropy、node2vec、batch composition、graph density、formal mechanism、response-aware、reward frontier guard 等。

主要入口：

- `experiments/run_main.py`：合成/MVP 主入口。
- `experiments/run_movielens.py`：MovieLens 主实验入口。
- `experiments/run_ablation.py`：消融实验。
- `experiments/run_robustness.py`：稳健性实验。
- `experiments/run_policy_variants.py`：policy variant 对比。
- `experiments/plot_movielens_figures.py`、`plot_mechanism_diagnostics.py`、`plot_three_dataset_main.py`：图表生成。

对应产物主要在：

- `outputs/tables/*_metrics.csv`
- `outputs/tables/*_final_summary.csv`
- `outputs/tables/*_diagnostics*.csv`
- `outputs/figures/*_w_rmse.png`
- `outputs/figures/*_cumulative_reward.png`
- `reports/douban_*`
- `reports/flixster_*`
- `reports/movielens_*`

### 5. Figure 4a / Figure 4b 复现推进

项目对论文 Figure 4a/4b 做了多轮专门推进：

- 初始 Figure 4a 目标报告、距离评分、metric alignment、metric formula visual audit。
- MovieLens Table 1 初始诊断、reward-weight metric、observational entropy、Linear UCB comparator diagnostic。
- backend fidelity、model backend fidelity、mechanism evaluation audit。
- Figure 4 digitization precision audit，用可视化 digitization 结果衡量曲线误差。
- `figure4a_submission_bundle_*` 多版交付包构建。

关键脚本示例：

- `experiments/build_figure4a_metric_alignment_audit.py`
- `experiments/build_figure4a_metric_formula_visual_audit.py`
- `experiments/build_figure4a_reward_weight_metric_20seeds.py`
- `experiments/build_figure4a_linear_ucb_comparator_diagnostic.py`
- `experiments/build_figure4_digitization_precision_audit_v2.py`
- `experiments/build_figure4a_submission_bundle.py`

关键报告入口：

- `reports/figure4a_handoff_index.md`
- `reports/figure4a_delivery_quality_gate.md`
- `reports/figure4a_submission_bundle_latest.md`
- `reports/figure4_digitization_precision_v2.md`
- `reports/metric_reconstruction_audit_v4.md`

### 6. Paper-faithful MovieLens 主线

从 2026-07-09 起，项目重点转向更严格的 paper-faithful MovieLens 复现：

- rank 20 / epochs 20 / log Eq.3.5 / Schur entropy。
- ELPH role/proxy 诊断。
- tune-every-cycle / every-5-cycle 诊断。
- fold sensitivity。
- selection policy confirmation。
- reward-RMSE tradeoff。
- Schur loglambda top-k 与 full-pool softmax tau 分支。

主要入口：

- `experiments/run_paper_faithful_movielens.py`
- `experiments/run_paper_faithful_movielens_fold_sensitivity_v1.py`
- `experiments/build_paper_faithful_figure4a_rank20_ep20.py`
- `experiments/build_paper_faithful_eq35_log_diagnostic.py`
- `experiments/build_paper_faithful_embedding_phi_diagnostic.py`
- `experiments/build_selection_policy_confirmation_v1.py`
- `experiments/build_softmax_stochastic_main_freeze_v1.py`
- `experiments/build_softmax_final_delivery_v1.py`

关键报告入口：

- `reports/paper_faithful_figure4a_rank20_ep20_report.md`
- `reports/paper_faithful_movielens_alignment_20260709.md`
- `reports/paper_faithful_movielens_selection_policy_confirmation_v1_20seeds.md`
- `reports/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1.md`
- `reports/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1_figure4_digitized_alignment_v1.md`

### 7. Exact-attempt v1/v2 和严格 gate

当前最新阶段是 `paper_exact_attempt_movielens_v2`，目标是判断在只依赖论文和本地可恢复信息的情况下，是否能达到 exact replication。

冻结主线见：

- `configs/paper_exact_attempt_movielens_v2.json`
- `reports/paper_exact_attempt_movielens_v2_protocol_lock.md`

v2 冻结配置摘要：

- Seeds：`0..19`
- Split：`movielens_u1_fallback_table1_density`
- Shape：`943` users、`1681` items、`1364` initial observations
- Test rows：约 `20000`
- MF：rank `20`、epochs `20`、learning rate `0.015`、regularization `0.06`、algorithm `svdpp`
- Acquisition：`20` cycles，每轮 batch `100`
- DU-Net exact-attempt mainline：`schur_loglambda_tuned_full_pool_softmax_tau1_literal`
- Entropy：`schur_conditional_variance`
- Selection：full-pool exponential sampling，`tau=1`
- Tuning：`standard_mask_rmse`，每 `5` cycles

最新严格结论见：

- `reports/paper_exact_attempt_movielens_v2_final_exact_gate.md`

当前 gate 摘要：

- 已本地通过 / 关闭：
  - Schur conditional entropy clean-room 实现。
  - Tau=1 full-pool softmax 选择主线。
  - Row-level completeness，关键方法 seeds `0..19` final-cycle row-level predictions 完整。
- 未能 exact 关闭：
  - w-RMSE metric exactness。
  - exact author split/mask provenance。
  - paper Linear UCB 协议。
  - paper DLS/ELPH 后端。
  - 调参 cadence/objective/grid/tie-break protocol。
  - Figure 4a endpoint scale。
  - Figure 4b reward gap magnitude。

关键 v2 脚本：

- `experiments/run_paper_exact_movielens_v2_fold_sensitivity.py`
- `experiments/run_paper_exact_movielens_v2_elph_feasibility.py`
- `experiments/build_paper_exact_attempt_v2_protocol_lock.py`
- `experiments/build_paper_exact_attempt_v2_final_gate_report.py`
- `experiments/build_paper_exact_attempt_v2_source_recovery_packet.py`
- `experiments/build_metric_reconstruction_audit_v4.py`
- `experiments/build_schur_shortlist_vs_full_audit_v2.py`
- `experiments/build_lambda_noise_sensitivity_audit_v2.py`
- `experiments/build_tuning_frequency_objective_audit_v2.py`
- `experiments/build_external_linucb_protocol_audit_v2.py`

关键 v2 产物：

- `outputs/tables/paper_exact_attempt_movielens_v2_metrics.csv`
- `outputs/tables/paper_exact_attempt_movielens_v2_final_summary.csv`
- `outputs/tables/paper_exact_attempt_movielens_v2_allcycles_metrics.csv`
- `outputs/tables/paper_exact_attempt_movielens_v2_final_exact_gate.csv`
- `outputs/tables/paper_exact_attempt_movielens_v2_same_config_figure_gate.csv`
- `outputs/tables/metric_curve_alignment_v4.csv`
- `outputs/tables/figure4_digitization_precision_v2_targets.csv`
- `outputs/row_level_predictions/*.csv.gz`

### 8. Inverse reconstruction / mask screening

最新追加工作包括 `inverse_reconstruction_v1/v2`：

- 反向搜索可能解释 Figure 4a/4b 的 metric family、mask candidate、Linear UCB proxy、DU-Net knobs。
- 构建 mask screening queue 和 candidate ranking。
- 对 50 个 Table-1-compatible random masks 做 screening。
- 明确结论：这些是 empirical inverse-fit diagnostics，不是 author mask 证明。

关键脚本：

- `experiments/build_inverse_reconstruction_leaderboard.py`
- `experiments/build_inverse_reconstruction_v2_joint_leaderboard.py`
- `experiments/run_inverse_reconstruction_v2_mask_screening.py`
- `experiments/run_inverse_reconstruction_v2_linucb_candidate_confirm.py`
- `experiments/build_inverse_reconstruction_v2_metric_curve_audit.py`

关键报告：

- `reports/inverse_reconstruction_v1_report.md`
- `reports/inverse_reconstruction_v2_metric_curve_audit.md`
- `reports/inverse_reconstruction_v2_mask_screening_report.md`

最新 mask screening 结论：

- 当前 top candidate 是 `table1_random_000`。
- `author_mask_reconstructed_candidate = no`。
- exact mask provenance 仍然需要作者源码、作者 seed/split artifact，或等价生成脚本。

### 9. 作者源码 / 缺失协议请求材料

因为 exact replication 被多个 source-blocked gate 卡住，项目已经准备了作者源码恢复和请求材料：

- `reports/paper_exact_attempt_movielens_v2_source_recovery_packet.md`
- `reports/author_source_request_email.md`
- `reports/author_source_request_send_checklist.md`
- `reports/author_source_response_intake_template.md`
- `reports/source_artifact_inspection_protocol.md`
- `reports/source_trigger_monitor_snapshot.md`
- `reports/source_trigger_intake_check.md`

这些材料用于在获得作者代码、mask、metric formula、Linear UCB 参数、ELPH/DLS backend 或调参协议后，触发下一轮 source-grounded exact attempt。

## 如何运行

### 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 运行测试

```powershell
python -m pytest tests
```

当前测试覆盖：

- 核心 acquisition / metric / data loader 行为。
- Figure 4a delivery gate 和 submission bundle。
- 作者源码请求材料生成。
- source trigger monitor / intake check。
- inverse reconstruction leaderboard 和 v2 mask/metric audit。

### 常用实验入口

```powershell
# 最小/基础实验
python experiments/run_main.py --config configs/mvp_synthetic.json

# MovieLens 主实验
python experiments/run_movielens.py --config configs/movielens_100k_svdpp_optimized.json

# paper-faithful MovieLens
python experiments/run_paper_faithful_movielens.py --config configs/paper_faithful_movielens_v1.json

# exact-attempt v2 fold sensitivity
python experiments/run_paper_exact_movielens_v2_fold_sensitivity.py

# 生成 exact gate 报告
python experiments/build_paper_exact_attempt_v2_final_gate_report.py
```

不同脚本的参数并不完全一致。继续运行前建议先看目标脚本 `argparse` 定义，或从对应 `configs/*.json` 和已有 `outputs/tables/*_metrics.csv` 反推已使用的配置。

## 关键阅读顺序

如果要快速理解“这个文件夹现在做了什么”，建议按这个顺序看：

1. `README.md`：当前总览。
2. `reports/paper_exact_attempt_movielens_v2_protocol_lock.md`：最新 exact-attempt v2 主线和冻结协议。
3. `reports/paper_exact_attempt_movielens_v2_final_exact_gate.md`：当前最严格结论。
4. `reports/paper_exact_attempt_movielens_v2_source_recovery_packet.md`：下一步需要作者材料解决的 blocker。
5. `reports/metric_reconstruction_audit_v4.md`：w-RMSE metric exactness 为什么没关上。
6. `reports/figure4_digitization_precision_v2.md`：Figure 4 digitization 精度和误差边界。
7. `reports/inverse_reconstruction_v2_mask_screening_report.md`：最新反向 mask screening。
8. `src/acquisition.py`：核心 policy loop 和 comparator/proxy 实现。
9. `tests/test_core.py`：核心行为的回归测试入口。

## 当前边界和风险

- 这是一个**最大努力 paper-only clean-room reproduction**，不是 exact replication。
- 本地可以复现很多趋势、诊断和 proxy gap，但论文没有公开足够信息来唯一确定：
  - exact w-RMSE 权重和归一化。
  - 作者 initial/candidate/test mask 和随机 seed protocol。
  - Linear UCB feature map、alpha/reg、reward transform、update protocol。
  - ELPH backend、DLS embedding/source 参数。
  - 调参 cadence、inner objective、grid、tie-break。
- 不建议在没有新 source trigger 的情况下继续启动大规模 local proxy sweep；最新 v2 报告已经明确建议先使用 source recovery packet。

## 整理说明

本次整理没有移动已有实验文件，原因是：

- `experiments/` 中大量脚本硬编码或约定读取 `configs/`、`outputs/`、`reports/` 的现有相对路径。
- `outputs/` 中虽然文件很多，但包含重要证据链，尤其是 exact gate、row-level predictions、mask screening 和 submission bundle。
- `reports/` 已经是人类可读审计记录的集中入口，保留原文件名比按日期重排更利于追溯。

新增整理内容：

- `README.md`：项目全局说明、工作阶段、目录索引、关键产物、运行方法和当前结论。
- `.gitignore`：忽略 Python 缓存、pytest 缓存、临时 pid/out/err/log、OS 噪音文件。
