# Figure 4a 复现成果与论文差距报告

生成日期：2026-07-08

## 1. 结论摘要

当前复现可以作为 **strong partial reproduction / 有条件交付复现**，但不能声明为无保留的 exact Figure 4a replication。

最新冻结候选为 `graph_late_lambda_half_start16`。在当前透明的 reward-tail / paper-like proxy 指标 `linear_reward_norm_w_rmse` 下，本地 DU-Net-style 方法优于 MF-centered Linear UCB proxy：

| 对照 | DU-Net | Comparator | Linear/Myopic-minus-DU gap | 解释 |
|---|---:|---:|---:|---|
| DU-Net vs Linear UCB proxy | 0.9811 | 1.0018 | +0.0207 | 正 gap 表示 DU-Net 更低、更好 |
| DU-Net vs Myopic | 0.9811 | 1.0021 | +0.0210 | sanity comparator 不推翻 DU-Net 优势 |

论文 Figure 4a 的视觉估计 gap 约为 `1.814 - 1.788 = 0.0260`。本地 `+0.0207` gap 达到论文视觉 gap 的约 `79.5%`，剩余短板约 `0.0053`，即还差约 `20.5%` 的 gap。

因此，最稳妥的表述是：

> 当前结果复现了论文 Figure 4a 中 DU-Net 相对 Linear UCB 更优的方向，并在本地 reward-tail proxy 上达到强部分复现；但由于 exact w-RMSE、MovieLens split/mask、DU-Net backend、Linear UCB protocol 仍未恢复，不能称为 exact Figure 4a 复现。

## 2. 对照论文 Figure 4a 的已完成部分

本地复现已经对齐了论文 Figure 4a 的核心实验形状：

| 论文要求 / 论文线索 | 本地状态 | 证据口径 |
|---|---|---|
| MovieLens 主实验 | 已聚焦 MovieLens 100K | final package 与 source-trigger 报告均围绕 MovieLens Figure 4a |
| 20 acquisition cycles | 已覆盖 | final reports 明确以 20 cycles 为主实验形状 |
| batch size 100 | 已覆盖 | final reports 与 paper PDF audit 均记录该协议 |
| 20 random initializations / 20 seeds | 已完成 | final package 使用 20-seed 表 |
| DU-Net 优于 Linear UCB 的方向 | 已在主 proxy 下复现 | DU-Net 0.9811 vs Linear proxy 1.0018 |
| 累计 reward 对照 | 已报告 | DU-Net 2762.55 vs Linear proxy 2717.85 |
| Myopic sanity comparator | 已补齐 | Myopic 1.0021；DU-Net 仍更好 |

本地交付状态为 `ready_for_caveated_submission`。交付包为：

- `outputs/figure4a_submission_bundle_20260708_006.zip`
- SHA256：`44B81698B28E21D3A2DBACE07A7B0C10A943A3C06F4F92CC9B95CB0A928B989B`

## 3. 最终 20-seed 结果

最新冻结 20-seed 总表如下：

| 方法 | reward | primary w-RMSE | sum-norm w-RMSE | sigmoid w-RMSE | conventional RMSE |
|---|---:|---:|---:|---:|---:|
| DU-Net best (`graph_late_lambda_half_start16`) | 2762.55 | 0.9811 | 0.8147 | 0.9092 | 1.2102 |
| MF-centered Linear UCB proxy | 2717.85 | 1.0018 | 0.8264 | 0.9088 | 1.2025 |
| Myopic | 2734.45 | 1.0021 | 0.8266 | 0.9089 | 1.2027 |
| DU-Net graph-base diagnostic | 2745.35 | 0.9827 | 0.8157 | 0.9092 | 1.2096 |

关键差值：

| 指标 | DU-Net vs Linear UCB proxy | DU-Net vs Myopic | 结论 |
|---|---:|---:|---|
| primary w-RMSE gap | +0.0207 | +0.0210 | 支持 DU-Net |
| cumulative reward gap | +44.70 | +28.10 | 支持 DU-Net |
| sum-norm w-RMSE gap | +0.0117 | +0.0119 | 支持 DU-Net |
| sigmoid w-RMSE gap | -0.0004 | -0.0002 | 近似持平，略偏 comparator |
| conventional RMSE gap | -0.0077 | -0.0076 | 不支持 DU-Net，Linear/Myopic 更好 |

这里的 gap 约定为 comparator-minus-DU；正数代表 DU-Net 更好，负数代表 comparator 更好。

## 4. seed 稳定性

DU-Net 在主 proxy 上的优势不是单 seed 偶然值：

| 对照 | DU-Net 赢的 seed 数 | 解释 |
|---|---:|---|
| vs MF-centered Linear UCB proxy | 19/20 | 只有 seed 15 负向 |
| vs Myopic | 18/20 | 主结论稳定 |

对 Linear UCB proxy 的 seed-level 诊断显示，失败 seed 为 `15`：

| seed | primary gap | reward gap | RMSE gap |
|---:|---:|---:|---:|
| 15 | -0.0063 | -59.0 | +0.0024 |

最强正向 seed 包括 seed `10`、`14`、`16`，primary gap 分别约 `+0.0519`、`+0.0494`、`+0.0417`。这说明主结果由多数 seed 支撑，但也存在个别初始化下 reward-tail acquisition 失败。

## 5. 机制层面：优势集中在高 reward 尾部

当前复现最像论文 Figure 4a 的地方，是它在高评分/高 reward 区域更少犯错：

| 分区 RMSE | DU-Net | Linear UCB proxy | Linear-minus-DU gap | 结论 |
|---|---:|---:|---:|---|
| high-rating RMSE >= 4 | 0.6413 | 0.6544 | +0.0131 | DU-Net 更好 |
| rating-5 RMSE | 0.9698 | 0.9911 | +0.0213 | DU-Net 更好 |
| low-rating RMSE <= 3 | 1.6776 | 1.6583 | -0.0193 | Linear UCB proxy 更好 |
| conventional full RMSE | 1.2102 | 1.2025 | -0.0077 | Linear UCB proxy 更好 |

这解释了为什么 reward-tail proxy 支持 DU-Net，而 conventional RMSE 不支持 DU-Net：本地 DU-Net 更偏向高评分 acquisition 和高 reward 尾部，代价是低评分区域与全局均方误差略差。

这个现象和论文中强调 reward / information value 的方向相容，但还不能证明本地实现等同于作者 DU-Net，因为 exact entropy、DLS、diffusion、动态调参协议仍缺失。

## 6. 与论文 Figure 4a 的主要差距

### 6.1 绝对 w-RMSE 尺度不一致

论文 Figure 4a 视觉估计终点约为：

| 方法 | 论文 Figure 4a 视觉估计 w-RMSE |
|---|---:|
| DU-Net | 1.788 |
| Linear UCB | 1.814 |
| gap | 0.026 |

本地最新主 proxy 为：

| 方法 | 本地主 proxy |
|---|---:|
| DU-Net-style | 0.9811 |
| MF-centered Linear UCB proxy | 1.0018 |
| gap | 0.0207 |

因此，当前最接近论文的是“相对 gap 方向和 gap 大小”，而不是“绝对 endpoint 数值”。由于论文没有公开足够可执行的 Figure 4a w-RMSE 权重、normalization、mask、aggregation 细节，绝对尺度不能被可靠对齐。

### 6.2 Linear UCB 仍是 proxy，不是作者源码级 comparator

当前 Linear UCB 为 MF-centered Linear UCB proxy。它用于测试“DU-Net 是否至少能超过论文第二强基线的本地近似版本”，但不能替代论文作者实现。

缺失项包括：

- feature map；
- alpha schedule；
- regularization；
- update frequency；
- reward transform。

### 6.3 DU-Net backend 仍是本地重建，不是作者 exact implementation

当前 `graph_late_lambda_half_start16` 是按论文机制动机重建出来的最佳候选，但仍不是作者源码级 DU-Net。

仍缺：

- exact observational entropy estimator；
- DLS smoothing rule；
- sigma schedule；
- network diffusion rule；
- feature normalization；
- `{lambda, sigma_t, p_t}` per-cycle tuning protocol；
- SVD++/backend hyperparameters。

### 6.4 MovieLens split / mask / seed protocol 未精确恢复

PDF 证据支持“initial/candidate/test sets、1% initial、20 cycles、batch 100、20 random initializations”这些高层协议，但没有给出：

- train/test split files；
- initial observed sets；
- candidate masks；
- exact held-out test masks；
- seed mapping；
- filtering details。

这会直接影响 w-RMSE endpoint 和 per-seed 结果。

## 7. 已尝试但拒绝晋级的改进分支

以下分支已经运行并拒绝晋级，原因是它们没有让结果更接近当前主目标，或造成 reward-tail collapse：

| 分支 | 结果 | 决策 |
|---|---|---|
| Phase 1 seed-15 micro screen | start17 / start15 有局部改善，但损害稳定 seed | 不晋级 |
| Phase 4 adaptive probe | primary delta `-0.0086`，reward delta `-58.75` | 不晋级 |
| no-network fidelity probe | primary delta `-0.0055`，reward delta `-11.25` | 不晋级 |
| exponential sampling tau=1 | primary delta `-0.2115`，reward delta `-945.50` | 不晋级 |

这说明剩余差距不太可能靠继续本地 proxy sweep 稳定解决。继续调参可能提高某个 proxy，却会增加偏离论文真实协议的风险。

## 8. 当前可交付声明与禁止声明

### 可以声明

- 本地 DU-Net-style candidate 在 primary reward-tail proxy 下稳定优于 MF-centered Linear UCB proxy。
- 本地 gap `+0.0207` 达到论文视觉 gap `+0.026` 的 `79.5%`。
- DU-Net 对 Linear UCB proxy 赢 `19/20` seeds，对 Myopic 赢 `18/20` seeds。
- 优势主要来自高 rating / rating-5 区域，符合 reward-tail acquisition 的机制直觉。
- 这是 strong partial reproduction / caveated reproduction。

### 不应声明

- 不应说已经 exact reproduction Figure 4a。
- 不应说当前 Linear UCB 是论文 exact Linear UCB。
- 不应说当前 w-RMSE 就是论文 exact w-RMSE。
- 不应忽略 conventional RMSE 反向结果。
- 不应把 no-network 或 exponential sampling 分支作为更 faithful 的最终结果。

## 9. 当前阻塞项与下一步

当前 source-trigger gate 状态为 `blocked_by_missing_source`。下一步不是继续本地大规模调参，而是获取 source-backed 信息。

优先级最高的材料：

| 优先级 | 需要的材料 | 为什么重要 |
|---:|---|---|
| P0 | exact Figure 4a w-RMSE definition | 解决绝对尺度和指标口径问题 |
| P0 | MovieLens split / initial / candidate / test masks | 解决数据人口和 seed 对齐问题 |
| P0 | DU-Net entropy / DLS / diffusion / tuning implementation | 解决 backend fidelity |
| P0 | per-cycle `{lambda, sigma_t, p_t}` tuning protocol | 解决动态调参协议 |
| P1 | paper Linear UCB protocol | 解决 comparator exactness |

已经准备好的外部请求材料：

- `reports/author_source_request_email.md`
- `reports/author_source_request_send_checklist.md`
- `reports/author_source_response_intake_template.md`

收到任何作者回复或源码/附件后，第一步不是直接跑 20 seeds，而是先按 response intake template 分类；如果触发条件成立，只跑 `seeds 0,1,2` 最小诊断。只有 3-seed gate 通过后，才扩展到 20 seeds。

## 10. 最终判断

如果交付标准是“严谨、透明、带 caveat 的部分复现”，当前结果已经可交付。

如果交付标准是“无保留精确复现论文 Figure 4a”，当前仍未完成。主要瓶颈不是算力或 seed 数，而是论文公开材料不足以恢复 exact metric、exact split、exact DU-Net backend 和 exact Linear UCB comparator。

当前最佳行动是冻结 `_006` 交付包，发送作者/source request，等待 source-backed trigger。继续本地 proxy sweep 不应作为下一步。


