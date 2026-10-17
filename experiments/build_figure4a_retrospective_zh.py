"""Build a Chinese retrospective for the Figure 4a replication branch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
PHASE2_GATE = TABLE_DIR / "figure4a_observational_entropy_gate.csv"
PHASE3_GATE = TABLE_DIR / "figure4a_adaptive_grid_gate.csv"
CLAIM_FREEZE = TABLE_DIR / "figure4a_claim_freeze.csv"
READINESS_AUDIT = TABLE_DIR / "figure4a_readiness_audit.csv"

OUT_CSV = TABLE_DIR / "figure4a_retrospective_zh.csv"
OUT_REPORT = REPORT_DIR / "figure4a_retrospective_zh.md"

PAPER_GAP = 1.814 - 1.788


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def retrospective_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "section": "进展",
                "point": "本地复现实验已闭环",
                "evidence": "Phase 1-5、readiness、claim-freeze、handoff、checksum 均已生成并验证。",
            },
            {
                "section": "进展",
                "point": "当前结果方向正但不足以声称 exact replication",
                "evidence": "20-seed Linear-minus-DU local w-RMSE gap 为 +0.0083，低于 +0.010 次级阈值。",
            },
            {
                "section": "有效判断",
                "point": "metric 是核心分歧",
                "evidence": "reward-weighted 指标支持 DU-Net，conventional RMSE 仍支持 Linear UCB proxy。",
            },
            {
                "section": "有效判断",
                "point": "observational entropy 和 adaptive tuning 没有本地晋级",
                "evidence": "两类方向都改善 conventional RMSE 的倾向更强，但损伤主 w-RMSE 或 reward。",
            },
            {
                "section": "停止原因",
                "point": "继续本地 sweep 的边际价值很低",
                "evidence": "失败候选没有通过 1/3-seed gate；20-seed 当前基线已经稳定。",
            },
            {
                "section": "下一步",
                "point": "只在源头信息出现后重开实验",
                "evidence": "重开触发：精确 w-RMSE、作者 split/mask、DU-Net entropy/DLS/tuning、paper Linear UCB 协议。",
            },
        ]
    )


def markdown_table(rows: pd.DataFrame) -> str:
    lines = [
        "| 类别 | 判断 | 证据 |",
        "|---|---|---|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(f"| {row.section} | {row.point} | {row.evidence} |")
    return "\n".join(lines)


def write_report(rows: pd.DataFrame) -> None:
    final = pd.read_csv(FINAL_SUMMARY)
    phase2 = pd.read_csv(PHASE2_GATE).sort_values("w_rmse").iloc[0]
    phase3 = pd.read_csv(PHASE3_GATE).sort_values("w_rmse").iloc[0]
    claims = pd.read_csv(CLAIM_FREEZE)
    readiness = pd.read_csv(READINESS_AUDIT)
    du = final[final["variant"] == "du_net"].iloc[0]
    linear = final[final["variant"] == "mf_linear_ucb_a002"].iloc[0]
    gap = float(linear.local_reward_weighted_w_rmse) - float(du.local_reward_weighted_w_rmse)
    ready = bool(readiness["exists"].all())
    forbidden = int((claims["claim_type"] == "forbidden").sum())

    report = f"""# Figure 4a 复盘与下一步

生成脚本：`experiments/build_figure4a_retrospective_zh.py`。

## 当前进展

本地 Figure 4a 复现分支已经闭环，handoff ready: `{str(ready).lower()}`。

最终 20-seed 主结果：

- DU-Net local reward-weighted w-RMSE：`{fmt(du.local_reward_weighted_w_rmse)}`
- MF-centered Linear UCB proxy：`{fmt(linear.local_reward_weighted_w_rmse)}`
- Linear-minus-DU gap：`{fmt(gap)}`
- 相对论文视觉 gap：`{fmt(100.0 * gap / PAPER_GAP, 1)}%`

## 复盘表

{markdown_table(rows)}

## 反思

这轮推进的价值不在于把曲线“调得更像”，而在于把哪些方向不该继续跑变成了证据：

- selection 的论文式指数抽样更接近某些常规 RMSE 行为，但损伤 reward-weighted 目标。
- observational entropy/logdet 方向改善 conventional RMSE，却没有提升主 Figure 4a proxy。
- adaptive tuning、guarded adaptive、reward-floor adaptive 都没有超过当前固定 DU-Net。
- Linear UCB 已经被固定为 MF-centered proxy，不能移动 comparator 来制造更大的 gap。

因此，当前结果应表述为：**reward-weighted/high-rating 区域方向性部分复现，但不是精确复现**。
claim-freeze 中已有 `{forbidden}` 类禁止表述，后续报告必须遵守。

## 下一步

下一步不是继续本地实验，而是等待源头触发。只有拿到以下任一信息，才重新进入最小验证：

1. 精确 Figure 4a w-RMSE 定义；
2. 作者 MovieLens split、initial observed mask 或 candidate mask；
3. DU-Net observational entropy、DLS、network diffusion、SVD++ 后端或 per-cycle tuning 细节；
4. 论文 Linear UCB feature/update protocol。

触发后只先跑 seeds `0,1,2`。只有 3-seed 通过 gate，才允许 20-seed。

## 已知最佳本地候选

- 当前 DU-Net：最终保留。
- Phase 2 最佳观测熵行：`{phase2.variant}`，w-RMSE `{fmt(phase2.w_rmse)}`，未晋级。
- Phase 3 最佳 adaptive 行：`{phase3.variant}`，w-RMSE `{fmt(phase3.w_rmse)}`，未晋级。
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = retrospective_rows()
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
