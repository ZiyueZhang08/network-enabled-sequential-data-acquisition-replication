"""Build a concise Chinese status handoff for Figure 4a replication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
SOURCE_BLOCKERS = TABLE_DIR / "figure4a_source_recovery_blockers.csv"
PHASE2_GATE = TABLE_DIR / "figure4a_observational_entropy_gate.csv"
PHASE3_GATE = TABLE_DIR / "figure4a_adaptive_grid_gate.csv"

OUT_REPORT = REPORT_DIR / "figure4a_current_status_zh.md"
OUT_CSV = TABLE_DIR / "figure4a_current_status_zh.csv"

PAPER_GAP = 1.814 - 1.788


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def status_rows() -> pd.DataFrame:
    final = pd.read_csv(FINAL_SUMMARY)
    du = final[final["variant"] == "du_net"].iloc[0]
    linear = final[final["variant"] == "mf_linear_ucb_a002"].iloc[0]
    myopic = final[final["variant"] == "myopic"].iloc[0]
    gap = float(linear.local_reward_weighted_w_rmse) - float(du.local_reward_weighted_w_rmse)
    return pd.DataFrame(
        [
            {
                "item": "复现阶段",
                "value": "本地可执行复现路径已闭环；进入源头信息恢复阶段",
            },
            {
                "item": "DU-Net 20-seed local w-RMSE",
                "value": fmt(du.local_reward_weighted_w_rmse),
            },
            {
                "item": "MF-centered Linear UCB proxy 20-seed local w-RMSE",
                "value": fmt(linear.local_reward_weighted_w_rmse),
            },
            {
                "item": "Myopic 20-seed local w-RMSE",
                "value": fmt(myopic.local_reward_weighted_w_rmse),
            },
            {
                "item": "Linear-minus-DU gap",
                "value": fmt(gap),
            },
            {
                "item": "相对论文视觉 gap",
                "value": f"{fmt(100.0 * gap / PAPER_GAP, 1)}%",
            },
            {
                "item": "结论",
                "value": "方向正，但低于 +0.010 次级阈值；不能声称 exact replication",
            },
        ]
    )


def blockers_table(blockers: pd.DataFrame) -> str:
    lines = [
        "| 优先级 | 缺口 | 最小需要信息 |",
        "|---:|---|---|",
    ]
    for row in blockers.sort_values("priority").itertuples(index=False):
        lines.append(f"| {row.priority} | {row.blocker} | {row.minimum_needed} |")
    return "\n".join(lines)


def write_report(rows: pd.DataFrame) -> None:
    final = pd.read_csv(FINAL_SUMMARY)
    du = final[final["variant"] == "du_net"].iloc[0]
    linear = final[final["variant"] == "mf_linear_ucb_a002"].iloc[0]
    gap = float(linear.local_reward_weighted_w_rmse) - float(du.local_reward_weighted_w_rmse)
    phase2 = pd.read_csv(PHASE2_GATE).sort_values("w_rmse").iloc[0]
    phase3 = pd.read_csv(PHASE3_GATE).sort_values("w_rmse").iloc[0]
    blockers = pd.read_csv(SOURCE_BLOCKERS)

    status_lines = [
        "| 项目 | 当前值 |",
        "|---|---|",
    ]
    for row in rows.itertuples(index=False):
        status_lines.append(f"| {row.item} | {row.value} |")

    report = f"""# Figure 4a 当前复现状态

生成脚本：`experiments/build_figure4a_current_status_zh.py`。

## 进度判定

本地复现路径已经完成闭环。当前不再建议继续扩大 seed、扩大 adaptive
grid、或继续调 selection/entropy proxy；这些方向已经被 3-seed/1-seed
gate 排除。

{chr(10).join(status_lines)}

## 已执行但未晋级的路线

- 指标对齐：完成；主指标采用本地 linear reward-weighted w-RMSE，同时报告
  conventional RMSE 和 sigmoid w-RMSE。
- observational entropy：未晋级；最佳候选 `{phase2.variant}`，w-RMSE
  `{fmt(phase2.w_rmse)}`，没有超过当前 DU-Net。
- adaptive tuning：未晋级；最佳 adaptive 候选 `{phase3.variant}`，w-RMSE
  `{fmt(phase3.w_rmse)}`，仍差于当前 DU-Net。
- Linear UCB comparator：保留 `mf_linear_ucb_a002`，但必须标注为
  MF-centered Linear UCB proxy。

## 当前结论

当前 DU-Net 对 MF-centered Linear UCB proxy 的 20-seed gap 是
`{fmt(gap)}`。这个方向与论文 Figure 4a 的 DU-Net 优于 Linear UCB 一致，
但只达到论文视觉 gap 的 `{fmt(100.0 * gap / PAPER_GAP, 1)}%`，且低于
计划的 `+0.010` 次级成功阈值。因此结论是：**方向性部分复现，不是精确复现**。

## 下一步

下一步已经不是本地实验，而是源头恢复。只有拿到以下任一类信息，才值得开启新的
3-seed 最小验证：

{blockers_table(blockers)}

## 收到源头信息后的执行规则

1. 只先跑 seeds `0,1,2`，不直接跑 20-seed。
2. 同场比较 current DU-Net、MF-centered Linear UCB proxy、Myopic。
3. 只有当主阻塞项改善且不引入新的指标矛盾时，才晋级 20-seed。
4. 20-seed 最终包继续固定为：best recovered DU-Net、current DU-Net、
   MF-centered Linear UCB proxy、Myopic。

## 关键交付物

- `reports/figure4a_final_package.md`
- `reports/figure4a_source_recovery_packet.md`
- `outputs/tables/figure4a_final_package_summary.csv`
- `outputs/tables/figure4a_source_recovery_blockers.csv`
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = status_rows()
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")
    print(rows.to_string(index=False))


if __name__ == "__main__":
    main()
