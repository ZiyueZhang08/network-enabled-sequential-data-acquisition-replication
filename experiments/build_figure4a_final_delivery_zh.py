"""Build the final Chinese delivery summary for the Figure 4a replication."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
LATEST_COMPARISONS = TABLE_DIR / "figure4a_latest_execution_comparisons.csv"
NO_NETWORK_GATE = TABLE_DIR / "figure4a_no_network_fidelity_gate.csv"
EXP_GATE = TABLE_DIR / "figure4a_exp_fidelity_gate.csv"

OUT_CSV = TABLE_DIR / "figure4a_final_delivery_zh.csv"
OUT_REPORT = REPORT_DIR / "figure4a_final_delivery_zh.md"

PAPER_DU = 1.788
PAPER_LINEAR = 1.814
PAPER_GAP = PAPER_LINEAR - PAPER_DU


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def load_context() -> dict[str, object]:
    summary = pd.read_csv(FINAL_SUMMARY)
    comparisons = pd.read_csv(LATEST_COMPARISONS)
    no_network = pd.read_csv(NO_NETWORK_GATE).iloc[0]
    exp = pd.read_csv(EXP_GATE).iloc[0]

    du = summary[summary["variant"] == "du_net"].iloc[0]
    linear = summary[summary["variant"] == "mf_linear_ucb_a002"].iloc[0]
    myopic = summary[summary["variant"] == "myopic"].iloc[0]
    linear_cmp = comparisons[comparisons["comparator"] == "mf_linear_ucb_a002"].iloc[0]
    myopic_cmp = comparisons[comparisons["comparator"] == "myopic"].iloc[0]

    primary_gap = float(linear.linear_reward_norm_w_rmse) - float(du.linear_reward_norm_w_rmse)
    myopic_gap = float(myopic.linear_reward_norm_w_rmse) - float(du.linear_reward_norm_w_rmse)
    paper_share = 100.0 * primary_gap / PAPER_GAP

    return {
        "du": du,
        "linear": linear,
        "myopic": myopic,
        "linear_cmp": linear_cmp,
        "myopic_cmp": myopic_cmp,
        "no_network": no_network,
        "exp": exp,
        "primary_gap": primary_gap,
        "myopic_gap": myopic_gap,
        "paper_share": paper_share,
    }


def build_rows(ctx: dict[str, object]) -> pd.DataFrame:
    du = ctx["du"]
    linear = ctx["linear"]
    myopic = ctx["myopic"]
    linear_cmp = ctx["linear_cmp"]
    myopic_cmp = ctx["myopic_cmp"]
    no_network = ctx["no_network"]
    exp = ctx["exp"]
    primary_gap = float(ctx["primary_gap"])
    myopic_gap = float(ctx["myopic_gap"])
    paper_share = float(ctx["paper_share"])

    rows = [
        {
            "section": "交付结论",
            "evidence": (
                "当前结果应交付为 strong partial reproduction，而不是 exact Figure 4a replication。"
            ),
            "decision": "冻结当前 best candidate，不再做无来源的新调参分支。",
        },
        {
            "section": "论文对照",
            "evidence": (
                "论文 Figure 4a 目标是 MovieLens、20 acquisition cycles、batch 100、"
                "20 random initializations，并显示 DU-Net 优于 Linear UCB。"
            ),
            "decision": "本地协议已覆盖 20 seeds / 20 cycles / batch 100 的核心对照。",
        },
        {
            "section": "主结果",
            "evidence": (
                f"DU-Net primary={fmt(du.linear_reward_norm_w_rmse)}，"
                f"MF-centered Linear UCB proxy={fmt(linear.linear_reward_norm_w_rmse)}，"
                f"Linear-minus-DU gap={fmt(primary_gap)}，达到论文视觉 gap 的 {fmt(paper_share, 1)}%。"
            ),
            "decision": "方向与论文一致，且超过本地 +0.020 成功门槛。",
        },
        {
            "section": "Myopic 对照",
            "evidence": (
                f"Myopic primary={fmt(myopic.linear_reward_norm_w_rmse)}，"
                f"Myopic-minus-DU gap={fmt(myopic_gap)}。"
            ),
            "decision": "Myopic 不改变主结论；DU-Net 在主 proxy 下仍优于 sanity comparator。",
        },
        {
            "section": "seed 稳定性",
            "evidence": (
                f"DU-Net 对 Linear UCB proxy 赢 {int(linear_cmp.primary_seed_wins)}/"
                f"{int(linear_cmp.seed_count)} seeds；对 Myopic 赢 {int(myopic_cmp.primary_seed_wins)}/"
                f"{int(myopic_cmp.seed_count)} seeds。"
            ),
            "decision": "主 proxy 优势不是单 seed 偶然波动。",
        },
        {
            "section": "机制解释",
            "evidence": (
                f"rating-5 RMSE gap={fmt(float(linear.rating5_rmse) - float(du.rating5_rmse))}，"
                f"high>=4 RMSE gap={fmt(float(linear.high_rating_rmse_ge4) - float(du.high_rating_rmse_ge4))}，"
                f"low<=3 RMSE gap={fmt(float(linear.low_rating_rmse_le3) - float(du.low_rating_rmse_le3))}。"
            ),
            "decision": "优势集中在高 reward 尾部；低分区域和普通 RMSE 必须作为 caveat 报告。",
        },
        {
            "section": "必要 caveat",
            "evidence": (
                f"conventional RMSE: DU-Net={fmt(du.rmse)}，Linear UCB proxy={fmt(linear.rmse)}；"
                f"sigmoid w-RMSE: DU-Net={fmt(du.sigmoid_reward_w_rmse)}，"
                f"Linear UCB proxy={fmt(linear.sigmoid_reward_w_rmse)}。"
            ),
            "decision": "不能声称所有 RMSE 口径都复现论文优势；只能声称主 reward-tail proxy 下方向一致。",
        },
        {
            "section": "no-network probe",
            "evidence": (
                f"primary delta={fmt(no_network.primary_delta_vs_frozen)}，"
                f"reward delta={fmt(no_network.reward_delta_vs_frozen, 2)}，"
                f"decision={no_network.decision}。"
            ),
            "decision": "拒绝晋级到 20-seed；它改善部分 RMSE 检查但削弱主结果。",
        },
        {
            "section": "exp-sampling probe",
            "evidence": (
                f"primary delta={fmt(exp.primary_delta_vs_frozen)}，"
                f"reward delta={fmt(exp.reward_delta_vs_frozen, 2)}，"
                f"decision={exp.decision}。"
            ),
            "decision": "拒绝晋级到 20-seed；指数采样 tau=1 会造成 reward-tail collapse。",
        },
        {
            "section": "计划调整",
            "evidence": (
                "当前剩余差距主要来自 exact w-RMSE、MovieLens split/mask、DU-Net backend、"
                "paper Linear UCB protocol，而不是 seed 数不足。"
            ),
            "decision": "停止 broad tuning；后续只接受 source-gated 最小诊断。",
        },
        {
            "section": "下一步",
            "evidence": (
                "若拿到精确 metric/split/backend/Linear UCB 信息，先跑 seeds 0,1,2；"
                "只有通过 gate 后才扩展到 20 seeds。"
            ),
            "decision": "现在进入最终报告交付与证据索引维护。",
        },
    ]
    return pd.DataFrame(rows)


def final_table(ctx: dict[str, object]) -> str:
    rows = [
        ("DU-Net best", ctx["du"]),
        ("Linear UCB proxy", ctx["linear"]),
        ("Myopic", ctx["myopic"]),
    ]
    lines = [
        "| 方法 | reward | primary w-RMSE | sigmoid w-RMSE | conventional RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in rows:
        lines.append(
            f"| {label} | {fmt(row.cumulative_reward, 2)} | "
            f"{fmt(row.linear_reward_norm_w_rmse)} | {fmt(row.sigmoid_reward_w_rmse)} | "
            f"{fmt(row.rmse)} |"
        )
    return "\n".join(lines)


def rows_table(rows: pd.DataFrame) -> str:
    lines = [
        "| section | evidence | decision |",
        "|---|---|---|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(f"| {row.section} | {row.evidence} | {row.decision} |")
    return "\n".join(lines)


def write_report(ctx: dict[str, object], rows: pd.DataFrame) -> None:
    du = ctx["du"]
    linear = ctx["linear"]
    primary_gap = float(ctx["primary_gap"])
    paper_share = float(ctx["paper_share"])

    report = f"""# Figure 4a 最终中文交付摘要

Generated by `experiments/build_figure4a_final_delivery_zh.py`.

## 短结论

当前 best candidate `graph_late_lambda_half_start16` 可以作为可交付的
**strong partial reproduction**，但不能写成无保留的 exact Figure 4a
replication。对照论文 Figure 4a 的 DU-Net vs Linear UCB 方向，本地 DU-Net
在主 reward-tail proxy 上优于 MF-centered Linear UCB proxy：
DU-Net `{fmt(du.linear_reward_norm_w_rmse)}` vs Linear UCB proxy
`{fmt(linear.linear_reward_norm_w_rmse)}`，Linear-minus-DU gap
`{fmt(primary_gap)}`，约为论文视觉 gap `0.0260` 的 `{fmt(paper_share, 1)}%`。

## 最终指标表

{final_table(ctx)}

## 交付口径

可以写：本地 DU-Net 在 reward-tail-weighted proxy 下稳定优于
MF-centered Linear UCB proxy 和 Myopic，方向与论文 Figure 4a 一致。

必须同时写：Linear UCB 仍是 proxy，w-RMSE 是 transparent local proxy；
conventional RMSE 和 sigmoid w-RMSE 不支持同等强度结论，因此结果是
partial/caveated reproduction。

## 执行后反思

本轮不应继续启动新模型实验。no-network 和 exp-sampling 两个最像论文机制的
fidelity probe 都未通过晋级门槛：前者削弱主 gap 和 reward，后者虽然改善普通
RMSE，却造成 reward-tail collapse。继续无来源调参更可能扩大 proxy 假设，
而不是更接近论文实现。

## 结构化证据

{rows_table(rows)}

## 下一步

冻结当前 best candidate，整理最终报告和证据索引。只有在获得 exact w-RMSE、
MovieLens split/mask、DU-Net backend 细节或 paper Linear UCB protocol 时，
才按 source-gated protocol 从 seeds `0,1,2` 重新开启最小诊断。

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8-sig")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    rows = build_rows(ctx)
    rows.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    write_report(ctx, rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
