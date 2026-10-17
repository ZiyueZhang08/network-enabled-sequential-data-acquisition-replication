"""Build a Figure 4a evidence appendix for report writing."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
CLAIM_FREEZE = TABLE_DIR / "figure4a_claim_freeze.csv"
NO_SOURCE_CLOSURE = TABLE_DIR / "figure4a_no_author_source_closure.csv"

OUT_CSV = TABLE_DIR / "figure4a_evidence_appendix_zh.csv"
OUT_REPORT = REPORT_DIR / "figure4a_evidence_appendix_zh.md"


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def build_rows() -> list[dict[str, str]]:
    final = pd.read_csv(FINAL_SUMMARY)
    claims = pd.read_csv(CLAIM_FREEZE)
    closure = read_rows(NO_SOURCE_CLOSURE)

    du = final[final["variant"] == "du_net"].iloc[0]
    linear = final[final["variant"] == "mf_linear_ucb_a002"].iloc[0]
    myopic = final[final["variant"] == "myopic"].iloc[0]
    gate = row_by(closure, "item", "rerun_gate")
    no_source = row_by(closure, "item", "author_source_status")

    local_gap = float(linear.local_reward_weighted_w_rmse) - float(du.local_reward_weighted_w_rmse)
    rmse_gap = float(linear.rmse) - float(du.rmse)
    myopic_gap = float(myopic.local_reward_weighted_w_rmse) - float(du.local_reward_weighted_w_rmse)

    allowed = claims[claims["claim_type"] == "allowed"].reset_index(drop=True)
    forbidden = claims[claims["claim_type"] == "forbidden"].reset_index(drop=True)

    rows = [
        {
            "claim_level": "primary_allowed",
            "report_claim": "本地 DU-Net 在 local reward-weighted w-RMSE 下相对 MF-centered Linear UCB proxy 有方向性优势。",
            "numeric_evidence": (
                f"DU-Net={fmt(du.local_reward_weighted_w_rmse)}, "
                f"Linear UCB proxy={fmt(linear.local_reward_weighted_w_rmse)}, "
                f"Linear-minus-DU={fmt(local_gap)}"
            ),
            "artifact_evidence": "reports/figure4a_final_package.md; outputs/tables/figure4a_final_package_summary.csv",
            "required_caveat": "必须称 comparator 为 MF-centered Linear UCB proxy，不得称为论文 exact Linear UCB。",
        },
        {
            "claim_level": "primary_allowed",
            "report_claim": "该结果是 partial/caveated reproduction，不是 exact Figure 4a replication。",
            "numeric_evidence": "Linear-minus-DU gap=0.0083，低于 +0.010 次级阈值，约为论文视觉 gap 的 31.9%。",
            "artifact_evidence": "reports/figure4a_no_author_source_closure.md; reports/figure4a_claim_freeze.md",
            "required_caveat": "必须报告失败阈值和 paper-gap share。",
        },
        {
            "claim_level": "counter_evidence",
            "report_claim": "conventional full-test RMSE 仍然支持 Linear UCB proxy，而不是 DU-Net。",
            "numeric_evidence": (
                f"DU-Net RMSE={fmt(du.rmse)}, Linear UCB proxy RMSE={fmt(linear.rmse)}, "
                f"Linear-minus-DU RMSE gap={fmt(rmse_gap)}"
            ),
            "artifact_evidence": "reports/figure4a_final_package.md",
            "required_caveat": "必须与 reward-weighted 指标并列呈现，不能只呈现有利指标。",
        },
        {
            "claim_level": "context",
            "report_claim": "Myopic 是 sanity comparator；DU-Net 也只是在本地 reward-weighted proxy 下略优。",
            "numeric_evidence": f"Myopic={fmt(myopic.local_reward_weighted_w_rmse)}, Myopic-minus-DU={fmt(myopic_gap)}",
            "artifact_evidence": "outputs/tables/figure4a_final_package_summary.csv",
            "required_caveat": "不要把小幅差距包装成强优势。",
        },
        {
            "claim_level": "stop_rule",
            "report_claim": "在没有作者源码的条件下，不再执行新的模型实验。",
            "numeric_evidence": f"rerun_gate={gate['status']}; author_source_status={no_source['status']}",
            "artifact_evidence": "reports/figure4a_no_author_source_closure.md; reports/source_trigger_intake_check.md",
            "required_caveat": "只允许非源码精确信息触发 seeds 0,1,2 最小诊断。",
        },
    ]

    for row in allowed.itertuples(index=False):
        rows.append(
            {
                "claim_level": "frozen_allowed_claim",
                "report_claim": row.claim,
                "numeric_evidence": row.evidence,
                "artifact_evidence": "reports/figure4a_claim_freeze.md",
                "required_caveat": row.condition,
            }
        )
    for row in forbidden.itertuples(index=False):
        rows.append(
            {
                "claim_level": "frozen_forbidden_claim",
                "report_claim": row.claim,
                "numeric_evidence": row.evidence,
                "artifact_evidence": "reports/figure4a_claim_freeze.md",
                "required_caveat": row.condition,
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| level | claim | numeric evidence | artifact evidence | required caveat |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['claim_level']} | {row['report_claim']} | {row['numeric_evidence']} | "
            f"`{row['artifact_evidence']}` | {row['required_caveat']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Figure 4a 证据附录

Generated by `experiments/build_figure4a_evidence_appendix_zh.py`.

## 用途

这份附录把 Figure 4a 最终可写入报告的每个结论绑定到数值证据、文件证据和必须保留的限定条件。它的目标不是新增实验结果，而是防止最终写作时把 partial reproduction 写成 exact replication。

## Evidence Matrix

{markdown_table(rows)}

## 写作建议

最终报告应采用三段式结构：

1. 先给出 20-seed local reward-weighted 结果和 `+0.0083` gap。
2. 随即说明该 gap 低于 `+0.010` 阈值，且 conventional RMSE 反向。
3. 最后说明作者源码不可用，因此 exact Figure 4a replication 在公开材料条件下不可达；后续仅可由精确 metric/split/protocol 信息触发。

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8-sig")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
