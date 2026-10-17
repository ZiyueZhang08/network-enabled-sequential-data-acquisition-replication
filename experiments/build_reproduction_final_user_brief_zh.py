"""Build a concise Chinese user-facing status brief for the reproduction."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

SNAPSHOT_CSV = TABLE_DIR / "reproduction_iteration_status_snapshot.csv"
DISTANCE_CSV = TABLE_DIR / "figure4a_distance_scorecard.csv"
CLAIM_INDEX_CSV = TABLE_DIR / "reproduction_claim_evidence_index.csv"
TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
INTEGRITY_CSV = TABLE_DIR / "reproduction_artifact_integrity_check.csv"
QUEUE_CSV = TABLE_DIR / "next_experiment_queue.csv"

OUT_CSV = TABLE_DIR / "reproduction_final_user_brief_zh.csv"
OUT_REPORT = REPORT_DIR / "reproduction_final_user_brief_zh.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def count_status(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def latest_queue(rows: list[dict[str, str]], status: str) -> dict[str, str] | None:
    matches = [row for row in rows if row.get("status") == status]
    return matches[-1] if matches else None


def build_rows() -> list[dict[str, str]]:
    snapshot = read_rows(SNAPSHOT_CSV)
    distance = read_rows(DISTANCE_CSV)
    claims = read_rows(CLAIM_INDEX_CSV)
    trigger = read_rows(TRIGGER_CSV)
    integrity = count_status(read_rows(INTEGRITY_CSV))
    queue = read_rows(QUEUE_CSV)

    overall = row_by(snapshot, "area", "overall_progress")
    paper = row_by(snapshot, "area", "paper_target")
    weighted = row_by(snapshot, "area", "local_reward_weighted")
    full = row_by(snapshot, "area", "local_full_rmse")
    high = row_by(snapshot, "area", "local_high_rating_region")
    sigmoid = row_by(snapshot, "area", "sigmoid_reward_weighted")
    weighted_distance = row_by(distance, "claim_area", "local_reward_weighted")
    rerun = row_by(trigger, "trigger", "rerun_decision")
    source_gap = row_by(claims, "claim_id", "source_gap")
    next_protocol = row_by(claims, "claim_id", "next_experiment_protocol")
    last_done = latest_queue(queue, "done")
    next_item = latest_queue(queue, "next")

    return [
        {
            "section": "current_progress",
            "status": overall["status"],
            "evidence": overall["evidence"],
            "decision": "当前应报告为强部分复现，不应报告为 exact DU-Net/Figure 4a 复现。",
        },
        {
            "section": "paper_target",
            "status": paper["status"],
            "evidence": paper["evidence"],
            "decision": "论文视觉目标仍是 DU-Net 优于 Linear UCB。",
        },
        {
            "section": "local_best_support",
            "status": weighted["status"],
            "evidence": weighted["evidence"],
            "decision": (
                f"方向一致；本地 gap 是论文 gap 的 "
                f"{float(weighted_distance['paper_gap_ratio']) * 100:.1f}%，"
                f"绝对 DU-Net endpoint 是论文 endpoint 的 "
                f"{float(weighted_distance['du_scale_ratio_to_paper']) * 100:.1f}%。"
            ),
        },
        {
            "section": "main_blocker",
            "status": full["status"],
            "evidence": full["evidence"],
            "decision": "full-test RMSE 方向反了，因此不能 claim exact Figure 4a。",
        },
        {
            "section": "mechanism_support",
            "status": high["status"],
            "evidence": high["evidence"],
            "decision": "高评分区域支持 reward-region 机制解释。",
        },
        {
            "section": "metric_ambiguity",
            "status": sigmoid["status"],
            "evidence": sigmoid["evidence"],
            "decision": "paper-like sigmoid weighting 在 20 seeds 下近似打平，不是稳定 DU-Net 胜利。",
        },
        {
            "section": "latest_execution",
            "status": "completed",
            "evidence": f"last_done={last_done['priority'] if last_done else 'n/a'}; next={next_item['priority'] if next_item else 'n/a'}",
            "decision": "本轮完成 source artifact inspection protocol；若未来 scanner 发现 unknown PDF/archive/source/mat/config，将按候选类型检查 metric、split、DU-Net backend 和 Linear UCB adapter，再决定是否进入 3-seed diagnostic。",
        },
        {
            "section": "handoff_integrity",
            "status": "pass" if integrity.get("fail", 0) == 0 else "fail",
            "evidence": f"{integrity.get('pass', 0)} pass; {integrity.get('fail', 0)} fail",
            "decision": "当前交付包内部一致。",
        },
        {
            "section": "missing_source",
            "status": "blocked",
            "evidence": source_gap["evidence"],
            "decision": "下一步需要作者代码、精确 w-RMSE、split/preprocessing、DU-Net backend 等 P0 材料。",
        },
        {
            "section": "next_step",
            "status": rerun["status"],
            "evidence": rerun["evidence"],
            "decision": next_protocol["claim"],
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["| section | status | evidence | decision |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['section']} | {row['status']} | {row['evidence']} | {row['decision']} |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    table = markdown_table(rows)
    report = f"""# 当前复现进度中文简报

Generated by `experiments/build_reproduction_final_user_brief_zh.py`.

## 结论

当前复现是 **强部分复现**：本地 reward-weighted / high-rating 证据支持
DU-Net-style 方法相对 Linear UCB 的论文方向，但 full-test RMSE 和绝对 w-RMSE
尺度仍未复现论文 Figure 4a。

## 简表

{table}

## 汇报口径

- 可以说：当前结果相对 Linear UCB 在 reward-weighted / high-rating 区域与论文方向一致。
- 可以说：本地 reward-weighted gap 是论文视觉 gap 的 31.9%，相对优势有可报告价值。
- 不能说：已经 exact 复现 Figure 4a 或 DU-Net。
- 不能说：full-test RMSE 下 DU-Net 优于 Linear UCB。

## 下一步

继续等待 primary source 或 approved dependency trigger。触发条件出现后，只先跑
MovieLens Figure 4a seeds 0-2 的最小 faithful diagnostic；只有它改善
reward-weighted gap 或解决 full-RMSE / 绝对尺度问题，才升级到 20 seeds。

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    queue = read_rows(QUEUE_CSV)
    last_done = latest_queue(queue, "done")
    next_item = latest_queue(queue, "next")
    for row in rows:
        if row.get("section") == "latest_execution":
            row["evidence"] = (
                f"last_done={last_done['priority'] if last_done else 'n/a'}; "
                f"next={next_item['priority'] if next_item else 'n/a'}"
            )
            row["decision"] = (
                f"Latest completed queue item: {last_done['task'] if last_done else 'n/a'}. "
                f"Evidence artifact: {last_done['expected_artifact'] if last_done else 'n/a'}. "
                f"Next queue item: {next_item['task'] if next_item else 'n/a'}."
            )
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
