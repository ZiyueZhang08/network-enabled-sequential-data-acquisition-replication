"""Build a readable Chinese delivery package for the reproduction."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FIGURE4A = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
HANDOFF = TABLE_DIR / "final_reproduction_handoff.csv"
DISTANCE = TABLE_DIR / "figure4a_distance_scorecard.csv"
INTEGRITY = TABLE_DIR / "reproduction_artifact_integrity_check.csv"
TRIGGER = TABLE_DIR / "paper_faithful_trigger_status.csv"

OUT_MANIFEST = TABLE_DIR / "reproduction_delivery_manifest.csv"
OUT_REPORT = REPORT_DIR / "reproduction_delivery_package_zh.md"


ARTIFACTS = [
    ("paper_pdf_evidence_audit", "PDF evidence audit for Figure 4a blockers", REPORT_DIR / "paper_pdf_evidence_audit.md"),
    ("paper_pdf_evidence_audit_csv", "PDF evidence audit table for Figure 4a blockers", TABLE_DIR / "paper_pdf_evidence_audit.csv"),
    ("paper_method_detail_keyword_audit", "PDF keyword-context audit for method detail blockers", REPORT_DIR / "paper_method_detail_keyword_audit.md"),
    ("paper_method_detail_keyword_audit_csv", "PDF keyword-context audit table for method detail blockers", TABLE_DIR / "paper_method_detail_keyword_audit.csv"),
    ("paper_url_dependency_audit", "PDF embedded URL and dependency audit", REPORT_DIR / "paper_url_dependency_audit.md"),
    ("paper_url_dependency_audit_csv", "PDF embedded URL and dependency audit table", TABLE_DIR / "paper_url_dependency_audit.csv"),
    ("local_source_artifact_inventory_audit", "Local source artifact inventory audit", REPORT_DIR / "local_source_artifact_inventory_audit.md"),
    ("local_source_artifact_inventory_audit_csv", "Local source artifact inventory audit table", TABLE_DIR / "local_source_artifact_inventory_audit.csv"),
    ("movielens_protocol_alignment_audit", "MovieLens data/protocol alignment audit", REPORT_DIR / "movielens_protocol_alignment_audit.md"),
    ("movielens_protocol_alignment_audit_csv", "MovieLens data/protocol alignment audit table", TABLE_DIR / "movielens_protocol_alignment_audit.csv"),
    ("movielens_split_overlap_audit", "MovieLens random-vs-official split overlap audit", REPORT_DIR / "movielens_split_overlap_audit.md"),
    ("movielens_split_overlap_audit_csv", "MovieLens random-vs-official split overlap table", TABLE_DIR / "movielens_split_overlap_audit.csv"),
    ("gcmc_monti_split_provenance_audit", "GCMC/Monti split provenance audit", REPORT_DIR / "gcmc_monti_split_provenance_audit.md"),
    ("gcmc_monti_split_provenance_audit_csv", "GCMC/Monti split provenance audit table", TABLE_DIR / "gcmc_monti_split_provenance_audit.csv"),
    ("movielens_official_u1_sensitivity", "MovieLens official u1 fold sensitivity diagnostic", REPORT_DIR / "movielens_100k_svdpp_official_u1_sensitivity_3seeds_report.md"),
    ("movielens_official_u1_sensitivity_summary", "MovieLens official u1 fold final summary", TABLE_DIR / "movielens_100k_svdpp_official_u1_sensitivity_3seeds_final_summary.csv"),
    ("movielens_official_u1_sensitivity_gaps", "MovieLens official u1 fold gap table", TABLE_DIR / "movielens_100k_svdpp_official_u1_sensitivity_3seeds_gaps.csv"),
    ("movielens_official_u1_sensitivity_metrics", "MovieLens official u1 fold trajectory metrics", TABLE_DIR / "movielens_100k_svdpp_official_u1_sensitivity_3seeds_metrics.csv"),
    ("movielens_official_folds_sensitivity", "MovieLens official u1-u5 folds sensitivity diagnostic", REPORT_DIR / "movielens_100k_svdpp_official_folds_sensitivity_report.md"),
    ("movielens_official_folds_sensitivity_summary", "MovieLens official u1-u5 folds final summary", TABLE_DIR / "movielens_100k_svdpp_official_folds_sensitivity_final_summary.csv"),
    ("movielens_official_folds_sensitivity_gaps", "MovieLens official u1-u5 folds gap table", TABLE_DIR / "movielens_100k_svdpp_official_folds_sensitivity_fold_gaps.csv"),
    ("movielens_official_folds_sensitivity_metrics", "MovieLens official u1-u5 folds trajectory metrics", TABLE_DIR / "movielens_100k_svdpp_official_folds_sensitivity_metrics.csv"),
    ("final_handoff", "全项目最终状态", REPORT_DIR / "final_reproduction_handoff.md"),
    ("figure4a_delivery", "Figure 4a 可报告数字和 caveat", REPORT_DIR / "final_figure4a_caveated_delivery.md"),
    ("figure4a_target", "DU-Net 对 Linear UCB 的最终目标比较", REPORT_DIR / "figure4a_final_target_report.md"),
    ("figure4a_user_goal_comparison_brief", "Figure 4a user-goal comparison brief", REPORT_DIR / "figure4a_user_goal_comparison_brief.md"),
    ("figure4a_user_goal_comparison_brief_csv", "Figure 4a user-goal comparison brief table", TABLE_DIR / "figure4a_user_goal_comparison_brief.csv"),
    ("figure4a_metric_formula_visual_audit", "Figure 4a rendered PDF metric formula audit", REPORT_DIR / "figure4a_metric_formula_visual_audit.md"),
    ("figure4a_metric_formula_visual_audit_csv", "Figure 4a rendered PDF metric formula audit table", TABLE_DIR / "figure4a_metric_formula_visual_audit.csv"),
    ("figure4a_primary_source", "本地 PDF 主来源恢复", REPORT_DIR / "figure4a_primary_source_recovery.md"),
    ("figure4a_external_source", "外部 source/supplement 恢复检查", REPORT_DIR / "figure4a_external_source_recovery.md"),
    ("public_source_rescan_audit", "Latest public source/supplement rescan audit", REPORT_DIR / "public_source_rescan_audit.md"),
    ("public_source_rescan_audit_csv", "Latest public source/supplement rescan audit table", TABLE_DIR / "public_source_rescan_audit.csv"),
    ("trigger_status", "paper-faithful 复跑触发门控", REPORT_DIR / "paper_faithful_trigger_status.md"),
    ("paper_quality", "三数据集最终 paper-quality 综合", REPORT_DIR / "final_paper_quality_results.md"),
    ("benchmark_caveat", "14+2 benchmark caveat", REPORT_DIR / "final_benchmark_claim_freeze.md"),
    ("progress_plan", "当前进度和下一步计划", REPORT_DIR / "reproduction_progress_next_execution.md"),
    ("iteration_snapshot", "最新迭代状态快照", REPORT_DIR / "reproduction_iteration_status_snapshot.md"),
    ("figure4a_distance", "Figure 4a 复现距离 scorecard", REPORT_DIR / "figure4a_distance_scorecard.md"),
    ("rerun_readiness", "paper-faithful 复跑 readiness checklist", REPORT_DIR / "paper_faithful_rerun_readiness.md"),
    ("paper_faithful_rerun_playbook", "Paper-faithful rerun playbook", REPORT_DIR / "paper_faithful_rerun_playbook.md"),
    ("paper_faithful_rerun_playbook_csv", "Paper-faithful rerun playbook table", TABLE_DIR / "paper_faithful_rerun_playbook.csv"),
    ("source_artifact_inspection_protocol", "Source artifact inspection protocol", REPORT_DIR / "source_artifact_inspection_protocol.md"),
    ("source_artifact_inspection_protocol_csv", "Source artifact inspection protocol table", TABLE_DIR / "source_artifact_inspection_protocol.csv"),
    ("source_trigger_monitor_snapshot", "Source trigger monitor snapshot", REPORT_DIR / "source_trigger_monitor_snapshot.md"),
    ("source_trigger_monitor_snapshot_csv", "Source trigger monitor snapshot table", TABLE_DIR / "source_trigger_monitor_snapshot.csv"),
    ("source_trigger_intake_check", "One-command source trigger intake check", REPORT_DIR / "source_trigger_intake_check.md"),
    ("source_trigger_intake_check_csv", "One-command source trigger intake check table", TABLE_DIR / "source_trigger_intake_check.csv"),
    ("source_request_packet", "paper-faithful 缺失来源请求清单", REPORT_DIR / "paper_source_request_packet.md"),
    ("author_source_request_email", "Author source request email", REPORT_DIR / "author_source_request_email.md"),
    ("author_source_request_email_csv", "Author source request email table", TABLE_DIR / "author_source_request_email.csv"),
    ("remaining_gap_priority_audit", "Remaining exact-reproduction gap priority audit", REPORT_DIR / "remaining_gap_priority_audit.md"),
    ("remaining_gap_priority_audit_csv", "Remaining exact-reproduction gap priority table", TABLE_DIR / "remaining_gap_priority_audit.csv"),
    ("minimal_diagnostic_protocol", "paper-faithful 最小 3-seed 诊断 protocol", REPORT_DIR / "minimal_faithful_diagnostic_protocol.md"),
    ("continuation_audit", "继续执行决策审计", REPORT_DIR / "reproduction_continuation_audit.md"),
    ("execution_gate_closure_audit", "Execution gate closure audit", REPORT_DIR / "execution_gate_closure_audit.md"),
    ("claim_evidence_index", "复现 claim-evidence 索引", REPORT_DIR / "reproduction_claim_evidence_index.md"),
    ("final_user_brief_zh", "当前复现进度中文简报", REPORT_DIR / "reproduction_final_user_brief_zh.md"),
    ("figure4a_delivery_csv", "Figure 4a 交付表", TABLE_DIR / "final_figure4a_caveated_delivery.csv"),
    ("handoff_csv", "全项目 handoff 表", TABLE_DIR / "final_reproduction_handoff.csv"),
    ("trigger_status_csv", "触发门控表", TABLE_DIR / "paper_faithful_trigger_status.csv"),
    ("iteration_snapshot_csv", "最新迭代状态快照表", TABLE_DIR / "reproduction_iteration_status_snapshot.csv"),
    ("figure4a_distance_csv", "Figure 4a 复现距离表", TABLE_DIR / "figure4a_distance_scorecard.csv"),
    ("rerun_readiness_csv", "paper-faithful 复跑 readiness 表", TABLE_DIR / "paper_faithful_rerun_readiness.csv"),
    ("source_request_packet_csv", "paper-faithful 缺失来源请求表", TABLE_DIR / "paper_source_request_packet.csv"),
    ("minimal_diagnostic_protocol_csv", "paper-faithful 最小 3-seed 诊断 protocol 表", TABLE_DIR / "minimal_faithful_diagnostic_protocol.csv"),
    ("continuation_audit_csv", "继续执行决策审计表", TABLE_DIR / "reproduction_continuation_audit.csv"),
    ("execution_gate_closure_audit_csv", "Execution gate closure audit table", TABLE_DIR / "execution_gate_closure_audit.csv"),
    ("claim_evidence_index_csv", "复现 claim-evidence 索引表", TABLE_DIR / "reproduction_claim_evidence_index.csv"),
    ("final_user_brief_zh_csv", "当前复现进度中文简报表", TABLE_DIR / "reproduction_final_user_brief_zh.csv"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for item in rows:
        if item[key] == value:
            return item
    raise KeyError((key, value))


def manifest_rows() -> list[dict[str, str]]:
    rows = []
    for artifact_id, role, path in ARTIFACTS:
        exists = path.exists()
        rows.append(
            {
                "artifact_id": artifact_id,
                "role": role,
                "path": str(path.relative_to(ROOT)),
                "exists": "yes" if exists else "no",
                "size_bytes": str(path.stat().st_size if exists else 0),
            }
        )
    return rows


def write_manifest(rows: list[dict[str, str]]) -> None:
    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_artifact_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| artifact | role | exists | path |",
        "|---|---|---|---|",
    ]
    for item in rows:
        lines.append(f"| {item['artifact_id']} | {item['role']} | {item['exists']} | `{item['path']}` |")
    return "\n".join(lines)


def status_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in read_rows(path):
        status = item["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_report(manifest: list[dict[str, str]]) -> None:
    figure = read_rows(FIGURE4A)
    handoff = read_rows(HANDOFF)
    distance = read_rows(DISTANCE)
    trigger = read_rows(TRIGGER)
    integrity_counts = status_counts(INTEGRITY)

    paper = row(figure, "claim_area", "paper_target")
    weighted = row(figure, "claim_area", "local_reward_weighted")
    full = row(figure, "claim_area", "local_full_rmse")
    high = row(figure, "claim_area", "local_high_rating_region")
    low = row(figure, "claim_area", "local_low_rating_region")
    sigmoid = row(figure, "claim_area", "sigmoid_reward_weighted")

    weighted_distance = row(distance, "claim_area", "local_reward_weighted")
    full_distance = row(distance, "claim_area", "local_full_rmse")
    high_distance = row(distance, "claim_area", "local_high_rating_region")

    flixster = row(handoff, "section", "flixster")
    douban = row(handoff, "section", "douban")
    movielens = row(handoff, "section", "movielens")
    rerun = row(trigger, "trigger", "rerun_decision")

    report = f"""# 复现交付包

Generated by `experiments/build_reproduction_delivery_package.py`.

## 一句话结论

当前结果是 **strong partial reproduction（强部分复现）**，不是 exact DU-Net replication。
最稳妥的表述是：本地 DU-Net-style 方法在 Figure 4a 的 reward-weighted / high-rating
评价区域相对 Linear UCB 的方向与论文一致，但没有复现论文 Figure 4a 的绝对 w-RMSE
尺度，也没有在 full-test conventional RMSE 下复现 DU-Net 优于 Linear UCB 的排序。

## Figure 4a 进度和距离

| 比较 | DU-Net | Linear UCB | Linear - DU | 判断 |
|---|---:|---:|---:|---|
| 论文 Figure 4a 视觉估计 | {paper['du_net']} | {paper['linear_ucb']} | {paper['linear_minus_du']} | 论文方向：DU-Net 更好 |
| 本地 reward-weighted w-RMSE | {weighted['du_net']} | {weighted['linear_ucb']} | {weighted['linear_minus_du']} | 方向一致，但尺度不一致 |
| 本地 full RMSE | {full['du_net']} | {full['linear_ucb']} | {full['linear_minus_du']} | 方向不一致，Linear UCB 略好 |
| 本地 high-rating RMSE | {high['du_net']} | {high['linear_ucb']} | {high['linear_minus_du']} | 机制支持 DU-Net |
| 本地 low-rating RMSE | {low['du_net']} | {low['linear_ucb']} | {low['linear_minus_du']} | 解释 full RMSE 反转 |
| sigmoid reward-weighted RMSE | {sigmoid['du_net']} | {sigmoid['linear_ucb']} | {sigmoid['linear_minus_du']} | 20 seeds 下近似打平 |

距离论文的核心事实：

- 本地 reward-weighted gap 是论文视觉 gap 的 `{float(weighted_distance['paper_gap_ratio']) * 100:.1f}%`。
- 本地 reward-weighted 的相对 DU-Net 优势是论文相对优势的 `{float(weighted_distance['relative_improvement_vs_paper']) * 100:.1f}%`。
- 本地 reward-weighted DU-Net endpoint 只有论文视觉 DU-Net endpoint 的 `{float(weighted_distance['du_scale_ratio_to_paper']) * 100:.1f}%`。
- full-test RMSE 的 gap 是 `{float(full_distance['linear_minus_du']):.4f}`，排序与论文相反。
- high-rating 区域 gap 是论文 gap 的 `{float(high_distance['paper_gap_ratio']) * 100:.1f}%`，支持 reward-region 机制解释。

## 全项目状态

| 数据集 / 分支 | 当前状态 | 解释 |
|---|---|---|
| MovieLens | {movielens['status']} | {movielens['key_result']} |
| Flixster | {flixster['status']} | {flixster['key_result']} |
| Douban | {douban['status']} | {douban['key_result']} |
| Benchmark | 14+2 caveat | 14 个论文显式 benchmark slots 已实现或 proxy-covered；2 个 paper-count slots 仍未能从主来源确认。 |
| Artifact integrity | {integrity_counts.get('pass', 0)} pass / {integrity_counts.get('fail', 0)} fail | 交付包和最新补充报告内部一致。 |
| Rerun gate | {rerun['status']} | {rerun['decision']} |
| Rerun readiness | not ready | 触发条件出现前不复跑；触发后先做最小 MovieLens Figure 4a 3-seed diagnostic。 |
| Source request | prepared | 已列出作者代码、精确 w-RMSE、split/config、DU-Net backend、Linear UCB adapter 等缺失项和验收规则。 |
| Minimal diagnostic protocol | defined | 触发后只先跑 MovieLens Figure 4a seeds 0-2；满足 promotion rule 后才升级 20 seeds。 |
| Continuation audit | do not run experiment | 当前证据不支持继续本地 proxy 复跑；下一步仍是等待 primary source 或 approved dependency trigger。 |
| Claim-evidence index | prepared | 当前可报告 claim、不可报告 claim、source gap 和下一步 protocol 已映射到证据文件。 |
| Final user brief | prepared | 已生成直接回答当前进度、执行结果、分析和下一步的中文简报。 |

## 可以说什么

- 可以说：这是强部分复现。
- 可以说：Figure 4a 相对 Linear UCB 的方向，在本地 reward-weighted / high-rating 区域一致。
- 可以说：Flixster 是最干净的本地结果，同时匹配 reward 和 w-RMSE 方向。
- 可以说：Douban 有正向 diagnosed candidate，但不是 exact DU-Net。

## 不要说什么

- 不要说 exact DU-Net replication。
- 不要说 exact Figure 4a curve 或 exact w-RMSE scale 已复现。
- 不要说 full-test RMSE 下 DU-Net 优于 Linear UCB。
- 不要把两个 unresolved benchmark labels 从依赖库中反推出来。

## 交付入口

{markdown_artifact_table(manifest)}

## 下一步

当前不应继续本地 Figure 4a proxy sweep。下一步只在以下触发条件出现时执行：

1. 作者代码、supplement、split files 或 exact configs；
2. Figure 4a w-RMSE 的精确定义；
3. 明确批准引入 validated external mechanism dependency，并先跑最小 3-seed faithful diagnostic。

Manifest artifact: `{OUT_MANIFEST.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = manifest_rows()
    write_manifest(manifest)
    write_report(manifest)
    print(f"Wrote {OUT_MANIFEST}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
