"""Audit paper-specific benchmark definitions against current implementation.

This script is deliberately stricter than the implemented-baseline coverage
audit: it follows the benchmark taxonomy stated in paper Section 4.2.1 and
keeps unresolved paper-count slots visible instead of silently mapping them to
local convenience baselines.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import (  # noqa: E402
    IMPLEMENTED_EXTRA_BASELINE_POLICIES,
    PAPER_BENCHMARKS_EXPLICIT,
    PAPER_BENCHMARK_TARGET_COUNT,
    unresolved_paper_benchmark_slots,
)


TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

RESULT_TABLES = [
    ("20-seed", "movielens_100k_svdpp_baselines_20seeds_final_summary.csv"),
    ("20-seed", "movielens_100k_svdpp_baselines_missing_20seeds_final_summary.csv"),
    ("20-seed", "flixster_svdpp_baselines_20seeds_final_summary.csv"),
    ("20-seed", "douban_svdpp_baselines_netobs_20seeds_final_summary.csv"),
    ("3-seed", "flixster_svdpp_oful_3seeds_final_summary.csv"),
    ("3-seed", "flixster_svdpp_paper_bandit_repair_3seeds_final_summary.csv"),
    ("3-seed", "flixster_svdpp_mf_oful_beta_zoom_3seeds_final_summary.csv"),
    ("3-seed", "flixster_svdpp_paper_bandit_gap_3seeds_final_summary.csv"),
    ("3-seed", "flixster_svdpp_logistic_bandit_scale_3seeds_final_summary.csv"),
    ("3-seed", "movielens_100k_svdpp_bandit_upgrade_3seeds_final_summary.csv"),
    ("3-seed", "movielens_100k_svdpp_active_learning_baselines_3seeds_final_summary.csv"),
    ("3-seed", "movielens_100k_svdpp_social_rec_baselines_3seeds_final_summary.csv"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_result_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scope, table_name in RESULT_TABLES:
        for row in read_rows(TABLE_DIR / table_name):
            row = dict(row)
            row["_scope"] = scope
            row["_source_table"] = table_name
            rows.append(row)
    return rows


def metric(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def row_name(row: dict[str, str]) -> str:
    return row.get("variant") or row.get("policy") or ""


def evidence_for_policy(policy: str | None, result_rows: list[dict[str, str]]) -> dict[str, str]:
    if policy is None:
        return {
            "result_scope": "not implemented",
            "source_tables": "",
            "best_w_rmse_evidence": "",
            "best_reward_evidence": "",
        }
    matches = [row for row in result_rows if row.get("policy") == policy or row_name(row) == policy]
    if not matches:
        return {
            "result_scope": "implemented without listed result row",
            "source_tables": "",
            "best_w_rmse_evidence": "",
            "best_reward_evidence": "",
        }
    scopes = {row["_scope"] for row in matches}
    result_scope = "20-seed evidence" if "20-seed" in scopes else "3-seed evidence only"
    evidence_matches = [row for row in matches if row["_scope"] == "20-seed"] or matches
    source_tables = ", ".join(sorted({row["_source_table"] for row in matches}))
    best_w = min(evidence_matches, key=lambda row: metric(row, "w_rmse"))
    best_reward = max(evidence_matches, key=lambda row: metric(row, "cumulative_reward"))
    best_w_text = (
        f"{row_name(best_w)} in {best_w['_source_table']} "
        f"w_rmse={metric(best_w, 'w_rmse'):.4f}"
    )
    best_reward_text = (
        f"{row_name(best_reward)} in {best_reward['_source_table']} "
        f"reward={metric(best_reward, 'cumulative_reward'):.2f}"
    )
    return {
        "result_scope": result_scope,
        "source_tables": source_tables,
        "best_w_rmse_evidence": best_w_text,
        "best_reward_evidence": best_reward_text,
    }


def next_action(status: str, paper_name: str) -> str:
    if paper_name == "OFUL" and status == "implemented":
        return "Keep as 3-seed implementation evidence; repair bandit adaptation before 20-seed promotion."
    if status == "implemented":
        return "Keep frozen as a covered benchmark row."
    if status == "proxy":
        return "Use as interim evidence only; replace or relabel before paper-level benchmark claims."
    if paper_name == "OFUL":
        return "Implement first: paper says OFUL is the best acquisition-reward comparator on Flixster."
    if "Logistic" in paper_name:
        return "Implement after OFUL if binary/response-transformed reward modeling is needed."
    if paper_name in {"Epsilon Greedy", "Adaptive Greedy", "Explore-Then-Exploit"}:
        return "Implement as finite-arm stress baselines only after OFUL/logistic rows are settled."
    return "Add a paper-faithful implementation or document why the benchmark is non-applicable."


def audit_rows() -> list[dict[str, str]]:
    result_rows = load_result_rows()
    rows: list[dict[str, str]] = []
    for idx, spec in enumerate(PAPER_BENCHMARKS_EXPLICIT, start=1):
        evidence = evidence_for_policy(spec.current_policy, result_rows)
        rows.append(
            {
                "paper_slot": str(idx),
                "paper_family": spec.family,
                "paper_name": spec.paper_name,
                "paper_target_count": str(PAPER_BENCHMARK_TARGET_COUNT),
                "current_policy": spec.current_policy or "",
                "implementation_status": spec.implementation_status,
                "result_scope": evidence["result_scope"],
                "source_tables": evidence["source_tables"],
                "best_w_rmse_evidence": evidence["best_w_rmse_evidence"],
                "best_reward_evidence": evidence["best_reward_evidence"],
                "note": spec.note,
                "next_action": next_action(spec.implementation_status, spec.paper_name),
            }
        )
    for offset in range(unresolved_paper_benchmark_slots()):
        slot = len(PAPER_BENCHMARKS_EXPLICIT) + offset + 1
        rows.append(
            {
                "paper_slot": str(slot),
                "paper_family": "unresolved from local paper text",
                "paper_name": f"Unresolved paper benchmark slot {offset + 1}",
                "paper_target_count": str(PAPER_BENCHMARK_TARGET_COUNT),
                "current_policy": "",
                "implementation_status": "unresolved",
                "result_scope": "not implemented",
                "source_tables": "",
                "best_w_rmse_evidence": "",
                "best_reward_evidence": "",
                "note": "Paper Section 4.2.1 states 16 benchmarks, but the local full-text, footnote, and public source audit identifies 14 confirmed benchmark slots; see reports/paper_benchmark_slot_recovery_audit.md and reports/author_bandit_source_comparison.md.",
                "next_action": "Keep unresolved unless original PDF/supplement/author code confirms a name; final reporting should use the frozen 14+2 benchmark caveat.",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row["implementation_status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_report(rows: list[dict[str, str]], path: Path, csv_path: Path) -> None:
    counts = status_counts(rows)
    explicit = len(PAPER_BENCHMARKS_EXPLICIT)
    unresolved = unresolved_paper_benchmark_slots()
    covered_or_proxy = counts.get("implemented", 0) + counts.get("proxy", 0)
    missing_total = counts.get("missing", 0) + counts.get("unresolved", 0)
    lines = [
        "# Paper Benchmark Definition Audit",
        "",
        "Generated by `experiments/build_paper_benchmark_definition_audit.py`.",
        "",
        "## Summary",
        "",
        f"- Paper benchmark target: {PAPER_BENCHMARK_TARGET_COUNT}.",
        f"- Explicitly named benchmark slots recovered from local paper Section 4.2.1: {explicit}.",
        f"- Unresolved slots from the paper's 16-count claim: {unresolved}.",
        f"- Covered or proxy-covered explicit slots: {covered_or_proxy}.",
        f"- Missing or unresolved slots: {missing_total}.",
        "",
        "This means the current project should still be reported as a strong partial reproduction, not a full paper benchmark reproduction.",
        "",
        "## Paper Slots",
        "",
        "| slot | family | paper benchmark | status | current policy | result evidence | next action |",
        "|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = row["result_scope"]
        if row["best_w_rmse_evidence"]:
            evidence += f"; {row['best_w_rmse_evidence']}"
        lines.append(
            f"| {row['paper_slot']} | {row['paper_family']} | {row['paper_name']} | "
            f"{row['implementation_status']} | {row['current_policy'] or '-'} | "
            f"{evidence} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Current Extra Rows",
            "",
            "The following implemented policies are useful diagnostics or stronger local comparators, but they are not explicit replacements for missing paper benchmark names:",
            "",
            ", ".join(f"`{policy}`" for policy in IMPLEMENTED_EXTRA_BASELINE_POLICIES) + ".",
            "",
            "## Interpretation",
            "",
            "- The strongest immediate gap is not another entropy or embedding sweep; it is paper-faithful benchmark definition and author-code comparability.",
            "- OFUL is implemented and screened, but the current adaptation is weak; MF-centered repair helps and still does not reach the Flixster frontier.",
            "- Logistic UCB/Thompson and finite-arm policies are now covered as clearly labeled proxy rows; they close catalog coverage but not paper-faithful benchmark strength.",
            "- A focused local source audit still finds only 14 explicit names; public footnote-code inspection makes active-learning exploration and SoftMax chooser plausible count-mismatch candidates, but does not confirm them as paper benchmarks.",
            "- The benchmark claim is now frozen as 14 explicit slots covered/proxy-covered plus 2 unresolved paper-count slots unless original PDF, supplement, or author code becomes available.",
            "- Slot recovery audit: `reports/paper_benchmark_slot_recovery_audit.md`.",
            "- Author bandit source comparison: `reports/author_bandit_source_comparison.md`.",
            "- Frozen benchmark claim: `reports/final_benchmark_claim_freeze.md`.",
            "",
            f"CSV artifact: `{csv_path.relative_to(ROOT)}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = audit_rows()
    csv_path = TABLE_DIR / "paper_benchmark_definition_audit.csv"
    report_path = REPORT_DIR / "paper_benchmark_definition_audit.md"
    write_csv(rows, csv_path)
    write_report(rows, report_path, csv_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
