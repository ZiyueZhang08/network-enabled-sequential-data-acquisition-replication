"""Record whether paper-faithful rerun triggers are currently available."""

from __future__ import annotations

import csv
from pathlib import Path

from build_local_source_artifact_inventory_audit import build_rows as build_inventory_rows


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
OUT_REPORT = REPORT_DIR / "paper_faithful_trigger_status.md"


def actionable_inventory_rows() -> list[dict[str, str]]:
    return [
        row
        for row in build_inventory_rows()
        if row["artifact"] != "execution_decision" and row["actionable_trigger_candidate"] == "yes"
    ]


def inventory_evidence(rows: list[dict[str, str]]) -> str:
    if rows:
        return "; ".join(f"{row['artifact']} ({row['classification']})" for row in rows[:10])
    return "Inventory-aware scan found no new actionable PDF/archive/source/config candidate."


def split_or_config_candidate_count(rows: list[dict[str, str]]) -> int:
    return sum(
        1
        for row in rows
        if row["classification"] in {"unknown_mat_candidate", "unknown_config_candidate"}
        or any(term in row["artifact"].lower() for term in ("split", "config"))
    )


def build_rows() -> list[dict[str, str]]:
    inventory_candidates = actionable_inventory_rows()
    local_status = "inspection_required" if inventory_candidates else "not_triggered"
    local_evidence = inventory_evidence(inventory_candidates)
    split_status = "inspection_required" if split_or_config_candidate_count(inventory_candidates) else "not_triggered"
    rerun_status = "inspection_required" if inventory_candidates else "blocked_by_missing_source"
    rerun_evidence = (
        "Actionable local source-like artifact(s) require inspection before rerun."
        if inventory_candidates
        else "All rerun triggers are currently absent."
    )
    return [
        {
            "trigger": "author_code_or_supplement",
            "status": local_status,
            "evidence": local_evidence,
            "decision": "Inspect actionable local artifacts before deciding whether a faithful rerun is unlocked.",
        },
        {
            "trigger": "exact_figure4a_metric_definition",
            "status": "not_triggered",
            "evidence": "Local PDF and extracted text still do not disambiguate w-RMSE weighting/normalization.",
            "decision": "Keep the Figure 4a claim caveated.",
        },
        {
            "trigger": "exact_split_or_config_files",
            "status": split_status,
            "evidence": (
                "Actionable split/config-like local artifact requires inspection."
                if split_status == "inspection_required"
                else "No author split/config files are present beyond local reproduction configs."
            ),
            "decision": "Do not rerun 20 seeds under a guessed split.",
        },
        {
            "trigger": "validated_external_mechanism_dependency",
            "status": "not_triggered",
            "evidence": "No explicit approval has been given to add a validated external Node2Vec/ELPH/DLS dependency.",
            "decision": "Keep dependency-free proxy results frozen.",
        },
        {
            "trigger": "public_web_recovery",
            "status": "not_triggered",
            "evidence": "Exact title/DU-Net/author web queries did not recover a public DU-Net author repository or supplement in this check.",
            "decision": "Treat external source recovery as unavailable for now.",
        },
        {
            "trigger": "local_source_inventory",
            "status": "clear" if not inventory_candidates else "inspection_required",
            "evidence": local_evidence,
            "decision": "The trigger gate now uses the local source artifact inventory, including PDFs and archives.",
        },
        {
            "trigger": "rerun_decision",
            "status": rerun_status,
            "evidence": rerun_evidence,
            "decision": (
                "Inspect candidate artifacts before rerun."
                if rerun_status == "inspection_required"
                else "No further local proxy sweep; wait for primary source or approved faithful dependency."
            ),
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| trigger | status | evidence | decision |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trigger']} | {row['status']} | {row['evidence']} | {row['decision']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Paper-Faithful Trigger Status

Generated by `experiments/build_paper_faithful_trigger_status.py`.

## Current Progress

The reproduction remains at a stable strong-partial state. Figure 4a has a final
caveated delivery and the whole project has a final handoff. This status check
asks whether the waiting trigger for a paper-faithful rerun is now satisfied.

## Trigger Table

{markdown_table(rows)}

## Analysis

No trigger is currently active. The gate now uses the local source artifact
inventory, so new PDFs, archives, source-like files, matrix artifacts, and
unclassified configs are treated as inspection candidates instead of being
silently ignored. The only current local PDF is the already-audited paper PDF,
which recovers the Figure 4a setup and broad SVD++/Linear-UCB context, but not
enough to rebuild the exact metric/backend/tuning pipeline.

Therefore, the next step remains unchanged: do not run more local Figure 4a
proxy experiments. Resume only when author code, supplement, exact split/config
files, precise w-RMSE definition, or approved validated mechanism dependencies
become available.

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


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
