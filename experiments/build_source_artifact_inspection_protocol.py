"""Build the inspection protocol for source-like artifact candidates.

The trigger gate can now mark unknown PDFs, archives, source-like files, matrix
artifacts, and configs as inspection candidates. This report defines what to do
next for each candidate class before a faithful rerun can be unlocked.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

TRIGGER_STATUS = TABLE_DIR / "paper_faithful_trigger_status.csv"
INVENTORY = TABLE_DIR / "local_source_artifact_inventory_audit.csv"
SOURCE_PACKET = TABLE_DIR / "paper_source_request_packet.csv"
PLAYBOOK = TABLE_DIR / "paper_faithful_rerun_playbook.csv"

OUT_CSV = TABLE_DIR / "source_artifact_inspection_protocol.csv"
OUT_REPORT = REPORT_DIR / "source_artifact_inspection_protocol.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def actionable_count(rows: list[dict[str, str]]) -> int:
    return sum(
        1
        for row in rows
        if row["artifact"] != "execution_decision" and row["actionable_trigger_candidate"] == "yes"
    )


def build_rows() -> list[dict[str, str]]:
    trigger = read_rows(TRIGGER_STATUS)
    inventory = read_rows(INVENTORY)
    source = read_rows(SOURCE_PACKET)
    playbook = read_rows(PLAYBOOK)

    rerun = row_by(trigger, "trigger", "rerun_decision")
    source_intake = row_by(playbook, "step", "1_source_intake")
    metric = row_by(source, "request_id", "exact_figure4a_metric")
    split = row_by(source, "request_id", "exact_split_and_preprocessing")
    backend = row_by(source, "request_id", "du_net_backend_config")
    linear = row_by(source, "request_id", "linear_ucb_adapter")
    count = actionable_count(inventory)

    return [
        {
            "candidate_class": "unknown_pdf_candidate",
            "inspection_status": "pending_when_present",
            "inspection_action": "Extract text/tables/attachments and search for Figure 4a metric, MovieLens split, DU-Net backend, and Linear UCB adapter details.",
            "unlocks": "exact_figure4a_metric; exact_split_and_preprocessing; du_net_backend_config; linear_ucb_adapter",
            "acceptance_rule": f"{metric['acceptance_test']} OR {split['acceptance_test']} OR {backend['acceptance_test']}",
            "current_state": f"present_actionable_candidates={count}; rerun_decision={rerun['status']}",
        },
        {
            "candidate_class": "unknown_archive_candidate",
            "inspection_status": "pending_when_present",
            "inspection_action": "List archive contents without overwriting local files; classify code, configs, split files, notebooks, and metric scripts.",
            "unlocks": "author_code_or_supplement; exact_split_and_preprocessing; du_net_backend_config",
            "acceptance_rule": "Archive contains runnable Figure 4a code/config or source-backed files that map to playbook steps 2-5.",
            "current_state": f"source_intake={source_intake['status']}",
        },
        {
            "candidate_class": "source_like_candidate",
            "inspection_status": "pending_when_present",
            "inspection_action": "Read source or notebook for data paths, split construction, w-RMSE implementation, DU-Net scoring, and Linear UCB implementation.",
            "unlocks": "author_code_or_supplement; du_net_backend_config; linear_ucb_adapter",
            "acceptance_rule": "Source maps directly to local code/config changes without guessed masks or guessed normalization.",
            "current_state": backend["current_status"],
        },
        {
            "candidate_class": "unknown_mat_candidate",
            "inspection_status": "pending_when_present",
            "inspection_action": "Inspect matrix keys/shapes and determine whether it contains MovieLens initial/candidate/test masks or rating matrices.",
            "unlocks": "exact_split_and_preprocessing",
            "acceptance_rule": split["acceptance_test"],
            "current_state": split["current_status"],
        },
        {
            "candidate_class": "unknown_config_candidate",
            "inspection_status": "pending_when_present",
            "inspection_action": "Compare config fields with requested metric, split, backend, SVD++, and Linear UCB settings.",
            "unlocks": "exact_figure4a_metric; exact_split_and_preprocessing; du_net_backend_config; linear_ucb_adapter",
            "acceptance_rule": "Config gives enough exact values to run the 3-seed diagnostic without introducing new guessed settings.",
            "current_state": linear["current_status"],
        },
        {
            "candidate_class": "known_paper_or_dataset_artifact",
            "inspection_status": "already_classified",
            "inspection_action": "Do not treat the already-audited paper PDF, MovieLens raw archive, known Monti Flixster/Douban mats, or local reproduction configs as rerun triggers.",
            "unlocks": "none",
            "acceptance_rule": "Known artifacts remain non-actionable unless replaced by a new unclassified artifact.",
            "current_state": "clear",
        },
        {
            "candidate_class": "execution_decision",
            "inspection_status": "ready_but_idle" if count == 0 else "inspection_required",
            "inspection_action": (
                "Keep protocol ready; no candidate is present."
                if count == 0
                else "Inspect actionable candidates before any rerun."
            ),
            "unlocks": "minimal_3seed_diagnostic_after_source_mapping",
            "acceptance_rule": "Proceed to playbook only after an inspected artifact satisfies at least one P0/P1 acceptance rule.",
            "current_state": rerun["status"],
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| candidate class | status | inspection action | unlocks | acceptance rule |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate_class']} | {row['inspection_status']} | "
            f"{row['inspection_action']} | {row['unlocks']} | {row['acceptance_rule']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    decision = row_by(rows, "candidate_class", "execution_decision")
    report = f"""# Source Artifact Inspection Protocol

Generated by `experiments/build_source_artifact_inspection_protocol.py`.

## Current Progress

The reproduction remains a strong partial result. The trigger gate can now mark
new local source-like materials as inspection candidates; this protocol defines
how those candidates should be inspected before any rerun is authorized.

## Inspection Table

{markdown_table(rows)}

## Decision

Status: `{decision['inspection_status']}`.

{decision['inspection_action']} The next executable model run remains the
minimal 3-seed diagnostic, and only after an inspected artifact satisfies the
playbook acceptance criteria.

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
