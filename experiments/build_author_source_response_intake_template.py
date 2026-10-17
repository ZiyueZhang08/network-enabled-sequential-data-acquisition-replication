"""Build the response-intake template for author/source replies.

This artifact is used after the prepared source request has been sent. It maps
possible replies or attachments to the smallest allowed inspection and rerun
path, keeping the frozen Figure 4a result unchanged until a source-backed
trigger appears.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

SEND_CHECKLIST = TABLE_DIR / "author_source_request_send_checklist.csv"
NEXT_TRIGGER = TABLE_DIR / "figure4a_next_trigger_protocol.csv"
SOURCE_PACKET = TABLE_DIR / "paper_source_request_packet.csv"

OUT_CSV = TABLE_DIR / "author_source_response_intake_template.csv"
OUT_REPORT = REPORT_DIR / "author_source_response_intake_template.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def trigger_by_prefix(rows: list[dict[str, str]], prefix: str) -> dict[str, str]:
    for row in rows:
        if row["trigger"].startswith(prefix):
            return row
    raise KeyError(f"Missing trigger prefix={prefix}")


def build_rows() -> list[dict[str, str]]:
    send = read_rows(SEND_CHECKLIST)
    triggers = read_rows(NEXT_TRIGGER)
    source = read_rows(SOURCE_PACKET)

    waiting = row_by(send, "step", "waiting_state")
    metric = trigger_by_prefix(triggers, "Exact Figure 4a w-RMSE")
    split = trigger_by_prefix(triggers, "Author MovieLens split")
    backend = trigger_by_prefix(triggers, "DU-Net entropy")
    tuning = trigger_by_prefix(triggers, "Per-cycle lambda")
    linear = trigger_by_prefix(triggers, "Paper Linear UCB")

    metric_request = row_by(source, "request_id", "exact_figure4a_metric")
    split_request = row_by(source, "request_id", "exact_split_and_preprocessing")
    backend_request = row_by(source, "request_id", "du_net_backend_config")
    linear_request = row_by(source, "request_id", "linear_ucb_adapter")

    return [
        {
            "response_item": "exact_metric_reply",
            "status": "waiting",
            "minimum_evidence": metric_request["needed_artifact"],
            "intake_action": "Transcribe the formula, denominator, masks, clipping, and seed/cycle aggregation before touching model code.",
            "first_validation": metric["validation"],
            "promotion_rule": metric["promote_to_20seed_if"],
        },
        {
            "response_item": "split_or_masks_reply",
            "status": "waiting",
            "minimum_evidence": split_request["needed_artifact"],
            "intake_action": "Store recovered masks/splits separately and build a source-faithful 3-seed config.",
            "first_validation": split["validation"],
            "promotion_rule": split["promote_to_20seed_if"],
        },
        {
            "response_item": "du_net_backend_or_code_reply",
            "status": "waiting",
            "minimum_evidence": backend_request["needed_artifact"],
            "intake_action": "Map code/config fields to local DU-Net modules before adding a source-faithful candidate.",
            "first_validation": backend["validation"],
            "promotion_rule": backend["promote_to_20seed_if"],
        },
        {
            "response_item": "dynamic_tuning_reply",
            "status": "waiting",
            "minimum_evidence": tuning["minimum_input"],
            "intake_action": "Replace only the proxy tuner branch after recording the recovered grid, loss, and tie-break rules.",
            "first_validation": tuning["validation"],
            "promotion_rule": tuning["promote_to_20seed_if"],
        },
        {
            "response_item": "linear_ucb_reply",
            "status": "waiting",
            "minimum_evidence": linear_request["needed_artifact"],
            "intake_action": "Add the source-faithful Linear UCB as an appendix comparator while preserving the frozen proxy.",
            "first_validation": linear["validation"],
            "promotion_rule": linear["promote_to_20seed_if"],
        },
        {
            "response_item": "non_actionable_reply",
            "status": "keep_frozen",
            "minimum_evidence": "Thanks, general advice, or no concrete metric/split/code/config details.",
            "intake_action": "Record the reply as correspondence only.",
            "first_validation": "No model rerun.",
            "promotion_rule": "No 20-seed promotion.",
        },
        {
            "response_item": "intake_decision",
            "status": waiting["status"],
            "minimum_evidence": "At least one source-backed trigger must satisfy a listed minimum evidence field.",
            "intake_action": "Inspect and classify the reply before opening any experiment branch.",
            "first_validation": "The first executable run remains seeds 0,1,2 unless metric-only recomputation is possible.",
            "promotion_rule": "Promote to 20 seeds only after the matching 3-seed rule passes.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| response item | status | minimum evidence | intake action | first validation | promotion rule |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['response_item']} | {row['status']} | {row['minimum_evidence']} | "
            f"{row['intake_action']} | {row['first_validation']} | {row['promotion_rule']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    decision = row_by(rows, "response_item", "intake_decision")
    report = f"""# Author Source Response Intake Template

Generated by `experiments/build_author_source_response_intake_template.py`.

## Current Progress

The outbound author/source request is ready or waiting externally. This template
defines how to classify a future reply before any local rerun is authorized.

## Intake Table

{markdown_table(rows)}

## Decision

Status: `{decision['status']}`.

{decision['intake_action']} {decision['first_validation']} {decision['promotion_rule']}

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
