"""Inventory local source-like artifacts that could unlock a faithful rerun.

The paper-faithful gate depends on whether new local materials exist: author
code, supplements, split/config files, metric definitions, or source archives.
This audit makes that local inventory explicit, including PDFs that the trigger
scanner should not silently ignore.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_CSV = TABLE_DIR / "local_source_artifact_inventory_audit.csv"
OUT_REPORT = REPORT_DIR / "local_source_artifact_inventory_audit.md"

EXCLUDED_PARTS = {".agents", ".codex", ".git", ".omx", "__pycache__", "outputs", "reports", "tmp"}
ARCHIVE_SUFFIXES = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"}
SOURCE_SUFFIXES = {".m", ".ipynb", ".jl", ".r"}
CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"}


def is_generated(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return any(part in EXCLUDED_PARTS for part in rel.parts)


def classify(path: Path) -> tuple[str, str]:
    rel_text = str(path.relative_to(ROOT)).lower()
    name = path.name.lower()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        if "network-enabled sequential data acquisition" in name:
            return "known_paper_pdf", "Primary paper PDF already audited; not a supplement or author code package."
        return "unknown_pdf_candidate", "Unclassified PDF; inspect before keeping the rerun gate closed."
    if suffix in ARCHIVE_SUFFIXES:
        if name == "ml-100k.zip":
            return "known_dataset_archive", "MovieLens raw dataset archive; not source/supplement/config evidence."
        return "unknown_archive_candidate", "Archive could contain supplement/source/config material."
    if suffix == ".mat":
        if "data\\raw\\gcmc" in rel_text or "data/raw/gcmc" in rel_text:
            return "known_monti_split_artifact", "Monti/GCMC artifact already audited; useful for Flixster/Douban, not exact MovieLens Figure 4a."
        return "unknown_mat_candidate", "Unclassified matrix artifact; inspect for split or metric evidence."
    if suffix in SOURCE_SUFFIXES:
        return "source_like_candidate", "Source-like file that may need manual classification."
    if suffix in CONFIG_SUFFIXES:
        if rel_text.startswith("configs/") or rel_text.startswith("configs\\"):
            return "local_reproduction_config", "Local reproduction config, not author/supplement source."
        return "unknown_config_candidate", "Unclassified config file."
    return "other", "Not a source-unlock artifact."


def candidate_files() -> list[Path]:
    suffixes = {".pdf", ".mat"} | ARCHIVE_SUFFIXES | SOURCE_SUFFIXES | CONFIG_SUFFIXES
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or is_generated(path):
            continue
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in CONFIG_SUFFIXES and rel.parts and rel.parts[0] == "configs":
            continue
        if path.suffix.lower() in suffixes:
            files.append(path)
    return sorted(files, key=lambda item: str(item.relative_to(ROOT)).lower())


def local_config_count() -> int:
    config_dir = ROOT / "configs"
    if not config_dir.exists():
        return 0
    return sum(1 for path in config_dir.glob("*") if path.is_file() and path.suffix.lower() in CONFIG_SUFFIXES)


def timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in candidate_files():
        classification, interpretation = classify(path)
        actionable = classification.startswith("unknown") or classification == "source_like_candidate"
        rows.append(
            {
                "artifact": str(path.relative_to(ROOT)),
                "classification": classification,
                "size_bytes": str(path.stat().st_size),
                "last_write_time": timestamp(path),
                "actionable_trigger_candidate": "yes" if actionable else "no",
                "interpretation": interpretation,
            }
        )
    rows.append(
        {
            "artifact": "configs/*",
            "classification": "local_reproduction_config_summary",
            "size_bytes": "",
            "last_write_time": "",
            "actionable_trigger_candidate": "no",
            "interpretation": f"{local_config_count()} local reproduction config file(s) are intentionally excluded from trigger candidacy.",
        }
    )
    actionable_count = sum(1 for row in rows if row["actionable_trigger_candidate"] == "yes")
    rows.append(
        {
            "artifact": "execution_decision",
            "classification": "no_new_actionable_source_artifact" if actionable_count == 0 else "inspect_actionable_candidates",
            "size_bytes": "",
            "last_write_time": "",
            "actionable_trigger_candidate": "no" if actionable_count == 0 else "yes",
            "interpretation": (
                "Only known paper/data/Monti/local-config artifacts are present; no new source-backed rerun trigger."
                if actionable_count == 0
                else f"{actionable_count} unclassified local artifact(s) require inspection before the gate can stay closed."
            ),
        }
    )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| artifact | classification | actionable | interpretation |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['artifact']}` | {row['classification']} | "
            f"{row['actionable_trigger_candidate']} | {row['interpretation']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    decision = next(row for row in rows if row["artifact"] == "execution_decision")
    report = f"""# Local Source Artifact Inventory Audit

Generated by `experiments/build_local_source_artifact_inventory_audit.py`.

## Current Progress

This audit explicitly inventories local PDFs, archives, source-like files,
matrix artifacts, and configs that could unlock a paper-faithful rerun. It is a
guard against missing a newly added supplement or source package.

## Inventory

{markdown_table(rows)}

## Decision

Status: `{decision['classification']}`.

{decision['interpretation']} The current execution gate therefore remains
closed unless a new actionable artifact appears or an external mechanism
dependency is explicitly approved.

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
