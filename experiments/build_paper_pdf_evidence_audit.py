"""Audit what the local paper PDF can and cannot resolve for Figure 4a.

This is a source-evidence artifact, not a model experiment. It extracts page
locations for the Figure 4a setup from the local PDF and records which blockers
remain after reading the paper itself.
"""

from __future__ import annotations

import csv
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
PDF = ROOT / "Network-Enabled Sequential Data Acquisition (1).pdf"

OUT_CSV = TABLE_DIR / "paper_pdf_evidence_audit.csv"
OUT_REPORT = REPORT_DIR / "paper_pdf_evidence_audit.md"


def load_pdf_pages() -> list[str]:
    reader = PdfReader(str(PDF))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def find_pages(pages: list[str], terms: list[str]) -> str:
    hits: list[int] = []
    lowered_terms = [term.lower() for term in terms]
    for index, text in enumerate(pages, start=1):
        lowered = text.lower()
        if any(term in lowered for term in lowered_terms):
            hits.append(index)
    return ",".join(str(page) for page in hits)


def build_rows() -> list[dict[str, str]]:
    pages = load_pdf_pages()
    return [
        {
            "item": "pdf_available",
            "status": "available" if PDF.exists() and len(pages) > 0 else "missing",
            "pdf_pages": str(len(pages)),
            "pdf_evidence": "Local paper PDF is readable with pypdf.",
            "decision": "Use the PDF as primary-text evidence before assuming missing details.",
        },
        {
            "item": "figure4a_target",
            "status": "available",
            "pdf_pages": find_pages(pages, ["Figure 4", "MovieLens", "w-RMSE"]),
            "pdf_evidence": "Figure 4 identifies MovieLens w-RMSE and shows DU-Net below Linear UCB at the final epoch.",
            "decision": "The visual target remains DU-Net around 1.788 and Linear UCB around 1.814.",
        },
        {
            "item": "evaluation_metric_family",
            "status": "partial_available",
            "pdf_pages": find_pages(pages, ["weighted root mean squared error", "w-RMSE", "sigmoid"]),
            "pdf_evidence": "The PDF names cumulative reward and w-RMSE, and defines acquisition reward with a sigmoid transform around rating 3.",
            "decision": "This helps interpret reward weighting, but does not fully specify an executable Figure 4a w-RMSE recomputation.",
        },
        {
            "item": "split_protocol",
            "status": "partial_available",
            "pdf_pages": find_pages(pages, ["Monti", "initial set", "candidate acquisition set", "test set", "1%"]),
            "pdf_evidence": "The PDF states that datasets use initial, candidate acquisition, and hold-out test subsets, with 1% of the training set as initial data.",
            "decision": "This is enough to constrain the local protocol, but not enough to recover exact split files or seed mappings.",
        },
        {
            "item": "cycle_and_batch_protocol",
            "status": "partial_available",
            "pdf_pages": find_pages(pages, ["20 acquisition cycles", "batch", "100"]),
            "pdf_evidence": "The PDF describes 20 acquisition cycles and batch acquisition of 100 data points.",
            "decision": "The high-level cycle schedule is known; exact per-seed candidate masks still remain source-dependent.",
        },
        {
            "item": "randomization_protocol",
            "status": "partial_available",
            "pdf_pages": find_pages(pages, ["20 random", "randomly initiated", "different splits"]),
            "pdf_evidence": "The PDF reports averaging over 20 random initializations or split instances.",
            "decision": "This supports 20-seed reporting, but does not identify the exact random seeds or split artifacts.",
        },
        {
            "item": "exact_figure4a_metric",
            "status": "missing",
            "pdf_pages": find_pages(pages, ["w-RMSE"]),
            "pdf_evidence": "The PDF names w-RMSE but does not provide enough executable detail for weights, normalization, masking, and aggregation.",
            "decision": "Do not claim exact Figure 4a metric reproduction from the PDF alone.",
        },
        {
            "item": "exact_split_files",
            "status": "missing",
            "pdf_pages": find_pages(pages, ["Monti", "split"]),
            "pdf_evidence": "The PDF references dataset splits but does not include split files, initial masks, or candidate masks.",
            "decision": "A paper-faithful rerun still needs source split artifacts or an equivalent supplement.",
        },
        {
            "item": "du_net_backend_config",
            "status": "missing",
            "pdf_pages": find_pages(pages, ["DU-Net", "DLS", "observational entropy", "sigma"]),
            "pdf_evidence": "The PDF describes the method, but does not provide runnable architecture, hyperparameter, and backend configuration.",
            "decision": "The local DU-Net-style implementation remains a proxy unless author code or exact config appears.",
        },
        {
            "item": "linear_ucb_adapter",
            "status": "missing",
            "pdf_pages": find_pages(pages, ["Linear UCB", "contextualbandits"]),
            "pdf_evidence": "The PDF identifies Linear UCB as a benchmark, but does not fully specify features, regularization, alpha schedule, or update protocol.",
            "decision": "Linear UCB comparison is directionally useful, but not an exact paper adapter.",
        },
        {
            "item": "execution_decision",
            "status": "blocked_by_missing_source",
            "pdf_pages": "",
            "pdf_evidence": "The local PDF narrows the protocol but does not clear the exact metric/split/backend blockers.",
            "decision": "Keep the next executable experiment gated on author code, supplement, split/config files, exact metric definition, or approved dependency.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| item | status | pdf pages | evidence | decision |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['item']} | {row['status']} | {row['pdf_pages']} | "
            f"{row['pdf_evidence']} | {row['decision']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    blocked = next(row for row in rows if row["item"] == "execution_decision")
    report = f"""# Paper PDF Evidence Audit

Generated by `experiments/build_paper_pdf_evidence_audit.py`.

## Current Progress

The local PDF was rechecked as primary-text evidence for the Figure 4a rerun
gate. It confirms several high-level protocol facts, but it does not remove the
source-backed blockers that prevent an exact DU-Net/Figure 4a replication.

## Audit Table

{markdown_table(rows)}

## Decision

Status: `{blocked['status']}`.

The PDF supports the current strong-partial-reproduction conclusion. It does
not justify another local proxy sweep or a full 20-seed paper-faithful rerun.
The next experiment remains the smallest MovieLens Figure 4a 3-seed diagnostic,
and only after a real source or approved-dependency trigger appears.

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
