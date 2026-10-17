"""Extract method-detail keyword evidence from the local paper PDF.

This audit is intentionally source-only. It does not run another proxy model;
it checks whether the PDF contains executable details that could clear the
Figure 4a rerun gate.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
PDF = ROOT / "Network-Enabled Sequential Data Acquisition (1).pdf"

OUT_CSV = TABLE_DIR / "paper_method_detail_keyword_audit.csv"
OUT_REPORT = REPORT_DIR / "paper_method_detail_keyword_audit.md"


KEYWORD_GROUPS = {
    "figure4a_metric": [
        "w-RMSE",
        "weighted root mean squared error",
        "root mean squared error",
        "RMSE",
        "sigmoid",
    ],
    "reward_definition": [
        "cumulative reward",
        "reward",
        "sigmoid",
        "rating",
        "preference",
    ],
    "split_protocol": [
        "initial set",
        "candidate acquisition set",
        "test set",
        "training set",
        "Monti",
        "1%",
    ],
    "cycle_batch_seed_protocol": [
        "20 acquisition cycles",
        "100 data",
        "batch",
        "20 random",
        "randomly initiated",
        "different splits",
    ],
    "du_net_backend": [
        "DU-Net",
        "observational entropy",
        "DLS",
        "diffusion",
        "SVD++",
        "sigma",
        "hyperparameter",
    ],
    "linear_ucb_adapter": [
        "Linear UCB",
        "UCB",
        "contextualbandits",
        "regularization",
        "alpha",
    ],
}


GROUP_DECISIONS = {
    "figure4a_metric": (
        "partial_available",
        "Metric names appear, but the PDF does not give executable Figure 4a weights, normalization, mask, and aggregation details.",
    ),
    "reward_definition": (
        "partial_available",
        "Reward semantics are visible, but they are not enough to reconstruct the exact Figure 4a w-RMSE metric.",
    ),
    "split_protocol": (
        "partial_available",
        "The split structure is described, but exact split files, initial masks, candidate masks, and seed mappings are absent.",
    ),
    "cycle_batch_seed_protocol": (
        "partial_available",
        "The high-level cycle/batch/randomization protocol is visible, but exact per-seed artifacts are absent.",
    ),
    "du_net_backend": (
        "missing_executable_detail",
        "The method is described, but runnable DU-Net backend, architecture, entropy/DLS settings, and tuned config are absent.",
    ),
    "linear_ucb_adapter": (
        "missing_executable_detail",
        "Linear UCB is named as a comparator, but feature construction, tuning, and update protocol remain absent.",
    ),
}


def load_pages() -> list[str]:
    reader = PdfReader(str(PDF))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def snippets_for_term(page_text: str, term: str, max_snippets: int = 2) -> list[str]:
    snippets: list[str] = []
    pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
    normalized = compact(page_text)
    for match in pattern.finditer(normalized):
        start = max(0, match.start() - 90)
        end = min(len(normalized), match.end() + 90)
        snippets.append(normalized[start:end])
        if len(snippets) >= max_snippets:
            break
    return snippets


def build_rows() -> list[dict[str, str]]:
    pages = load_pages()
    rows: list[dict[str, str]] = []
    for group, terms in KEYWORD_GROUPS.items():
        status, decision = GROUP_DECISIONS[group]
        hits: list[tuple[int, str, str]] = []
        hit_pages: set[int] = set()
        for page_index, page_text in enumerate(pages, start=1):
            lowered = page_text.lower()
            for term in terms:
                if term.lower() not in lowered:
                    continue
                hit_pages.add(page_index)
                for snippet in snippets_for_term(page_text, term):
                    hits.append((page_index, term, snippet))
        preview = " || ".join(
            f"p{page}: {term}: {snippet}" for page, term, snippet in hits[:5]
        )
        rows.append(
            {
                "keyword_group": group,
                "status": status if hits else "not_found",
                "terms": "; ".join(terms),
                "hit_pages": ",".join(str(page) for page in sorted(hit_pages)),
                "hit_count_previewed": str(min(len(hits), 5)),
                "snippet_preview": preview,
                "decision": decision if hits else "No matching PDF text was extracted for this group.",
            }
        )
    rows.append(
        {
            "keyword_group": "execution_decision",
            "status": "blocked_by_missing_source",
            "terms": "",
            "hit_pages": "",
            "hit_count_previewed": "0",
            "snippet_preview": "",
            "decision": (
                "Keyword-level PDF extraction finds protocol clues but no executable exact metric, split/config, "
                "DU-Net backend, or Linear UCB adapter detail."
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
        "| group | status | pages | decision |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['keyword_group']} | {row['status']} | {row['hit_pages']} | {row['decision']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    previews = "\n".join(
        f"- `{row['keyword_group']}`: {row['snippet_preview'] or 'no snippet'}"
        for row in rows
        if row["keyword_group"] != "execution_decision"
    )
    report = f"""# Paper Method Detail Keyword Audit

Generated by `experiments/build_paper_method_detail_keyword_audit.py`.

## Current Progress

The reproduction remains a strong partial reproduction. This audit checks the
local PDF one level deeper than page hits by extracting short keyword-context
snippets for the details needed to clear the Figure 4a rerun gate.

## Summary

{markdown_table(rows)}

## Snippet Preview

{previews}

## Decision

The PDF contains useful high-level protocol clues, but it still does not provide
the executable details needed for an exact Figure 4a rerun. Continue to gate any
paper-faithful execution on author code, supplement, exact metric/split/config,
or an approved validated mechanism dependency.

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
