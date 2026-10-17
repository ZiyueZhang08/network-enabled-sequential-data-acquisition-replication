"""Recover Figure 4a details from the local paper PDF.

This script does not run a new policy experiment. It turns the local PDF
evidence and existing Figure 4a diagnostics into an auditable source-recovery
report, so the remaining reproduction gap is explicit.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "Network-Enabled Sequential Data Acquisition (1).pdf"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_CSV = TABLE_DIR / "figure4a_primary_source_recovery.csv"
OUT_REPORT = REPORT_DIR / "figure4a_primary_source_recovery.md"
OUT_HITS = TABLE_DIR / "figure4a_primary_source_pdf_hits.txt"
OUT_PAGES = TABLE_DIR / "figure4a_primary_source_pages_24_29.txt"

FINAL_TARGET_CSV = TABLE_DIR / "figure4a_final_target_report.csv"
CONFIG_PATH = ROOT / "configs" / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds.json"

EXPORT_PAGES = [22, 24, 25, 27, 28, 29, 42, 43]
SEARCH_TERMS = [
    "SVDpp",
    "SVD",
    "w-RMSE",
    "weighted root",
    "MovieLens",
    "initial set",
    "test set",
    "Linear UCB",
    "homogenous",
    "homogeneous",
    "response",
]


def import_pdfplumber():
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pdfplumber is required to extract the local paper PDF. "
            "Install it or rerun in the configured workspace environment."
        ) from exc
    return pdfplumber


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def squashed(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def text_contains(text: str, term: str) -> bool:
    normalized = normalize_spaces(text).lower()
    compact = squashed(text)
    term_lower = term.lower()
    return term_lower in normalized or term_lower.replace(" ", "") in compact


def extract_pdf_pages() -> dict[int, str]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)
    pdfplumber = import_pdfplumber()
    page_text: dict[int, str] = {}
    with pdfplumber.open(PDF_PATH) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            page_text[index] = page.extract_text(layout=True) or ""
    return page_text


def write_pdf_evidence(page_text: dict[int, str]) -> None:
    page_lines: list[str] = []
    for page_no in EXPORT_PAGES:
        page_lines.append(f"===== PAGE {page_no} =====")
        page_lines.append("")
        page_lines.append(page_text.get(page_no, ""))
        page_lines.append("")
    OUT_PAGES.write_text("\n".join(page_lines), encoding="utf-8")

    hit_lines = [f"pages {len(page_text)}", ""]
    for page_no, text in sorted(page_text.items()):
        terms = [term for term in SEARCH_TERMS if text_contains(text, term)]
        if not terms:
            continue
        hit_lines.append(f"--- PAGE {page_no} HITS {terms} ---")
        for raw_line in text.splitlines():
            line = normalize_spaces(raw_line)
            if not line:
                continue
            if any(text_contains(line, term) for term in terms):
                hit_lines.append(line)
        hit_lines.append("")
    OUT_HITS.write_text("\n".join(hit_lines), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def final_metric(rows: list[dict[str, str]], name: str) -> dict[str, str]:
    for row in rows:
        if row.get("comparison") == name:
            return row
    raise KeyError(name)


def build_rows() -> list[dict[str, str]]:
    final_rows = read_csv_rows(FINAL_TARGET_CSV)
    config = read_json(CONFIG_PATH)
    acquisition = config["acquisition"]  # type: ignore[index]
    mf = config["mf"]  # type: ignore[index]

    local_weighted = final_metric(final_rows, "local_reward_weighted_w_rmse")
    full_rmse = final_metric(final_rows, "rmse_all")
    high_rmse = final_metric(final_rows, "rmse_high_rating_ge4")
    sigmoid_rmse = final_metric(final_rows, "sigmoid_reward_w_rmse")

    local_params = (
        f"rank={mf['rank']}, epochs={mf['epochs']}, lr={mf['learning_rate']}, "
        f"reg={mf['regularization']}, lambda_entropy={acquisition['lambda_entropy']}, "
        f"sigma={acquisition['sigma']}, diffusion_prob={acquisition['diffusion_prob']}"
    )

    return [
        {
            "topic": "Figure 4a setup",
            "source_pages": "28-29",
            "recovered_detail": (
                "Figure 4 reports w-RMSE on left panels and cumulative reward/profit on right panels, "
                "over 20 acquisition cycles, under homogeneous response rates, averaged over 20 "
                "randomly initialized training/acquisition splits. The requested target is panel (a), "
                "MovieLens w-RMSE."
            ),
            "local_status": (
                "A 20-seed DU-Net versus MF-centered Linear UCB comparison is frozen in the final "
                "Figure 4a target report."
            ),
            "remaining_gap": "The PDF figure gives curves, not exact tabulated endpoints.",
            "next_action": "Use visual endpoint estimates with caveats; do not overstate exactness.",
        },
        {
            "topic": "Dataset and split",
            "source_pages": "24-25",
            "recovered_detail": (
                "Table 1 lists MovieLens as 943 users, 1681 items, 0.086% initial-set density, "
                "and 1.26% test-set density. The prose on page 24/25 also mentions 945 users, so "
                "the PDF itself is internally inconsistent on the user count."
            ),
            "local_status": (
                "The local loader has 943 users and 1682 items. Test density is aligned; the default "
                "initial density is smaller, and a Table-1-density diagnostic did not close the scale gap."
            ),
            "remaining_gap": (
                "The exact Monti et al. split files or paper split seed are not recovered from the PDF."
            ),
            "next_action": "Treat split/preprocessing as close but not exact unless source files are obtained.",
        },
        {
            "topic": "Backend",
            "source_pages": "22,43",
            "recovered_detail": (
                "The paper says noisy matrix completion supplies user/item representations and that SVDpp "
                "is used for those representations. Appendix notation and complexity also point to SVD/SVDpp "
                "representations."
            ),
            "local_status": (
                "The local backend is an in-house SVD++ proxy with fixed settings: "
                f"{local_params}. Backend screens did not recover the full conventional-RMSE ordering."
            ),
            "remaining_gap": (
                "The PDF does not specify enough to reproduce exact rank, optimization schedule, regularization, "
                "clipping, prediction treatment, or preprocessing details."
            ),
            "next_action": "Do not continue backend sweeps without source/supplement triggers.",
        },
        {
            "topic": "Tuning mechanism",
            "source_pages": "24",
            "recovered_detail": (
                "The paper describes a per-step procedure that partitions current observed data into training, "
                "acquisition, and test subsets, then tunes acquisition parameters by minimizing test loss."
            ),
            "local_status": (
                "The local Figure 4a run uses fixed tuned parameters rather than a recovered per-step "
                "paper tuning loop."
            ),
            "remaining_gap": (
                "The exact loss, grids, retuning frequency, and held-out construction are not specified enough "
                "for a faithful local implementation."
            ),
            "next_action": "Only implement this if the source code or supplement gives the concrete tuning protocol.",
        },
        {
            "topic": "w-RMSE definition",
            "source_pages": "27-28",
            "recovered_detail": (
                "The prose calls the implementation metric weighted root mean squared error and says it is "
                "computed on the test set. The extracted formula visibly uses an Omega_test mask, but no "
                "explicit reward-weight vector is recoverable from the PDF text extraction."
            ),
            "local_status": (
                "Local reward-weighted w-RMSE matches the paper direction: DU-Net "
                f"{float(local_weighted['du_net']):.4f} versus Linear UCB "
                f"{float(local_weighted['linear_ucb']):.4f}. Full conventional RMSE does not: DU-Net "
                f"{float(full_rmse['du_net']):.4f} versus Linear UCB {float(full_rmse['linear_ucb']):.4f}. "
                f"High-rating RMSE favors DU-Net ({float(high_rmse['linear_minus_du']):.4f} Linear-DU gap), "
                f"while sigmoid weighting is a near tie ({float(sigmoid_rmse['linear_minus_du']):.4f})."
            ),
            "remaining_gap": (
                "The PDF alone does not disambiguate whether Figure 4a uses reward-weighted, full-test, "
                "or another normalized w-RMSE variant."
            ),
            "next_action": "Report the current result as reward-region consistent, not paper-exact.",
        },
        {
            "topic": "Linear UCB comparator",
            "source_pages": "27-29",
            "recovered_detail": (
                "The benchmark section includes online linear bandits such as Linear UCB. The result text says "
                "Linear UCB stands out for MovieLens and Douban among bandits, and the reward-phase text says "
                "Linear UCB performs best on MovieLens cumulative reward."
            ),
            "local_status": (
                "The local comparator is MF-centered Linear UCB, which is the strongest recovered local Linear "
                "UCB-style row for Figure 4a. It is appropriate as the requested fallback comparator, but it is "
                "still a proxy."
            ),
            "remaining_gap": (
                "The PDF does not specify the exact feature map, confidence scaling, or offline adaptation used "
                "for the paper Linear UCB row."
            ),
            "next_action": "Keep comparing against local MF-centered Linear UCB and label it explicitly.",
        },
        {
            "topic": "Execution decision",
            "source_pages": "22,24-29,43",
            "recovered_detail": (
                "The local PDF recovers the broad Figure 4a setup and several key design choices, but not the "
                "missing implementation details needed to make an exact reproduction claim."
            ),
            "local_status": (
                "Queue item 32 is complete after this report. The strongest current claim remains a strong "
                "partial reproduction of the DU-Net versus Linear UCB relationship in reward-weighted/high-rating "
                "regions."
            ),
            "remaining_gap": (
                "Exact source code, supplement, or author implementation details are now the binding constraint."
            ),
            "next_action": "Stop local proxy sweeps; either seek external source details or finalize the caveated report.",
        },
    ]


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| topic | source pages | recovered | local status | remaining gap | next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["topic"],
                    row["source_pages"],
                    row["recovered_detail"],
                    row["local_status"],
                    row["remaining_gap"],
                    row["next_action"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Figure 4a Primary-Source Recovery",
        "",
        "Generated by `experiments/build_figure4a_primary_source_recovery.py`.",
        "",
        "## Answer",
        "",
        "The local PDF recovers enough to define the Figure 4a target and to justify comparison",
        "against Linear UCB, but it does not recover enough implementation detail to close the",
        "absolute-scale and full-test-RMSE gap. The current reproduction should remain labeled as",
        "a strong partial reproduction: DU-Net is consistent with the paper in reward-weighted and",
        "high-rating regions, but it is not an exact Figure 4a reproduction.",
        "",
        "## Recovery Table",
        "",
        markdown_table(rows),
        "",
        "## Decision",
        "",
        "Queue item 32 is complete. Further local proxy sweeps are low value because the PDF does",
        "not disclose the exact w-RMSE weighting/normalization, SVD++ backend settings, per-step",
        "tuning protocol, or Linear UCB feature adaptation. The next useful branch is external",
        "source/supplement/author-code recovery; without that, the final report should keep the",
        "current caveat and compare against the local MF-centered Linear UCB row.",
        "",
        "## Evidence Artifacts",
        "",
        f"- `{OUT_CSV.relative_to(ROOT)}`",
        f"- `{OUT_HITS.relative_to(ROOT)}`",
        f"- `{OUT_PAGES.relative_to(ROOT)}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    page_text = extract_pdf_pages()
    write_pdf_evidence(page_text)
    rows = build_rows()
    write_csv(rows, OUT_CSV)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_HITS}")
    print(f"Wrote {OUT_PAGES}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
