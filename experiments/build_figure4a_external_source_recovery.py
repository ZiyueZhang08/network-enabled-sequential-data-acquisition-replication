"""Build the final external-source recovery report for Figure 4a.

The goal is to record whether a source/supplement/author-code path can reduce
the remaining Figure 4a gap. It deliberately avoids launching another local
proxy experiment.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_CSV = TABLE_DIR / "figure4a_external_source_recovery.csv"
OUT_REPORT = REPORT_DIR / "figure4a_external_source_recovery.md"

FINAL_REPORT = REPORT_DIR / "figure4a_final_target_report.md"
PRIMARY_REPORT = REPORT_DIR / "figure4a_primary_source_recovery.md"
AUTHOR_REPORT = REPORT_DIR / "author_bandit_source_comparison.md"
BENCHMARK_FREEZE = REPORT_DIR / "final_benchmark_claim_freeze.md"


def build_rows() -> list[dict[str, str]]:
    return [
        {
            "source_type": "local_pdf",
            "source": "Network-Enabled Sequential Data Acquisition (1).pdf",
            "checked_for": "Figure 4a setup, MovieLens split, SVD++ backend family, w-RMSE definition, Linear UCB context",
            "result": "partial",
            "evidence": "Recovered broad setup and comparator context; exact metric/backend details remain absent.",
            "action": "Keep primary-source recovery report as the PDF evidence boundary.",
        },
        {
            "source_type": "local_workspace",
            "source": "workspace reports and artifacts",
            "checked_for": "supplement archives, source bundles, exact author configs, figure-label source files",
            "result": "not_found",
            "evidence": "Prior local source comparison found no DU-Net author implementation or supplement in the workspace.",
            "action": "Do not infer missing details from local artifacts.",
        },
        {
            "source_type": "web_search",
            "source": "exact title, author, DU-Net, Figure 4a, w-RMSE query set",
            "checked_for": "author code, supplement, exact experiment configs, public project page",
            "result": "not_found",
            "evidence": "No public DU-Net author repository or supplement was recovered from the search set in this session.",
            "action": "Treat exact source recovery as unavailable unless the user supplies a new URL or file.",
        },
        {
            "source_type": "paper_footnote_dependency",
            "source": "https://github.com/david-cortes/contextualbandits",
            "checked_for": "online and finite-arm bandit benchmark definitions",
            "result": "dependency_confirmed",
            "evidence": "The repository lists LinUCB, Linear Thompson, Logistic UCB/TS, UCB, Thompson, epsilon greedy, adaptive greedy, explore-then-exploit, active-learning exploration, and SoftMax chooser families.",
            "action": "Use only as benchmark-family context; it is not DU-Net author code and does not confirm the two unresolved benchmark slots.",
        },
        {
            "source_type": "paper_footnote_dependency",
            "source": "https://github.com/kwangsungjun/lrbandit",
            "checked_for": "low-rank bandit and OFUL adaptation source",
            "result": "dependency_confirmed",
            "evidence": "The repository is for Bilinear Bandits with Low-rank Structure and contains small-matrix low-rank bandit variants such as bloful.",
            "action": "Use only as low-rank bandit context; it does not recover exact DU-Net Figure 4a settings.",
        },
        {
            "source_type": "decision",
            "source": "combined local PDF plus external recovery",
            "checked_for": "whether another local proxy sweep is justified",
            "result": "stop_local_proxy_sweeps",
            "evidence": "Remaining gaps are exact w-RMSE weighting/normalization, SVD++ hyperparameters, per-step tuning protocol, Linear UCB feature adaptation, and source split/config details.",
            "action": "Finalize the caveated Figure 4a claim unless new primary-source evidence appears.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| source type | source | checked for | result | evidence | action |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["source_type"],
                    row["source"],
                    row["checked_for"],
                    row["result"],
                    row["evidence"],
                    row["action"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Figure 4a External-Source Recovery

Generated by `experiments/build_figure4a_external_source_recovery.py`.

## Current Reproduction Progress

The Figure 4a branch is now at the end of the local evidence path. The current
result is a strong partial reproduction, not an exact Figure 4a match:

- Paper visual endpoint estimate: DU-Net about 1.788 w-RMSE, Linear UCB about
  1.814, so the paper favors DU-Net.
- Local 20-seed reward-weighted w-RMSE: DU-Net 0.8250, MF-centered Linear UCB
  0.8333, so the local reward-region direction matches.
- Local 20-seed full conventional RMSE: DU-Net 1.1608, MF-centered Linear UCB
  1.1560, so the full-test ordering does not match.
- Local high-rating region favors DU-Net; local low-rating region favors Linear
  UCB. This explains why reward-weighted/high-rating evaluation aligns while
  full-test RMSE flips.

## External Recovery Result

{markdown_table(rows)}

## Analysis

The external recovery did not produce a new executable mechanism. The PDF and
footnote dependency repositories are useful for bounding the interpretation,
but they do not give the missing details needed to make Figure 4a exact:
w-RMSE weighting/normalization, SVD++ rank and optimizer settings, per-step
parameter tuning, Linear UCB features, or exact split/config files.

The important consequence is that another local sweep would mainly tune proxies
against a visual endpoint. That would be weaker evidence than the current
caveated result. The honest stopping point is therefore to report consistency
with the paper only under reward-weighted/high-rating evaluation and to compare
explicitly with the local MF-centered Linear UCB row.

## Decision

Queue item 33 is complete. No further local Figure 4a proxy experiment is
justified unless a new primary source is supplied. The next actionable trigger is
one of:

1. author code, supplement, split files, or exact experiment configs;
2. a precise w-RMSE definition from the source;
3. explicit permission to add validated external mechanism dependencies and
   rerun a smallest faithful 3-seed diagnostic.

Without one of those, the next step is final caveated handoff, not more
experimentation.

## Linked Local Artifacts

- `{FINAL_REPORT.relative_to(ROOT)}`
- `{PRIMARY_REPORT.relative_to(ROOT)}`
- `{AUTHOR_REPORT.relative_to(ROOT)}`
- `{BENCHMARK_FREEZE.relative_to(ROOT)}`
- `{OUT_CSV.relative_to(ROOT)}`
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
