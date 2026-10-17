"""Build a final handoff report for the reproduction project."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

PAPER_RESULTS = TABLE_DIR / "final_paper_quality_results.csv"
FIGURE4A = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
OUT_CSV = TABLE_DIR / "final_reproduction_handoff.csv"
OUT_REPORT = REPORT_DIR / "final_reproduction_handoff.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for item in rows:
        if item.get(key) == value:
            return item
    raise KeyError((key, value))


def fmt_float(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def build_rows() -> list[dict[str, str]]:
    paper = read_rows(PAPER_RESULTS)
    fig = read_rows(FIGURE4A)

    ml = row(paper, "dataset", "MovieLens 100K")
    flixster = row(paper, "dataset", "Flixster")
    douban = row(paper, "setting", "homogeneous final candidate")
    fig_weighted = row(fig, "claim_area", "local_reward_weighted")
    fig_full = row(fig, "claim_area", "local_full_rmse")
    fig_high = row(fig, "claim_area", "local_high_rating_region")

    return [
        {
            "section": "overall",
            "status": "strong_partial_reproduction",
            "key_result": "Main loop, real datasets, 20-cycle schedule, 20-seed evidence, and caveated final reports are complete.",
            "artifact": "reports/final_paper_quality_results.md",
            "next_action": "Report as strong partial reproduction, not exact DU-Net replication.",
        },
        {
            "section": "figure4a",
            "status": "caveated_complete",
            "key_result": (
                "Reward-weighted direction matches DU-Net over Linear UCB "
                f"({fig_weighted['du_net']} vs {fig_weighted['linear_ucb']}), but full RMSE reverses "
                f"({fig_full['du_net']} vs {fig_full['linear_ucb']})."
            ),
            "artifact": "reports/final_figure4a_caveated_delivery.md",
            "next_action": "Do not rerun local proxies without new source details.",
        },
        {
            "section": "figure4a_mechanism",
            "status": "mechanism_supported_but_limited",
            "key_result": (
                "High-rating region supports DU-Net over Linear UCB "
                f"({fig_high['du_net']} vs {fig_high['linear_ucb']}), while low-rating/full-test regions limit the claim."
            ),
            "artifact": "reports/figure4a_rating_tail_20seed_confirmation.md",
            "next_action": "Use this as the explanation for why reward-weighted and full-test metrics diverge.",
        },
        {
            "section": "movielens",
            "status": ml["verdict"],
            "key_result": (
                "DU-Net vs myopic has reward delta "
                f"{fmt_float(ml['reward_delta'], 2)} and w-RMSE delta {fmt_float(ml['w_rmse_delta'], 4)}."
            ),
            "artifact": ml["source_table"],
            "next_action": "Keep as w-RMSE-direction evidence, with reward tradeoff caveat.",
        },
        {
            "section": "flixster",
            "status": flixster["verdict"],
            "key_result": (
                "DU-Net vs myopic has reward delta "
                f"{fmt_float(flixster['reward_delta'], 2)} and w-RMSE delta {fmt_float(flixster['w_rmse_delta'], 4)}."
            ),
            "artifact": flixster["source_table"],
            "next_action": "Use as the cleanest local reproduction result.",
        },
        {
            "section": "douban",
            "status": douban["verdict"],
            "key_result": (
                "NetObs + reliability-DLS vs myopic has reward delta "
                f"{fmt_float(douban['reward_delta'], 2)} and w-RMSE delta {fmt_float(douban['w_rmse_delta'], 4)}."
            ),
            "artifact": douban["source_table"],
            "next_action": "Report as a positive diagnosed candidate, not an exact DU-Net match.",
        },
        {
            "section": "benchmarks",
            "status": "14_plus_2_caveat",
            "key_result": "Fourteen explicitly named benchmark slots are implemented or proxy-covered; two paper-count slots remain unresolved.",
            "artifact": "reports/final_benchmark_claim_freeze.md",
            "next_action": "Do not infer the two unresolved labels from dependency repositories.",
        },
        {
            "section": "blocked_next",
            "status": "needs_primary_source_or_dependency_approval",
            "key_result": "Remaining gaps are exact w-RMSE, SVD++ backend settings, DLS/diffusion details, split/config files, and author-code benchmark adapters.",
            "artifact": "reports/figure4a_external_source_recovery.md",
            "next_action": "Resume experiments only with author source/supplement/exact metric or approved validated dependencies.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| section | status | key result | artifact | next action |",
        "|---|---|---|---|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['section']} | {item['status']} | {item['key_result']} | "
            f"{item['artifact']} | {item['next_action']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Final Reproduction Handoff

Generated by `experiments/build_final_reproduction_handoff.py`.

## Current Status

The reproduction has reached a stable local stopping point. The right status is
strong partial reproduction, not exact DU-Net replication.

The codebase now has frozen 20-seed evidence, Figure 4a-specific caveats,
benchmark-coverage caveats, and primary-source recovery reports. Further local
experiments would not reduce the main uncertainty unless they are triggered by
new paper-faithful information or approved validated dependencies.

## Handoff Table

{markdown_table(rows)}

## Reporting Wording

Use this wording:

> We obtain a strong partial reproduction. The local DU-Net-style method is
> directionally consistent with Figure 4a relative to Linear UCB under
> reward-weighted/high-rating evaluation, and Flixster gives the cleanest
> reward plus w-RMSE match. However, the result is not paper-exact: the Figure
> 4a absolute scale and full-test RMSE ordering are not recovered, Douban remains
> a diagnosed sparse-network case, and the benchmark suite carries a 14+2 label
> caveat.

## Stop Condition

Do not run more local Figure 4a proxy sweeps. Resume only if at least one of the
following becomes available:

1. author code, supplement, split files, or exact experiment configs;
2. precise source definition of Figure 4a w-RMSE;
3. explicit approval to add validated external mechanism dependencies, followed
   by a smallest faithful 3-seed diagnostic before any 20-seed rerun.

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
