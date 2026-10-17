"""Build a copy-ready author/source request email.

The exact Figure 4a reproduction remains blocked by source-backed details. This
script converts the frozen Figure 4a package and current source-recovery blocker
list into a concise email that can be sent to authors or anyone who can provide
supplement/source artifacts.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
SOURCE_BLOCKERS = TABLE_DIR / "figure4a_source_recovery_blockers.csv"
NEXT_TRIGGER = TABLE_DIR / "figure4a_next_trigger_protocol.csv"

OUT_CSV = TABLE_DIR / "author_source_request_email.csv"
OUT_REPORT = REPORT_DIR / "author_source_request_email.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


PAPER_GAP = 1.814 - 1.788


def fmt(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def p0_requests(blockers: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in blockers
        if int(row["priority"]) <= 4
    ]


def p1_requests(blockers: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in blockers
        if int(row["priority"]) > 4
    ]


def bullet_lines(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        lines.append(f"- {row['blocker']}: {row['minimum_needed']}")
    return "\n".join(lines)


def trigger_summary(triggers: list[dict[str, str]]) -> str:
    lines = []
    for row in triggers:
        lines.append(
            f"- If {row['trigger']}: {row['first_action']}; validate by {row['validation']}; promote only if {row['promote_to_20seed_if']}."
        )
    return "\n".join(lines)


def build_email(
    final: list[dict[str, str]],
    blockers: list[dict[str, str]],
    triggers: list[dict[str, str]],
) -> tuple[str, str]:
    du = row_by(final, "variant", "du_net")
    linear = row_by(final, "variant", "mf_linear_ucb_a002")
    myopic = row_by(final, "variant", "myopic")
    gap = float(linear["local_reward_weighted_w_rmse"]) - float(du["local_reward_weighted_w_rmse"])
    target_pct = 100.0 * gap / PAPER_GAP

    subject = "Request for DU-Net Figure 4a MovieLens reproduction details"
    body = f"""Dear authors,

I am trying to reproduce the DU-Net MovieLens 100K result reported in Figure 4a of "Network-Enabled Sequential Data Acquisition".

My current reproduction is directionally consistent with the Linear UCB comparison under the best local reward-weighted evaluation, but it is not an exact Figure 4a match:

- Local reward-weighted comparison: DU-Net-style {fmt(du['local_reward_weighted_w_rmse'])} vs MF-centered Linear UCB proxy {fmt(linear['local_reward_weighted_w_rmse'])}; Linear-minus-DU gap {gap:.4f}.
- This recovers about {target_pct:.1f}% of the Figure 4a visual gap estimate (~{PAPER_GAP:.3f}), but it remains below an exact reproduction target.
- DU-Net also beats Myopic under the same local proxy: Myopic {fmt(myopic['local_reward_weighted_w_rmse'])}; Myopic-minus-DU gap {float(myopic['local_reward_weighted_w_rmse']) - float(du['local_reward_weighted_w_rmse']):.4f}.
- Full conventional RMSE still reverses the ordering: DU-Net-style {fmt(du['rmse'])} vs MF-centered Linear UCB proxy {fmt(linear['rmse'])}; Linear-minus-DU gap {float(linear['rmse']) - float(du['rmse']):.4f}.

Would you be able to share any of the following Figure 4a reproduction materials?

Highest-priority materials:
{bullet_lines(p0_requests(blockers))}

Helpful comparator details:
{bullet_lines(p1_requests(blockers))}

If the full source code or supplement archive cannot be shared, the minimum useful subset would be:

- Exact Figure 4a w-RMSE definition, including weighting, normalization, test mask, clipping, and aggregation.
- MovieLens 100K split files or exact split/preprocessing/seed protocol, including the initial observed set and candidate mask.
- DU-Net backend/config details, including architecture, observational entropy estimator, DLS/network diffusion settings, SVD++ hyperparameters, and per-cycle lambda/sigma_t/p_t tuning.
- Linear UCB feature construction, reward transform, regularization, alpha schedule, and update/update-frequency protocol.

My planned validation after receiving any material is intentionally small:
{trigger_summary(triggers)}

Thank you very much for any guidance or artifacts you can share.
"""
    return subject, body


def build_rows() -> list[dict[str, str]]:
    final = read_rows(FINAL_SUMMARY)
    blockers = read_rows(SOURCE_BLOCKERS)
    triggers = read_rows(NEXT_TRIGGER)
    subject, body = build_email(final, blockers, triggers)
    rows = [
        {
            "item": "email_subject",
            "status": "ready",
            "content": subject,
        },
        {
            "item": "email_body",
            "status": "ready",
            "content": body,
        },
        {
            "item": "p0_request_count",
            "status": "ready",
            "content": str(len(p0_requests(blockers))),
        },
        {
            "item": "p1_request_count",
            "status": "ready",
            "content": str(len(p1_requests(blockers))),
        },
        {
            "item": "execution_decision",
            "status": "ready_to_send",
            "content": "Use this email to request the source-backed trigger; do not run another local proxy sweep while waiting.",
        },
    ]
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, str]]) -> None:
    subject = row_by(rows, "item", "email_subject")["content"]
    body = row_by(rows, "item", "email_body")["content"]
    decision = row_by(rows, "item", "execution_decision")
    report = f"""# Author Source Request Email

Generated by `experiments/build_author_source_request_email.py`.

## Current Progress

The current reproduction is a strong partial result. It is directionally
consistent with Linear UCB under the best local reward-weighted metric, but it
does not match the paper's absolute DU-Net Figure 4a scale or full-RMSE
ordering. The next useful action is to request source-backed reproduction
details.

## Subject

{subject}

## Email Body

```text
{body}
```

## Decision

Status: `{decision['status']}`.

{decision['content']}

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
