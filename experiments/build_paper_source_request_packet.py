"""Build a source request packet for closing the paper-faithful gap.

The packet turns the current blocked rerun gate into a concrete checklist of
source artifacts, definitions, and acceptance tests needed before a faithful
Figure 4a rerun can start.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

TRIGGER_CSV = TABLE_DIR / "paper_faithful_trigger_status.csv"
READINESS_CSV = TABLE_DIR / "paper_faithful_rerun_readiness.csv"
DISTANCE_CSV = TABLE_DIR / "figure4a_distance_scorecard.csv"

OUT_CSV = TABLE_DIR / "paper_source_request_packet.csv"
OUT_REPORT = REPORT_DIR / "paper_source_request_packet.md"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"Missing {key}={value}")


def build_rows() -> list[dict[str, str]]:
    trigger_rows = read_rows(TRIGGER_CSV)
    readiness_rows = read_rows(READINESS_CSV)
    distance_rows = read_rows(DISTANCE_CSV)

    rerun = row_by(trigger_rows, "trigger", "rerun_decision")
    reward = row_by(distance_rows, "claim_area", "local_reward_weighted")
    full = row_by(distance_rows, "claim_area", "local_full_rmse")
    diagnostic = row_by(readiness_rows, "item", "minimum_next_diagnostic")
    promotion = row_by(readiness_rows, "item", "promotion_rule")

    return [
        {
            "request_id": "author_code_or_supplement",
            "priority": "P0",
            "needed_artifact": "Official DU-Net source code, supplement archive, or reproduction package.",
            "why_needed": "The local PDF and public checks do not disclose enough implementation detail for exact Figure 4a.",
            "acceptance_test": "Artifact contains runnable DU-Net/MovieLens Figure 4a pipeline or enough pseudocode/config to map every local proxy to paper components.",
            "current_status": row_by(trigger_rows, "trigger", "author_code_or_supplement")["status"],
        },
        {
            "request_id": "exact_figure4a_metric",
            "priority": "P0",
            "needed_artifact": "Exact Figure 4a w-RMSE definition, weighting, normalization, test mask, and aggregation rule.",
            "why_needed": (
                f"Local reward-weighted direction matches, but scale is {float(reward['du_scale_ratio_to_paper']) * 100:.1f}% "
                "of the paper DU-Net endpoint and full RMSE reverses."
            ),
            "acceptance_test": "Metric recomputation explains the 1.788/1.814 visual target scale or identifies the missing normalization.",
            "current_status": row_by(trigger_rows, "trigger", "exact_figure4a_metric_definition")["status"],
        },
        {
            "request_id": "exact_split_and_preprocessing",
            "priority": "P0",
            "needed_artifact": "MovieLens 100K split files, initial observed set, candidate mask, item filtering, cycle count, and seed protocol.",
            "why_needed": "Initial-density and item-count diagnostics did not close the absolute-scale gap.",
            "acceptance_test": "Local data loader reproduces the paper Table 1 setup and Figure 4a evaluation population without guessed masks.",
            "current_status": row_by(trigger_rows, "trigger", "exact_split_or_config_files")["status"],
        },
        {
            "request_id": "du_net_backend_config",
            "priority": "P0",
            "needed_artifact": "DU-Net architecture details, observational entropy estimator, DLS settings, network diffusion rule, and SVD++ hyperparameters.",
            "why_needed": "Current local DU-Net is a modular proxy; backend/mechanism fidelity remains the main bottleneck.",
            "acceptance_test": "A 3-seed diagnostic can isolate which recovered mechanism changes reward-weighted gap, full RMSE, or scale.",
            "current_status": "missing",
        },
        {
            "request_id": "linear_ucb_adapter",
            "priority": "P1",
            "needed_artifact": "Paper's Linear UCB feature construction, reward transform, regularization, alpha schedule, and update protocol.",
            "why_needed": "The target comparison depends on Linear UCB being adapted the same way as in the paper.",
            "acceptance_test": "MF-centered Linear UCB can be replaced or verified against the paper comparator before final reporting.",
            "current_status": "missing",
        },
        {
            "request_id": "validated_external_dependency_approval",
            "priority": "P1",
            "needed_artifact": "Explicit approval to use validated external mechanism dependencies if no author source appears.",
            "why_needed": "Dependency-free Node2Vec/ELPH/DLS proxies are useful diagnostics but not paper-faithful mechanisms.",
            "acceptance_test": "Approved dependency has a clear mapping to paper mechanism and passes a smallest 3-seed diagnostic before 20 seeds.",
            "current_status": row_by(trigger_rows, "trigger", "validated_external_mechanism_dependency")["status"],
        },
        {
            "request_id": "minimum_rerun_protocol",
            "priority": "P0",
            "needed_artifact": "Agreement on smallest rerun protocol after any trigger appears.",
            "why_needed": rerun["evidence"],
            "acceptance_test": diagnostic["action"],
            "current_status": diagnostic["status"],
        },
        {
            "request_id": "promotion_rule",
            "priority": "P0",
            "needed_artifact": "Agreement not to promote directly from source recovery to 20-seed rerun.",
            "why_needed": "Avoid expensive reruns unless the recovered mechanism moves the current blocker.",
            "acceptance_test": promotion["action"],
            "current_status": promotion["status"],
        },
        {
            "request_id": "current_distance_baseline",
            "priority": "reference",
            "needed_artifact": "Frozen baseline numbers for comparing any future recovered mechanism.",
            "why_needed": "Future diagnostics need a fixed threshold.",
            "acceptance_test": (
                f"Beat or explain reward gap={reward['linear_minus_du']} and full-RMSE gap={full['linear_minus_du']}."
            ),
            "current_status": "available",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| request | priority | needed artifact | acceptance test | current status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['request_id']} | {row['priority']} | {row['needed_artifact']} | "
            f"{row['acceptance_test']} | {row['current_status']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    report = f"""# Paper Source Request Packet

Generated by `experiments/build_paper_source_request_packet.py`.

## Current Progress

The reproduction is reportable as a strong partial reproduction, but exact
Figure 4a reproduction is blocked by missing paper-faithful source information.
This packet lists the smallest set of artifacts or definitions needed to resume
scientific execution rather than running more local proxy sweeps.

## Request Checklist

{markdown_table(rows)}

## Suggested Author/Supplement Request

Please provide any available Figure 4a reproduction materials for the DU-Net
MovieLens 100K experiment: source code or supplement archive, exact w-RMSE
definition, MovieLens split/config files, DU-Net architecture and DLS/network
mechanism settings, SVD++ hyperparameters, and the Linear UCB feature/tuning
protocol. If full code cannot be shared, the exact metric definition plus split
files and per-policy hyperparameters would be the minimum useful subset.

## Execution Rule After Receipt

Do not launch a full 20-seed rerun immediately. First run the smallest
MovieLens Figure 4a 3-seed diagnostic against DU-Net-style, MF-centered Linear
UCB, and Myopic. Promote to 20 seeds only if the recovered material improves the
current reward-weighted gap or resolves the full-RMSE ordering / absolute-scale
mismatch.

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
