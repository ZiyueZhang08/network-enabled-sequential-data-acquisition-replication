"""Build the recovery plan for the remaining paper-faithfulness gap."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_CSV = TABLE_DIR / "paper_faithful_gap_priority.csv"
OUT_REPORT = REPORT_DIR / "paper_faithful_mechanism_recovery_plan.md"
QUEUE_CSV = TABLE_DIR / "next_experiment_queue.csv"


GAP_ROWS = [
    {
        "priority": 1,
        "gap": "Primary source recovery",
        "current_proxy": "Local paper text, local reports, and public footnote-code comparison",
        "evidence": "reports/paper_benchmark_slot_recovery_audit.md; reports/author_bandit_source_comparison.md; reports/final_benchmark_claim_freeze.md",
        "why_it_matters": "The current local evidence confirms 14 explicit benchmark names under a 16-benchmark paper claim, but it cannot recover the two unresolved labels or exact author settings.",
        "recovery_trigger": "Original PDF figure labels, supplement, author implementation, or experiment config becomes available.",
        "success_criterion": "All 16 benchmark labels and exact DU-Net experiment settings are mapped to implemented or intentionally excluded rows.",
        "next_action": "Recover or ingest primary-source artifacts before launching another full rerun.",
        "rerun_scope": "Regenerate benchmark audit, final synthesis, and any newly confirmed benchmark rows.",
    },
    {
        "priority": 2,
        "gap": "Exact observational entropy estimator",
        "current_proxy": "Log-det, network-observational, expected-network, stochastic, weighted, and reward-frontier approximations",
        "evidence": "src/entropy.py; reports/exact_vs_proxy_appendix.md; outputs/tables/douban_svdpp_weighted_network_entropy_3seeds_final_summary.csv; outputs/tables/douban_svdpp_reward_frontier_guard_3seeds_final_summary.csv",
        "why_it_matters": "Entropy is central to DU-Net's information value. Local closer-looking variants have not beaten the current netobs top-k candidate.",
        "recovery_trigger": "Paper formula, pseudo-code, author code, or a validated estimator specification is recovered.",
        "success_criterion": "The estimator can be unit-tested on small matrices and produces non-identical selections against the current proxy in a diagnostic pilot.",
        "next_action": "Stop entropy-gate sweeps until the estimator itself changes.",
        "rerun_scope": "Run a 3-seed diagnostic on Douban and Flixster, then promote only if it reaches the current frontier.",
    },
    {
        "priority": 3,
        "gap": "Exact DLS dynamic neighborhood update",
        "current_proxy": "Gaussian smoothing over graph-embedding neighbors plus dynamic sigma and reliability shrinkage",
        "evidence": "src/dls.py; outputs/tables/douban_svdpp_ablation_netobs_20seeds_final_summary.csv; outputs/tables/flixster_svdpp_ablation_20seeds_final_summary.csv",
        "why_it_matters": "Douban shows that the DLS proxy can propagate sparse-graph noise; no-DLS still slightly wins one sparse ablation.",
        "recovery_trigger": "Paper or author implementation clarifies DLS neighborhood construction, update timing, and smoothing weights.",
        "success_criterion": "DLS improves or ties no-DLS on sparse Douban without losing the current Flixster result.",
        "next_action": "Keep reliability-DLS and no-DLS as formal ablations until the exact DLS rule is recovered.",
        "rerun_scope": "Run Douban ablation first; expand to full 20-seed only after a positive 3-seed mechanism screen.",
    },
    {
        "priority": 4,
        "gap": "Network reward diffusion",
        "current_proxy": "Local same-item neighbor reward diffusion with optional normalization",
        "evidence": "src/reward.py; outputs/tables/flixster_svdpp_ablation_20seeds_final_summary.csv; outputs/tables/douban_svdpp_ablation_netobs_20seeds_final_summary.csv",
        "why_it_matters": "Flixster DU-Net currently equals no-network in the tuned ablation, so the diffusion proxy is not carrying the paper mechanism.",
        "recovery_trigger": "Exact diffusion rule, cascade probability, normalization, or author reward code is recovered.",
        "success_criterion": "Network-enabled reward changes selections and improves at least one main dataset without collapsing sparse Douban.",
        "next_action": "Do not tune diffusion constants blindly; rebuild only from a clearer mechanism specification.",
        "rerun_scope": "Run Flixster no-network ablation and Douban sparse-graph ablation before a final matrix.",
    },
    {
        "priority": 5,
        "gap": "Validated ELPH or Node2Vec representation",
        "current_proxy": "Spectral, dependency-free Node2Vec-style, ELPH-role, and local SGNS Node2Vec-style embeddings",
        "evidence": "src/graph_embedding.py; reports/embedding_fidelity_feasibility.md; reports/douban_node2vec_sgns_findings.md",
        "why_it_matters": "The strongest Douban candidate is still spectral/reliability-based; local SGNS Node2Vec is positive but not strongest, and true ELPH remains absent.",
        "recovery_trigger": "New dependencies are explicitly allowed, or a local faithful ELPH/Node2Vec implementation is provided.",
        "success_criterion": "The validated embedding beats or ties spectral on Douban 3-seed while preserving graph-density diagnostics.",
        "next_action": "Do not continue embedding-only local sweeps under the no-new-dependency constraint.",
        "rerun_scope": "Run Douban embedding matrix, then repeat final candidate only if it reaches the frontier.",
    },
    {
        "priority": 6,
        "gap": "Author-code-level benchmark adapters",
        "current_proxy": "Implemented or proxy-covered 14 explicit benchmark slots; two paper-count slots unresolved",
        "evidence": "reports/paper_benchmark_definition_audit.md; outputs/tables/paper_benchmark_definition_audit.csv; reports/final_benchmark_claim_freeze.md",
        "why_it_matters": "The benchmark catalog is broad but not author-identical. OFUL/logistic/finite-arm rows are local adaptations and should not be overclaimed.",
        "recovery_trigger": "Author benchmark code, exact baseline hyperparameters, or exact missing labels are recovered.",
        "success_criterion": "Each benchmark row has a named paper source, implementation mapping, and seed-level result artifact.",
        "next_action": "Keep the final claim at 14 explicit covered/proxy-covered plus 2 unresolved slots.",
        "rerun_scope": "Regenerate benchmark audit and run only confirmed paper-faithful rows before full 20-seed promotion.",
    },
]


NEXT_QUEUE_ROW = {
    "priority": "22",
    "phase": "long-term",
    "status": "next",
    "task": "Acquire primary source or approve dependency-enabled faithful mechanisms",
    "command": "TBD",
    "expected_artifact": "reports/paper_faithful_mechanism_recovery_plan.md",
    "reason": "The remaining gap cannot be materially narrowed without original PDF/supplement/author code or an explicitly approved faithful ELPH/Node2Vec/exact-mechanism dependency path.",
}


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_priority_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "priority",
        "gap",
        "current proxy",
        "why it matters",
        "success criterion",
        "next action",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["priority"]),
                    str(row["gap"]),
                    str(row["current_proxy"]),
                    str(row["why_it_matters"]),
                    str(row["success_criterion"]),
                    str(row["next_action"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(path: Path, csv_path: Path) -> None:
    report = f"""# Paper-Faithful Mechanism Recovery Plan

Generated by `experiments/build_paper_faithful_mechanism_recovery_plan.py`.

## Current Reproduction Position

The reproduction is a **strong partial reproduction**, not an exact DU-Net
replication. The main loop, real datasets, 20-cycle schedule, batch-size-100
setting, and 20-seed evidence exist for the core candidates. MovieLens and
Flixster reproduce the main weighted-RMSE direction; Flixster also reproduces
the reward direction. Douban has a positive network-observational candidate,
but it remains a sparse-graph mechanism recovery rather than a clean paper
match.

The next useful execution is therefore not another local convenience sweep. The
remaining bottleneck is paper-faithfulness: exact source details, exact
mechanisms, exact embeddings, and exact benchmark definitions.

## Priority Gap Table

{markdown_priority_table(GAP_ROWS)}

## Execution Decision

Queue item 21 is complete as a planning/execution artifact: the remaining
paper-faithfulness work has been decomposed into ranked gaps with explicit
triggers and success criteria. No new 20-seed matrix was launched because the
current local evidence already shows that additional proxy sweeps are unlikely
to reduce the real distance to the paper.

## Next Executable Trigger

Proceed only when at least one of these is true:

1. Original PDF figure labels, supplement, author code, or exact configs are
   available.
2. New dependencies are explicitly allowed for a validated sparse Node2Vec or
   ELPH path.
3. The exact observational entropy, DLS, or network diffusion rule is recovered
   from a primary source.

When triggered, run the smallest paper-faithful 3-seed diagnostic first. Promote
to a final 20-seed matrix only if that diagnostic reaches or improves the
current frozen frontier.

CSV artifact: `{csv_path.relative_to(ROOT)}`.
"""
    path.write_text(report, encoding="utf-8")


def update_queue(path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    def clean_row(row: dict[str, object]) -> dict[str, str]:
        cleaned = {key: str(row.get(key, "")) for key in fieldnames}
        extra_fields = row.get(None)
        if extra_fields:
            extras = [str(value) for value in extra_fields]
            last_key = fieldnames[-1]
            cleaned[last_key] = ",".join([cleaned[last_key], *extras])
        return cleaned

    updated_rows: list[dict[str, str]] = []
    saw_next = False
    for row in rows:
        row = clean_row(row)
        if row["task"] == "Paper-faithful mechanism or primary-source recovery":
            row = dict(row)
            row["status"] = "done"
            row["command"] = r"python experiments\build_paper_faithful_mechanism_recovery_plan.py"
            row["expected_artifact"] = "reports/paper_faithful_mechanism_recovery_plan.md"
            row["reason"] = "Converted the remaining fidelity bottleneck into a ranked recovery plan with triggers and success criteria."
        if row["task"] == NEXT_QUEUE_ROW["task"]:
            saw_next = True
            row = {key: str(NEXT_QUEUE_ROW[key]) for key in fieldnames}
        updated_rows.append(row)

    if not saw_next:
        updated_rows.append({key: str(NEXT_QUEUE_ROW[key]) for key in fieldnames})

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    tmp_path.replace(path)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(GAP_ROWS, OUT_CSV)
    write_report(OUT_REPORT, OUT_CSV)
    update_queue(QUEUE_CSV)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")
    print(f"Updated {QUEUE_CSV}")


if __name__ == "__main__":
    main()
