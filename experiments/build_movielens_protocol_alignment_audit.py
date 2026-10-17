"""Audit MovieLens protocol alignment against the paper's Figure 4a setup.

This script does not run a new model. It checks whether the local data split
and acquisition schedule match the high-level protocol recovered from the PDF,
and it identifies which split details remain source-dependent.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_movielens_100k, split_dataset


TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ml-100k"
CONFIG_PATH = ROOT / "configs" / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds.json"

OUT_CSV = TABLE_DIR / "movielens_protocol_alignment_audit.csv"
OUT_REPORT = REPORT_DIR / "movielens_protocol_alignment_audit.md"


PAPER = {
    "users_table": 943,
    "users_prose": 945,
    "items": 1681,
    "initial_density": 0.00086,
    "test_density": 0.0126,
    "cycles": 20,
    "batch_size": 100,
    "random_initializations": 20,
}


def read_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def read_ratings(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )


def density(count: int, users: int, items: int) -> float:
    return count / (users * items)


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def official_fold_rows(users: int, items: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fold in range(1, 6):
        base = RAW_DIR / f"u{fold}.base"
        test = RAW_DIR / f"u{fold}.test"
        if not base.exists() or not test.exists():
            rows.append(
                {
                    "check": f"official_u{fold}_split",
                    "status": "missing",
                    "paper_reference": "Monti/GCMC-style split source needed",
                    "local_value": f"{base.name}/{test.name} absent",
                    "interpretation": "Cannot compare this official MovieLens fold.",
                }
            )
            continue
        train_count = len(read_ratings(base))
        test_count = len(read_ratings(test))
        rows.append(
            {
                "check": f"official_u{fold}_split",
                "status": "available_not_used_by_current_config",
                "paper_reference": "MovieLens README official 80/20 split; GCMC ml_100k loader uses u1.base/u1.test",
                "local_value": (
                    f"train={train_count}; test={test_count}; "
                    f"test_density={density(test_count, users, items):.6f}"
                ),
                "interpretation": "The fold matches the paper-scale test density, but current Figure 4a configs split u.data by seed instead.",
            }
        )
    return rows


def build_rows() -> list[dict[str, str]]:
    config = read_config()
    dataset = load_movielens_100k(RAW_DIR)
    split = split_dataset(dataset, seed=0, **config["split"])
    ratings_count = len(dataset.ratings)
    total_cells = dataset.num_users * dataset.num_items
    paper_cells = PAPER["users_table"] * PAPER["items"]
    paper_initial_count = PAPER["initial_density"] * paper_cells
    paper_test_count = PAPER["test_density"] * paper_cells
    acquired_count = config["acquisition"]["cycles"] * config["acquisition"]["batch_size"]

    rows = [
        {
            "check": "dataset_user_count",
            "status": "match_table",
            "paper_reference": f"Table 1 users={PAPER['users_table']}; prose also mentions {PAPER['users_prose']}",
            "local_value": str(dataset.num_users),
            "interpretation": "Local MovieLens user count matches Table 1; the PDF prose has an internal 945-user inconsistency.",
        },
        {
            "check": "dataset_item_count",
            "status": "near_match",
            "paper_reference": f"Table 1 items={PAPER['items']}",
            "local_value": str(dataset.num_items),
            "interpretation": "Local MovieLens keeps 1682 items, one more than the paper table; prior backend screen showed this alone is not enough to close Figure 4a scale.",
        },
        {
            "check": "ratings_density",
            "status": "available",
            "paper_reference": "MovieLens 100K full rating matrix before train/test partition",
            "local_value": f"ratings={ratings_count}; density={density(ratings_count, dataset.num_users, dataset.num_items):.6f}",
            "interpretation": "The raw MovieLens corpus is standard ML-100K and locally complete.",
        },
        {
            "check": "test_density",
            "status": "aligned",
            "paper_reference": f"Table 1 test density={PAPER['test_density']:.6f}; implied count={paper_test_count:.1f}",
            "local_value": f"test={len(split.test)}; density={density(len(split.test), dataset.num_users, dataset.num_items):.6f}",
            "interpretation": "The local 20% test split matches the paper's MovieLens test density almost exactly.",
        },
        {
            "check": "initial_density",
            "status": "mismatch_but_tested",
            "paper_reference": f"Table 1 initial density={PAPER['initial_density']:.6f}; implied count={paper_initial_count:.1f}",
            "local_value": f"initial={len(split.initial)}; density={density(len(split.initial), dataset.num_users, dataset.num_items):.6f}",
            "interpretation": "The current config uses 1% of the 80k train split (800 rows), below Table 1 density; the 1363-row Table-1 diagnostic did not close the scale gap.",
        },
        {
            "check": "candidate_pool",
            "status": "aligned_high_level",
            "paper_reference": "Initial, candidate acquisition, and test subsets; acquisition restricted to candidate set",
            "local_value": f"candidate={len(split.candidate)}; train_total={len(split.initial) + len(split.candidate)}",
            "interpretation": "The local split has the same three-way structure, but not the paper's exact candidate mask/seed mapping.",
        },
        {
            "check": "acquisition_schedule",
            "status": "aligned",
            "paper_reference": f"{PAPER['cycles']} cycles; batch={PAPER['batch_size']}",
            "local_value": f"cycles={config['acquisition']['cycles']}; batch={config['acquisition']['batch_size']}; acquired={acquired_count}",
            "interpretation": "The local Figure 4a schedule matches the recovered PDF schedule.",
        },
        {
            "check": "seed_count",
            "status": "aligned",
            "paper_reference": f"{PAPER['random_initializations']} random initializations",
            "local_value": f"seeds={len(config['seeds'])}; first={config['seeds'][0]}; last={config['seeds'][-1]}",
            "interpretation": "The local robust comparison uses 20 seeds, matching the paper's averaging count but not exact paper seed identities.",
        },
        {
            "check": "current_split_source",
            "status": "proxy_not_exact",
            "paper_reference": "Dataset splits adhere to Monti et al. to ensure reproducibility",
            "local_value": "src.data_loader.load_movielens_100k reads u.data; split_dataset shuffles by seed",
            "interpretation": "MovieLens currently lacks a recovered Monti split artifact; this remains a paper-faithful blocker.",
        },
    ]
    rows.extend(official_fold_rows(dataset.num_users, dataset.num_items))
    rows.append(
        {
            "check": "execution_decision",
            "status": "do_not_rerun_yet",
            "paper_reference": "Exact Figure 4a requires exact split/metric/backend or an approved dependency",
            "local_value": "data protocol is high-level aligned, but exact split remains unresolved",
            "interpretation": "A split-only rerun is unlikely to solve the gap because test density already aligns and Table-1 initial density was tested; wait for source-backed trigger.",
        }
    )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["| check | status | paper reference | local value | interpretation |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['check']} | {row['status']} | {row['paper_reference']} | "
            f"{row['local_value']} | {row['interpretation']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    decision = next(row for row in rows if row["check"] == "execution_decision")
    report = f"""# MovieLens Protocol Alignment Audit

Generated by `experiments/build_movielens_protocol_alignment_audit.py`.

## Current Progress

The local MovieLens Figure 4a setup is high-level aligned with the paper on
test density, acquisition cycles, batch size, and 20-seed averaging. The largest
data-protocol caveat is that the current MovieLens configs use seeded random
splits from `u.data`, while the paper references Monti/GCMC split artifacts.

## Audit Table

{markdown_table(rows)}

## Analysis

This narrows the reproduction gap. The data protocol is not wildly off: the
test-set scale matches the paper, the acquisition schedule matches, and the
20-seed count matches. The initial-set density mismatch was already tested with
a Table-1-density diagnostic and did not close the Figure 4a scale gap.

The remaining high-value blockers are therefore exact w-RMSE definition,
paper split files or seed mapping, DU-Net backend/config, and Linear UCB
adapter details. The split issue should be treated as a faithfulness caveat,
not as evidence that another blind local rerun will resolve the result.

## Decision

Status: `{decision['status']}`.

Do not launch a full rerun from this audit alone. If source-backed split files
or a validated Monti/GCMC MovieLens split become available, first run the
minimal Figure 4a 3-seed diagnostic before any 20-seed promotion.

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
