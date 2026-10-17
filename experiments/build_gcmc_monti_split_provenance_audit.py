"""Audit local GCMC/Monti split provenance for MovieLens Figure 4a.

This is a source/provenance audit, not a model rerun. It checks whether the
bundled GCMC/Monti files can recover the exact MovieLens split/preprocessing
protocol needed for a paper-faithful Figure 4a reproduction.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

RAW_DIR = ROOT / "data" / "raw"
ML_DIR = RAW_DIR / "ml-100k"
README = RAW_DIR / "gcmc_README.md"
PREPROCESSING = RAW_DIR / "gcmc_preprocessing.py"
DATA_UTILS = RAW_DIR / "gcmc_data_utils.py"
GCMC_DIR = RAW_DIR / "gcmc"

OUT_CSV = TABLE_DIR / "gcmc_monti_split_provenance_audit.csv"
OUT_REPORT = REPORT_DIR / "gcmc_monti_split_provenance_audit.md"
U1_FILES_NEEDLE = "files = ['/u1.base', '/u1.test', '/u.item', '/u.user']"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def line_for(text: str, needle: str) -> str:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return f"line {line_no}: {line.strip()}"
    return "not found"


def present(paths: list[Path]) -> str:
    found = [path.name for path in paths if path.exists()]
    missing = [path.name for path in paths if not path.exists()]
    parts = []
    if found:
        parts.append("present=" + ",".join(found))
    if missing:
        parts.append("missing=" + ",".join(missing))
    return "; ".join(parts) if parts else "no paths checked"


def local_monti_mat_files() -> str:
    files = sorted(GCMC_DIR.glob("*/training_test_dataset.mat"))
    if not files:
        return "none"
    return "; ".join(str(path.relative_to(ROOT)) for path in files)


def build_rows() -> list[dict[str, str]]:
    readme = read_text(README)
    preprocessing = read_text(PREPROCESSING)
    data_utils = read_text(DATA_UTILS)
    official_fold_paths = [
        *(ML_DIR / f"u{fold}.base" for fold in range(1, 6)),
        *(ML_DIR / f"u{fold}.test" for fold in range(1, 6)),
    ]
    ua_ub_paths = [ML_DIR / name for name in ("ua.base", "ua.test", "ub.base", "ub.test")]

    return [
        {
            "check_id": "gcmc_readme_movielens_official_split",
            "status": "available_partial",
            "evidence": (
                f"{line_for(readme, 'python train.py -d ml_100k')}; "
                f"{line_for(readme, '--features --feat_hidden 10 --testing')}"
            ),
            "interpretation": "The GCMC README confirms ML-100K official-split commands, but not DU-Net Figure 4a masks or metrics.",
            "next_action": "Use only as provenance context, not as a rerun trigger.",
        },
        {
            "check_id": "gcmc_movielens_loader",
            "status": "available_partial",
            "evidence": (
                f"{line_for(preprocessing, 'def load_official_trainvaltest_split')}; "
                f"{line_for(preprocessing, U1_FILES_NEEDLE)}; "
                f"{line_for(preprocessing, 'np.random.seed(42)')}"
            ),
            "interpretation": "The local GCMC MovieLens loader uses u1.base/u1.test and a seed-42 validation split, not the paper's exact 20-random-initialization protocol.",
            "next_action": "Keep exact split/preprocessing as an unresolved P0 gap.",
        },
        {
            "check_id": "gcmc_random_loader",
            "status": "proxy_not_exact",
            "evidence": (
                f"{line_for(data_utils, 'def load_data(fname, seed=1234')}; "
                f"{line_for(data_utils, 'random.seed(seed)')}"
            ),
            "interpretation": "The generic GCMC loader shuffles raw ratings by seed, similar to the local proxy split approach, but it is not a recovered Monti split artifact.",
            "next_action": "Do not treat seeded random splits as exact Figure 4a evidence.",
        },
        {
            "check_id": "monti_mat_loader",
            "status": "available_for_flixster_douban",
            "evidence": (
                f"{line_for(preprocessing, 'def load_data_monti')}; "
                f"{line_for(preprocessing, 'training_test_dataset.mat')}; "
                f"local_files={local_monti_mat_files()}"
            ),
            "interpretation": "Monti-style .mat train/test masks exist locally for Flixster and Douban, but no MovieLens training_test_dataset.mat is bundled.",
            "next_action": "Do not infer MovieLens Figure 4a masks from Flixster/Douban artifacts.",
        },
        {
            "check_id": "local_official_fold_files",
            "status": "available_not_exact",
            "evidence": f"{present(official_fold_paths)}; {present(ua_ub_paths)}",
            "interpretation": "Local MovieLens official folds are present, but prior u1 and u1-u5 sensitivity diagnostics did not make the Figure 4a comparison robust.",
            "next_action": "Use official folds only as sensitivity evidence unless paper split mapping is recovered.",
        },
        {
            "check_id": "execution_decision",
            "status": "blocked_by_missing_source",
            "evidence": "GCMC/Monti local provenance confirms official-fold and .mat loader context, but not exact MovieLens Figure 4a split/preprocessing, initial mask, candidate mask, or seed protocol.",
            "interpretation": "This narrows the split story without clearing the exact-reproduction blocker.",
            "next_action": "Wait for author code/supplement/split files/exact metric/backend trigger; then run the minimal 3-seed faithful diagnostic first.",
        },
    ]


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| check | status | evidence | interpretation | next action |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['check_id']} | {row['status']} | {row['evidence']} | "
            f"{row['interpretation']} | {row['next_action']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    decision = next(row for row in rows if row["check_id"] == "execution_decision")
    report = f"""# GCMC/Monti Split Provenance Audit

Generated by `experiments/build_gcmc_monti_split_provenance_audit.py`.

## Current Progress

This audit checks whether local GCMC/Monti files can recover the exact
MovieLens split and preprocessing protocol needed for paper-faithful Figure 4a.
They cannot: the local files provide useful provenance, but not the exact
MovieLens masks/configuration used by DU-Net in the paper.

## Audit Table

{markdown_table(rows)}

## Analysis

The GCMC README and preprocessing code are helpful because they anchor the
MovieLens official split story: the bundled GCMC loader uses `u1.base` and
`u1.test`, then creates a validation split with seed 42. Separately, Monti-style
`training_test_dataset.mat` artifacts exist locally for Flixster and Douban.

That still does not recover the missing MovieLens Figure 4a details: exact
initial observed set, candidate mask, test mask, preprocessing/filtering, seed
mapping, w-RMSE aggregation, DU-Net backend/config, and Linear UCB adapter.
The earlier official-fold sensitivity diagnostics already showed that switching
to bundled official MovieLens folds is not a robust split-only fix.

## Decision

Status: `{decision['status']}`.

{decision['next_action']}

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
