from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
OUT_CSV = TABLE_DIR / "split_mask_search_audit_v3.csv"
OUT_REPORT = REPORT_DIR / "split_mask_search_audit_v3.md"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.fillna("").copy()

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(col) for col in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(clean(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def local_file_summary() -> tuple[str, str]:
    raw = ROOT / "data" / "raw"
    official_folds = sorted(path.name for path in (raw / "ml-100k").glob("u[1-5].*"))
    official_ab = sorted(path.name for path in (raw / "ml-100k").glob("u[ab].*"))
    mat_files = sorted(rel(path) for path in (raw / "gcmc").glob("**/training_test_dataset.mat"))
    fold_text = ", ".join(official_folds + official_ab)
    mat_text = ", ".join(mat_files)
    return fold_text, mat_text


def build_rows() -> pd.DataFrame:
    folds, mats = local_file_summary()
    rows = [
        {
            "search_group": "exact_title_public_code",
            "query_or_source": '"Network-Enabled Sequential Data Acquisition" GitHub / MovieLens / author / code',
            "status": "not_found",
            "evidence": "Web search returned no author repository, code bundle, split file, or mask artifact for the exact title.",
            "source_url": "",
            "local_artifact": "",
            "decision": "Exact MovieLens masks remain source-blocked.",
        },
        {
            "search_group": "exact_title_preprint_or_supplement",
            "query_or_source": "OpenReview, arXiv, SSRN, PapersWithCode exact-title searches",
            "status": "not_found",
            "evidence": "Exact-title targeted searches did not expose ancillary data, code, or numeric Figure 4 tables.",
            "source_url": "",
            "local_artifact": "",
            "decision": "Do not infer masks from visual Figure 4 curves.",
        },
        {
            "search_group": "gcmc_repository",
            "query_or_source": "riannevdberg/gc-mc README",
            "status": "available_partial",
            "evidence": "README lists MovieLens 100K commands on the official split, with and without features.",
            "source_url": "https://github.com/riannevdberg/gc-mc",
            "local_artifact": "data/raw/gcmc_README.md; data/raw/gcmc_data_utils.py",
            "decision": "Supports u1 official-split provenance only; not DU-Net sequential initial/candidate/test masks.",
        },
        {
            "search_group": "gcmc_paper",
            "query_or_source": "Graph Convolutional Matrix Completion",
            "status": "context_only",
            "evidence": "Paper/repo are related matrix-completion context, not a source for Network-Enabled Sequential Data Acquisition masks.",
            "source_url": "https://arxiv.org/abs/1706.02263",
            "local_artifact": "data/raw/gcmc_data_utils.py",
            "decision": "Use as loader/preprocessing context, not exact mask evidence.",
        },
        {
            "search_group": "monti_rmgcnn_repository",
            "query_or_source": "fmonti/mgcnn README",
            "status": "available_partial",
            "evidence": "README states the repository has data for five datasets including MovieLens 100K, Douban, and Flixster.",
            "source_url": "https://github.com/fmonti/mgcnn",
            "local_artifact": "data/raw/gcmc/douban/training_test_dataset.mat; data/raw/gcmc/flixster/training_test_dataset.mat",
            "decision": "Monti-style .mat masks found locally for Douban/Flixster, not MovieLens.",
        },
        {
            "search_group": "monti_rmgcnn_paper",
            "query_or_source": "Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks",
            "status": "context_only",
            "evidence": "Related recommender matrix-completion work; no recovered DU-Net MovieLens sequential mask artifact.",
            "source_url": "https://arxiv.org/abs/1704.06803",
            "local_artifact": "",
            "decision": "Do not transfer masks from related datasets or papers.",
        },
        {
            "search_group": "local_movielens_official_folds",
            "query_or_source": "data/raw/ml-100k",
            "status": "available_not_exact",
            "evidence": f"Official fold files present: {folds}",
            "source_url": "",
            "local_artifact": "data/raw/ml-100k",
            "decision": "Use u1-u5 only for sensitivity, never as exact author masks without paper/source linkage.",
        },
        {
            "search_group": "local_monti_style_masks",
            "query_or_source": "data/raw/gcmc/**/training_test_dataset.mat",
            "status": "available_for_other_datasets_not_movielens",
            "evidence": f"Local .mat masks present: {mats}",
            "source_url": "",
            "local_artifact": "data/raw/gcmc",
            "decision": "MovieLens training_test_dataset.mat is absent, so this cannot close the MovieLens mask gate.",
        },
        {
            "search_group": "prior_local_overlap",
            "query_or_source": "split_search_audit_v2 and gcmc_monti_split_provenance_audit",
            "status": "consistent_with_v3",
            "evidence": "Prior local audits also found official-fold/GCMC/Monti provenance but no exact author MovieLens masks.",
            "source_url": "",
            "local_artifact": "outputs/tables/split_search_audit_v2.csv; outputs/tables/gcmc_monti_split_provenance_audit.csv",
            "decision": "Carry split/mask exactness as failed/source-blocked in v2.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(rows: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fail_rows = rows[rows["status"].isin(["not_found", "available_not_exact", "available_for_other_datasets_not_movielens"])]
    lines = [
        "# Split/Mask Search Audit v3",
        "",
        "Scope: exact-title public-source search plus local data/mask provenance for the MovieLens Figure 4 exact-reproduction gate.",
        "",
        "## Result",
        "",
        "No author MovieLens initial/candidate/test masks, seed protocol, or Figure 4 split script were recovered. Official MovieLens u1-u5 folds and related GCMC/Monti artifacts remain useful sensitivity context, but they do not close exact mask provenance.",
        "",
        "## Evidence Table",
        "",
        md_table(rows),
        "",
        "## Failed/Substitute Sources",
        "",
        md_table(fail_rows[["search_group", "status", "evidence", "decision"]]),
        "",
        "## Decision",
        "",
        "The v2 protocol may stay Table-1-shape aligned through the u1 fallback, but split/mask exactness must remain `source_blocked`. A perfect/exact reproduction claim requires author masks or an author-equivalent generation script.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(OUT_REPORT)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
