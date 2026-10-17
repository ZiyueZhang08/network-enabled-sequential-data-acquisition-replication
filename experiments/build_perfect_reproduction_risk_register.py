from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_faithful_movielens_remaining_risk_register_v2"
TABLE_PATH = ROOT / "outputs" / "tables" / f"{PREFIX}.csv"
REPORT_PATH = ROOT / "reports" / f"{PREFIX}.md"
SOURCE_TABLE_PATH = ROOT / "outputs" / "tables" / f"{PREFIX}_source_refresh.csv"

P0_GATE_PATH = ROOT / "outputs" / "tables" / "paper_faithful_movielens_tau1_literal_p0_reconstruction_v1_gate_summary.csv"
LOCAL_SCAN_PATH = ROOT / "data" / "processed" / "movielens_monti_scan" / "scan_report.json"
FALLBACK_PROVENANCE_PATH = (
    ROOT / "data" / "processed" / "movielens_u1_fallback_table1_density" / "provenance.json"
)
EQ36_REPORT = ROOT / "reports" / "paper_faithful_entropy_eq36_equivalence_v1.md"
P0_REPORT = ROOT / "reports" / "paper_faithful_movielens_tau1_literal_p0_reconstruction_v1.md"


def read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_gate_status() -> dict[str, str]:
    if not P0_GATE_PATH.exists():
        return {}
    frame = pd.read_csv(P0_GATE_PATH)
    return {str(row.gate): str(row.status) for row in frame.itertuples(index=False)}


def source_refresh_rows() -> list[dict[str, str]]:
    local_scan = read_json_if_exists(LOCAL_SCAN_PATH)
    fallback = read_json_if_exists(FALLBACK_PROVENANCE_PATH)
    candidate_count = int(local_scan.get("candidate_count", 0) or 0)
    fallback_source = str(fallback.get("source", "MovieLens 100K official u1.base/u1.test fallback"))
    return [
        {
            "source_item": "target DU-Net paper public artifact search",
            "status": "not_found",
            "source_url": "",
            "evidence": (
                "Manual web refresh for exact phrases around DU-Net, Linear UCB, MovieLens, "
                "and network-enabled sequential data acquisition did not recover author code, "
                "masks, or Figure 4 point data."
            ),
        },
        {
            "source_item": "Monti et al. RMGCNN paper",
            "status": "paper_available_no_masks",
            "source_url": "https://arxiv.org/abs/1704.06803",
            "evidence": (
                "The arXiv page exposes the paper/PDF/TeX source, while the paper text gives "
                "real-data graph construction and benchmark RMSEs but not MovieLens masks or "
                "seed mappings for this clean-room Figure 4 protocol."
            ),
        },
        {
            "source_item": "MovieLens 100K official fallback",
            "status": "available",
            "source_url": "https://grouplens.org/datasets/movielens/100k/",
            "evidence": fallback_source,
        },
        {
            "source_item": "local Monti mask scan",
            "status": "not_found" if candidate_count == 0 else "candidate_found",
            "source_url": str(LOCAL_SCAN_PATH.relative_to(ROOT)).replace("\\", "/"),
            "evidence": f"local exact 943x1681 candidate count: {candidate_count}",
        },
    ]


def risk_rows() -> list[dict[str, str]]:
    gates = read_gate_status()
    return [
        {
            "risk": "Tau1 literal softmax selection",
            "status": "closed",
            "evidence": "tau1 literal mainline is in the P0 bundle and one-config Figure 4 table",
            "next_action": "Do not tune selection temperature for the main claim unless the paper source says so.",
        },
        {
            "risk": "Schur / Eq3.6 formula implementation",
            "status": gates.get("Entropy Eq3.6", "closed_cleanroom" if EQ36_REPORT.exists() else "unresolved"),
            "evidence": "explicit observed-kernel Schur complement equivalence report plus unit test",
            "next_action": "Keep the explicit kernel helper as an audit path; production should stay on feature-space scoring.",
        },
        {
            "risk": "Figure 4a ranking",
            "status": gates.get("Figure 4a", "strong_pattern_recovery"),
            "evidence": "tau1 DU-Net is lower than both clean-room Linear UCB proxies under executable standard RMSE",
            "next_action": "Stop broad hyperparameter search; only source-backed metric/split changes are justified.",
        },
        {
            "risk": "Figure 4b cumulative reward scale",
            "status": gates.get("Figure 4b", "partial"),
            "evidence": "ranking direction matches the supplied visual extraction, but local reward gap is too large",
            "next_action": "Treat as partial recovery until exact reward definition and comparator protocol are available.",
        },
        {
            "risk": "Exact Figure 4 metric normalization",
            "status": gates.get("Metric", "unresolved"),
            "evidence": "seed0 row-level audit is complete for tau1 DU-Net vs linucb_concat; direction matches but no metric candidate passes the scale/gap gate",
            "next_action": "Run remaining seeds only if exact metric recovery is still needed; otherwise keep metric exactness as unresolved.",
        },
        {
            "risk": "Exact MovieLens split/masks",
            "status": gates.get("Split", "unresolved"),
            "evidence": "local mask scan found no Monti/DU-Net Figure 4 masks; fallback is official MovieLens u1 split",
            "next_action": "Keep fallback label unless a public mask artifact or author split is recovered.",
        },
        {
            "risk": "Author Linear UCB protocol",
            "status": gates.get("Linear UCB", "proxy_set"),
            "evidence": "concat, MF-interaction, and bias-interaction proxies are implemented; author feature/update protocol unavailable",
            "next_action": "Report Linear UCB as a proxy set and keep strongest proxy caveat visible.",
        },
        {
            "risk": "ELPH / exact author backend",
            "status": "open_source_dependent",
            "evidence": "local ELPH-style embeddings are dependency-free proxies and have not beaten the main spectral/user-feature backend",
            "next_action": "Do not promote ELPH proxy without source-backed backend details or a clean validation win.",
        },
    ]


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(str(column) for column in frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    return "\n".join(lines)


def write_report(risks: pd.DataFrame, sources: pd.DataFrame) -> None:
    lines = [
        f"# Remaining Risk Register: {PREFIX}",
        "",
        "## Position",
        "",
        "The current repository now has a strong clean-room reproduction path, but not an exact public-author replication. The controllable formula risk is closed for Schur Eq3.6 equivalence; the remaining blockers are source availability, exact metric normalization, exact masks, and the author Linear UCB protocol.",
        "",
        "## Risk Register",
        "",
        md_table(risks),
        "",
        "## Source Refresh",
        "",
        md_table(sources),
        "",
        "## Hard Boundary",
        "",
        "Do not call the result perfect exact replication unless the unresolved source-dependent gates are supplied or recovered. The strongest truthful label is: paper-faithful clean-room reproduction with closed tau1 selection and Eq3.6 formula risks, plus explicit unresolved metric/split/comparator caveats.",
        "",
        "## Primary Artifacts",
        "",
        f"- P0 bundle: `{P0_REPORT.relative_to(ROOT).as_posix()}`",
        f"- Eq3.6 note: `{EQ36_REPORT.relative_to(ROOT).as_posix()}`",
        f"- Risk table: `{TABLE_PATH.relative_to(ROOT).as_posix()}`",
        f"- Source table: `{SOURCE_TABLE_PATH.relative_to(ROOT).as_posix()}`",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    risks = pd.DataFrame(risk_rows())
    sources = pd.DataFrame(source_refresh_rows())
    risks.to_csv(TABLE_PATH, index=False)
    sources.to_csv(SOURCE_TABLE_PATH, index=False)
    write_report(risks, sources)
    print(REPORT_PATH)
    print(TABLE_PATH)
    print(SOURCE_TABLE_PATH)


if __name__ == "__main__":
    main()
