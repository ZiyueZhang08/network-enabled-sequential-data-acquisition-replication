"""Build a reproducible audit of implemented baseline coverage.

The audit is intentionally read-only with respect to experiments: it consumes
existing 20-seed final-summary CSVs and writes a compact coverage report.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class DatasetBaselineSpec:
    dataset: str
    source_table: str
    extension_tables: tuple[str, ...]
    expected_families: tuple[str, ...]
    note: str


SPECS = [
    DatasetBaselineSpec(
        dataset="MovieLens 100K",
        source_table="movielens_100k_svdpp_baselines_20seeds_final_summary.csv",
        extension_tables=("movielens_100k_svdpp_baselines_missing_20seeds_final_summary.csv",),
        expected_families=(
            "random",
            "myopic",
            "entropy",
            "linear_ucb",
            "low_rank_bandit",
            "thompson_sampling",
            "graph_social",
            "du_net",
        ),
        note="Implemented 20-seed coverage now has the cross-dataset baseline rows via the separate missing-baselines extension table.",
    ),
    DatasetBaselineSpec(
        dataset="Flixster",
        source_table="flixster_svdpp_baselines_20seeds_final_summary.csv",
        extension_tables=(),
        expected_families=(
            "random",
            "myopic",
            "entropy",
            "linear_ucb",
            "low_rank_bandit",
            "thompson_sampling",
            "graph_social",
            "du_net",
        ),
        note="Cleanest implemented-baseline result: DU-Net leads implemented baselines on reward and w-RMSE; graph_social is close.",
    ),
    DatasetBaselineSpec(
        dataset="Douban",
        source_table="douban_svdpp_baselines_netobs_20seeds_final_summary.csv",
        extension_tables=(),
        expected_families=(
            "random",
            "myopic",
            "entropy",
            "linear_ucb",
            "low_rank_bandit",
            "thompson_sampling",
            "graph_social",
            "du_net",
            "netobs_reliability_spectral",
        ),
        note="NetObs reliability-DLS leads implemented baselines on w-RMSE, but no-DLS remains a stronger sparse-graph mechanism fallback in ablations.",
    ),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def variant_name(row: dict[str, str]) -> str:
    variant = str(row.get("variant", "")).strip()
    policy = str(row.get("policy", "")).strip()
    return variant or policy


def metric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def classify_family(name: str) -> str:
    normalized = name.lower()
    if normalized in {"du_net", "current_du_net"}:
        return "du_net"
    return normalized


def audit_rows() -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    for spec in SPECS:
        rows = read_rows(TABLE_DIR / spec.source_table)
        for extension_table in spec.extension_tables:
            extension_path = TABLE_DIR / extension_table
            if extension_path.exists():
                rows.extend(read_rows(extension_path))
        implemented = {classify_family(variant_name(row)) for row in rows}
        missing_expected = [family for family in spec.expected_families if family not in implemented]
        best_w_rmse = min(rows, key=lambda row: metric(row, "w_rmse"))
        best_reward = max(rows, key=lambda row: metric(row, "cumulative_reward"))
        rows_out.append(
            {
                "dataset": spec.dataset,
                "source_table": "; ".join((spec.source_table, *spec.extension_tables)),
                "implemented_count": len(rows),
                "implemented_families": ", ".join(sorted(implemented)),
                "missing_expected_in_matrix": ", ".join(missing_expected),
                "best_w_rmse_variant": variant_name(best_w_rmse),
                "best_w_rmse": metric(best_w_rmse, "w_rmse"),
                "best_reward_variant": variant_name(best_reward),
                "best_reward": metric(best_reward, "cumulative_reward"),
                "paper_16_baseline_status": "partial",
                "note": spec.note,
            }
        )
    return rows_out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return ""
    return f"{numeric:.{digits}f}"


def write_report(rows: list[dict[str, object]], path: Path, csv_path: Path) -> None:
    lines = [
        "# Baseline Coverage Audit",
        "",
        "Generated by `experiments/build_baseline_coverage_audit.py` from existing 20-seed final summaries.",
        "",
        "## Coverage Table",
        "",
        "| dataset | implemented rows | best w-RMSE | best reward | status |",
        "|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['implemented_count']} | "
            f"{row['best_w_rmse_variant']} ({fmt(row['best_w_rmse'])}) | "
            f"{row['best_reward_variant']} ({fmt(row['best_reward'], 2)}) | "
            f"{row['paper_16_baseline_status']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The implemented baseline matrices are now useful evidence, but they are not the paper's full 16-baseline benchmark.",
            "- MovieLens now includes the two previously missing cross-dataset rows (`graph_social` and `thompson_sampling`) through a separate extension table.",
            "- Flixster is the cleanest qualitative match: DU-Net leads implemented baselines on both reward and w-RMSE.",
            "- Douban is improved but should remain conservative: NetObs reliability-DLS leads the implemented baseline matrix, while no-DLS is still a crucial ablation comparator.",
            "- The weak bandit rows are evidence of baseline incompleteness, not strong evidence that the paper's benchmark suite is fully beaten.",
            "- This remains partial paper coverage because the paper benchmark suite has 16 stronger baselines, while these matrices cover implemented baseline families only.",
            "",
            "## Dataset Notes",
            "",
        ]
    )
    for row in rows:
        missing = str(row["missing_expected_in_matrix"]) or "none within implemented matrix target"
        lines.extend(
            [
                f"### {row['dataset']}",
                "",
                f"- Source: `outputs/tables/{row['source_table']}`",
                f"- Implemented families: {row['implemented_families']}",
                f"- Missing from current implemented matrix target: {missing}",
                f"- Note: {row['note']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Next Step",
            "",
            "The next execution branch should strengthen only the remaining paper-specific baseline definitions before another final paper-quality claim; bandit, active-learning, and social-rec pilots should be promoted to 20 seeds only if a focused 3-seed screen reaches the DU-Net/graph-social frontier.",
            "",
            f"CSV artifact: `{csv_path.relative_to(ROOT)}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = audit_rows()
    csv_path = TABLE_DIR / "baseline_coverage_audit.csv"
    report_path = REPORT_DIR / "baseline_coverage_audit.md"
    write_csv(rows, csv_path)
    write_report(rows, report_path, csv_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
