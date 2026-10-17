"""Validate the final reproduction delivery artifacts and key claims."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

MANIFEST = TABLE_DIR / "reproduction_delivery_manifest.csv"
FIGURE4A = TABLE_DIR / "final_figure4a_caveated_delivery.csv"
TRIGGER_STATUS = TABLE_DIR / "paper_faithful_trigger_status.csv"

OUT_CSV = TABLE_DIR / "reproduction_artifact_integrity_check.csv"
OUT_REPORT = REPORT_DIR / "reproduction_artifact_integrity_check.md"


EXTRA_ARTIFACTS = [
    ("delivery_manifest", "delivery package manifest", MANIFEST),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def row(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for item in rows:
        if item[key] == value:
            return item
    raise KeyError((key, value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_artifacts() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    manifest_items = [
        (item["artifact_id"], item["role"], ROOT / item["path"])
        for item in read_rows(MANIFEST)
    ]
    for artifact_id, role, path in manifest_items + EXTRA_ARTIFACTS:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        rows.append(
            {
                "check_type": "artifact",
                "check_name": artifact_id,
                "status": "pass" if exists and size > 0 else "fail",
                "observed": f"exists={exists}; size={size}; sha256={sha256(path) if exists else ''}",
                "expected": "exists=True; size>0",
                "interpretation": role,
            }
        )
    return rows


def check_claims() -> list[dict[str, str]]:
    figure = read_rows(FIGURE4A)
    trigger = read_rows(TRIGGER_STATUS)
    weighted = row(figure, "claim_area", "local_reward_weighted")
    full = row(figure, "claim_area", "local_full_rmse")
    high = row(figure, "claim_area", "local_high_rating_region")
    rerun = row(trigger, "trigger", "rerun_decision")

    checks = [
        {
            "check_type": "claim",
            "check_name": "figure4a_reward_weighted_direction",
            "observed_value": float(weighted["linear_minus_du"]),
            "expected_pass": lambda value: value > 0,
            "expected_text": "Linear - DU > 0",
            "interpretation": "Local reward-weighted Figure 4a direction favors DU-Net over Linear UCB.",
        },
        {
            "check_type": "claim",
            "check_name": "figure4a_full_rmse_reversal",
            "observed_value": float(full["linear_minus_du"]),
            "expected_pass": lambda value: value < 0,
            "expected_text": "Linear - DU < 0",
            "interpretation": "Full-test RMSE reverses, so the exact Figure 4a claim remains caveated.",
        },
        {
            "check_type": "claim",
            "check_name": "figure4a_high_rating_support",
            "observed_value": float(high["linear_minus_du"]),
            "expected_pass": lambda value: value > 0,
            "expected_text": "Linear - DU > 0",
            "interpretation": "High-rating region supports the DU-Net reward-region mechanism.",
        },
    ]
    rows: list[dict[str, str]] = []
    for check in checks:
        value = float(check["observed_value"])
        rows.append(
            {
                "check_type": str(check["check_type"]),
                "check_name": str(check["check_name"]),
                "status": "pass" if check["expected_pass"](value) else "fail",
                "observed": f"{value:.6f}",
                "expected": str(check["expected_text"]),
                "interpretation": str(check["interpretation"]),
            }
        )
    rows.append(
        {
            "check_type": "claim",
            "check_name": "paper_faithful_rerun_gate",
            "status": "pass" if rerun["status"] == "blocked_by_missing_source" else "fail",
            "observed": rerun["status"],
            "expected": "blocked_by_missing_source",
            "interpretation": "No paper-faithful rerun should start without new primary source or approved dependency.",
        }
    )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        "| type | check | status | observed | expected | interpretation |",
        "|---|---|---|---|---|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['check_type']} | {item['check_name']} | {item['status']} | "
            f"`{item['observed']}` | `{item['expected']}` | {item['interpretation']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]]) -> None:
    passed = sum(1 for item in rows if item["status"] == "pass")
    failed = len(rows) - passed
    report = f"""# Reproduction Artifact Integrity Check

Generated by `experiments/build_reproduction_artifact_integrity_check.py`.

## Summary

- Checks passed: {passed}
- Checks failed: {failed}

The final delivery package and latest status addenda are internally consistent
if all checks pass. This does not change the scientific claim: the project
remains a strong partial reproduction, not an exact DU-Net/Figure 4a
reproduction.

## Checks

{markdown_table(rows)}

## Decision

The artifact set is ready for handoff when all checks pass. Future execution
should still wait for author code, supplement, split files, exact w-RMSE
definition, or approved validated external mechanism dependencies.

CSV artifact: `{OUT_CSV.relative_to(ROOT)}`.
"""
    OUT_REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = check_artifacts() + check_claims()
    write_csv(rows)
    write_report(rows)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
