"""Build a reproducible zip bundle for the Figure 4a submission packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import build_figure4a_artifact_checksums as artifact_checksums
import build_figure4a_readiness_audit as readiness_audit


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
TABLE_DIR = OUT_DIR / "tables"
REPORT_DIR = ROOT / "reports"

OUT_LATEST_CSV = TABLE_DIR / "figure4a_submission_bundle_latest.csv"
OUT_LATEST_REPORT = REPORT_DIR / "figure4a_submission_bundle_latest.md"

BUNDLE_FILES = [
    "reports/figure4a_submission_packet.md",
    "reports/figure4a_final_package.md",
    "reports/figure4a_final_delivery_zh.md",
    "reports/figure4a_delivery_quality_gate.md",
    "reports/figure4a_readiness_audit.md",
    "reports/figure4a_handoff_index.md",
    "reports/figure4a_artifact_checksums.md",
    "reports/author_source_request_email.md",
    "reports/figure4a_next_trigger_protocol.md",
    "outputs/tables/figure4a_submission_packet.csv",
    "outputs/tables/figure4a_final_package_summary.csv",
    "outputs/tables/figure4a_delivery_quality_gate.csv",
    "outputs/tables/figure4a_readiness_audit.csv",
    "outputs/tables/figure4a_handoff_index.csv",
    "outputs/tables/author_source_request_email.csv",
    "outputs/figures/movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_w_rmse.png",
    "outputs/figures/movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds_cumulative_reward.png",
]


@dataclass(frozen=True)
class BundlePaths:
    bundle_dir: Path
    zip_path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_signature() -> tuple[tuple[str, int, str], ...]:
    paths = [
        REPORT_DIR / "figure4a_readiness_audit.md",
        TABLE_DIR / "figure4a_readiness_audit.csv",
        REPORT_DIR / "figure4a_artifact_checksums.md",
        TABLE_DIR / "figure4a_artifact_checksums.csv",
    ]
    signature = []
    for path in paths:
        signature.append(
            (
                str(path.relative_to(ROOT)),
                path.stat().st_size if path.exists() else 0,
                sha256(path) if path.exists() else "",
            )
        )
    return tuple(signature)


def prepare_audit_artifacts(max_iterations: int = 5) -> None:
    """Regenerate mutually-referencing audit files until their hashes settle."""
    previous = None
    for _ in range(max_iterations):
        readiness_audit.main()
        artifact_checksums.main()
        current = audit_signature()
        if current == previous:
            return
        previous = current
    raise RuntimeError("Figure 4a audit artifacts did not stabilize before packaging")


def bundle_rows() -> list[dict[str, str]]:
    rows = []
    for rel in BUNDLE_FILES:
        path = ROOT / rel
        rows.append(
            {
                "path": rel,
                "exists": str(path.exists()).lower(),
                "bytes": str(path.stat().st_size if path.exists() else 0),
                "sha256": sha256(path) if path.exists() else "",
            }
        )
    return rows


def next_bundle_paths(stamp: str) -> BundlePaths:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_name = f"figure4a_submission_bundle_{stamp}"
    index = 1
    while (OUT_DIR / f"{base_name}_{index:03d}.zip").exists():
        index += 1
    return BundlePaths(
        bundle_dir=OUT_DIR / f"{base_name}_{index:03d}",
        zip_path=OUT_DIR / f"{base_name}_{index:03d}.zip",
    )


def write_manifest(bundle_dir: Path, rows: list[dict[str, str]]) -> Path:
    manifest = bundle_dir / "bundle_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "exists", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def copy_files(bundle_dir: Path, rows: list[dict[str, str]]) -> None:
    missing = [row["path"] for row in rows if row["exists"] != "true"]
    if missing:
        raise FileNotFoundError("Missing bundle files: " + ", ".join(missing))
    for row in rows:
        src = ROOT / row["path"]
        dst = bundle_dir / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def zip_bundle(bundle_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_dir))


def write_latest_report(paths: BundlePaths, rows: list[dict[str, str]]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    zip_hash = sha256(paths.zip_path)
    metadata = [
        {
            "item": "bundle_dir",
            "value": str(paths.bundle_dir.relative_to(ROOT)),
        },
        {
            "item": "zip_path",
            "value": str(paths.zip_path.relative_to(ROOT)),
        },
        {
            "item": "zip_bytes",
            "value": str(paths.zip_path.stat().st_size),
        },
        {
            "item": "zip_sha256",
            "value": zip_hash,
        },
        {
            "item": "entry_count",
            "value": str(len(rows)),
        },
        {
            "item": "submission_status",
            "value": "ready_for_caveated_submission",
        },
        {
            "item": "primary_gap",
            "value": "0.0207",
        },
        {
            "item": "paper_gap_share",
            "value": "79.5%",
        },
    ]
    with OUT_LATEST_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["item", "value"])
        writer.writeheader()
        writer.writerows(metadata)

    lines = [
        "# Figure 4a Submission Bundle Latest",
        "",
        "Generated by `experiments/build_figure4a_submission_bundle.py`.",
        "",
        "## Bundle",
        "",
        f"- Zip: `{paths.zip_path.relative_to(ROOT)}`",
        f"- Directory: `{paths.bundle_dir.relative_to(ROOT)}`",
        f"- Entry count: `{len(rows)}`",
        f"- Zip bytes: `{paths.zip_path.stat().st_size}`",
        f"- Zip SHA256: `{zip_hash}`",
        "",
        "## Submission Status",
        "",
        "- Status: `ready_for_caveated_submission`.",
        "- Primary gap: `0.0207`.",
        "- Paper visual gap share: `79.5%`.",
        "- Use as a caveated partial reproduction package, not an exact Figure 4a replication.",
        "",
        "## Bundle Manifest",
        "",
        "| path | exists | bytes | sha256 |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['path']}` | {row['exists']} | {row['bytes']} | `{row['sha256']}` |")
    OUT_LATEST_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def build_bundle(stamp: str) -> tuple[BundlePaths, list[dict[str, str]]]:
    paths = next_bundle_paths(stamp)
    prepare_audit_artifacts()
    rows = bundle_rows()
    paths.bundle_dir.mkdir(parents=True, exist_ok=False)
    copy_files(paths.bundle_dir, rows)
    manifest = write_manifest(paths.bundle_dir, rows)
    rows.append(
        {
            "path": str(manifest.relative_to(paths.bundle_dir)).replace("\\", "/"),
            "exists": "true",
            "bytes": str(manifest.stat().st_size),
            "sha256": sha256(manifest),
        }
    )
    zip_bundle(paths.bundle_dir, paths.zip_path)
    write_latest_report(paths, rows)
    return paths, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stamp",
        default=datetime.now().strftime("%Y%m%d"),
        help="Date stamp used in the output bundle name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths, rows = build_bundle(args.stamp)
    print(f"bundle_dir={paths.bundle_dir}")
    print(f"zip_path={paths.zip_path}")
    print(f"zip_bytes={paths.zip_path.stat().st_size}")
    print(f"zip_sha256={sha256(paths.zip_path)}")
    print(f"entry_count={len(rows)}")


if __name__ == "__main__":
    main()
