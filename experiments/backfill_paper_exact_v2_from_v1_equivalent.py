from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1 = "paper_exact_attempt_movielens_v1"
V2 = "paper_exact_attempt_movielens_v2"
CONFIG_V1 = ROOT / "configs" / f"{V1}.json"
CONFIG_V2 = ROOT / "configs" / f"{V2}.json"
ROW_V1 = ROOT / "outputs" / "row_level_predictions" / V1
ROW_V2 = ROOT / "outputs" / "row_level_predictions" / V2
TABLE_DIR = ROOT / "outputs" / "tables"
MANIFEST = TABLE_DIR / f"{V2}_backfill_manifest.csv"


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_base_config_equivalent(v1: dict, v2: dict) -> None:
    for key in ["dataset", "split", "mf", "acquisition", "seeds"]:
        if v1[key] != v2[key]:
            raise ValueError(f"Base config differs for {key}; refusing artifact backfill.")


def overlapping_variants(v1: dict, v2: dict) -> list[str]:
    v2_by_label = {variant["label"]: variant for variant in v2["variants"]}
    labels: list[str] = []
    for variant in v1["variants"]:
        label = variant["label"]
        if label not in v2_by_label:
            continue
        if variant != v2_by_label[label]:
            raise ValueError(f"Variant config differs for {label}; refusing artifact backfill.")
        labels.append(label)
    return labels


def copy_row_level(labels: list[str]) -> list[dict[str, object]]:
    ROW_V2.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for label in labels:
        src = ROW_V1 / f"{safe_label(label)}__seed_00.csv"
        dst = ROW_V2 / f"{safe_label(label)}__seed_00.csv"
        status = "missing_source"
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
                status = "copied"
            else:
                status = "already_exists"
        rows.append(
            {
                "artifact_type": "row_level_predictions",
                "variant": label,
                "seed": 0,
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                "destination": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "status": status,
            }
        )
    return rows


def copy_metrics(labels: list[str]) -> list[dict[str, object]]:
    src = TABLE_DIR / f"{V1}_metrics.csv"
    dst = TABLE_DIR / f"{V2}_metrics.csv"
    rows: list[dict[str, object]] = []
    if not src.exists():
        return [
            {
                "artifact_type": "metrics",
                "variant": "",
                "seed": "",
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                "destination": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "status": "missing_source",
            }
        ]
    source = pd.read_csv(src)
    copied = source[(source["variant"].isin(labels)) & (source["seed"] == 0)].copy()
    if dst.exists():
        existing = pd.read_csv(dst)
        keys = set((str(row.variant), int(row.seed), int(row.cycle)) for row in existing.itertuples(index=False))
        copied = copied[
            [
                (str(row.variant), int(row.seed), int(row.cycle)) not in keys
                for row in copied.itertuples(index=False)
            ]
        ]
        out = pd.concat([existing, copied], ignore_index=True, sort=False)
    else:
        out = copied
    out.to_csv(dst, index=False)
    for label in labels:
        rows.append(
            {
                "artifact_type": "metrics",
                "variant": label,
                "seed": 0,
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                "destination": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "status": "copied_or_already_present",
            }
        )
    return rows


def main() -> None:
    v1 = load_json(CONFIG_V1)
    v2 = load_json(CONFIG_V2)
    assert_base_config_equivalent(v1, v2)
    labels = overlapping_variants(v1, v2)
    manifest_rows = copy_row_level(labels) + copy_metrics(labels)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(MANIFEST, index=False)
    print(MANIFEST)
    print(f"backfilled_labels={labels}")


if __name__ == "__main__":
    main()
