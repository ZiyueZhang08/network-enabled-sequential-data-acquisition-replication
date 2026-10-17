from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ml-100k"
PROCESSED_DIR = ROOT / "data" / "processed"
RATING_COLUMNS = ["user_id", "item_id", "rating"]


def _read_fold(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    frame["user_id"] = frame["user_id"].astype(int) - 1
    frame["item_id"] = frame["item_id"].astype(int) - 1
    return frame[RATING_COLUMNS].reset_index(drop=True)


def _candidate_artifacts(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if not any(token in lower for token in ("movielens", "ml_100k", "ml-100k", "gcmc", "monti")):
            continue
        if path.suffix.lower() not in {".mat", ".npy", ".npz", ".pkl", ".csv", ".txt"}:
            continue
        row: dict[str, object] = {
            "path": str(path.relative_to(ROOT)),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "exact_943x1681_mask": False,
        }
        if path.suffix.lower() == ".npy":
            try:
                arr = np.load(path, mmap_mode="r")
                row["shape"] = list(arr.shape)
                row["exact_943x1681_mask"] = tuple(arr.shape) == (943, 1681)
            except Exception as exc:  # pragma: no cover - diagnostic script path
                row["load_error"] = str(exc)
        elif path.suffix.lower() == ".npz":
            try:
                with np.load(path) as archive:
                    row["keys"] = sorted(archive.files)
                    row["shapes"] = {key: list(archive[key].shape) for key in archive.files}
                    row["exact_943x1681_mask"] = any(tuple(archive[key].shape) == (943, 1681) for key in archive.files)
            except Exception as exc:  # pragma: no cover - diagnostic script path
                row["load_error"] = str(exc)
        rows.append(row)
    return rows


def _mask_from_ratings(ratings: pd.DataFrame, num_users: int, num_items: int) -> np.ndarray:
    mask = np.zeros((num_users, num_items), dtype=bool)
    filtered = ratings[(ratings["user_id"] < num_users) & (ratings["item_id"] < num_items)]
    mask[filtered["user_id"].to_numpy(dtype=int), filtered["item_id"].to_numpy(dtype=int)] = True
    return mask


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_u1_fallback(raw_dir: Path, output_dir: Path, initial_size: int, seed: int) -> dict[str, object]:
    train = _read_fold(raw_dir / "u1.base")
    test = _read_fold(raw_dir / "u1.test")
    train_1681 = train[train["item_id"] < 1681].reset_index(drop=True)
    test_1681 = test[test["item_id"] < 1681].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    initial_size = min(int(initial_size), len(train_1681))
    initial_pos = rng.choice(len(train_1681), size=initial_size, replace=False)
    initial_mask_rows = np.zeros(len(train_1681), dtype=bool)
    initial_mask_rows[initial_pos] = True
    initial = train_1681.loc[initial_mask_rows].reset_index(drop=True)
    candidate = train_1681.loc[~initial_mask_rows].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "initial_mask.npy", _mask_from_ratings(initial, 943, 1681))
    np.save(output_dir / "train_mask.npy", _mask_from_ratings(candidate, 943, 1681))
    np.save(output_dir / "val_mask.npy", np.zeros((943, 1681), dtype=bool))
    np.save(output_dir / "test_mask.npy", _mask_from_ratings(test_1681, 943, 1681))
    _write_json(output_dir / "user_mapping.json", {str(idx + 1): idx for idx in range(943)})
    _write_json(output_dir / "item_mapping.json", {str(idx + 1): idx for idx in range(1681)})
    provenance = {
        "split_name": "movielens_u1_fallback_table1_density",
        "source": "MovieLens 100K official u1.base/u1.test",
        "num_users": 943,
        "num_items": 1681,
        "dropped_raw_item_ids": [1682],
        "dropped_train_rows": int(len(train) - len(train_1681)),
        "dropped_test_rows": int(len(test) - len(test_1681)),
        "initial_size": int(len(initial)),
        "candidate_size": int(len(candidate)),
        "test_size": int(len(test_1681)),
        "initial_density": float(len(initial) / (943 * 1681)),
        "test_density": float(len(test_1681) / (943 * 1681)),
        "seed": int(seed),
        "caveat": "No exact Monti/DU-Net MovieLens Figure 4a masks were found locally; this is an explicitly named fallback.",
    }
    _write_json(output_dir / "provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "movielens_u1_fallback_table1_density")
    parser.add_argument("--initial-size", type=int, default=1364)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifacts = _candidate_artifacts(ROOT / "data")
    exact = [row for row in artifacts if row.get("exact_943x1681_mask")]
    audit = {
        "checked_roots": ["data/raw", "data/processed", "data/splits"],
        "candidate_count": len(artifacts),
        "exact_943x1681_candidates": exact,
        "candidate_artifacts": artifacts,
    }
    (PROCESSED_DIR / "movielens_monti_scan").mkdir(parents=True, exist_ok=True)
    _write_json(PROCESSED_DIR / "movielens_monti_scan" / "scan_report.json", audit)

    if exact:
        print("Found possible exact 943x1681 local masks; inspect data/processed/movielens_monti_scan/scan_report.json")
        return
    provenance = prepare_u1_fallback(args.raw_dir, args.output_dir, args.initial_size, args.seed)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
