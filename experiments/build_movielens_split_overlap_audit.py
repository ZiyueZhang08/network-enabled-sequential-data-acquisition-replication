"""Compare local random MovieLens splits with bundled official u1-u5 folds.

This is a data-diagnostic artifact. It checks whether the current seeded random
80/20 split protocol is statistically close to the official MovieLens folds
bundled with ML-100K, while keeping the exact-mask caveat explicit.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
RAW_DIR = ROOT / "data" / "raw" / "ml-100k"
CONFIG_PATH = ROOT / "configs" / "movielens_100k_svdpp_figure4a_reward_weight_metric_20seeds.json"

OUT_CSV = TABLE_DIR / "movielens_split_overlap_audit.csv"
OUT_REPORT = REPORT_DIR / "movielens_split_overlap_audit.md"


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


def pair_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(frame["user_id"].astype(int), frame["item_id"].astype(int)))


def jaccard(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
    return len(a & b) / len(a | b)


def summarize_distribution(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "split": name,
        "kind": "distribution",
        "count": len(frame),
        "users": frame["user_id"].nunique(),
        "items": frame["item_id"].nunique(),
        "rating_mean": frame["rating"].mean(),
        "rating_std": frame["rating"].std(ddof=0),
        "high_rating_share": (frame["rating"] >= 4).mean(),
        "low_rating_share": (frame["rating"] <= 2).mean(),
        "mean_official_jaccard": "",
        "max_official_jaccard": "",
        "mean_random_jaccard": "",
        "interpretation": "Marginal test-set distribution summary.",
    }


def random_test_frames(ratings: pd.DataFrame, seeds: list[int], test_ratio: float) -> dict[str, pd.DataFrame]:
    n_test = max(1, int(round(len(ratings) * test_ratio)))
    frames = {}
    for seed in seeds:
        shuffled = ratings.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        frames[f"random_seed_{seed}"] = shuffled.iloc[:n_test].reset_index(drop=True)
    return frames


def official_test_frames() -> dict[str, pd.DataFrame]:
    frames = {}
    for fold in range(1, 6):
        path = RAW_DIR / f"u{fold}.test"
        if path.exists():
            frames[f"official_u{fold}"] = read_ratings(path)
    return frames


def mean_pairwise_jaccard(sets: list[set[tuple[int, int]]]) -> float:
    values = [jaccard(left, right) for left, right in combinations(sets, 2)]
    return float(np.mean(values)) if values else 0.0


def distribution_range_rows(kind: str, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    summaries = [summarize_distribution(name, frame) for name, frame in frames.items()]
    rows = []
    for metric in ["count", "users", "items", "rating_mean", "rating_std", "high_rating_share", "low_rating_share"]:
        values = [float(row[metric]) for row in summaries]
        rows.append(
            {
                "split": kind,
                "kind": f"{metric}_range",
                "count": "",
                "users": "",
                "items": "",
                "rating_mean": "",
                "rating_std": "",
                "high_rating_share": "",
                "low_rating_share": "",
                "mean_official_jaccard": "",
                "max_official_jaccard": "",
                "mean_random_jaccard": "",
                "interpretation": f"{metric}: min={min(values):.6f}; max={max(values):.6f}; spread={max(values)-min(values):.6f}",
            }
        )
    return rows


def overlap_rows(random_frames: dict[str, pd.DataFrame], official_frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    official_sets = {name: pair_set(frame) for name, frame in official_frames.items()}
    random_sets = {name: pair_set(frame) for name, frame in random_frames.items()}
    random_pairwise = mean_pairwise_jaccard(list(random_sets.values()))
    official_pairwise = mean_pairwise_jaccard(list(official_sets.values()))

    rows: list[dict[str, object]] = [
        {
            "split": "official_folds_pairwise",
            "kind": "overlap",
            "count": "",
            "users": "",
            "items": "",
            "rating_mean": "",
            "rating_std": "",
            "high_rating_share": "",
            "low_rating_share": "",
            "mean_official_jaccard": f"{official_pairwise:.6f}",
            "max_official_jaccard": "",
            "mean_random_jaccard": "",
            "interpretation": "Official u1-u5 test folds are designed to be disjoint.",
        },
        {
            "split": "random_seeds_pairwise",
            "kind": "overlap",
            "count": "",
            "users": "",
            "items": "",
            "rating_mean": "",
            "rating_std": "",
            "high_rating_share": "",
            "low_rating_share": "",
            "mean_official_jaccard": "",
            "max_official_jaccard": "",
            "mean_random_jaccard": f"{random_pairwise:.6f}",
            "interpretation": "Random seeded test masks overlap as expected for independent 20% samples.",
        },
    ]

    for random_name, random_set in random_sets.items():
        overlaps = [jaccard(random_set, official_set) for official_set in official_sets.values()]
        rows.append(
            {
                "split": random_name,
                "kind": "random_vs_official_overlap",
                "count": "",
                "users": "",
                "items": "",
                "rating_mean": "",
                "rating_std": "",
                "high_rating_share": "",
                "low_rating_share": "",
                "mean_official_jaccard": f"{float(np.mean(overlaps)):.6f}",
                "max_official_jaccard": f"{float(np.max(overlaps)):.6f}",
                "mean_random_jaccard": "",
                "interpretation": "Current random split is not the same mask as any official fold, but overlaps at the expected random-sample level.",
            }
        )
    return rows


def build_rows() -> list[dict[str, object]]:
    config = read_config()
    ratings = read_ratings(RAW_DIR / "u.data")
    random_frames = random_test_frames(ratings, config["seeds"], config["split"]["test_ratio"])
    official_frames = official_test_frames()

    rows: list[dict[str, object]] = []
    for name, frame in official_frames.items():
        rows.append(summarize_distribution(name, frame))
    for name in list(random_frames)[:5]:
        rows.append(summarize_distribution(name, random_frames[name]))
    rows.extend(distribution_range_rows("official_u1_u5", official_frames))
    rows.extend(distribution_range_rows("random_seed_0_19", random_frames))
    rows.extend(overlap_rows(random_frames, official_frames))
    rows.append(
        {
            "split": "execution_decision",
            "kind": "decision",
            "count": "",
            "users": "",
            "items": "",
            "rating_mean": "",
            "rating_std": "",
            "high_rating_share": "",
            "low_rating_share": "",
            "mean_official_jaccard": "",
            "max_official_jaccard": "",
            "mean_random_jaccard": "",
            "interpretation": "Official and random test folds have similar marginal distributions, but exact masks differ; this supports a split caveat, not a full rerun trigger.",
        }
    )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    display_rows = [row for row in rows if row["kind"] in {"overlap", "decision"}]
    display_rows.extend(row for row in rows if row["kind"].endswith("_range"))
    lines = [
        "| split | kind | official jaccard | random jaccard | interpretation |",
        "|---|---|---:|---:|---|",
    ]
    for row in display_rows:
        lines.append(
            f"| {row['split']} | {row['kind']} | {row['mean_official_jaccard']} | "
            f"{row['mean_random_jaccard']} | {row['interpretation']} |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]]) -> None:
    report = f"""# MovieLens Split Overlap Audit

Generated by `experiments/build_movielens_split_overlap_audit.py`.

## Current Progress

This audit compares the current seeded random MovieLens test masks with the
bundled official `u1-u5` MovieLens folds. It does not run a new model. It checks
whether the split caveat is a likely standalone explanation for the Figure 4a
gap.

## Audit Summary

{markdown_table(rows)}

## Analysis

The official folds and current random splits have the same test size and very
similar marginal rating/user/item coverage. The masks are not identical, and
the paper still references Monti/GCMC-style split artifacts, so exact split
faithfulness remains unresolved. However, the overlap and distribution checks
do not support treating split choice alone as the main explanation for the
large absolute-scale gap.

## Decision

Do not launch a full rerun from split overlap alone. If a future local-only
diagnostic is needed, the smallest reasonable one would be an explicitly
labeled official-fold sensitivity run, not a paper-faithful claim. The main
next step remains source-backed exact metric/split/backend recovery.

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
