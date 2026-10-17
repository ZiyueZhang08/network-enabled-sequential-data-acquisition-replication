from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, run_policy
from src.data_loader import RATING_COLUMNS, RatingSplit, load_movielens_100k
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label)


def parse_prediction_cycles(spec: str, total_cycles: int) -> set[int] | None:
    value = str(spec).strip().lower()
    if value in {"all", "*"}:
        return None
    if value in {"final", "last"}:
        return {int(total_cycles)}
    cycles: set[int] = set()
    for part in value.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        cycles.add(int(part))
    return cycles


def read_fold(path: Path, num_items: int) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    frame["user_id"] = frame["user_id"].astype(int) - 1
    frame["item_id"] = frame["item_id"].astype(int) - 1
    frame = frame[frame["item_id"] < int(num_items)].copy()
    return frame[RATING_COLUMNS].reset_index(drop=True)


def make_u1_fallback_split(raw_dir: Path, seed: int, initial_size: int, num_items: int) -> RatingSplit:
    train = read_fold(raw_dir / "u1.base", num_items=num_items)
    test = read_fold(raw_dir / "u1.test", num_items=num_items)
    rng = np.random.default_rng(seed)
    size = min(int(initial_size), len(train))
    initial_idx = rng.choice(len(train), size=size, replace=False)
    initial_mask = np.zeros(len(train), dtype=bool)
    initial_mask[initial_idx] = True
    return RatingSplit(
        initial=train.loc[initial_mask, RATING_COLUMNS].reset_index(drop=True),
        candidate=train.loc[~initial_mask, RATING_COLUMNS].reset_index(drop=True),
        test=test[RATING_COLUMNS].reset_index(drop=True),
    )


def summarize_final(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (variant, policy), group in final.groupby(["variant", "policy"], sort=False):
        row: dict[str, float | str] = {
            "variant": str(variant),
            "policy": str(policy),
            "seed_count": float(group["seed"].nunique()),
        }
        for metric in [
            "pdf_literal_norm_rmse",
            "standard_mask_rmse",
            "cumulative_reward",
            "centered_cumulative_reward",
            "reward_tail_proxy_rmse",
            "rmse",
            "w_rmse",
            "observed_size",
            "candidate_size",
        ]:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[metric] = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_faithful_movielens_v1.json"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-prefix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-row-level-predictions", action="store_true")
    parser.add_argument("--prediction-cycles", default="final", help="final, all, or comma-separated cycle numbers")
    parser.add_argument("--row-level-dir")
    parser.add_argument(
        "--row-level-compression",
        choices=["none", "gzip"],
        default="none",
        help="Compression for per variant/seed row-level prediction shards.",
    )
    parser.add_argument("--variants", nargs="+", help="Optional variant labels to run from the config")
    args = parser.parse_args()

    config = load_json(args.config)
    table_dir = ensure_dir(ROOT / "outputs" / "tables")
    dataset_cfg = config["dataset"]
    raw_dir = ROOT / dataset_cfg["raw_dir"]
    dataset = load_movielens_100k(raw_dir, network_mode=dataset_cfg.get("network_mode", "gcmc_features"))
    network_embedding = None
    embedding_mode = str(dataset_cfg.get("network_embedding_mode", "graph")).lower()
    if embedding_mode in {"user_features", "metadata_features", "monti_features", "gcmc_features"}:
        if dataset.user_features is None:
            raise ValueError(f"Dataset {dataset.name} does not expose user_features for network_embedding_mode={embedding_mode}.")
        network_embedding = dataset.user_features

    split_cfg = config["split"]
    num_items = int(split_cfg.get("num_items", 1681))
    num_users = int(split_cfg.get("num_users", dataset.num_users))
    seeds = args.seeds if args.seeds is not None else config["seeds"]
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    variants = config["variants"]
    if args.variants:
        requested = set(args.variants)
        variants = [variant for variant in variants if str(variant["label"]) in requested]
        missing = sorted(requested - {str(variant["label"]) for variant in variants})
        if missing:
            raise ValueError(f"Requested variants are not in the config: {missing}")
    prefix = args.output_prefix or config.get("output_prefix", "paper_faithful_movielens_cleanroom_v1")
    metrics_path = table_dir / f"{prefix}_metrics.csv"
    row_level_dir = None
    if args.save_row_level_predictions:
        row_level_dir = ensure_dir(Path(args.row_level_dir) if args.row_level_dir else ROOT / "outputs" / "row_level_predictions" / prefix)
    prediction_cycles = parse_prediction_cycles(args.prediction_cycles, int(base_acquisition.cycles))

    all_metrics: list[pd.DataFrame] = []
    completed: set[tuple[str, int]] = set()
    wrote_header = False
    if args.resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        all_metrics.append(existing)
        completed = {
            (str(row.variant), int(row.seed))
            for row in existing[["variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True

    for seed in seeds:
        split = make_u1_fallback_split(
            raw_dir,
            seed=int(seed),
            initial_size=int(split_cfg.get("initial_size", 1364)),
            num_items=num_items,
        )
        for variant in variants:
            label = str(variant["label"])
            row_level_suffix = ".csv.gz" if args.row_level_compression == "gzip" else ".csv"
            row_level_path = row_level_dir / f"{safe_label(label)}__seed_{int(seed):02d}{row_level_suffix}" if row_level_dir is not None else None
            metrics_done = (label, int(seed)) in completed
            row_level_done = row_level_path is not None and row_level_path.exists()
            if metrics_done and (row_level_dir is None or row_level_done):
                continue
            acquisition = base_acquisition
            if variant.get("acquisition_overrides"):
                acquisition = replace(base_acquisition, **variant["acquisition_overrides"])
            prediction_sink: list[pd.DataFrame] | None = [] if row_level_dir is not None else None
            metrics = run_policy(
                policy=str(variant["policy"]),
                split=split,
                adjacency=dataset.adjacency,
                num_users=num_users,
                num_items=num_items,
                mf_config=mf_config,
                acquisition_config=acquisition,
                seed=int(seed),
                network_embedding=network_embedding,
                prediction_sink=prediction_sink,
                prediction_cycles=prediction_cycles,
            )
            if prediction_sink is not None and row_level_path is not None:
                predictions = pd.concat(prediction_sink, ignore_index=True) if prediction_sink else pd.DataFrame()
                if not predictions.empty:
                    predictions["method"] = label
                    predictions.insert(0, "protocol", str(split_cfg.get("mode", "movielens_u1_fallback_table1_density")))
                    predictions.insert(1, "variant", label)
                    predictions.insert(2, "seed", int(seed))
                    predictions.insert(3, "initial_size", len(split.initial))
                    predictions.insert(4, "test_size", len(split.test))
                    predictions.to_csv(
                        row_level_path,
                        index=False,
                        compression="gzip" if args.row_level_compression == "gzip" else None,
                    )
            if not metrics_done:
                metrics.insert(0, "protocol", str(split_cfg.get("mode", "movielens_u1_fallback_table1_density")))
                metrics.insert(1, "variant", label)
                metrics.insert(2, "seed", int(seed))
                metrics.insert(3, "initial_size", len(split.initial))
                metrics.insert(4, "test_size", len(split.test))
                all_metrics.append(metrics)
                metrics.to_csv(metrics_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
                wrote_header = True

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    final = metrics_df.sort_values("cycle").groupby(["variant", "seed"], as_index=False).tail(1)
    summary = summarize_final(final)
    summary.to_csv(table_dir / f"{prefix}_final_summary.csv", index=False)
    print(f"Wrote metrics: {metrics_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
