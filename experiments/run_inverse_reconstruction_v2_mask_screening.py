from __future__ import annotations

import argparse
import math
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


PREFIX = "inverse_reconstruction_v2"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
MASK_MANIFEST_PATH = TABLE_DIR / "inverse_reconstruction_v1_mask_candidates_50.csv"
MASK_NPZ_PATH = ROOT / "outputs" / "masks" / "inverse_reconstruction_v1" / "table1_random_masks_v1.npz"
TARGET_PATH = (
    TABLE_DIR
    / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
    / "tau1_figure4_digitized_targets.csv"
)
U1_BASELINE_PATH = TABLE_DIR / "paper_exact_attempt_movielens_v2_allcycles_metrics.csv"
RESULTS_PATH = TABLE_DIR / f"{PREFIX}_mask_screening_results.csv"
RANKING_PATH = TABLE_DIR / f"{PREFIX}_mask_candidate_ranking.csv"
REPORT_PATH = REPORT_DIR / f"{PREFIX}_mask_screening_report.md"
CONFIG_PATH = ROOT / "configs" / "paper_exact_attempt_movielens_v2.json"

DU_METHOD = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
LINEAR_METHODS = ["linucb_concat", "linucb_mf_interaction"]
METHODS = [DU_METHOD, *LINEAR_METHODS]
BEST_DIAGNOSTIC_METRIC = "reward_tail_proxy_rmse"
PRIMARY_FIG4A_METRIC = "standard_mask_rmse"


def md_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.6g}" if pd.notna(value) else "")
        else:
            view[col] = view[col].fillna("").astype(str)

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(col) for col in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(clean(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def rmse(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan
    return float(np.sqrt(np.mean(arr * arr)))


def read_movielens_ratings(raw_dir: Path, num_items: int) -> pd.DataFrame:
    frame = pd.read_csv(
        raw_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    frame["source_row_id"] = np.arange(len(frame), dtype=int)
    frame["user_id"] = frame["user_id"].astype(int) - 1
    frame["item_id"] = frame["item_id"].astype(int) - 1
    frame = frame[frame["item_id"] < int(num_items)].copy()
    return frame[["source_row_id", *RATING_COLUMNS]].reset_index(drop=True)


def load_mask_archive() -> tuple[pd.DataFrame, np.lib.npyio.NpzFile]:
    if not MASK_MANIFEST_PATH.exists() or not MASK_NPZ_PATH.exists():
        raise FileNotFoundError("Run build_inverse_reconstruction_leaderboard.py first to create the 50 mask archive.")
    return pd.read_csv(MASK_MANIFEST_PATH), np.load(MASK_NPZ_PATH)


def mask_indices(mask_ids: list[str], manifest: pd.DataFrame) -> list[int]:
    id_to_index = {str(row.mask_id): idx for idx, row in enumerate(manifest.itertuples(index=False))}
    return [id_to_index[mask_id] for mask_id in mask_ids]


def make_mask_split(ratings: pd.DataFrame, masks: np.lib.npyio.NpzFile, mask_index: int) -> RatingSplit:
    initial_ids = set(np.asarray(masks["initial_source_row_id"][mask_index], dtype=int).tolist())
    test_ids = set(np.asarray(masks["test_source_row_id"][mask_index], dtype=int).tolist())
    if initial_ids & test_ids:
        raise ValueError(f"Mask {mask_index} has overlapping initial and test source row ids.")
    source_ids = ratings["source_row_id"].astype(int)
    initial = ratings[source_ids.isin(initial_ids)][RATING_COLUMNS].reset_index(drop=True)
    test = ratings[source_ids.isin(test_ids)][RATING_COLUMNS].reset_index(drop=True)
    candidate = ratings[~source_ids.isin(initial_ids | test_ids)][RATING_COLUMNS].reset_index(drop=True)
    return RatingSplit(initial=initial, candidate=candidate, test=test)


def load_dataset_and_config(config_path: Path) -> tuple[dict, object, np.ndarray | None]:
    config = load_json(config_path)
    dataset_cfg = config["dataset"]
    dataset = load_movielens_100k(ROOT / dataset_cfg["raw_dir"], network_mode=dataset_cfg.get("network_mode", "gcmc_features"))
    embedding_mode = str(dataset_cfg.get("network_embedding_mode", "graph")).lower()
    network_embedding = None
    if embedding_mode in {"user_features", "metadata_features", "monti_features", "gcmc_features"}:
        if dataset.user_features is None:
            raise ValueError(f"Dataset {dataset.name} does not expose user_features for {embedding_mode}.")
        network_embedding = dataset.user_features
    return config, dataset, network_embedding


def output_path_for_batch(batch_label: str | None) -> Path:
    if batch_label:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in batch_label)
        return TABLE_DIR / f"{PREFIX}_mask_screening_results_{safe}.csv"
    return RESULTS_PATH


def run_screening(
    config_path: Path = CONFIG_PATH,
    mask_start: int = 0,
    mask_count: int = 50,
    mask_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
    batch_label: str | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2] if seeds is None else [int(seed) for seed in seeds]
    methods = METHODS if methods is None else [str(method) for method in methods]
    config, dataset, network_embedding = load_dataset_and_config(config_path)
    variants = {str(variant["label"]): variant for variant in config["variants"]}
    missing = sorted(set(methods) - set(variants))
    if missing:
        raise ValueError(f"Requested methods are not in config variants: {missing}")

    manifest, masks = load_mask_archive()
    if mask_ids:
        selected_indices = mask_indices(mask_ids, manifest)
    else:
        stop = min(mask_start + mask_count, len(manifest))
        selected_indices = list(range(mask_start, stop))
    raw_dir = ROOT / config["dataset"]["raw_dir"]
    num_items = int(config["split"].get("num_items", 1681))
    ratings = read_movielens_ratings(raw_dir, num_items=num_items)
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    num_users = int(config["split"].get("num_users", dataset.num_users))
    out_path = output_path_for_batch(batch_label)
    completed: set[tuple[str, str, int]] = set()
    existing = pd.DataFrame()
    wrote_header = False
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        completed = {
            (str(row.mask_id), str(row.variant), int(row.seed))
            for row in existing[["mask_id", "variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True
    frames: list[pd.DataFrame] = [existing] if not existing.empty else []

    for mask_index in selected_indices:
        mask_row = manifest.iloc[int(mask_index)]
        mask_id = str(mask_row["mask_id"])
        split = make_mask_split(ratings, masks, int(mask_index))
        for seed in seeds:
            for method in methods:
                if (mask_id, method, int(seed)) in completed:
                    continue
                variant = variants[method]
                acquisition = base_acquisition
                if variant.get("acquisition_overrides"):
                    acquisition = replace(base_acquisition, **variant["acquisition_overrides"])
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
                )
                metrics.insert(0, "protocol", "random_table1_global_mask_screen")
                metrics.insert(1, "mask_id", mask_id)
                metrics.insert(2, "mask_index", int(mask_index))
                metrics.insert(3, "mask_seed", int(mask_row["seed"]))
                metrics.insert(4, "mask_type", str(mask_row["type"]))
                metrics.insert(5, "density_match", str(mask_row["density_match"]))
                metrics.insert(6, "candidate_xor_hash", int(mask_row["candidate_xor_hash"]))
                metrics.insert(7, "variant", method)
                metrics.insert(8, "seed", int(seed))
                metrics.insert(9, "initial_size", len(split.initial))
                metrics.insert(10, "test_size", len(split.test))
                frames.append(metrics)
                metrics.to_csv(out_path, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
                wrote_header = True
    masks.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_screening_results() -> pd.DataFrame:
    files = sorted(TABLE_DIR.glob(f"{PREFIX}_mask_screening_results*.csv"))
    frames = [pd.read_csv(path) for path in files if path.exists() and path.stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(subset=["mask_id", "variant", "seed", "cycle"], keep="last").reset_index(drop=True)


def local_curve(metrics: pd.DataFrame, method: str, value_col: str) -> pd.DataFrame:
    subset = metrics[metrics["variant"].astype(str) == method].copy()
    if subset.empty:
        return pd.DataFrame()
    return subset.groupby("cycle", as_index=False).agg(local_value=(value_col, "mean"), seed_count=("seed", "nunique"))


def target_curve(targets: pd.DataFrame, figure: str, method: str) -> pd.DataFrame:
    return targets[(targets["figure"] == figure) & (targets["method"] == method)][["cycle", "paper_value"]].copy()


def curve_error(local: pd.DataFrame, target: pd.DataFrame) -> float:
    merged = local.merge(target, on="cycle", how="inner")
    if merged.empty:
        return math.nan
    return rmse(merged["local_value"] - merged["paper_value"])


def final_value(local: pd.DataFrame) -> float:
    if local.empty:
        return math.nan
    return float(local.sort_values("cycle").tail(1)["local_value"].iloc[0])


def baseline_rows(targets: pd.DataFrame) -> pd.DataFrame:
    if not U1_BASELINE_PATH.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(U1_BASELINE_PATH)
    metrics = metrics[metrics["seed"].astype(int).isin([0, 1, 2])].copy()
    rows = []
    for linear in LINEAR_METHODS:
        rows.append(score_one_candidate("u1_fallback", linear, metrics, targets, protocol_penalty=0.0))
    return pd.DataFrame(rows)


def score_one_candidate(
    mask_id: str,
    linear_method: str,
    metrics: pd.DataFrame,
    targets: pd.DataFrame,
    protocol_penalty: float,
) -> dict[str, object]:
    du_a = local_curve(metrics, DU_METHOD, PRIMARY_FIG4A_METRIC)
    linear_a = local_curve(metrics, linear_method, PRIMARY_FIG4A_METRIC)
    du_a_diag = local_curve(metrics, DU_METHOD, BEST_DIAGNOSTIC_METRIC)
    linear_a_diag = local_curve(metrics, linear_method, BEST_DIAGNOSTIC_METRIC)
    du_b = local_curve(metrics, DU_METHOD, "cumulative_reward")
    linear_b = local_curve(metrics, linear_method, "cumulative_reward")
    paper_du_a = target_curve(targets, "Figure 4a", "DU-Net")
    paper_linear_a = target_curve(targets, "Figure 4a", "Linear UCB")
    paper_du_b = target_curve(targets, "Figure 4b", "DU-Net")
    paper_linear_b = target_curve(targets, "Figure 4b", "Linear UCB")
    paper_final_gap = final_value(paper_linear_a.rename(columns={"paper_value": "local_value"})) - final_value(
        paper_du_a.rename(columns={"paper_value": "local_value"})
    )
    local_final_gap = final_value(linear_a) - final_value(du_a)
    fig4b_scale = max(
        abs(final_value(paper_du_b.rename(columns={"paper_value": "local_value"}))),
        abs(final_value(paper_linear_b.rename(columns={"paper_value": "local_value"}))),
        1.0,
    )
    fig4a_du_error = curve_error(du_a, paper_du_a)
    fig4a_linear_error = curve_error(linear_a, paper_linear_a)
    fig4b_du_error = curve_error(du_b, paper_du_b)
    fig4b_linear_error = curve_error(linear_b, paper_linear_b)
    final_gap_error = abs(local_final_gap - paper_final_gap) if np.isfinite(local_final_gap) else math.nan
    total = (
        fig4a_du_error
        + fig4a_linear_error
        + (fig4b_du_error / fig4b_scale)
        + (fig4b_linear_error / fig4b_scale)
        + 5.0 * final_gap_error
        + protocol_penalty
    )
    return {
        "mask_id": mask_id,
        "du_method": DU_METHOD,
        "linear_method": linear_method,
        "metric": PRIMARY_FIG4A_METRIC,
        "diagnostic_metric": BEST_DIAGNOSTIC_METRIC,
        "seed_count": int(metrics["seed"].nunique()) if not metrics.empty else 0,
        "cycle_count": int(metrics["cycle"].nunique()) if not metrics.empty else 0,
        "figure4a_du_curve_error": fig4a_du_error,
        "figure4a_linear_curve_error": fig4a_linear_error,
        "figure4a_du_diagnostic_curve_error": curve_error(du_a_diag, paper_du_a),
        "figure4a_linear_diagnostic_curve_error": curve_error(linear_a_diag, paper_linear_a),
        "figure4b_du_reward_error": fig4b_du_error,
        "figure4b_linear_reward_error": fig4b_linear_error,
        "figure4b_du_reward_error_norm": fig4b_du_error / fig4b_scale if np.isfinite(fig4b_du_error) else math.nan,
        "figure4b_linear_reward_error_norm": fig4b_linear_error / fig4b_scale if np.isfinite(fig4b_linear_error) else math.nan,
        "local_final_gap": local_final_gap,
        "paper_final_gap": paper_final_gap,
        "final_gap_error": final_gap_error,
        "protocol_penalty": protocol_penalty,
        "mask_screen_score": total,
    }


def build_ranking(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    targets = pd.read_csv(TARGET_PATH)
    rows: list[dict[str, object]] = []
    baseline = baseline_rows(targets)
    baseline_by_linear = {str(row.linear_method): row for row in baseline.itertuples(index=False)}
    for (mask_id, _), mask_metrics in results.groupby(["mask_id", "mask_index"], sort=False):
        density_ok = str(mask_metrics["density_match"].iloc[0]).lower() == "pass"
        protocol_penalty = 0.0 if density_ok else 0.5
        for linear in LINEAR_METHODS:
            row = score_one_candidate(str(mask_id), linear, mask_metrics, targets, protocol_penalty=protocol_penalty)
            base = baseline_by_linear.get(linear)
            improves = False
            if base is not None:
                improves = (
                    row["figure4a_du_curve_error"] < float(base.figure4a_du_curve_error)
                    and row["figure4a_linear_curve_error"] < float(base.figure4a_linear_curve_error)
                    and row["figure4b_du_reward_error"] < float(base.figure4b_du_reward_error)
                    and row["figure4b_linear_reward_error"] < float(base.figure4b_linear_reward_error)
                )
                row["u1_fallback_score"] = float(base.mask_screen_score)
                row["score_improvement_vs_u1"] = float(base.mask_screen_score) - float(row["mask_screen_score"])
            else:
                row["u1_fallback_score"] = math.nan
                row["score_improvement_vs_u1"] = math.nan
            row["density_match"] = "pass" if density_ok else "fail"
            row["author_mask_reconstructed_candidate"] = "yes_not_verified_author_mask" if improves and density_ok else "no"
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["mask_screen_score", "figure4a_du_curve_error", "figure4b_du_reward_error"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def write_report(results: pd.DataFrame, ranking: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    completed = (
        results.groupby("variant")["mask_id"].nunique().to_dict() if not results.empty else {}
    )
    promoted = bool((ranking.get("author_mask_reconstructed_candidate", pd.Series(dtype=str)).astype(str) == "yes_not_verified_author_mask").any())
    lines = [
        "# inverse_reconstruction_v2 mask screening report",
        "",
        "Scope: 50 Table-1-compatible random masks, 3 acquisition seeds, tau1 DU-Net plus two Linear UCB proxies.",
        "",
        f"- Results: `{RESULTS_PATH.relative_to(ROOT).as_posix()}`",
        f"- Ranking: `{RANKING_PATH.relative_to(ROOT).as_posix()}`",
        f"- Completed masks by method: `{completed}`",
        f"- Author-mask reconstructed candidate: `{'yes_not_verified_author_mask' if promoted else 'no'}`",
        "",
        "## Top Mask Candidates",
        "",
        md_table(ranking.head(20), max_rows=20),
        "",
        "## Interpretation",
        "",
        "Rows are empirical inverse-fit diagnostics only. A promoted mask candidate, if any, is explicitly `not verified author mask`; exact mask provenance still requires author source or an author-equivalent generation script.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize_rankings() -> dict[str, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    results = load_all_screening_results()
    if not results.empty:
        results.to_csv(RESULTS_PATH, index=False)
    ranking = build_ranking(results)
    ranking.to_csv(RANKING_PATH, index=False)
    write_report(results, ranking)
    return {"results": results, "ranking": ranking}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--mask-start", type=int, default=0)
    parser.add_argument("--mask-count", type=int, default=50)
    parser.add_argument("--mask-ids", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--batch-label")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-ranking", action="store_true")
    parser.add_argument("--rank-only", action="store_true")
    args = parser.parse_args()

    if not args.rank_only:
        run_screening(
            config_path=Path(args.config),
            mask_start=int(args.mask_start),
            mask_count=int(args.mask_count),
            mask_ids=args.mask_ids,
            seeds=[int(seed) for seed in args.seeds],
            methods=[str(method) for method in args.methods],
            batch_label=args.batch_label,
            resume=bool(args.resume),
        )
        print(output_path_for_batch(args.batch_label))
    if not args.skip_ranking:
        outputs = finalize_rankings()
        print(RESULTS_PATH)
        print(RANKING_PATH)
        print(REPORT_PATH)
        print(outputs["ranking"].head(10).to_string(index=False) if not outputs["ranking"].empty else "No ranking rows yet.")


if __name__ == "__main__":
    main()
