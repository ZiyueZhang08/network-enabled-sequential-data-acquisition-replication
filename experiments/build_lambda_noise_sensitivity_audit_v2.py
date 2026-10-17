from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.acquisition import AcquisitionConfig, _du_net_combined_score, _select_rows
from src.data_loader import RATING_COLUMNS, RatingSplit, load_movielens_100k
from src.dls_paper import scoped_gaussian_dls_user_factors
from src.entropy_schur import schur_conditional_variance_entropy_scores
from src.matrix_factorization import MFConfig, MatrixFactorization
from src.reward import predicted_reward_scores
from src.utils import load_json


PREFIX = "paper_exact_attempt_movielens_v2"
OUT_CSV = ROOT / "outputs" / "tables" / f"{PREFIX}_lambda_noise_sensitivity_audit.csv"
OUT_REPORT = ROOT / "reports" / f"{PREFIX}_lambda_noise_sensitivity_audit.md"


def read_fold(path: Path, num_items: int) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"], engine="python")
    frame["user_id"] = frame["user_id"].astype(int) - 1
    frame["item_id"] = frame["item_id"].astype(int) - 1
    frame = frame[frame["item_id"] < int(num_items)].copy()
    return frame[RATING_COLUMNS].reset_index(drop=True)


def make_u1_fallback_split(raw_dir: Path, seed: int, initial_size: int, num_items: int) -> RatingSplit:
    train = read_fold(raw_dir / "u1.base", num_items=num_items)
    test = read_fold(raw_dir / "u1.test", num_items=num_items)
    rng = np.random.default_rng(seed)
    initial_idx = rng.choice(len(train), size=min(int(initial_size), len(train)), replace=False)
    mask = np.zeros(len(train), dtype=bool)
    mask[initial_idx] = True
    return RatingSplit(
        initial=train.loc[mask, RATING_COLUMNS].reset_index(drop=True),
        candidate=train.loc[~mask, RATING_COLUMNS].reset_index(drop=True),
        test=test[RATING_COLUMNS].reset_index(drop=True),
    )


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.copy()
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


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(len(set(map(int, a)).intersection(set(map(int, b)))) / max(min(len(a), len(b)), 1))


def prepare_seed(seed: int, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, AcquisitionConfig]:
    dataset_cfg = config["dataset"]
    split_cfg = config["split"]
    raw_dir = ROOT / dataset_cfg["raw_dir"]
    dataset = load_movielens_100k(raw_dir, network_mode=dataset_cfg.get("network_mode", "gcmc_features"))
    if dataset.user_features is None:
        raise ValueError("MovieLens user_features are required for this audit.")
    split = make_u1_fallback_split(
        raw_dir,
        seed=seed,
        initial_size=int(split_cfg["initial_size"]),
        num_items=int(split_cfg["num_items"]),
    )
    mf_config = MFConfig(**config["mf"])
    acquisition = AcquisitionConfig(**config["acquisition"])
    model = MatrixFactorization(int(split_cfg["num_users"]), int(split_cfg["num_items"]), mf_config, seed=seed * 1009)
    model.fit(split.initial)
    base_user_factors = model.user_representations()
    dls_user_factors, _ = scoped_gaussian_dls_user_factors(
        base_user_factors,
        dataset.user_features,
        None,
        None,
        sigma=float(acquisition.sigma),
        scope=str(acquisition.paper_dls_scope),
    )
    reward = predicted_reward_scores(model, split.candidate, user_factors=dls_user_factors)
    return split.initial, split.candidate, dls_user_factors, model.item_factors, acquisition, reward


def run_seed(seed: int, config: dict, lambdas: list[float]) -> pd.DataFrame:
    observed, candidates, user_factors, item_factors, acquisition, reward = prepare_seed(seed, config)
    base_lambda = float(acquisition.entropy_logdet_regularization)
    base_rows = compute_for_lambda(seed, observed, candidates, user_factors, item_factors, acquisition, reward, base_lambda)
    base_top100 = base_rows.pop("_top100")
    base_batch = base_rows.pop("_batch")
    rows: list[dict[str, object]] = []
    for lambda_noise in lambdas:
        current = compute_for_lambda(seed, observed, candidates, user_factors, item_factors, acquisition, reward, lambda_noise)
        top100 = current.pop("_top100")
        batch = current.pop("_batch")
        current.update(
            {
                "base_lambda_noise": base_lambda,
                "top100_overlap_vs_base": overlap(top100, base_top100),
                "selected_batch_overlap_vs_base": overlap(batch, base_batch),
                "selected_changed_count_vs_base": int(len(set(map(int, batch)).symmetric_difference(set(map(int, base_batch))))),
            }
        )
        rows.append(current)
    return pd.DataFrame(rows)


def compute_for_lambda(
    seed: int,
    observed: pd.DataFrame,
    candidates: pd.DataFrame,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    acquisition: AcquisitionConfig,
    reward: np.ndarray,
    lambda_noise: float,
) -> dict[str, object]:
    entropy = schur_conditional_variance_entropy_scores(
        candidates,
        observed,
        user_factors,
        item_factors,
        lambda_noise=float(lambda_noise),
        shortlist_size=int(acquisition.paper_entropy_shortlist_size),
        batch_size=int(acquisition.paper_entropy_batch_size),
    )
    score = _du_net_combined_score(reward, entropy, acquisition)
    top100 = np.argsort(score)[-100:]
    rng = np.random.default_rng(seed)
    batch = _select_rows(
        score,
        int(acquisition.batch_size),
        rng,
        str(acquisition.selection),
        temperature=float(acquisition.selection_temperature),
    )
    return {
        "seed": seed,
        "lambda_noise": float(lambda_noise),
        "candidate_count": int(len(candidates)),
        "selected_mean_rating": float(np.mean(candidates.iloc[batch]["rating"])) if len(batch) else np.nan,
        "selected_mean_entropy": float(np.mean(entropy[batch])) if len(batch) else np.nan,
        "selected_mean_score": float(np.mean(score[batch])) if len(batch) else np.nan,
        "entropy_mean": float(np.mean(entropy)),
        "entropy_max": float(np.max(entropy)),
        "entropy_nonzero_share": float(np.mean(entropy > 0)),
        "score_mean": float(np.mean(score)),
        "score_std": float(np.std(score)),
        "_top100": top100,
        "_batch": batch,
    }


def write_report(rows: pd.DataFrame) -> None:
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        rows.groupby("lambda_noise", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            min_top100_overlap=("top100_overlap_vs_base", "min"),
            min_selected_batch_overlap=("selected_batch_overlap_vs_base", "min"),
            max_selected_changed_count=("selected_changed_count_vs_base", "max"),
            mean_selected_rating=("selected_mean_rating", "mean"),
            mean_selected_entropy=("selected_mean_entropy", "mean"),
        )
        .sort_values("lambda_noise")
    )
    lines = [
        "# Lambda Noise Sensitivity Audit v2",
        "",
        "Scope: Phase 4 diagnostic for the hidden eta^2/sigma^2 / lambda-noise convention in the Schur scorer. This is a cycle0 frozen-scorer audit, not a full endpoint rerun.",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Per-Seed Detail",
        "",
        md_table(rows),
        "",
        "## Decision",
        "",
        "If changing lambda_noise changes top100 or selected batches, the paper's unpublished noise convention remains a live exactness blocker. A source-grounded estimator or source disclosure is required before this gate can be closed.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / f"{PREFIX}.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.3, 1.0, 3.0, 10.0])
    args = parser.parse_args()

    config = load_json(args.config)
    rows = pd.concat([run_seed(seed, config, args.lambdas) for seed in args.seeds], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(OUT_REPORT)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
