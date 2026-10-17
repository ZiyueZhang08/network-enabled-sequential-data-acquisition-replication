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

from experiments.run_linucb_alpha_reg_sensitivity_v2 import float_token
from experiments.run_paper_faithful_movielens import make_u1_fallback_split
from src.acquisition import AcquisitionConfig, run_policy
from src.data_loader import load_movielens_100k
from src.matrix_factorization import MFConfig
from src.utils import ensure_dir, load_json


PREFIX = "inverse_reconstruction_v2"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
CONFIG_PATH = ROOT / "configs" / "paper_exact_attempt_movielens_v2.json"
TARGET_PATH = (
    TABLE_DIR
    / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
    / "tau1_figure4_digitized_targets.csv"
)
V2_METRICS_PATH = TABLE_DIR / "paper_exact_attempt_movielens_v2_metrics.csv"
ALPHA_REG_METRICS_PATH = TABLE_DIR / "linucb_alpha_reg_sensitivity_v2_metrics.csv"
CUSTOM_METRICS_PATH = TABLE_DIR / f"{PREFIX}_linucb_candidate_confirm_metrics.csv"
CONFIRM_PATH = TABLE_DIR / f"{PREFIX}_linucb_candidate_confirm.csv"
REPORT_PATH = REPORT_DIR / f"{PREFIX}_linucb_protocol_audit.md"

BASE_CANDIDATES = ["linucb_concat", "linucb_mf_interaction", "linucb_bias_interaction"]
POLICY = "linucb_mf_interaction"
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


def candidate_label(alpha: float, regularization: float) -> str:
    return f"linucb_mf_alpha_{float_token(alpha)}_reg_{float_token(regularization)}"


def target_curve(targets: pd.DataFrame, figure: str) -> pd.DataFrame:
    return targets[(targets["figure"] == figure) & (targets["method"] == "Linear UCB")][["cycle", "paper_value"]]


def curve_error(metrics: pd.DataFrame, value_col: str, targets: pd.DataFrame, figure: str) -> float:
    local = metrics.groupby("cycle", as_index=False).agg(local_value=(value_col, "mean"))
    merged = local.merge(target_curve(targets, figure), on="cycle", how="inner")
    if merged.empty:
        return math.nan
    return rmse(merged["local_value"] - merged["paper_value"])


def final_error(metrics: pd.DataFrame, value_col: str, targets: pd.DataFrame, figure: str) -> float:
    local = metrics.groupby("cycle", as_index=False).agg(local_value=(value_col, "mean")).sort_values("cycle")
    target = target_curve(targets, figure).sort_values("cycle")
    if local.empty or target.empty:
        return math.nan
    return float(abs(local.tail(1)["local_value"].iloc[0] - target.tail(1)["paper_value"].iloc[0]))


def select_alpha_reg_candidates() -> pd.DataFrame:
    if not ALPHA_REG_METRICS_PATH.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(ALPHA_REG_METRICS_PATH)
    if metrics.empty:
        return pd.DataFrame()
    targets = pd.read_csv(TARGET_PATH)
    rows: list[dict[str, object]] = []
    for variant, group in metrics.groupby("variant", sort=False):
        if "ucb_alpha" not in group.columns or "ucb_regularization" not in group.columns:
            continue
        fig4a = curve_error(group, PRIMARY_FIG4A_METRIC, targets, "Figure 4a")
        fig4b = curve_error(group, "cumulative_reward", targets, "Figure 4b")
        paper_final_reward = float(target_curve(targets, "Figure 4b").sort_values("cycle").tail(1)["paper_value"].iloc[0])
        rows.append(
            {
                "candidate": str(variant),
                "policy": POLICY,
                "ucb_alpha": float(group["ucb_alpha"].iloc[0]),
                "ucb_regularization": float(group["ucb_regularization"].iloc[0]),
                "selection_source": "alpha_reg_sensitivity_v2",
                "screen_seed_count": int(group["seed"].nunique()),
                "screen_fig4a_linear_curve_error": fig4a,
                "screen_fig4b_linear_reward_error": fig4b,
                "screen_fig4b_linear_reward_error_norm": fig4b / max(abs(paper_final_reward), 1.0),
            }
        )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    best_a = candidates.sort_values("screen_fig4a_linear_curve_error").head(1)
    best_b = candidates.sort_values("screen_fig4b_linear_reward_error_norm").head(1)
    out = pd.concat([best_a, best_b], ignore_index=True).drop_duplicates("candidate")
    return out.reset_index(drop=True)


def candidate_descriptors() -> pd.DataFrame:
    base_rows = [
        {
            "candidate": name,
            "policy": name,
            "ucb_alpha": np.nan,
            "ucb_regularization": np.nan,
            "selection_source": "v2_20seed_base",
            "screen_seed_count": 20,
        }
        for name in BASE_CANDIDATES
    ]
    custom = select_alpha_reg_candidates()
    return pd.concat([pd.DataFrame(base_rows), custom], ignore_index=True, sort=False).drop_duplicates("candidate")


def load_dataset_and_config(config_path: Path) -> tuple[dict, object, np.ndarray | None]:
    config = load_json(config_path)
    dataset_cfg = config["dataset"]
    dataset = load_movielens_100k(ROOT / dataset_cfg["raw_dir"], network_mode=dataset_cfg.get("network_mode", "gcmc_features"))
    network_embedding = None
    embedding_mode = str(dataset_cfg.get("network_embedding_mode", "graph")).lower()
    if embedding_mode in {"user_features", "metadata_features", "monti_features", "gcmc_features"}:
        if dataset.user_features is None:
            raise ValueError(f"Dataset {dataset.name} does not expose user_features for {embedding_mode}.")
        network_embedding = dataset.user_features
    return config, dataset, network_embedding


def existing_custom_metrics(selected: pd.DataFrame) -> pd.DataFrame:
    if not ALPHA_REG_METRICS_PATH.exists() or selected.empty:
        return pd.DataFrame()
    metrics = pd.read_csv(ALPHA_REG_METRICS_PATH)
    wanted = set(selected["candidate"].astype(str))
    metrics = metrics[metrics["variant"].astype(str).isin(wanted)].copy()
    return metrics


def run_custom_candidates(
    config_path: Path = CONFIG_PATH,
    seeds: list[int] | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    seeds = list(range(20)) if seeds is None else [int(seed) for seed in seeds]
    selected = candidate_descriptors()
    selected = selected[selected["selection_source"].astype(str) == "alpha_reg_sensitivity_v2"].copy()
    if selected.empty:
        return pd.DataFrame()
    config, dataset, network_embedding = load_dataset_and_config(config_path)
    split_cfg = config["split"]
    num_items = int(split_cfg.get("num_items", 1681))
    num_users = int(split_cfg.get("num_users", dataset.num_users))
    mf_config = MFConfig(**config["mf"])
    base_acquisition = AcquisitionConfig(**config["acquisition"])
    completed: set[tuple[str, int]] = set()
    existing = pd.DataFrame()
    wrote_header = False
    if resume and CUSTOM_METRICS_PATH.exists():
        existing = pd.read_csv(CUSTOM_METRICS_PATH)
        completed = {
            (str(row.variant), int(row.seed))
            for row in existing[["variant", "seed"]].drop_duplicates().itertuples(index=False)
        }
        wrote_header = True
    alpha_existing = existing_custom_metrics(selected)
    if not alpha_existing.empty:
        completed.update(
            (str(row.variant), int(row.seed))
            for row in alpha_existing[["variant", "seed"]].drop_duplicates().itertuples(index=False)
        )
    frames: list[pd.DataFrame] = [existing] if not existing.empty else []
    for seed in seeds:
        split = make_u1_fallback_split(
            ROOT / config["dataset"]["raw_dir"],
            seed=int(seed),
            initial_size=int(split_cfg.get("initial_size", 1364)),
            num_items=num_items,
        )
        for row in selected.itertuples(index=False):
            label = str(row.candidate)
            if (label, int(seed)) in completed:
                continue
            acquisition = replace(
                base_acquisition,
                selection="topk",
                ucb_alpha=float(row.ucb_alpha),
                ucb_regularization=float(row.ucb_regularization),
            )
            metrics = run_policy(
                policy=POLICY,
                split=split,
                adjacency=dataset.adjacency,
                num_users=num_users,
                num_items=num_items,
                mf_config=mf_config,
                acquisition_config=acquisition,
                seed=int(seed),
                network_embedding=network_embedding,
            )
            metrics.insert(0, "protocol", str(split_cfg.get("mode", "movielens_u1_fallback_table1_density")))
            metrics.insert(1, "variant", label)
            metrics.insert(2, "seed", int(seed))
            metrics.insert(3, "initial_size", len(split.initial))
            metrics.insert(4, "test_size", len(split.test))
            metrics.insert(5, "ucb_alpha", float(row.ucb_alpha))
            metrics.insert(6, "ucb_regularization", float(row.ucb_regularization))
            metrics.to_csv(CUSTOM_METRICS_PATH, mode="a" if wrote_header else "w", header=not wrote_header, index=False)
            wrote_header = True
            frames.append(metrics)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def all_candidate_metrics(descriptors: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if V2_METRICS_PATH.exists():
        base = pd.read_csv(V2_METRICS_PATH)
        base = base[base["variant"].astype(str).isin(BASE_CANDIDATES)].copy()
        frames.append(base)
    custom_existing = existing_custom_metrics(descriptors)
    if not custom_existing.empty:
        frames.append(custom_existing)
    if CUSTOM_METRICS_PATH.exists():
        custom = pd.read_csv(CUSTOM_METRICS_PATH)
        frames.append(custom)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(subset=["variant", "seed", "cycle"], keep="last").reset_index(drop=True)


def build_confirmation() -> pd.DataFrame:
    descriptors = candidate_descriptors()
    metrics = all_candidate_metrics(descriptors)
    if metrics.empty:
        return pd.DataFrame()
    targets = pd.read_csv(TARGET_PATH)
    paper_final_reward = float(target_curve(targets, "Figure 4b").sort_values("cycle").tail(1)["paper_value"].iloc[0])
    rows: list[dict[str, object]] = []
    descriptor_lookup = {str(row.candidate): row for row in descriptors.itertuples(index=False)}
    for candidate, group in metrics.groupby("variant", sort=False):
        desc = descriptor_lookup.get(str(candidate))
        fig4a = curve_error(group, PRIMARY_FIG4A_METRIC, targets, "Figure 4a")
        fig4b = curve_error(group, "cumulative_reward", targets, "Figure 4b")
        endpoint_a = final_error(group, PRIMARY_FIG4A_METRIC, targets, "Figure 4a")
        endpoint_b = final_error(group, "cumulative_reward", targets, "Figure 4b")
        score = fig4a + fig4b / max(abs(paper_final_reward), 1.0) + endpoint_a + endpoint_b / max(abs(paper_final_reward), 1.0)
        seed_count = int(group["seed"].nunique())
        cycle_count = int(group["cycle"].nunique())
        near_curve = fig4a <= 0.05 and fig4b / max(abs(paper_final_reward), 1.0) <= 0.05 and seed_count >= 20
        rows.append(
            {
                "candidate": str(candidate),
                "policy": str(group["policy"].iloc[0]) if "policy" in group.columns else POLICY,
                "selection_source": str(getattr(desc, "selection_source", "unknown")) if desc is not None else "unknown",
                "feature_map": "concat" if str(candidate) == "linucb_concat" else ("bias_interaction" if str(candidate) == "linucb_bias_interaction" else "mf_interaction"),
                "ucb_alpha": float(group["ucb_alpha"].iloc[0]) if "ucb_alpha" in group.columns and pd.notna(group["ucb_alpha"].iloc[0]) else np.nan,
                "ucb_regularization": float(group["ucb_regularization"].iloc[0]) if "ucb_regularization" in group.columns and pd.notna(group["ucb_regularization"].iloc[0]) else np.nan,
                "seed_count": seed_count,
                "cycle_count": cycle_count,
                "figure4a_linear_curve_error": fig4a,
                "figure4a_linear_endpoint_error": endpoint_a,
                "figure4b_linear_reward_error": fig4b,
                "figure4b_linear_reward_error_norm": fig4b / max(abs(paper_final_reward), 1.0),
                "figure4b_linear_endpoint_error": endpoint_b,
                "figure4b_linear_endpoint_error_norm": endpoint_b / max(abs(paper_final_reward), 1.0),
                "linear_ucb_joint_score": score,
                "protocol_status": "clean_room_proxy_unverified_protocol",
                "paper_linear_ucb_reconstructed_candidate": "yes_unverified_protocol" if near_curve else "no",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["linear_ucb_joint_score", "figure4a_linear_curve_error"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def write_report(confirm: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    promoted = bool((confirm.get("paper_linear_ucb_reconstructed_candidate", pd.Series(dtype=str)).astype(str) != "no").any())
    lines = [
        "# inverse_reconstruction_v2 Linear UCB protocol audit",
        "",
        "Scope: confirm the most relevant clean-room Linear UCB comparator candidates on the current u1 fallback mask.",
        "",
        f"- Confirm table: `{CONFIRM_PATH.relative_to(ROOT).as_posix()}`",
        f"- Custom metrics: `{CUSTOM_METRICS_PATH.relative_to(ROOT).as_posix()}`",
        f"- Paper Linear UCB reconstructed candidate: `{'yes_unverified_protocol' if promoted else 'no'}`",
        "",
        "## Candidate Ranking",
        "",
        md_table(confirm, max_rows=20),
        "",
        "## Claim Boundary",
        "",
        "The paper does not expose Linear UCB feature construction, alpha, regularization, reward transform, or update protocol. A close empirical row remains an unverified clean-room protocol candidate, not an exact comparator.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize() -> pd.DataFrame:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    confirm = build_confirmation()
    confirm.to_csv(CONFIRM_PATH, index=False)
    write_report(confirm)
    return confirm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    if not args.skip_run:
        run_custom_candidates(Path(args.config), seeds=[int(seed) for seed in args.seeds], resume=bool(args.resume))
        print(CUSTOM_METRICS_PATH)
    confirm = finalize()
    print(CONFIRM_PATH)
    print(REPORT_PATH)
    print(confirm.head(10).to_string(index=False) if not confirm.empty else "No Linear UCB confirmation rows.")


if __name__ == "__main__":
    main()
