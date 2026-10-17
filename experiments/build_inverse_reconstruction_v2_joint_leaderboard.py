from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PREFIX = "inverse_reconstruction_v2"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
TARGET_PATH = (
    TABLE_DIR
    / "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
    / "tau1_figure4_digitized_targets.csv"
)
U1_METRICS_PATH = TABLE_DIR / "paper_exact_attempt_movielens_v2_metrics.csv"
U1_ALLCYCLES_METRICS_PATH = TABLE_DIR / "paper_exact_attempt_movielens_v2_allcycles_metrics.csv"
MASK_RANKING_PATH = TABLE_DIR / f"{PREFIX}_mask_candidate_ranking.csv"
LINUCB_CONFIRM_PATH = TABLE_DIR / f"{PREFIX}_linucb_candidate_confirm.csv"
METRIC_TOP50_PATH = TABLE_DIR / f"{PREFIX}_metric_curve_family_top50.csv"
OUT_PATH = TABLE_DIR / f"{PREFIX}_joint_leaderboard.csv"
REPORT_PATH = REPORT_DIR / f"{PREFIX}_joint_leaderboard_report.md"

TAU1 = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
DU_CANDIDATES = [TAU1, "lambda0p3_diagnostic", "schur_loglambda_tuned_full_pool_softmax_tau0p3"]
BASE_LINEAR_CANDIDATES = ["linucb_concat", "linucb_mf_interaction"]


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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def target_curve(targets: pd.DataFrame, figure: str, method: str) -> pd.DataFrame:
    return targets[(targets["figure"] == figure) & (targets["method"] == method)][["cycle", "paper_value"]]


def figure4b_scale(targets: pd.DataFrame) -> float:
    values = targets[targets["figure"] == "Figure 4b"].sort_values("cycle").groupby("method", as_index=False).tail(1)
    if values.empty:
        return 1.0
    return max(float(values["paper_value"].abs().max()), 1.0)


def metric_to_column(metric: str) -> str:
    name = str(metric)
    mapping = {
        "standard_rmse": "standard_mask_rmse",
        "centered_rating_rmse": "standard_mask_rmse",
        "unclipped_standard_rmse": "standard_mask_rmse",
        "reward_tail_proxy": "reward_tail_proxy_rmse",
        "current_w_rmse": "w_rmse",
        "mild_reward_w_rmse": "mild_reward_w_rmse",
        "medium_reward_w_rmse": "medium_reward_w_rmse",
        "sigmoid_w_rmse": "sigmoid_reward_w_rmse",
        "rating_weighted_rmse": "w_rmse",
        "pdf_literal_norm_rmse": "pdf_literal_norm_rmse",
    }
    return mapping.get(name, "standard_mask_rmse")


def metric_candidates() -> pd.DataFrame:
    rows = [{"metric": "standard_rmse", "metric_column": "standard_mask_rmse", "metric_status": "standard_defensible", "metric_penalty": 0.0}]
    top50 = read_csv(METRIC_TOP50_PATH)
    if not top50.empty:
        best = str(top50.iloc[0]["metric"])
        rows.append(
            {
                "metric": best,
                "metric_column": metric_to_column(best),
                "metric_status": "best_v2_metric_no_promotion",
                "metric_penalty": 0.15 if best != "standard_rmse" else 0.0,
            }
        )
        promoted = top50[top50["paper_wrmse_reconstructed_candidate"].astype(str) == "yes"]
        if promoted.empty:
            rows.append(
                {
                    "metric": "no_promoted_paper_metric_gate_failed",
                    "metric_column": metric_to_column(best),
                    "metric_status": "metric_gate_failed",
                    "metric_penalty": 0.3,
                }
            )
    return pd.DataFrame(rows).drop_duplicates("metric")


def mask_candidates() -> pd.DataFrame:
    rows = [{"mask_id": "u1_fallback", "mask_source": "current_u1_fallback", "mask_penalty": 0.0}]
    ranking = read_csv(MASK_RANKING_PATH)
    if not ranking.empty:
        seen: set[str] = set()
        for row in ranking.itertuples(index=False):
            mask_id = str(row.mask_id)
            if mask_id in seen:
                continue
            seen.add(mask_id)
            rows.append(
                {
                    "mask_id": mask_id,
                    "mask_source": str(getattr(row, "author_mask_reconstructed_candidate", "not_verified_author_mask")),
                    "mask_penalty": 0.05 if str(getattr(row, "author_mask_reconstructed_candidate", "")) != "no" else 0.15,
                }
            )
            if len(seen) >= 5:
                break
    return pd.DataFrame(rows)


def linear_candidates() -> pd.DataFrame:
    rows = [{"linear_ucb": name, "linear_source": "base_20seed", "linear_penalty": 0.05} for name in BASE_LINEAR_CANDIDATES]
    confirm = read_csv(LINUCB_CONFIRM_PATH)
    if not confirm.empty:
        best = confirm.sort_values("linear_ucb_joint_score").head(1).iloc[0]
        candidate = str(best["candidate"])
        if candidate not in {row["linear_ucb"] for row in rows}:
            rows.append(
                {
                    "linear_ucb": candidate,
                    "linear_source": str(best.get("protocol_status", "clean_room_proxy_unverified_protocol")),
                    "linear_penalty": 0.1,
                }
            )
    return pd.DataFrame(rows)


def du_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"du_net": TAU1, "du_source": "tau1_mainline", "du_penalty": 0.0},
            {"du_net": "lambda0p3_diagnostic", "du_source": "lambda_entropy_diagnostic", "du_penalty": 0.08},
            {"du_net": "schur_loglambda_tuned_full_pool_softmax_tau0p3", "du_source": "tau0p3_robustness_low_seed", "du_penalty": 0.12},
        ]
    )


def load_u1_metrics() -> pd.DataFrame:
    frames = [read_csv(U1_METRICS_PATH), read_csv(U1_ALLCYCLES_METRICS_PATH)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates(subset=["variant", "seed", "cycle"], keep="last").reset_index(drop=True)


def local_error(metrics: pd.DataFrame, variant: str, value_col: str, targets: pd.DataFrame, figure: str, target_method: str) -> tuple[float, int, int]:
    if metrics.empty or value_col not in metrics.columns:
        return math.nan, 0, 0
    subset = metrics[metrics["variant"].astype(str) == str(variant)].copy()
    if subset.empty:
        return math.nan, 0, 0
    local = subset.groupby("cycle", as_index=False).agg(local_value=(value_col, "mean"), seed_count=("seed", "nunique"))
    merged = local.merge(target_curve(targets, figure, target_method), on="cycle", how="inner")
    if merged.empty:
        return math.nan, int(subset["seed"].nunique()), int(subset["cycle"].nunique())
    return rmse(merged["local_value"] - merged["paper_value"]), int(subset["seed"].nunique()), int(subset["cycle"].nunique())


def evidence_from_u1(
    metrics: pd.DataFrame,
    du: str,
    linear: str,
    metric_col: str,
    targets: pd.DataFrame,
) -> dict[str, object]:
    fig4a_du, du_seeds, du_cycles = local_error(metrics, du, metric_col, targets, "Figure 4a", "DU-Net")
    fig4a_linear, linear_seeds, linear_cycles = local_error(metrics, linear, metric_col, targets, "Figure 4a", "Linear UCB")
    fig4b_du, _, _ = local_error(metrics, du, "cumulative_reward", targets, "Figure 4b", "DU-Net")
    fig4b_linear, _, _ = local_error(metrics, linear, "cumulative_reward", targets, "Figure 4b", "Linear UCB")
    return {
        "figure4a_du_curve_error": fig4a_du,
        "figure4a_linear_curve_error": fig4a_linear,
        "figure4b_du_curve_error": fig4b_du,
        "figure4b_linear_curve_error": fig4b_linear,
        "du_seed_count": du_seeds,
        "linear_seed_count": linear_seeds,
        "du_cycle_count": du_cycles,
        "linear_cycle_count": linear_cycles,
        "evidence_source": "u1_fallback_metrics",
    }


def evidence_from_mask(mask_id: str, linear: str, metric: str) -> dict[str, object]:
    ranking = read_csv(MASK_RANKING_PATH)
    if ranking.empty:
        return {}
    subset = ranking[(ranking["mask_id"].astype(str) == mask_id) & (ranking["linear_method"].astype(str) == linear)]
    if subset.empty:
        return {}
    row = subset.sort_values("mask_screen_score").iloc[0]
    use_diag = metric == "reward_tail_proxy"
    return {
        "figure4a_du_curve_error": float(row["figure4a_du_diagnostic_curve_error"] if use_diag else row["figure4a_du_curve_error"]),
        "figure4a_linear_curve_error": float(row["figure4a_linear_diagnostic_curve_error"] if use_diag else row["figure4a_linear_curve_error"]),
        "figure4b_du_curve_error": float(row["figure4b_du_reward_error"]),
        "figure4b_linear_curve_error": float(row["figure4b_linear_reward_error"]),
        "du_seed_count": int(row["seed_count"]),
        "linear_seed_count": int(row["seed_count"]),
        "du_cycle_count": int(row["cycle_count"]),
        "linear_cycle_count": int(row["cycle_count"]),
        "evidence_source": "mask_screening_3seed",
    }


def normalize_error(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return math.inf
    return float(value) / max(float(scale), 1.0)


def claim_status(row: dict[str, object]) -> str:
    flags: list[str] = []
    if str(row["metric_status"]) != "standard_defensible":
        flags.append("metric_gate_not_promoted")
    if str(row["mask_id"]) != "u1_fallback" and "yes_not_verified_author_mask" not in str(row["mask_source"]):
        flags.append("cherry_picked_mask_not_author_verified")
    if "unverified" in str(row["linear_source"]):
        flags.append("linear_ucb_unverified_protocol")
    if str(row["evidence_source"]) != "u1_fallback_metrics" and str(row["du_net"]) != TAU1:
        flags.append("missing_same_config_du_mask_evidence")
    if int(row["du_seed_count"]) < 20 or int(row["linear_seed_count"]) < 20:
        flags.append("low_seed_or_screen_only")
    if not np.isfinite(float(row["total_error"])):
        flags.append("missing_same_config_evidence")
    if flags:
        return "inverse_fit_diagnostic__" + "__".join(flags[:3])
    close = (
        float(row["figure4a_du_curve_error"]) <= 0.05
        and float(row["figure4a_linear_curve_error"]) <= 0.05
        and float(row["figure4b_du_curve_error_norm"]) <= 0.05
        and float(row["figure4b_linear_curve_error_norm"]) <= 0.05
    )
    return "near_perfect_empirical_reconstruction_candidate" if close else "not_close"


def build_leaderboard() -> pd.DataFrame:
    targets = read_csv(TARGET_PATH)
    if targets.empty:
        return pd.DataFrame()
    u1_metrics = load_u1_metrics()
    f4b_scale = figure4b_scale(targets)
    rows: list[dict[str, object]] = []
    for metric in metric_candidates().itertuples(index=False):
        for mask in mask_candidates().itertuples(index=False):
            for linear in linear_candidates().itertuples(index=False):
                for du in du_candidates().itertuples(index=False):
                    if str(mask.mask_id) == "u1_fallback":
                        evidence = evidence_from_u1(u1_metrics, str(du.du_net), str(linear.linear_ucb), str(metric.metric_column), targets)
                    elif str(du.du_net) == TAU1 and str(linear.linear_ucb) in BASE_LINEAR_CANDIDATES:
                        evidence = evidence_from_mask(str(mask.mask_id), str(linear.linear_ucb), str(metric.metric))
                    else:
                        evidence = {}
                    if not evidence:
                        evidence = {
                            "figure4a_du_curve_error": math.inf,
                            "figure4a_linear_curve_error": math.inf,
                            "figure4b_du_curve_error": math.inf,
                            "figure4b_linear_curve_error": math.inf,
                            "du_seed_count": 0,
                            "linear_seed_count": 0,
                            "du_cycle_count": 0,
                            "linear_cycle_count": 0,
                            "evidence_source": "missing_same_config_evidence",
                        }
                    protocol_penalty = (
                        float(metric.metric_penalty)
                        + float(mask.mask_penalty)
                        + float(linear.linear_penalty)
                        + float(du.du_penalty)
                    )
                    fig4b_du_norm = normalize_error(float(evidence["figure4b_du_curve_error"]), f4b_scale)
                    fig4b_linear_norm = normalize_error(float(evidence["figure4b_linear_curve_error"]), f4b_scale)
                    total = (
                        0.35 * float(evidence["figure4a_du_curve_error"])
                        + 0.25 * float(evidence["figure4a_linear_curve_error"])
                        + 0.20 * fig4b_du_norm
                        + 0.10 * fig4b_linear_norm
                        + 0.10 * protocol_penalty
                    )
                    row = {
                        "metric": str(metric.metric),
                        "metric_column": str(metric.metric_column),
                        "metric_status": str(metric.metric_status),
                        "mask_id": str(mask.mask_id),
                        "mask_source": str(mask.mask_source),
                        "linear_ucb": str(linear.linear_ucb),
                        "linear_source": str(linear.linear_source),
                        "du_net": str(du.du_net),
                        "du_source": str(du.du_source),
                        **evidence,
                        "figure4b_du_curve_error_norm": fig4b_du_norm,
                        "figure4b_linear_curve_error_norm": fig4b_linear_norm,
                        "protocol_penalty": protocol_penalty,
                        "total_error": total,
                    }
                    row["claim_status"] = claim_status(row)
                    rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["total_error", "protocol_penalty"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.insert(1, "pipeline_id", [f"joint_v2_{idx:04d}" for idx in range(1, len(out) + 1)])
    return out


def write_report(leaderboard: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    top_status = str(leaderboard.iloc[0]["claim_status"]) if not leaderboard.empty else "missing"
    lines = [
        "# inverse_reconstruction_v2 joint leaderboard",
        "",
        "Scope: same-config Figure 4a + Figure 4b inverse-fit leaderboard over metric, mask, Linear UCB, and DU-Net candidates.",
        "",
        f"- Output: `{OUT_PATH.relative_to(ROOT).as_posix()}`",
        f"- Top status: `{top_status}`",
        "",
        "## Top Pipelines",
        "",
        md_table(leaderboard.head(30), max_rows=30),
        "",
        "## Decision Rule",
        "",
        "Rows requiring a non-promoted metric, a non-author-verified mask, an unverified Linear UCB protocol, or missing same-config evidence are inverse-fit diagnostics. They are not perfect reproductions.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = build_leaderboard()
    leaderboard.to_csv(OUT_PATH, index=False)
    write_report(leaderboard)
    print(OUT_PATH)
    print(REPORT_PATH)
    print(leaderboard.head(10).to_string(index=False) if not leaderboard.empty else "No joint leaderboard rows.")


if __name__ == "__main__":
    main()
