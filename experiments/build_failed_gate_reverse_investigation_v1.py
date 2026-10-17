from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports" / "paper_exact_attempt_movielens_v1_failed_gate_reverse_investigation.md"
RISK_PATH = ROOT / "outputs" / "tables" / "paper_exact_attempt_movielens_v1_failed_gate_risks.csv"
METRIC_PATH = ROOT / "outputs" / "tables" / "paper_exact_attempt_movielens_v1_failed_gate_metric_candidates.csv"
JOINT_AFFINE_PATH = ROOT / "outputs" / "tables" / "paper_exact_attempt_movielens_v1_failed_gate_joint_affine.csv"

DU = "schur_loglambda_tuned_full_pool_softmax_tau1_literal"
CONCAT = "linucb_concat"
MF = "linucb_mf_interaction"
PAPER_DU_4A = 1.788
PAPER_LINEAR_4A = 1.814
PAPER_GAP_4A = PAPER_LINEAR_4A - PAPER_DU_4A


def read_csv(rel: str) -> pd.DataFrame:
    path = ROOT / rel
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(val):
        return ""
    if abs(val) >= 100:
        return f"{val:.3f}"
    if abs(val) >= 10:
        return f"{val:.4f}"
    if abs(val) >= 1:
        return f"{val:.6f}"
    return f"{val:.8f}"


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows available._"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(fmt)
    view = view.fillna("")

    def clean(value: object) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = [clean(col) for col in view.columns]
    rows = [[clean(value) for value in row] for row in view.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def value_from_run_metric(final_rows: pd.DataFrame, method: str, col: str) -> float | None:
    rows = final_rows[final_rows["variant"] == method]
    if rows.empty or col not in rows.columns:
        return None
    return float(rows.iloc[0][col])


def value_from_row_metric(summary: pd.DataFrame, method: str, metric: str) -> float | None:
    rows = summary[(summary["method"] == method) & (summary["metric"] == metric)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["value"])


def build_metric_candidate_table() -> pd.DataFrame:
    run_metrics = read_csv("outputs/tables/paper_exact_attempt_movielens_v1_metrics.csv")
    row_summary = read_csv("outputs/tables/paper_exact_attempt_movielens_v1_metric_reconstruction_v3_summary.csv")
    final_rows = run_metrics[run_metrics["cycle"] == 20].copy() if not run_metrics.empty else pd.DataFrame()

    metric_specs = [
        ("standard_mask_rmse", "run", "standard_mask_rmse"),
        ("pdf_literal_norm_rmse", "run", "pdf_literal_norm_rmse"),
        ("current_w_rmse", "run", "w_rmse"),
        ("mild_reward_w_rmse", "run", "mild_reward_w_rmse"),
        ("medium_reward_w_rmse", "run", "medium_reward_w_rmse"),
        ("sigmoid_w_rmse", "run", "sigmoid_reward_w_rmse"),
        ("reward_tail_proxy", "run", "reward_tail_proxy_rmse"),
        ("rating_weighted_rmse", "row", "rating_weighted_rmse"),
        ("rating_normalized_weighted_rmse", "row", "normalized_rating_weighted_rmse"),
        ("centered_rating_rmse", "row", "centered_rating_rmse"),
        ("unclipped_standard_rmse", "row", "raw_unclipped_rmse"),
        ("0_1_normalized_rmse", "row", "rating_0_1_normalized_rmse"),
    ]

    rows: list[dict[str, object]] = []
    for label, source, key in metric_specs:
        getter = value_from_run_metric if source == "run" else value_from_row_metric
        du = getter(final_rows if source == "run" else row_summary, DU, key)
        concat = getter(final_rows if source == "run" else row_summary, CONCAT, key)
        mf = getter(final_rows if source == "run" else row_summary, MF, key)
        gap_concat = None if du is None or concat is None else concat - du
        gap_mf = None if du is None or mf is None else mf - du
        scale_dist_concat = (
            None
            if du is None or concat is None
            else (abs(du - PAPER_DU_4A) + abs(concat - PAPER_LINEAR_4A)) / 2.0
        )
        direction_concat = "paper_like" if gap_concat is not None and gap_concat > 0 else "reversed_or_tie"
        rows.append(
            {
                "metric": label,
                "du_net_tau1": du,
                "linucb_concat": concat,
                "linucb_mf": mf,
                "gap_vs_concat": gap_concat,
                "gap_vs_mf": gap_mf,
                "scale_dist_vs_paper_concat": scale_dist_concat,
                "gap_dist_vs_paper_concat": None if gap_concat is None else abs(gap_concat - PAPER_GAP_4A),
                "direction_vs_concat": direction_concat,
                "source": source,
            }
        )
    out = pd.DataFrame(rows)
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(METRIC_PATH, index=False)
    return out


def build_joint_affine_table() -> pd.DataFrame:
    curves = read_csv(
        "outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/"
        "tau1_figure4_digitized_local_vs_paper_curve.csv"
    )
    rows: list[dict[str, object]] = []
    if not curves.empty:
        for figure, metric in [
            ("Figure 4a", "standard_mask_rmse"),
            ("Figure 4a", "w_rmse"),
            ("Figure 4b", "cumulative_reward"),
        ]:
            df = curves[(curves["figure"] == figure) & (curves["local_metric"] == metric)].dropna(
                subset=["local_value", "paper_value"]
            )
            if len(df) >= 2:
                slope, intercept = np.polyfit(df["local_value"].to_numpy(float), df["paper_value"].to_numpy(float), 1)
                pred = slope * df["local_value"].to_numpy(float) + intercept
                rmse = float(np.sqrt(np.mean((pred - df["paper_value"].to_numpy(float)) ** 2)))
                rows.append(
                    {
                        "figure": figure,
                        "local_metric": metric,
                        "fit_scope": "DU-Net + Linear UCB joint",
                        "slope_a": float(slope),
                        "intercept_b": float(intercept),
                        "rmse": rmse,
                        "points": len(df),
                    }
                )
    out = pd.DataFrame(rows)
    JOINT_AFFINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(JOINT_AFFINE_PATH, index=False)
    return out


def build_risk_table() -> pd.DataFrame:
    risks = [
        ("20-seed row-level metric recomputation", "A", "Generate final-cycle, then optionally all-cycle row-level predictions for all required methods.", "high compute; hours to overnight single-process", "Closes partial evidence for metric candidates, but cannot fix a wrong paper formula by itself."),
        ("u1-u5 and random-mask sensitivity", "A", "Promote existing u1-u5 3-seed evidence only if expanded to 20 seeds or paired with author mask recovery.", "medium to high", "Reduces mask-overfitting concern; does not recover exact masks."),
        ("all-candidate Schur vs 20k shortlist", "A", "Run a small all-candidate diagnostic and compare selected batches and endpoints.", "medium to high memory/CPU", "Can close shortlist approximation risk for current implementation."),
        ("lambda_noise sensitivity", "A", "Grid or estimate eta^2/sigma^2 from residual variance and rating-noise assumptions.", "medium", "Tests whether backend/noise convention explains reward/RMSE tradeoff."),
        ("alpha sensitivity for local Linear UCB", "A", "Broaden alpha/regularization diagnostics beyond the current 3-seed grid if comparator remains important.", "medium", "Clarifies comparator range; does not identify author Linear UCB."),
        ("Figure 4 digitization precision", "A", "Use WebPlotDigitizer/manual axis calibration on the paper PDF crops and regenerate target CSVs.", "low to medium", "Improves target precision; still not author numeric data."),
        ("contextualbandits/lrbandit adapters", "B", "Install/test minimal adapters only as external comparator candidates.", "medium plus dependency risk", "May widen comparator range; not exact without paper linkage."),
        ("ELPH backend implementation", "B", "Implement or vendor a documented ELPH-like graph embedding only if paper/backend evidence appears.", "high", "Could affect DLS inputs materially; current exactness remains blocked."),
        ("tiled all-candidate Schur production path", "B", "Optimize the all-candidate Schur computation if diagnostics show shortlist changes decisions.", "high", "Needed only if shortlist materially changes selection."),
        ("estimated noise ratio backend", "B", "Add a source-grounded eta^2/sigma^2 estimator and freeze it in config.", "medium", "Could align Eq.3.6 convention, but paper does not expose enough details."),
        ("author exact MovieLens masks", "C", "Obtain initial/candidate/test masks or generation scripts.", "requires source", "Required for exact replication."),
        ("author exact w-RMSE code or numeric Figure 4a table", "C", "Obtain metric code, weights, normalization, aggregation, and Figure 4 numeric data.", "requires source", "Required to resolve the formula/plot contradiction."),
        ("author Linear UCB protocol", "C", "Obtain feature map, alpha, regularization, reward transform, and update protocol.", "requires source", "Required before naming any comparator paper_linear_ucb_reconstructed."),
        ("author DU-Net backend/ELPH parameters", "C", "Obtain backend details for DLS input graph/embedding, sigma/noise/tuning, and any approximation.", "requires source", "Required for perfect source-faithful DU-Net reproduction."),
    ]
    out = pd.DataFrame(risks, columns=["risk", "category", "next_action", "estimated_effort", "expected_impact"])
    RISK_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RISK_PATH, index=False)
    return out


def selected_behavior_table() -> pd.DataFrame:
    metrics = read_csv("outputs/tables/paper_exact_attempt_movielens_v1_metrics.csv")
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant in [DU, CONCAT, "schur_loglambda_tuned_full_pool_softmax_tau0p3", "schur_tuned_topk"]:
        part = metrics[(metrics["variant"] == variant) & (metrics["cycle"] < 20)]
        final = metrics[(metrics["variant"] == variant) & (metrics["cycle"] == 20)]
        if part.empty or final.empty:
            continue
        rows.append(
            {
                "variant": variant,
                "avg_selected_rating_c0_19": float(part["selected_avg_rating"].mean()),
                "selected_ge4_share_c0_19": float(part["selected_rating_ge4_share"].mean()),
                "avg_reward_percentile_c0_19": float(part["selected_avg_reward_percentile"].mean())
                if "selected_avg_reward_percentile" in part
                else np.nan,
                "avg_entropy_percentile_c0_19": float(part["selected_avg_entropy_percentile"].mean())
                if "selected_avg_entropy_percentile" in part
                else np.nan,
                "final_cumulative_reward": float(final.iloc[0]["cumulative_reward"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    config = json.loads((ROOT / "configs" / "paper_exact_attempt_movielens_v1.json").read_text())
    metric_candidates = build_metric_candidate_table()
    joint_affine = build_joint_affine_table()
    risks = build_risk_table()

    final_gate = read_csv("outputs/tables/paper_exact_attempt_movielens_v1_final_exact_gate.csv")
    row_completion = read_csv("outputs/tables/paper_exact_attempt_movielens_v1_row_level_completion_v3.csv")
    exact_summary = read_csv("outputs/tables/paper_exact_attempt_movielens_v1_final_summary.csv")
    twenty_summary = read_csv(
        "outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/"
        "tau1_final_summary.csv"
    )
    metric_comparison = read_csv("outputs/tables/paper_exact_attempt_movielens_v1_metric_reconstruction_v3_comparison.csv")
    targets = read_csv(
        "outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/"
        "tau1_figure4_digitized_targets.csv"
    )
    endpoint_alignment = read_csv(
        "outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/"
        "tau1_figure4_endpoint_alignment.csv"
    )
    curve_summary = read_csv(
        "outputs/tables/paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1/"
        "tau1_figure4_curve_alignment_summary.csv"
    )
    split_audit = read_csv("outputs/tables/split_search_audit_v2.csv")
    fold_gaps = read_csv("outputs/tables/paper_faithful_movielens_fold_sensitivity_v1_3seeds_fold_comparator_gaps.csv")
    linear_protocol = read_csv("outputs/tables/linear_ucb_protocol_audit_v1.csv")
    alpha = read_csv("outputs/tables/paper_faithful_movielens_gap_investigation_v1_linucb_mf_alpha_b3.csv")
    reward_tradeoff = read_csv("outputs/tables/paper_faithful_movielens_reward_tradeoff_v1_20seeds_final_summary.csv")
    selected_behavior = selected_behavior_table()

    fig4a_targets = pd.DataFrame()
    fig4b_targets = pd.DataFrame()
    if not targets.empty:
        fig4a = targets[targets["figure"] == "Figure 4a"].pivot(index="cycle", columns="method", values="paper_value")
        fig4a_targets = fig4a.reset_index()
        if {"DU-Net", "Linear UCB"}.issubset(fig4a_targets.columns):
            fig4a_targets["paper_gap"] = fig4a_targets["Linear UCB"] - fig4a_targets["DU-Net"]
        fig4b = targets[targets["figure"] == "Figure 4b"].pivot(index="cycle", columns="method", values="paper_value")
        fig4b_targets = fig4b.reset_index()
        if {"DU-Net", "Linear UCB"}.issubset(fig4b_targets.columns):
            fig4b_targets["paper_gap"] = fig4b_targets["Linear UCB"] - fig4b_targets["DU-Net"]

    exact_main = next(v for v in config["variants"] if v["claim_role"] == "mainline")
    acquisition = config["acquisition"]
    mf = config["mf"]
    split = config["split"]

    best_metric = metric_comparison.head(5) if not metric_comparison.empty else pd.DataFrame()
    best_requested_gap = metric_candidates.assign(
        abs_gap_distance=lambda d: (d["gap_vs_concat"] - PAPER_GAP_4A).abs()
    ).sort_values("abs_gap_distance")

    pdf_derivation = (
        "For N=20000 and standard RMSE about 1.07, "
        "sqrt(sum(err^2))/N = standard_RMSE/sqrt(N) = 1.07/sqrt(20000) = about 0.00757. "
        "Therefore a Figure 4a y-axis around 1.78-1.82 cannot be the literal PDF formula."
    )

    lines: list[str] = []
    lines.extend(
        [
            "# Failed Gate Reverse Investigation: paper_exact_attempt_movielens_v1",
            "",
            "Purpose: reverse-investigate each failed exact-replication gate. This is not a tuning report.",
            "",
            "## 1. Current mainline exact config",
            "",
            f"- Mainline variant: `{exact_main['label']}` with policy `{exact_main['policy']}`.",
            f"- Selection policy: `{exact_main['acquisition_overrides']['selection']}`; tau `{exact_main['acquisition_overrides']['selection_temperature']}`.",
            f"- Entropy mode: `{acquisition['entropy_mode']}`; entropy transform `{acquisition['entropy_score_transform']}`.",
            f"- Lambda tuning grid: `{acquisition['adaptive_lambda_grid']}`; diffusion p grid `{acquisition['adaptive_diffusion_prob_grid']}`.",
            f"- Sigma grid: `{acquisition['sigma_grid']}`.",
            f"- Tuning frequency: every `{acquisition['paper_tuning_frequency']}` cycles using `{acquisition['paper_tuning_metric']}`.",
            f"- DLS mode: scope `{acquisition['paper_dls_scope']}`, neighbor mode `{acquisition['dls_neighbor_mode']}`, top-k `{acquisition['dls_top_k']}`.",
            f"- MF backend: `{mf['algorithm']}`, rank `{mf['rank']}`, epochs `{mf['epochs']}`, lr `{mf['learning_rate']}`, reg `{mf['regularization']}`.",
            f"- Diffusion probability p is fixed to `{acquisition['diffusion_prob']}` in the mainline grid.",
            f"- Split: `{split['mode']}`, users `{split['num_users']}`, items `{split['num_items']}`, initial size `{split['initial_size']}`.",
            "",
            "The exact-attempt variants share the same executable path (`run_policy` / `paper_faithful_dunet`) and differ by config overrides: tau1 softmax, tau0.3 softmax, deterministic top-k, and a lambda0.3 diagnostic. The Linear UCB rows share the same acquisition loop but use local proxy scoring functions. In the exact-attempt replay, `lambda0p3_diagnostic` seed0 is numerically identical to tau1 because adaptive tuning can overwrite the initial lambda override.",
            "",
            "Within the exact-attempt seed0 final-cycle artifacts, standard RMSE, PDF-literal, and local w-RMSE come from the same run family. The 20-seed tau1 bundle is aggregate metrics only; arbitrary row-level metric formulas are currently available only for seed0 final-cycle files.",
            "",
            "Final strict gate table:",
            "",
            md_table(final_gate),
            "",
            "Row-level completion:",
            "",
            md_table(row_completion),
            "",
            "## 2. Metric exactness answers",
            "",
            pdf_derivation,
            "",
            "The paper may have plotted standard RMSE or a weighted RMSE while writing a PDF-literal formula, but the current evidence cannot prove which executable definition was used. The PDF text names w-RMSE and reward concepts, yet does not expose weights, normalization, mask, or aggregation details.",
            "",
            "Requested metric candidate table, seed0 final-cycle where row-level formulas are available:",
            "",
            md_table(metric_candidates),
            "",
            "Best candidates from the broader v3 row-level audit:",
            "",
            md_table(best_metric),
            "",
            "Best requested-table candidates by gap closeness to the visual 0.026 gap:",
            "",
            md_table(best_requested_gap[["metric", "du_net_tau1", "linucb_concat", "gap_vs_concat", "gap_dist_vs_paper_concat", "direction_vs_concat"]].head(6)),
            "",
            "Answers: no candidate simultaneously satisfies endpoint scale, gap, direction, and text-defensible definition. Standard RMSE has the right direction and a close gap versus `linucb_concat`, but its direct endpoint error is about 0.717 on the Figure 4a y-axis (the broader v3 audit's normalized scale-distance field is about 0.398). Some reward-weighted metrics can get closer to the gap, but often reverse direction or lack textual support. Exact w-RMSE is therefore unrecoverable from the paper plus current implementation.",
            "",
            "Row-level incompleteness matters for confidence intervals and formula sweeps across all seeds, but it is unlikely to move the Figure 4a scale from about 1.07 to about 1.79 without a different metric definition or mask. A full 20-seed row-level run is still worth doing if the goal is maximum-effort evidence; without it, the metric gate must remain partial evidence.",
            "",
            "## 3. Figure digitization answers",
            "",
            "Figure 4a MovieLens has been digitized as a cycle-level CSV from local PDF crops. These are visual approximations, not author data.",
            "",
            md_table(fig4a_targets),
            "",
            "Early/mid/final Figure 4a gaps:",
            "",
            md_table(fig4a_targets[fig4a_targets["cycle"].isin([0, 1, 5, 10, 20])] if not fig4a_targets.empty else fig4a_targets),
            "",
            "Figure 4a endpoint and curve diagnostics:",
            "",
            md_table(endpoint_alignment[endpoint_alignment["figure"] == "Figure 4a"] if not endpoint_alignment.empty else pd.DataFrame()),
            "",
            md_table(curve_summary[curve_summary["figure"] == "Figure 4a"] if not curve_summary.empty else pd.DataFrame()),
            "",
            "Joint affine fits using both DU-Net and Linear UCB points:",
            "",
            md_table(joint_affine[joint_affine["figure"] == "Figure 4a"] if not joint_affine.empty else pd.DataFrame()),
            "",
            "Separate affine fits can make individual shapes look close, especially local w-RMSE curves, but that does not recover the paper metric. A single shared affine transform is only a diagnostic and cannot close exactness without metric, mask, and comparator provenance.",
            "",
            "Figure 4b has also been digitized as a cycle-level CSV. Endpoint target: DU-Net about 1325, Linear UCB about 1390, gap about 65.",
            "",
            md_table(fig4b_targets),
            "",
            md_table(endpoint_alignment[endpoint_alignment["figure"] == "Figure 4b"] if not endpoint_alignment.empty else pd.DataFrame()),
            "",
            "The current tau1 reward ranking is paper-like, but the local gap is about 147.294 versus the visual paper gap of about 65, so the gap is too large by about 82.294. The DU-Net local reward endpoint is about 78.614 below paper, while `linucb_concat` is only about 3.680 above paper.",
            "",
            "## 4. Split/mask exactness answers",
            "",
            "The current MovieLens fallback uses official `u1.base`/`u1.test`, maps users/items to zero-based ids, filters to 1681 items by dropping the final MovieLens item id after zero-basing, samples the initial observed set of 1364 rows from `u1.base` by seed, defines candidates as `u1.base` minus the initial rows, and keeps `u1.test` fixed. Seeds affect only the initial/candidate partition, not the fixed test fold.",
            "",
            "Aligned with Table 1-style constraints: 943 users, 1681 items, 1364 initial observations, 20000 test rows, 20 cycles, batch size 100. Not aligned: exact train mask, exact initial mask, exact candidate pool, exact test mask provenance, and exact seed protocol.",
            "",
            "Split search audit:",
            "",
            md_table(split_audit),
            "",
            "Monti/RMGCNN-style `.mat` masks were found locally for Flixster and Douban, not MovieLens. GCMC provides an official `u1.base`/`u1.test` MovieLens loader and validation split convention; it does not recover DU-Net's Figure 4 masks. It is plausible the paper used official u1 because the dimensions and public-loader conventions fit, but this remains unproven because the paper does not publish the exact initial/candidate/test masks.",
            "",
            "Official fold sensitivity, 3 seeds per fold:",
            "",
            md_table(fold_gaps),
            "",
            "Fold interpretation: tau1 is stable against `linucb_concat` on most/all folds in this 3-seed diagnostic, but not stable against `linucb_mf_interaction`. This lowers pure mask-overfitting concern for the concat comparator, but it does not close exact mask provenance.",
            "",
            "## 5. Linear UCB exactness answers",
            "",
            "The PDF evidence audit finds Linear UCB named as a comparator and mentions the contextualbandits repository, but does not specify feature construction, alpha, regularization, reward transform, or update protocol. Local package checks found `contextualbandits` and `lrbandit` not installed at audit time.",
            "",
            "Linear UCB protocol audit:",
            "",
            md_table(linear_protocol),
            "",
            "Local proxies:",
            "",
            "- `linucb_concat`: ridge UCB on concatenated `[user_factors[u], item_factors[i]]`, target `rating - 3`.",
            "- `linucb_mf_interaction`: MF prediction mean plus alpha times z-scored uncertainty from elementwise interaction features `user_factors[u] * item_factors[i]`.",
            "- `linucb_bias_interaction`: ridge UCB on user factors, item factors, interaction, user bias, item bias, and intercept.",
            "",
            "They use the same SVD++ backend, candidate set, batch size, cycle update, and selected-row removal loop as the DU-Net runs. They are fair local comparators but not recovered author implementations.",
            "",
            "20-seed local proxy endpoints:",
            "",
            md_table(twenty_summary[twenty_summary["claim_role"].isin(["mainline", "linear_ucb_proxy"])] if "claim_role" in twenty_summary.columns else twenty_summary),
            "",
            "Closest Figure 4b Linear endpoint is `linucb_concat` (1393.680 versus visual 1390). Strongest reward proxy is `linucb_bias_interaction` but it overshoots reward. Strongest standard-RMSE proxy is `linucb_mf_interaction`, but it is not the closest reward comparator. The most defensible comparator for Figure 4b is `linucb_concat`; for MF-style uncertainty comparisons, `linucb_mf_interaction` remains a required sensitivity comparator. No `paper_linear_ucb_reconstructed` should be promoted.",
            "",
            "Alpha sensitivity, 3-seed MF-interaction diagnostic:",
            "",
            md_table(alpha),
            "",
            "Current alpha is 0.8 with regularization 1.0. It is a local choice, not recovered from the paper. Smaller alpha values increase immediate reward but worsen RMSE, which shows the comparator is sensitive to hidden protocol choices.",
            "",
            "## 6. DU-Net exactness answers",
            "",
            "Schur/Eq.3.6: the explicit-kernel audit used a non-orthogonal toy factorization and compared the production feature-space conditional variance with direct observed-kernel Schur complement. Maximum absolute difference was 2.220e-16. This proves implementation equivalence for the current clean-room Schur scorer, not full identity with every unpublished paper backend convention.",
            "",
            "The implementation uses `phi = kron(q_user[u], q_item[i])`, posterior precision `(Phi_obs.T Phi_obs + lambda I)^-1`, and returns `lambda * phi.T precision phi`. This is theoretically aligned with a low-rank Schur conditional variance, but the code does not expose every paper symbol as a named object (`mu_i(U) mu_j(V)`, cross-coherence `nu`, `L_t(O_t)`, and `eta^2/sigma^2`). A formal note is useful if writing a methods appendix.",
            "",
            "Shortlist: current Schur correction is applied to a coherence-base shortlist of 20000 candidates when the final candidate pool is about 76635. Non-shortlisted candidates receive zero entropy. This can change selected batches and has not been closed by an all-candidate diagnostic. Perfect reproduction requires at least a small all-candidate-vs-shortlist audit.",
            "",
            "lambda_noise: current `lambda_noise=1.0` is inherited from `entropy_logdet_regularization=1.0`. There is no data-estimated eta^2/sigma^2 version in the mainline, and no source-backed sensitivity closure. A reasonable estimator would use held-out/residual variance for eta^2 and latent-score variance or rating-noise scale for sigma^2, then freeze the ratio before selection.",
            "",
            "DLS: current DLS is full-neighborhood Gaussian smoothing when `paper_dls_scope=full`, with no residual blend unless shrinkage is configured. Sigma is dynamically selected from the configured grid every tuning interval. Its input `u_sn` is the configured network/user-feature embedding (`network_embedding_mode=user_features`, GCMC/user metadata features), not a recovered paper ELPH embedding.",
            "",
            "ELPH: the repo has dependency-free ELPH-style proxies in other diagnostics, but the exact paper ELPH backend and parameters are not recovered or validated for MovieLens. Treat ELPH as a remaining blocker, not a branch to keep tuning blindly.",
            "",
            "## 7. Selection exactness answers",
            "",
            "tau1 literal softmax is implemented as full-pool `exp(score)` sampling without replacement after subtracting the max score for numerical stability. The subtraction does not change probabilities; exponent clipping only prevents extreme underflow/overflow. The score itself is the DU-Net combined score after z-scoring reward and transformed entropy before adding them.",
            "",
            "The paper gives an exponential weighting idea, but does not specify all implementation choices: whether scores were z-normalized, whether batch sampling was without replacement, and whether numerical clipping was used. Thus selection is closed as a clean-room literal-tau gate, but not as author-source exactness. The 20-seed tau1 mainline is stable: it beats `linucb_concat` 20/20 seeds and `linucb_mf_interaction` 19/20 seeds under standard RMSE. tau1 reward is close to tau0.3 in the 20-seed bundle, so tau1 should replace tau0.3 as the formal mainline.",
            "",
            "## 8. Figure 4b joint reproduction answers",
            "",
            "Selected-sample behavior in the exact-attempt seed0 run:",
            "",
            md_table(selected_behavior),
            "",
            "tau1 selects lower immediate-reward rows than `linucb_concat` (mean selected rating about 3.5205 versus 3.9300). Its average entropy percentile is nontrivial, so the reward gap is consistent with stronger exploration/information-value pressure. Dynamic tuning is currently driven by `standard_mask_rmse`, which naturally prioritizes Figure 4a-style error rather than Figure 4b reward.",
            "",
            "Reward tradeoff diagnostics, 20 seeds:",
            "",
            md_table(reward_tradeoff[reward_tradeoff["variant"].isin(["linucb_concat", "schur_lambda_0p3", "schur_loglambda_tuned_standard_topk", "schur_lambda_1", "schur_lambda_0"])] if not reward_tradeoff.empty else pd.DataFrame()),
            "",
            "`schur_lambda_0p3` shows a more paper-like Figure 4b reward endpoint/gap than the tau1 mainline, but it worsens standard RMSE relative to tau1 and is not the frozen exact-attempt mainline. `schur_lambda_0` can exceed Linear reward but badly hurts RMSE. This indicates a local objective mismatch: no single current lambda simultaneously closes Figure 4a scale/metric and Figure 4b reward gap.",
            "",
            "## 9. Risks categorized by closability",
            "",
            "Categories: A = can be closed by more local experiments; B = requires installation/implementation before closure; C = cannot be closed by experiments alone and needs source material.",
            "",
            md_table(risks),
            "",
            "## 10. Final judgment and recommended next steps",
            "",
            "If the target is perfect exact replication, the three most critical failed gates are: metric exactness, split/mask exactness, and Linear UCB protocol exactness. The next tier is DU-Net backend/ELPH/noise/shortlist exactness plus Figure 4b same-config joint reproduction.",
            "",
            "Local experiments can close row-level completeness, fold robustness, all-candidate Schur risk, lambda_noise sensitivity, alpha sensitivity, and digitization precision. They cannot close author masks, author w-RMSE code/numeric Figure 4 data, or author Linear UCB hidden protocol.",
            "",
            "Maximum-effort paper-only work remaining: run full 20-seed row-level final-cycle predictions for required methods; optionally run all-cycle row-level only if storage is acceptable; expand fold/mask sensitivity; run all-candidate Schur diagnostics; run lambda_noise/eta-sigma sensitivity; and add dependency-backed Linear UCB adapters only as explicitly labeled comparator candidates.",
            "",
            "Current status: strong clean-room recovery, not near-perfect exact reproduction. It has not yet reached the strongest possible paper-only reproduction because several closable local experiments remain. Even after those are done, near-perfect replication still requires source conditions: exact metric code/numeric Figure 4 data, exact MovieLens masks/seed protocol, exact Linear UCB adapter, and DU-Net backend details. Without those, the remaining gap is source availability and paper ambiguity, not simply code quality.",
            "",
            "Generated artifacts:",
            "",
            f"- `{METRIC_PATH.relative_to(ROOT)}`",
            f"- `{JOINT_AFFINE_PATH.relative_to(ROOT)}`",
            f"- `{RISK_PATH.relative_to(ROOT)}`",
        ]
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(REPORT_PATH)
    print(METRIC_PATH)
    print(JOINT_AFFINE_PATH)
    print(RISK_PATH)


if __name__ == "__main__":
    main()
