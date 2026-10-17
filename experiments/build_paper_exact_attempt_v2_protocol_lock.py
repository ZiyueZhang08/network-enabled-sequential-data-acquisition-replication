from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_exact_attempt_movielens_v2"
CONFIG_PATH = ROOT / "configs" / f"{PREFIX}.json"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
GATE_PATH = TABLE_DIR / f"{PREFIX}_gate_status.csv"
REPORT_PATH = REPORT_DIR / f"{PREFIX}_protocol_lock.md"
SOURCE_RECOVERY_PATH = REPORT_DIR / f"{PREFIX}_source_recovery_packet.md"


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


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def variant_roles(config: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in config["variants"]:
        overrides = variant.get("acquisition_overrides", {})
        rows.append(
            {
                "variant": variant["label"],
                "policy": variant["policy"],
                "claim_role": variant["claim_role"],
                "selection": overrides.get("selection", config["acquisition"]["selection"]),
                "tau": overrides.get("selection_temperature", config["acquisition"]["selection_temperature"]),
                "lambda_entropy": overrides.get("lambda_entropy", config["acquisition"]["lambda_entropy"]),
                "exact_attempt_role": role_decision(str(variant["claim_role"])),
            }
        )
    return pd.DataFrame(rows)


def role_decision(claim_role: str) -> str:
    if claim_role == "mainline":
        return "only Figure 4 exact-attempt mainline"
    if claim_role == "linear_ucb_proxy":
        return "comparator candidate; not paper Linear UCB until source-verified"
    if claim_role == "calibrated_softmax_robustness":
        return "robustness branch; not promoted to mainline"
    if claim_role == "deterministic_ablation":
        return "deterministic ablation; not promoted to mainline"
    if claim_role == "lambda_entropy_diagnostic":
        return "reward/RMSE diagnostic; not promoted to mainline"
    return "diagnostic"


def existing_prior_text() -> dict[str, str]:
    paths = {
        "v1_final_gate": TABLE_DIR / "paper_exact_attempt_movielens_v1_final_exact_gate.csv",
        "v1_metric": TABLE_DIR / "paper_exact_attempt_movielens_v1_metric_reconstruction_v3_comparison.csv",
        "v1_rows": TABLE_DIR / "paper_exact_attempt_movielens_v1_row_level_completion_v3.csv",
        "split": TABLE_DIR / "split_mask_search_audit_v3.csv",
        "linear": TABLE_DIR / "external_linucb_protocol_audit_v2.csv",
        "tuning": TABLE_DIR / f"{PREFIX}_tuning_frequency_objective_audit.csv",
    }
    out: dict[str, str] = {}
    for key, path in paths.items():
        out[key] = str(path.relative_to(ROOT)).replace("\\", "/") if path.exists() else "missing"
    return out


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def row_level_evidence() -> tuple[str, bool, bool]:
    completion = read_csv(TABLE_DIR / "row_level_completeness_v2.csv")
    if completion.empty:
        return "v2 row-level completeness table missing", False, False
    required = (
        completion[completion["required_for_metric_gate"].astype(bool)]
        if "required_for_metric_gate" in completion.columns
        else completion
    )
    diagnostics = (
        completion[~completion["required_for_metric_gate"].astype(bool)]
        if "required_for_metric_gate" in completion.columns
        else pd.DataFrame()
    )
    complete = bool((required["status"] == "complete_20seed").all())
    partial = bool((required["available_seed_count"].astype(int) > 0).any())
    counts = "; ".join(
        f"{row.method}:{int(row.available_seed_count)}/{int(row.required_seed_count)} {row.status}"
        for row in required.itertuples(index=False)
    )
    if not diagnostics.empty:
        diag_counts = "; ".join(
            f"{row.method}:{int(row.available_seed_count)}/{int(row.required_seed_count)} {row.status}"
            for row in diagnostics.itertuples(index=False)
        )
        counts = f"metric_required={counts}; diagnostic_tracked={diag_counts}"
    return counts, complete, partial


def metric_evidence() -> tuple[str, bool, bool]:
    comparison = read_csv(TABLE_DIR / "metric_reconstruction_v4_comparison.csv")
    if comparison.empty:
        return "metric v4 comparison missing or empty", False, False
    metric_pass = bool((comparison["paper_wrmse_reconstructed"] == "yes").any())
    best = comparison.iloc[0]
    evidence = (
        f"best={best['metric']} vs {best['comparator']}; "
        f"scale_distance={float(best['scale_distance_to_paper']):.6g}; "
        f"gap_distance={float(best['gap_distance_to_paper']):.6g}; "
        f"seed_count={int(best['row_level_seed_count'])}; "
        f"paper_wrmse_reconstructed={best['paper_wrmse_reconstructed']}"
    )
    return evidence, metric_pass, True


def schur_shortlist_evidence() -> tuple[str, bool]:
    audit = read_csv(TABLE_DIR / f"{PREFIX}_schur_shortlist_vs_full_audit.csv")
    if audit.empty:
        return "shortlist-vs-full audit missing", False
    for col in [
        "top100_overlap",
        "selected_batch_overlap",
        "selected_changed_count",
        "one_step_standard_rmse_diff_full_minus_short",
    ]:
        if col in audit.columns:
            audit[col] = pd.to_numeric(audit[col], errors="coerce")
    min_top100 = float(audit["top100_overlap"].min())
    min_batch = float(audit["selected_batch_overlap"].min())
    max_changed = int(audit["selected_changed_count"].max())
    if "one_step_standard_rmse_diff_full_minus_short" in audit.columns:
        max_abs_rmse_diff = float(audit["one_step_standard_rmse_diff_full_minus_short"].abs().max())
    else:
        max_abs_rmse_diff = float("nan")
    seeds = " ".join(str(int(seed)) for seed in sorted(audit["seed"].dropna().unique()))
    cycles = " ".join(str(int(cycle)) for cycle in sorted(audit["cycle"].dropna().unique())) if "cycle" in audit.columns else "0"
    closed = bool(min_top100 >= 0.95 and min_batch >= 0.95 and (pd.isna(max_abs_rmse_diff) or max_abs_rmse_diff <= 1.0e-3))
    return (
        f"shortlist-vs-full seeds={seeds}; cycles={cycles}; min_top100_overlap={min_top100:.6g}; "
        f"min_selected_batch_overlap={min_batch:.6g}; max_selected_changed_count={max_changed}; "
        f"max_abs_one_step_rmse_diff={max_abs_rmse_diff:.6g}",
        closed,
    )


def lambda_noise_evidence() -> tuple[str, bool]:
    audit = read_csv(TABLE_DIR / f"{PREFIX}_lambda_noise_sensitivity_audit.csv")
    if audit.empty:
        return "lambda_noise sensitivity audit missing", False
    min_top100 = float(audit["top100_overlap_vs_base"].min())
    min_batch = float(audit["selected_batch_overlap_vs_base"].min())
    max_changed = int(audit["selected_changed_count_vs_base"].max())
    lambdas = " ".join(f"{float(value):g}" for value in sorted(audit["lambda_noise"].dropna().unique()))
    seeds = " ".join(str(int(seed)) for seed in sorted(audit["seed"].dropna().unique()))
    closed = bool(min_top100 >= 0.95 and min_batch >= 0.95 and max_changed <= 5)
    return (
        f"lambda_noise grid={lambdas}; seeds={seeds}; min_top100_overlap_vs_base={min_top100:.6g}; "
        f"min_selected_batch_overlap_vs_base={min_batch:.6g}; max_selected_changed_count_vs_base={max_changed}",
        closed,
    )


def dls_elph_evidence() -> tuple[str, bool]:
    dep = read_csv(TABLE_DIR / f"{PREFIX}_elph_feasibility_dependencies.csv")
    graph = read_csv(TABLE_DIR / f"{PREFIX}_elph_feasibility_movielens_graph.csv")
    embedding = read_csv(TABLE_DIR / f"{PREFIX}_elph_feasibility_embedding_distance.csv")
    summary = read_csv(TABLE_DIR / f"{PREFIX}_elph_feasibility.csv")
    parts: list[str] = []
    if not dep.empty:
        official = dep[dep["module"].astype(str) == "elph"]
        official_available = bool(official["available"].astype(bool).any()) if not official.empty else False
        support_modules = " ".join(
            f"{row.module}={bool(row.available)}"
            for row in dep[dep["kind"].astype(str) == "supporting_stack"].itertuples(index=False)
        )
        parts.append(f"official_elph_import_available={official_available}; support_stack={support_modules}")
    if not graph.empty:
        row = graph.iloc[0]
        parts.append(
            f"MovieLens graph={row['graph_builder']}; users={int(row['num_users'])}; "
            f"feature_dim={int(row['user_feature_dim'])}; edges={int(float(row['edge_count_undirected']))}; "
            f"degree_mean={float(row['degree_mean']):.6g}"
        )
    if not embedding.empty:
        proxy = embedding[embedding["embedding_method"].astype(str) == "elph_role"]
        if not proxy.empty:
            row = proxy.iloc[0]
            parts.append(
                f"elph_role_proxy top20_overlap_with_metadata={float(row['topk_overlap_with_metadata']):.6g}; "
                f"pairwise_dist_corr={float(row['pairwise_dist_corr_with_metadata']):.6g}; "
                f"graph_edge_hit_rate={float(row['graph_edge_hit_rate_in_embedding_topk']):.6g}"
            )
    if not summary.empty:
        proxy = summary[summary["embedding_method"].astype(str) == "elph_role"]
        if not proxy.empty:
            row = proxy.iloc[0]
            parts.append(
                f"3seed elph_role_proxy seed_count={int(row['seed_count'])}; "
                f"delta_standard_rmse={float(row['delta_standard_mask_rmse_vs_metadata']):.6g}; "
                f"delta_reward={float(row['delta_cumulative_reward_vs_metadata']):.6g}; "
                f"delta_reward_tail_rmse={float(row['delta_reward_tail_proxy_rmse_vs_metadata']):.6g}; "
                f"exact_elph_reconstructed={row['exact_elph_reconstructed']}"
            )
    closed = bool(not dep.empty and not graph.empty and not embedding.empty and not summary.empty)
    if not parts:
        return "ELPH feasibility diagnostic missing", False
    return "; ".join(parts), closed


def tuning_evidence() -> tuple[str, bool]:
    audit = read_csv(TABLE_DIR / f"{PREFIX}_tuning_frequency_objective_audit.csv")
    if audit.empty:
        return "Phase 5 tuning audit missing", False
    statuses = "; ".join(
        f"{row.audit_item}:{row.status}"
        for row in audit.itertuples(index=False)
    )
    closed = bool((audit["status"] == "pass").all())
    return statuses, closed


def fold_sensitivity_evidence() -> tuple[str, bool]:
    gaps = read_csv(TABLE_DIR / f"{PREFIX}_fold_sensitivity_5seeds_fold_comparator_gaps.csv")
    if gaps.empty:
        return "u1-u5 5-seed fold sensitivity missing", False
    for col in ["seeds", "standard_win_rate", "reward_win_rate", "mean_standard_gap", "mean_reward_gap"]:
        gaps[col] = pd.to_numeric(gaps[col], errors="coerce")
    folds = int(gaps["fold"].nunique())
    comparators = int(gaps["comparator"].nunique())
    min_seeds = int(gaps["seeds"].min())
    min_standard_win = float(gaps["standard_win_rate"].min())
    concat = gaps[gaps["comparator"] == "linucb_concat"]
    mf = gaps[gaps["comparator"] == "linucb_mf_interaction"]
    concat_reward = float(concat["reward_win_rate"].mean()) if not concat.empty else float("nan")
    mf_reward_min = float(mf["reward_win_rate"].min()) if not mf.empty else float("nan")
    closed = bool(folds == 5 and comparators >= 2 and min_seeds >= 5 and min_standard_win == 1.0)
    return (
        f"u1-u5 fold sensitivity folds={folds}/5; comparators={comparators}; "
        f"min_seeds={min_seeds}; min_standard_win_rate={min_standard_win:.6g}; "
        f"concat_reward_win_rate_mean={concat_reward:.6g}; mf_reward_win_rate_min={mf_reward_min:.6g}",
        closed,
    )


def linucb_alpha_reg_evidence() -> tuple[str, bool]:
    summary = read_csv(TABLE_DIR / "linucb_alpha_reg_sensitivity_v2.csv")
    if summary.empty:
        return "alpha/reg sensitivity v2 missing", False
    for col in [
        "ucb_alpha",
        "ucb_regularization",
        "seed_count",
        "reward_tail_proxy_rmse",
        "standard_mask_rmse",
        "cumulative_reward",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")
    grid_points = int(summary[["ucb_alpha", "ucb_regularization"]].drop_duplicates().shape[0])
    min_seeds = int(summary["seed_count"].min()) if "seed_count" in summary.columns else 0
    alphas = " ".join(f"{float(value):g}" for value in sorted(summary["ucb_alpha"].dropna().unique()))
    regs = " ".join(f"{float(value):g}" for value in sorted(summary["ucb_regularization"].dropna().unique()))
    best_tail = summary.sort_values("reward_tail_proxy_rmse").iloc[0]
    best_reward = summary.sort_values("cumulative_reward", ascending=False).iloc[0]
    current = summary[
        (summary["ucb_alpha"].sub(0.8).abs() < 1e-12)
        & (summary["ucb_regularization"].sub(1.0).abs() < 1e-12)
    ]
    current_text = "current_alpha0.8_reg1=missing"
    if not current.empty:
        row = current.iloc[0]
        current_text = (
            f"current_alpha0.8_reg1 reward_tail={float(row['reward_tail_proxy_rmse']):.6g}, "
            f"reward={float(row['cumulative_reward']):.6g}"
        )
    reward_tail_spread = float(summary["reward_tail_proxy_rmse"].max() - summary["reward_tail_proxy_rmse"].min())
    reward_spread = float(summary["cumulative_reward"].max() - summary["cumulative_reward"].min())
    reconstructed = (
        summary["paper_linear_ucb_reconstructed"].astype(str).str.lower().eq("yes").any()
        if "paper_linear_ucb_reconstructed" in summary.columns
        else False
    )
    closed = bool(grid_points >= 24 and min_seeds >= 3 and not reconstructed)
    return (
        f"alpha/reg grid={grid_points}/24; alphas={alphas}; regs={regs}; min_seeds={min_seeds}; "
        f"best_reward_tail=alpha{float(best_tail['ucb_alpha']):g}/reg{float(best_tail['ucb_regularization']):g}/"
        f"{float(best_tail['reward_tail_proxy_rmse']):.6g}; "
        f"best_reward=alpha{float(best_reward['ucb_alpha']):g}/reg{float(best_reward['ucb_regularization']):g}/"
        f"{float(best_reward['cumulative_reward']):.6g}; {current_text}; "
        f"reward_tail_spread={reward_tail_spread:.6g}; reward_spread={reward_spread:.6g}; "
        f"paper_linear_ucb_reconstructed={'yes' if reconstructed else 'no'}",
        closed,
    )


def figure4b_evidence() -> tuple[str, bool, bool]:
    curves = read_csv(TABLE_DIR / "metric_curve_alignment_v4.csv")
    if curves.empty:
        return "Figure 4b same-config curve alignment missing", False, False
    figure4b = curves[
        (curves["figure"].astype(str) == "Figure 4b")
        & (curves["metric"].astype(str) == "cumulative_reward")
        & (curves["status"].astype(str) == "computed")
    ].copy()
    if figure4b.empty:
        return "Figure 4b cumulative_reward alignment not computed", False, False
    for col in ["gap_abs_error", "paper_gap", "curve_rmse", "endpoint_abs_error_mean", "local_gap"]:
        figure4b[col] = pd.to_numeric(figure4b[col], errors="coerce")
    best_gap = figure4b.sort_values(["gap_abs_error", "curve_rmse"]).iloc[0]
    pass_gap = bool(
        pd.notna(best_gap["gap_abs_error"])
        and pd.notna(best_gap["paper_gap"])
        and float(best_gap["gap_abs_error"]) <= 0.2 * max(abs(float(best_gap["paper_gap"])), 1e-12)
        and float(best_gap["curve_rmse"]) <= 20.0
    )
    evidence = (
        f"same-config v2 reward computed; best_gap_comparator={best_gap['comparator']}; "
        f"local_gap={float(best_gap['local_gap']):.6g}; "
        f"paper_gap={float(best_gap['paper_gap']):.6g}; "
        f"gap_abs_error={float(best_gap['gap_abs_error']):.6g}; "
        f"curve_rmse={float(best_gap['curve_rmse']):.6g}; "
        f"endpoint_abs_error_mean={float(best_gap['endpoint_abs_error_mean']):.6g}"
    )
    return evidence, pass_gap, True


def figure4_digitization_evidence() -> tuple[str, bool]:
    audit = read_csv(TABLE_DIR / "figure4_digitization_precision_v2.csv")
    targets = read_csv(TABLE_DIR / "figure4_digitization_precision_v2_targets.csv")
    if audit.empty or targets.empty:
        return "Figure 4 digitization precision v2 audit missing", False
    for col in [
        "linear_minus_du_visual_gap",
        "endpoint_abs_error_mean",
        "gap_abs_error",
        "endpoint_error_to_stress_ratio",
        "gap_error_to_stress_ratio",
    ]:
        if col in audit.columns:
            audit[col] = pd.to_numeric(audit[col], errors="coerce")
        if col in targets.columns:
            targets[col] = pd.to_numeric(targets[col], errors="coerce")
    target_bits = []
    for row in targets.itertuples(index=False):
        target_bits.append(f"{row.figure}_gap={float(row.linear_minus_du_visual_gap):.6g}")
    status_bits = []
    for figure, group in audit.groupby("figure", sort=False):
        counts = group["precision_status"].astype(str).value_counts()
        status = ",".join(f"{key}:{int(value)}" for key, value in counts.items())
        best = group.sort_values(["gap_abs_error", "endpoint_abs_error_mean"]).iloc[0]
        status_bits.append(
            f"{figure} best_metric={best['metric']} best_comparator={best['comparator']} "
            f"gap_error={float(best['gap_abs_error']):.6g} endpoint_error={float(best['endpoint_abs_error_mean']):.6g} "
            f"statuses={status}"
        )
    return (
        "digitization_precision_v2 "
        + "; ".join(target_bits)
        + "; "
        + "; ".join(status_bits)
        + "; source=visual_digitization_only_not_author_data",
        True,
    )


def gate_rows(config: dict, roles: pd.DataFrame) -> pd.DataFrame:
    prior = existing_prior_text()
    row_text, row_complete, row_partial = row_level_evidence()
    metric_text, metric_pass, metric_available = metric_evidence()
    schur_text, schur_closed = schur_shortlist_evidence()
    lambda_text, lambda_closed = lambda_noise_evidence()
    dls_elph_text, dls_elph_closed = dls_elph_evidence()
    tuning_text, tuning_closed = tuning_evidence()
    fold_text, fold_closed = fold_sensitivity_evidence()
    linucb_alpha_text, _linucb_alpha_closed = linucb_alpha_reg_evidence()
    figure4b_text, figure4b_pass, figure4b_available = figure4b_evidence()
    digitization_text, _digitization_closed = figure4_digitization_evidence()
    mainline_count = int((roles["claim_role"] == "mainline").sum())
    diagnostics_locked = bool(
        set(roles["claim_role"]).issuperset(
            {
                "calibrated_softmax_robustness",
                "deterministic_ablation",
                "lambda_entropy_diagnostic",
            }
        )
    )
    linear_proxy_count = int((roles["claim_role"] == "linear_ucb_proxy").sum())

    rows = [
        {
            "gate": "Protocol role lock",
            "phase": "0",
            "status": "pass" if mainline_count == 1 and diagnostics_locked else "fail",
            "strict_condition": "Exactly one exact-attempt mainline; robustness, top-k, and lambda branches cannot be promoted by performance.",
            "current_evidence": f"mainline_count={mainline_count}; diagnostics_locked={diagnostics_locked}; config={CONFIG_PATH.relative_to(ROOT)}",
            "blocking_condition": "",
            "next_artifact": REPORT_PATH.relative_to(ROOT).as_posix(),
        },
        {
            "gate": "Tau=1 full-pool exponential sampling",
            "phase": "0/6",
            "status": "pass_clean_room",
            "strict_condition": "tau=1 literal softmax is the only DU-Net mainline selection rule.",
            "current_evidence": "mainline overrides selection=exp, selection_temperature=1.0; full candidate pool selection is in run_policy.",
            "blocking_condition": "Author did not publish every normalization/numerical detail, so this is clean-room exactness, not source exactness.",
            "next_artifact": "same-config Figure 4a/4b gate",
        },
        {
            "gate": "Schur conditional entropy",
            "phase": "0/4",
            "status": "pass_clean_room_source_caveat" if (schur_closed and lambda_closed) else "pass_clean_room_shortlist_risk",
            "strict_condition": "Use Schur conditional variance, no tuned entropy transform.",
            "current_evidence": f"entropy_mode={config['acquisition']['entropy_mode']}; entropy_score_transform={config['acquisition']['entropy_score_transform']}; {schur_text}; {lambda_text}",
            "blocking_condition": "Author all-candidate/source-grounded lambda-noise convention remains unpublished." if (schur_closed and lambda_closed) else "20k shortlist and/or unpublished lambda_noise convention changes selected batches beyond the safe-threshold audit.",
            "next_artifact": "reports/paper_exact_attempt_movielens_v2_schur_shortlist_vs_full_audit.md; reports/paper_exact_attempt_movielens_v2_lambda_noise_sensitivity_audit.md",
        },
        {
            "gate": "Gaussian DLS",
            "phase": "0/4",
            "status": "source_blocked_proxy_diagnostic_complete" if dls_elph_closed else "partial",
            "strict_condition": "DLS pipeline must match paper backend and embedding source.",
            "current_evidence": f"scope={config['acquisition']['paper_dls_scope']}; neighbor_mode={config['acquisition']['dls_neighbor_mode']}; sigma_grid={config['acquisition']['sigma_grid']}; {dls_elph_text}",
            "blocking_condition": "Paper ELPH/backend parameters are unrecovered; local ELPH-style proxy changes results but cannot count as exact.",
            "next_artifact": "reports/paper_exact_attempt_movielens_v2_elph_feasibility.md",
        },
        {
            "gate": "Table-1-aligned MovieLens protocol",
            "phase": "0/2",
            "status": "pass_shape_fail_mask",
            "strict_condition": "943 users, 1681 items, 1364 initial observations, 20000 test rows, exact author masks.",
            "current_evidence": f"split={config['split']}; split prior={prior['split']}; {fold_text}",
            "blocking_condition": "Exact author initial/candidate/test masks and seed protocol not recovered.",
            "next_artifact": "reports/split_mask_search_audit_v3.md; reports/paper_exact_attempt_movielens_v2_fold_sensitivity_5seeds.md",
        },
        {
            "gate": "w-RMSE metric exactness",
            "phase": "1",
            "status": "pass" if metric_pass else ("fail_v4_no_reconstruction" if metric_available else "fail_pending_v4"),
            "strict_condition": "Endpoint scale, gap, direction, curve shape, and text-defensible formula must all match.",
            "current_evidence": f"{metric_text}; v1 metric prior={prior['v1_metric']}",
            "blocking_condition": "" if metric_pass else "Paper formula and Figure 4a y-axis conflict; exact weights/normalization unpublished.",
            "next_artifact": "reports/metric_reconstruction_audit_v4.md",
        },
        {
            "gate": "Row-level completeness",
            "phase": "1",
            "status": "pass" if row_complete else ("partial_pending_20seed" if row_partial else "fail_pending_20seed"),
            "strict_condition": "The metric-gate-required v2 methods have final-cycle row-level predictions for seeds 0..19; diagnostic branches are tracked but cannot promote or block Figure 4a.",
            "current_evidence": f"{row_text}; v1 seed0 prior={prior['v1_rows']}",
            "blocking_condition": "" if row_complete else "20-seed row-level runs are not yet complete.",
            "next_artifact": "outputs/tables/row_level_completeness_v2.csv",
        },
        {
            "gate": "Linear UCB exactness",
            "phase": "3",
            "status": "proxy_only",
            "strict_condition": "Feature map, alpha/reg, reward transform, update protocol, and package implementation must be source-defensible.",
            "current_evidence": f"linear_proxy_count={linear_proxy_count}; prior={prior['linear']}; {linucb_alpha_text}",
            "blocking_condition": "Paper names Linear UCB but does not expose exact protocol; local rows remain comparator candidates.",
            "next_artifact": "reports/external_linucb_protocol_audit_v2.md; reports/linucb_alpha_reg_sensitivity_v2.md",
        },
        {
            "gate": "Tuning frequency/objective exactness",
            "phase": "5",
            "status": "pass" if tuning_closed else "source_blocked_not_closed",
            "strict_condition": "Retuning cadence, inner objective, held-out construction, lambda/sigma/p grid or optimizer, and tie-breaks must match the author protocol.",
            "current_evidence": f"{tuning_text}; prior={prior['tuning']}",
            "blocking_condition": "" if tuning_closed else "The paper does not expose enough tuning-protocol detail; current v2 uses every-5-cycle standard_mask_rmse tuning as a clean-room fixed choice.",
            "next_artifact": "reports/paper_exact_attempt_movielens_v2_tuning_frequency_objective_audit.md",
        },
        {
            "gate": "Figure 4a same-config reproduction",
            "phase": "6",
            "status": "fail_v4_metric_gate" if metric_available else "fail_pending_metric",
            "strict_condition": "Same v2 config must match Figure 4a DU-Net and Linear UCB curves under reconstructed metric.",
            "current_evidence": f"{metric_text}; {digitization_text}",
            "blocking_condition": "" if metric_pass else "Depends on metric exactness; row-level completeness is closed for metric-gate-required methods.",
            "next_artifact": "outputs/tables/metric_curve_alignment_v4.csv; reports/figure4_digitization_precision_v2.md",
        },
        {
            "gate": "Figure 4b same-config reproduction",
            "phase": "6",
            "status": "pass" if figure4b_pass else ("fail_reward_gap_magnitude" if figure4b_available else "fail_pending_joint_gate"),
            "strict_condition": "Same v2 config must match cumulative reward direction, endpoint, and gap.",
            "current_evidence": f"{figure4b_text}; {digitization_text}",
            "blocking_condition": "" if figure4b_pass else "Same-config reward is computed, but gap/curve magnitude and Linear UCB protocol exactness remain unresolved.",
            "next_artifact": "reports/figure4_digitization_precision_v2.md; final v2 exact gate report",
        },
    ]
    return pd.DataFrame(rows)


def write_report(config: dict, roles: pd.DataFrame, gates: pd.DataFrame) -> None:
    pass_like = {"pass", "pass_clean_room", "pass_clean_room_source_caveat", "pass_shape_fail_mask"}
    pass_count = int(gates["status"].isin(pass_like).sum())
    hard_fail_count = int(gates["status"].astype(str).str.startswith("fail").sum())
    lines = [
        "# paper_exact_attempt_movielens_v2 Protocol Lock",
        "",
        "This lock freezes the next exact-reproduction attempt. It is not a tuning branch: performance cannot promote a diagnostic branch to the mainline.",
        "",
        "## Phase 0 Decision",
        "",
        "- `schur_loglambda_tuned_full_pool_softmax_tau1_literal` is the only DU-Net exact-attempt mainline.",
        "- `schur_loglambda_tuned_full_pool_softmax_tau0p3` is calibrated robustness evidence only.",
        "- `schur_tuned_topk` is a deterministic selection ablation only.",
        "- `lambda0p3_diagnostic` is a reward/RMSE diagnostic only.",
        "- Linear UCB rows are comparator candidates until an author/source-defensible protocol is recovered.",
        "",
        "## Frozen Config",
        "",
        f"- Config: `{CONFIG_PATH.relative_to(ROOT).as_posix()}`",
        f"- Seeds: `{config['seeds']}`",
        f"- Split: `{config['split']}`",
        f"- MF: `{config['mf']}`",
        f"- Acquisition cycles/batch: `{config['acquisition']['cycles']}` / `{config['acquisition']['batch_size']}`",
        f"- Entropy: `{config['acquisition']['entropy_mode']}`",
        f"- DLS scope: `{config['acquisition']['paper_dls_scope']}`",
        f"- Tuning metric/frequency: `{config['acquisition']['paper_tuning_metric']}` every `{config['acquisition']['paper_tuning_frequency']}` cycles",
        "",
        "## Variant Roles",
        "",
        md_table(roles),
        "",
        "## Gate Status",
        "",
        f"- Pass-like gates: `{pass_count}`",
        f"- Hard fail/pending gates: `{hard_fail_count}`",
        "",
        md_table(gates),
        "",
        "## Claim Boundary",
        "",
        "A future `perfect/exact reproduction` claim requires every strict gate to pass in the same v2 configuration. Until then, the defensible claim remains maximum-effort paper-only clean-room reproduction with explicit source-blocked gates.",
        "",
        f"Use `{SOURCE_RECOVERY_PATH.relative_to(ROOT).as_posix()}` as the recovery handoff before launching any new local proxy or tuning branch.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    roles = variant_roles(config)
    gates = gate_rows(config, roles)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    gates.to_csv(GATE_PATH, index=False)
    write_report(config, roles, gates)
    print(REPORT_PATH)
    print(GATE_PATH)


if __name__ == "__main__":
    main()
