from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "paper_exact_attempt_movielens_v2"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_GATE_PATH = TABLE_DIR / f"{PREFIX}_final_exact_gate.csv"
FIGURE_GATE_PATH = TABLE_DIR / f"{PREFIX}_same_config_figure_gate.csv"
GATE_STATUS_PATH = TABLE_DIR / f"{PREFIX}_gate_status.csv"
ELPH_SUMMARY_PATH = TABLE_DIR / f"{PREFIX}_elph_feasibility.csv"
ELPH_DEP_PATH = TABLE_DIR / f"{PREFIX}_elph_feasibility_dependencies.csv"
LINUCB_ALPHA_PATH = TABLE_DIR / "linucb_alpha_reg_sensitivity_v2.csv"
TUNING_AUDIT_PATH = TABLE_DIR / f"{PREFIX}_tuning_frequency_objective_audit.csv"
PUBLIC_SOURCE_RESCAN_PATH = TABLE_DIR / "public_source_rescan_audit.csv"
PAPER_URL_AUDIT_PATH = TABLE_DIR / "paper_url_dependency_audit.csv"
PUBLIC_SOURCE_RESCAN_REPORT = REPORT_DIR / "public_source_rescan_audit.md"
PAPER_URL_AUDIT_REPORT = REPORT_DIR / "paper_url_dependency_audit.md"

OUT_REQUESTS = TABLE_DIR / f"{PREFIX}_source_recovery_requests.csv"
OUT_PLAYBOOK = TABLE_DIR / f"{PREFIX}_source_trigger_playbook.csv"
OUT_PACKET = TABLE_DIR / f"{PREFIX}_source_recovery_packet.csv"
OUT_REPORT = REPORT_DIR / f"{PREFIX}_source_recovery_packet.md"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def first_value(frame: pd.DataFrame, column: str, default: str = "missing") -> str:
    if frame.empty or column not in frame.columns:
        return default
    value = frame.iloc[0][column]
    if pd.isna(value):
        return default
    return str(value)


def compact_text(text: object, max_len: int = 900) -> str:
    value = str(text).strip()
    if value.lower() in {"", "nan", "none"}:
        return ""
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def protocol_gate_evidence(gate: str) -> str:
    gates = read_csv(GATE_STATUS_PATH)
    row = gates[gates["gate"].astype(str) == gate]
    if row.empty:
        return f"{gate} protocol-lock row missing"
    item = row.iloc[0]
    blocker = compact_text(item.get("blocking_condition", ""), max_len=240)
    evidence = compact_text(item.get("current_evidence", ""), max_len=700)
    if blocker:
        return f"status={item.status}; {evidence}; blocker={blocker}"
    return f"status={item.status}; {evidence}"


def figure_evidence() -> dict[str, str]:
    figure = read_csv(FIGURE_GATE_PATH)
    out: dict[str, str] = {}
    for row in figure.itertuples(index=False):
        out[str(row.figure)] = (
            f"metric={row.metric}; comparator={row.comparator}; "
            f"local_du={float(row.local_du_net):.6g}; local_linear={float(row.local_linear_ucb):.6g}; "
            f"paper_du={float(row.paper_du_net):.6g}; paper_linear={float(row.paper_linear_ucb):.6g}; "
            f"endpoint_error={float(row.endpoint_error):.6g}; gap_error={float(row.gap_error):.6g}; "
            f"direction={bool(row.direction)}"
        )
    return out


def linucb_evidence() -> str:
    frame = read_csv(LINUCB_ALPHA_PATH)
    if frame.empty:
        return "Linear UCB alpha/reg sensitivity missing"
    for col in ["ucb_alpha", "ucb_regularization", "reward_tail_proxy_rmse", "cumulative_reward"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    grid = int(frame[["ucb_alpha", "ucb_regularization"]].drop_duplicates().shape[0])
    best_tail = frame.sort_values("reward_tail_proxy_rmse").iloc[0]
    best_reward = frame.sort_values("cumulative_reward", ascending=False).iloc[0]
    return (
        f"alpha/reg grid={grid}; best_reward_tail=alpha{float(best_tail.ucb_alpha):g}/"
        f"reg{float(best_tail.ucb_regularization):g}/{float(best_tail.reward_tail_proxy_rmse):.6g}; "
        f"best_reward=alpha{float(best_reward.ucb_alpha):g}/"
        f"reg{float(best_reward.ucb_regularization):g}/{float(best_reward.cumulative_reward):.6g}; "
        "paper_linear_ucb_reconstructed=no"
    )


def elph_evidence() -> str:
    summary = read_csv(ELPH_SUMMARY_PATH)
    deps = read_csv(ELPH_DEP_PATH)
    if summary.empty or deps.empty:
        return "ELPH feasibility evidence missing"
    official = deps[deps["module"].astype(str) == "elph"]
    official_available = bool(official["available"].astype(bool).any()) if not official.empty else False
    proxy = summary[summary["embedding_method"].astype(str) == "elph_role"]
    if proxy.empty:
        return f"official_elph_import_available={official_available}; proxy diagnostic missing"
    row = proxy.iloc[0]
    return (
        f"official_elph_import_available={official_available}; elph_role_proxy seed_count={int(row.seed_count)}; "
        f"delta_standard_rmse={float(row.delta_standard_mask_rmse_vs_metadata):.6g}; "
        f"delta_reward={float(row.delta_cumulative_reward_vs_metadata):.6g}; "
        f"exact_elph_reconstructed={row.exact_elph_reconstructed}"
    )


def tuning_evidence() -> str:
    audit = read_csv(TUNING_AUDIT_PATH)
    if audit.empty:
        return "tuning audit missing"
    statuses = "; ".join(f"{row.audit_item}:{row.status}" for row in audit.itertuples(index=False))
    return statuses


def source_search_evidence() -> str:
    parts = []
    rescan = read_csv(PUBLIC_SOURCE_RESCAN_PATH)
    if not rescan.empty:
        check_col = "check_id" if "check_id" in rescan.columns else "check"
        decision = rescan[rescan[check_col].astype(str) == "execution_decision"] if check_col in rescan.columns else pd.DataFrame()
        if not decision.empty:
            row = decision.iloc[0]
            parts.append(f"public_rescan={row.result}; decision={row.decision}")
        else:
            parts.append("public_rescan=completed_without_execution_decision")
    else:
        parts.append("public_rescan=missing")
    url_audit = read_csv(PAPER_URL_AUDIT_PATH)
    if not url_audit.empty:
        decision = url_audit[url_audit["url"].astype(str) == "execution_decision"]
        if not decision.empty:
            row = decision.iloc[0]
            parts.append(f"paper_url_audit={row.category}; decision={row.decision}")
        else:
            parts.append("paper_url_audit=completed_without_execution_decision")
    else:
        parts.append("paper_url_audit=missing")
    return "; ".join(parts)


def final_gate_evidence(gate_name: str) -> str:
    gates = read_csv(FINAL_GATE_PATH)
    row = gates[gates["Gate"].astype(str) == gate_name]
    if row.empty:
        return f"{gate_name} final gate row missing"
    return first_value(row, "Evidence")


def build_requests() -> pd.DataFrame:
    fig = figure_evidence()
    rows = [
        {
            "request_id": "official_code_or_supplement_bundle",
            "priority": "P0",
            "unlocks_gates": "metric; split/mask; Linear UCB; DLS/ELPH; tuning; Figure 4a; Figure 4b",
            "needed_material": "Official DU-Net MovieLens source, supplement archive, configs, split files, metric scripts, and comparator code.",
            "current_evidence": f"v2 final exact gate has 3 pass and 7 fail/source-blocked rows; {source_search_evidence()}",
            "acceptance_rule": "Artifact maps every hidden protocol choice to code/config without guessed masks, guessed metric weights, or proxy ELPH.",
            "first_action_after_recovery": "Inspect source tree read-only, map files to local protocol, then run the matching smallest diagnostic only.",
            "three_seed_validation": "Run seeds 0,1,2 only unless the source supplies exact author seeds and numeric outputs.",
            "promotion_rule": "Promote to 20 seeds only after the 3-seed diagnostic closes or materially explains at least one active exact gate.",
        },
        {
            "request_id": "exact_figure4a_metric_or_numeric_curve",
            "priority": "P0",
            "unlocks_gates": "metric; Figure 4a",
            "needed_material": "Exact w-RMSE formula, weights, normalization, aggregation, test population, and/or author numeric Figure 4a curve data.",
            "current_evidence": f"{protocol_gate_evidence('w-RMSE metric exactness')}; {fig.get('Figure 4a', 'Figure 4a figure gate missing')}",
            "acceptance_rule": "Recomputed metric explains the 1.788/1.814 visual endpoint scale and same-config gap without choosing a convenient proxy.",
            "first_action_after_recovery": "If metric-only, recompute available row/curve metrics before any model rerun; otherwise run the minimal 3-seed diagnostic.",
            "three_seed_validation": "Only required if the recovered metric needs per-cycle predictions not already available.",
            "promotion_rule": "Promote if Figure 4a endpoint, gap, direction, and curve-shape gate pass under the recovered metric.",
        },
        {
            "request_id": "author_movielens_split_masks_seed_protocol",
            "priority": "P0",
            "unlocks_gates": "split/mask; Figure 4a; Figure 4b",
            "needed_material": "Initial observed masks, candidate/test masks, item filtering convention, fold/seed mapping, and preprocessing scripts.",
            "current_evidence": protocol_gate_evidence("Table-1-aligned MovieLens protocol"),
            "acceptance_rule": "Local loader reproduces the author MovieLens protocol without u1 fallback or random guessed masks.",
            "first_action_after_recovery": "Add a source-faithful split loader and run DU-Net mainline plus required Linear UCB comparator on seeds 0,1,2.",
            "three_seed_validation": "Compare source split against current v2 on standard RMSE, reward-tail RMSE, cumulative reward, and selected-row composition.",
            "promotion_rule": "Promote only if split recovery improves or explains the Figure 4 endpoint/gap mismatch without reward collapse.",
        },
        {
            "request_id": "paper_linear_ucb_protocol",
            "priority": "P0",
            "unlocks_gates": "Linear UCB; Figure 4a; Figure 4b",
            "needed_material": "Feature construction, alpha/regularization, reward transform, update protocol, package/backend, and tie-break behavior.",
            "current_evidence": f"{protocol_gate_evidence('Linear UCB exactness')}; {linucb_evidence()}",
            "acceptance_rule": "Recovered adapter names exactly one paper Linear UCB row and explains why local concat/MF/bias proxies differ.",
            "first_action_after_recovery": "Implement as a separately labeled source adapter and run seeds 0,1,2 beside the frozen DU-Net mainline.",
            "three_seed_validation": "Check Figure 4a gap and Figure 4b reward gap against current comparator range before any 20-seed promotion.",
            "promotion_rule": "Promote if the recovered adapter materially reduces comparator ambiguity and does not introduce a new metric contradiction.",
        },
        {
            "request_id": "du_net_backend_elph_tuning_protocol",
            "priority": "P0",
            "unlocks_gates": "DLS/ELPH; tuning; Figure 4a; Figure 4b",
            "needed_material": "ELPH backend/repo, MovieLens input graph, DLS sigma/noise convention, entropy backend, tuning cadence/objective/grid, and SVD++ details.",
            "current_evidence": f"{elph_evidence()}; {tuning_evidence()}",
            "acceptance_rule": "Recovered backend replaces local metadata/proxy embedding and source-blocked tuning choices without relabeling a proxy as ELPH.",
            "first_action_after_recovery": "Map recovered backend into a new source-faithful DU-Net branch and run seeds 0,1,2.",
            "three_seed_validation": "Require improvements or explanations for both Figure 4a endpoint scale and Figure 4b reward-gap magnitude.",
            "promotion_rule": "Promote only if backend/tuning recovery closes a named gate or clearly narrows the remaining source blocker.",
        },
        {
            "request_id": "exact_figure4b_reward_or_numeric_curve",
            "priority": "P1",
            "unlocks_gates": "Figure 4b",
            "needed_material": "Cumulative reward definition, per-cycle numeric Figure 4b data, reward transform, and comparator identity.",
            "current_evidence": fig.get("Figure 4b", "Figure 4b figure gate missing"),
            "acceptance_rule": "Recovered reward curve explains why local same-config gap is 147.294 vs visual paper gap 65 under the best current comparator.",
            "first_action_after_recovery": "Recompute Figure 4b alignment first; rerun models only if reward definition depends on selection-time hidden state.",
            "three_seed_validation": "Run seeds 0,1,2 only if reward recomputation alone cannot test the recovered definition.",
            "promotion_rule": "Promote if Figure 4b ranking, endpoint, gap magnitude, and curve shape pass under the recovered definition.",
        },
    ]
    return pd.DataFrame(rows)


def build_playbook() -> pd.DataFrame:
    rows = [
        {
            "trigger_type": "metric_only",
            "minimum_evidence": "Executable metric code or enough formula/numeric data to reproduce Figure 4a endpoints.",
            "inspection_step": "Compare formula to current metric_v4 and row-level prediction columns; identify whether model rerun is necessary.",
            "first_run": "Recompute existing row-level/curve metrics when possible.",
            "stop_rule": "Stop if endpoint scale and gap still fail after exact metric recomputation.",
        },
        {
            "trigger_type": "split_or_mask",
            "minimum_evidence": "Files or code for initial/candidate/test masks and item filtering.",
            "inspection_step": "Build a source split manifest with counts, overlap with u1 fallback, and hash of each recovered mask.",
            "first_run": "DU-Net mainline plus recovered/closest Linear UCB comparator, seeds 0,1,2.",
            "stop_rule": "Stop at 3 seeds unless split recovery improves/explains Figure 4a or Figure 4b gates.",
        },
        {
            "trigger_type": "linear_ucb_adapter",
            "minimum_evidence": "Paper Linear UCB feature/update/alpha protocol or runnable comparator code.",
            "inspection_step": "Implement adapter under a source-labeled variant; keep local proxy variants unchanged.",
            "first_run": "Recovered Linear UCB adapter against frozen DU-Net mainline, seeds 0,1,2.",
            "stop_rule": "Stop if comparator remains outside the paper gap/endpoint tolerances.",
        },
        {
            "trigger_type": "du_net_backend_or_elph",
            "minimum_evidence": "Official ELPH/backend/tuning details sufficient to replace local metadata/proxy choices.",
            "inspection_step": "Audit dependency/source provenance and document why it is paper ELPH rather than same-name or local proxy ELPH.",
            "first_run": "Recovered DU-Net branch against current comparator range, seeds 0,1,2.",
            "stop_rule": "Stop if it improves one metric by tuning but worsens the same-config Figure 4 joint gate.",
        },
        {
            "trigger_type": "full_source_bundle",
            "minimum_evidence": "Runnable source tree or supplement with configs/scripts for MovieLens Figure 4.",
            "inspection_step": "Run source in isolation or map line-by-line to local modules; record all file hashes and commands.",
            "first_run": "Source-faithful smoke run, then seeds 0,1,2 using recovered protocol.",
            "stop_rule": "Promote to 20 seeds only after the source-faithful smoke and 3-seed diagnostic pass their gate-specific checks.",
        },
    ]
    return pd.DataFrame(rows)


def build_packet_rows(requests: pd.DataFrame, playbook: pd.DataFrame) -> pd.DataFrame:
    final_gate = read_csv(FINAL_GATE_PATH)
    failed = "; ".join(final_gate[final_gate["Status"] == "fail"]["Gate"].astype(str).tolist())
    passed = "; ".join(final_gate[final_gate["Status"] == "pass"]["Gate"].astype(str).tolist())
    rows = [
        {
            "item": "current_exact_gate_state",
            "status": "blocked_by_source_or_numeric_targets",
            "evidence": f"passed={passed}; failed={failed}",
            "next_action": "Use the source recovery request table; do not run another local proxy sweep without a trigger.",
        },
        {
            "item": "p0_request_count",
            "status": str(int((requests["priority"] == "P0").sum())),
            "evidence": "; ".join(requests[requests["priority"] == "P0"]["request_id"].tolist()),
            "next_action": "Prioritize P0 source requests before any new tuning branch.",
        },
        {
            "item": "trigger_playbook_count",
            "status": str(len(playbook)),
            "evidence": "; ".join(playbook["trigger_type"].tolist()),
            "next_action": "Run only the matching playbook branch after source inspection clears a trigger.",
        },
        {
            "item": "execution_decision",
            "status": "ready_but_blocked",
            "evidence": "v2 source recovery packet is current with final exact gate and ELPH feasibility evidence.",
            "next_action": "Wait for author/source/numeric materials or explicit approved dependency trigger.",
        },
    ]
    return pd.DataFrame(rows)


def write_report(packet: pd.DataFrame, requests: pd.DataFrame, playbook: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# paper_exact_attempt_movielens_v2 Source Recovery Packet",
        "",
        "This packet updates the older Figure 4a source-trigger materials with the completed v2 exact-gate evidence. It is a recovery protocol, not a request to run more proxy experiments.",
        "",
        "## Current State",
        "",
        md_table(packet),
        "",
        "## Source Requests",
        "",
        md_table(requests),
        "",
        "## Trigger Playbook",
        "",
        md_table(playbook),
        "",
        "## Execution Rule",
        "",
        "Do not launch another broad local tuning/proxy sweep while the gate remains source-blocked. If source material appears, inspect it first, then run only the smallest matching branch in the trigger playbook. Promote to 20 seeds only after the 3-seed or metric-only validation closes or materially explains an active exact gate.",
        "",
        "## Artifacts",
        "",
        f"- `{OUT_PACKET.relative_to(ROOT)}`",
        f"- `{OUT_REQUESTS.relative_to(ROOT)}`",
        f"- `{OUT_PLAYBOOK.relative_to(ROOT)}`",
        f"- `{FINAL_GATE_PATH.relative_to(ROOT)}`",
        f"- `{FIGURE_GATE_PATH.relative_to(ROOT)}`",
        f"- `{PUBLIC_SOURCE_RESCAN_REPORT.relative_to(ROOT)}`",
        f"- `{PAPER_URL_AUDIT_REPORT.relative_to(ROOT)}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if read_csv(FINAL_GATE_PATH).empty:
        raise FileNotFoundError(f"Missing final exact gate table: {FINAL_GATE_PATH}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    requests = build_requests()
    playbook = build_playbook()
    packet = build_packet_rows(requests, playbook)
    requests.to_csv(OUT_REQUESTS, index=False)
    playbook.to_csv(OUT_PLAYBOOK, index=False)
    packet.to_csv(OUT_PACKET, index=False)
    write_report(packet, requests, playbook)
    print(OUT_REPORT)
    print(OUT_PACKET)
    print(OUT_REQUESTS)
    print(OUT_PLAYBOOK)


if __name__ == "__main__":
    main()
