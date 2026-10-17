from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "figure4_digitization_precision_v2"
V1_PREFIX = "paper_faithful_movielens_schur_loglambda_tuned_full_pool_softmax_tau1_literal_v1"
TARGET_PATH = ROOT / "outputs" / "tables" / V1_PREFIX / "tau1_figure4_digitized_targets.csv"
ALIGNMENT_PATH = ROOT / "outputs" / "tables" / "metric_curve_alignment_v4.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
AUDIT_PATH = TABLE_DIR / f"{PREFIX}.csv"
TARGET_SUMMARY_PATH = TABLE_DIR / f"{PREFIX}_targets.csv"
REPORT_PATH = REPORT_DIR / f"{PREFIX}.md"

# Conservative stress bands for visual extraction noise. These are not author
# uncertainty intervals; they are deliberately broad sanity checks.
STRESS_BANDS = {
    "Figure 4a": 0.005,
    "Figure 4b": 10.0,
}


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


def load_targets() -> pd.DataFrame:
    if not TARGET_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TARGET_PATH)


def load_alignment() -> pd.DataFrame:
    if not ALIGNMENT_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ALIGNMENT_PATH)


def target_summary(targets: pd.DataFrame) -> pd.DataFrame:
    if targets.empty:
        return pd.DataFrame()
    endpoints = targets.sort_values("cycle").groupby(["figure", "method"], as_index=False).tail(1)
    rows: list[dict[str, object]] = []
    for figure, group in endpoints.groupby("figure", sort=False):
        du = group[group["method"].astype(str) == "DU-Net"]
        linear = group[group["method"].astype(str) == "Linear UCB"]
        if du.empty or linear.empty:
            continue
        stress = float(STRESS_BANDS.get(str(figure), 0.0))
        rows.append(
            {
                "figure": figure,
                "paper_metric": str(group["paper_metric"].iloc[0]),
                "du_endpoint_visual": float(du["paper_value"].iloc[0]),
                "linear_endpoint_visual": float(linear["paper_value"].iloc[0]),
                "linear_minus_du_visual_gap": float(linear["paper_value"].iloc[0] - du["paper_value"].iloc[0]),
                "digitization_source": str(group["digitization_source"].iloc[0]),
                "digitization_status": str(group["digitization_status"].iloc[0]),
                "conservative_endpoint_stress_band": stress,
                "conservative_gap_stress_band": 2.0 * stress,
            }
        )
    return pd.DataFrame(rows)


def classify_precision(row: pd.Series) -> str:
    endpoint_error = float(row["endpoint_abs_error_mean"])
    gap_error = float(row["gap_abs_error"])
    endpoint_band = float(row["conservative_endpoint_stress_band"])
    gap_band = float(row["conservative_gap_stress_band"])
    endpoint_ok = endpoint_error <= endpoint_band
    gap_ok = gap_error <= gap_band
    if endpoint_ok and gap_ok:
        return "within_conservative_digitization_stress"
    if gap_ok:
        return "gap_within_stress_but_endpoint_scale_not"
    return "not_explained_by_digitization_stress"


def build_audit(alignment: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = target_summary(targets)
    if alignment.empty or summary.empty:
        return pd.DataFrame(), summary
    audit = alignment.merge(
        summary[
            [
                "figure",
                "paper_metric",
                "du_endpoint_visual",
                "linear_endpoint_visual",
                "linear_minus_du_visual_gap",
                "digitization_source",
                "digitization_status",
                "conservative_endpoint_stress_band",
                "conservative_gap_stress_band",
            ]
        ],
        on="figure",
        how="left",
    )
    numeric_cols = [
        "curve_rmse",
        "endpoint_abs_error_mean",
        "local_gap",
        "paper_gap",
        "gap_abs_error",
        "conservative_endpoint_stress_band",
        "conservative_gap_stress_band",
    ]
    for col in numeric_cols:
        audit[col] = pd.to_numeric(audit[col], errors="coerce")
    audit["endpoint_error_to_stress_ratio"] = (
        audit["endpoint_abs_error_mean"] / audit["conservative_endpoint_stress_band"].replace(0.0, pd.NA)
    )
    audit["gap_error_to_stress_ratio"] = audit["gap_abs_error"] / audit["conservative_gap_stress_band"].replace(0.0, pd.NA)
    audit["precision_status"] = audit.apply(classify_precision, axis=1)
    audit["paper_numeric_source_status"] = "visual_digitization_only_not_author_data"
    return audit.sort_values(["figure", "precision_status", "gap_abs_error"]), summary


def write_report(audit: pd.DataFrame, targets: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if audit.empty:
        REPORT_PATH.write_text(
            "# figure4_digitization_precision_v2\n\nMissing target or alignment inputs.\n",
            encoding="utf-8",
        )
        return
    status_counts = (
        audit.groupby(["figure", "precision_status"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["figure", "precision_status"])
    )
    best_by_figure = audit.sort_values(["figure", "gap_abs_error", "endpoint_abs_error_mean"]).groupby("figure").head(3)
    lines = [
        "# figure4_digitization_precision_v2",
        "",
        "This audit formalizes the Figure 4 visual-target boundary for the v2 exact-attempt gates. It does not introduce new tuning, and it does not treat visually digitized values as author data.",
        "",
        "## Inputs",
        "",
        f"- Digitized target table: `{TARGET_PATH.relative_to(ROOT).as_posix()}`",
        f"- Same-config alignment table: `{ALIGNMENT_PATH.relative_to(ROOT).as_posix()}`",
        "- Stress bands: Figure 4a endpoint +/-0.005 w-RMSE; Figure 4b endpoint +/-10 reward units; gap band is twice the endpoint band.",
        "",
        "## Visual Targets",
        "",
        md_table(targets),
        "",
        "## Precision Stress Verdict",
        "",
        md_table(status_counts),
        "",
        "## Best Local Alignments Under Current Visual Targets",
        "",
        md_table(best_by_figure),
        "",
        "## Claim Boundary",
        "",
        "Digitization precision can explain small gap-level differences in some Figure 4a rows, but it cannot explain the endpoint-scale mismatch. For Figure 4b, the same-config reward gap error remains larger than the conservative digitization stress band. Exact Figure 4 reproduction therefore still requires author numeric data or source code.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    targets = load_targets()
    alignment = load_alignment()
    audit, target_endpoints = build_audit(alignment, targets)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    target_endpoints.to_csv(TARGET_SUMMARY_PATH, index=False)
    audit.to_csv(AUDIT_PATH, index=False)
    write_report(audit, target_endpoints)
    print(AUDIT_PATH)
    print(TARGET_SUMMARY_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
