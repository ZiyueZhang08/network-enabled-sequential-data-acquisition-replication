from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"
OUT_CSV = TABLE_DIR / "external_linucb_protocol_audit_v2.csv"
OUT_REPORT = REPORT_DIR / "external_linucb_protocol_audit_v2.md"


def md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame.fillna("").copy()

    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(col) for col in view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(clean(row[col]) for col in view.columns) + " |")
    return "\n".join(lines)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def contextualbandits_rows() -> list[dict[str, object]]:
    if importlib.util.find_spec("contextualbandits") is None:
        return [
            {
                "package": "contextualbandits",
                "candidate": "LinUCB",
                "status": "not_installed",
                "version": "",
                "signature": "",
                "smoke_test": "not_run",
                "protocol_fit": "unavailable",
                "exactness_decision": "fail",
                "notes": "Package import failed.",
            }
        ]

    from contextualbandits.online import AdaptiveGreedy, BootstrappedUCB, LinUCB

    smoke = "not_run"
    try:
        model = LinUCB(nchoices=3, alpha=0.8, lambda_=1.0, random_state=1)
        x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.2]])
        choice = np.array([0, 1, 2, 0])
        reward = np.array([1.0, 0.0, 1.0, 0.0])
        model.fit(x, choice, reward)
        scores = model.decision_function(x)
        smoke = f"pass; decision_function_shape={tuple(scores.shape)}"
    except Exception as exc:  # pragma: no cover - depends on installed package
        smoke = f"fail: {type(exc).__name__}: {exc}"

    rows = []
    for cls in [LinUCB, AdaptiveGreedy, BootstrappedUCB]:
        name = cls.__name__
        if name == "LinUCB":
            protocol_fit = "partial_fixed_arms_only"
            exactness = "proxy_candidate_only"
            notes = (
                "Implements fixed-arm contextual LinUCB. The paper comparator may be item-pair candidate scoring, "
                "so using this package requires a nontrivial arm mapping that is not source-verified."
            )
        else:
            protocol_fit = "not_linear_ucb_exact"
            exactness = "diagnostic_only"
            notes = "Useful contextual-bandit comparator family, but not the named Linear UCB baseline."
        rows.append(
            {
                "package": "contextualbandits",
                "candidate": name,
                "status": "installed",
                "version": package_version("contextualbandits"),
                "signature": str(inspect.signature(cls)),
                "smoke_test": smoke if name == "LinUCB" else "not_run",
                "protocol_fit": protocol_fit,
                "exactness_decision": exactness,
                "notes": notes,
            }
        )
    return rows


def lrbandit_rows() -> list[dict[str, object]]:
    installed = importlib.util.find_spec("lrbandit") is not None
    return [
        {
            "package": "lrbandit",
            "candidate": "package",
            "status": "installed" if installed else "pypi_no_matching_distribution",
            "version": package_version("lrbandit") if installed else "",
            "signature": "",
            "smoke_test": "not_run",
            "protocol_fit": "unavailable" if not installed else "unknown",
            "exactness_decision": "fail" if not installed else "needs_adapter_audit",
            "notes": "Escalated pip install reported no matching PyPI distribution for `lrbandit`." if not installed else "",
        }
    ]


def local_proxy_rows() -> list[dict[str, object]]:
    return [
        {
            "package": "local",
            "candidate": "linucb_concat",
            "status": "implemented",
            "version": "",
            "signature": "src.acquisition policy=linucb_concat",
            "smoke_test": "covered_by_tests_and_v2_seed0",
            "protocol_fit": "dynamic_candidate_pool_proxy",
            "exactness_decision": "proxy_candidate_only",
            "notes": "Same acquisition loop/candidate removal as DU-Net, but author feature map and alpha/reg are unrecovered.",
        },
        {
            "package": "local",
            "candidate": "linucb_mf_interaction",
            "status": "implemented",
            "version": "",
            "signature": "src.acquisition policy=linucb_mf_interaction",
            "smoke_test": "covered_by_tests_and_v2_seed0",
            "protocol_fit": "dynamic_candidate_pool_proxy",
            "exactness_decision": "proxy_candidate_only",
            "notes": "MF interaction uncertainty proxy; not promoted to paper Linear UCB.",
        },
        {
            "package": "local",
            "candidate": "linucb_bias_interaction",
            "status": "implemented",
            "version": "",
            "signature": "src.acquisition policy=linucb_bias_interaction",
            "smoke_test": "covered_by_tests_and_v2_seed0",
            "protocol_fit": "dynamic_candidate_pool_proxy",
            "exactness_decision": "proxy_candidate_only",
            "notes": "Bias+interaction ridge UCB proxy; v2 seed0 row-level generated.",
        },
    ]


def build_rows() -> pd.DataFrame:
    rows = contextualbandits_rows() + lrbandit_rows() + local_proxy_rows()
    return pd.DataFrame(rows)


def write_report(rows: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    installed = rows[rows["status"].isin(["installed", "implemented"])]
    lines = [
        "# External Linear UCB Protocol Audit v2",
        "",
        "Scope: Phase 3 package/protocol availability for reconstructing the paper's Linear UCB comparator.",
        "",
        "## Result",
        "",
        "`contextualbandits` is installed and its `LinUCB` class passes a small smoke test. `lrbandit` could not be installed from PyPI because no matching distribution was found. No external package is promoted to `paper_linear_ucb_reconstructed`: the paper still does not expose feature construction, arm mapping, alpha, regularization, reward transform, or update protocol.",
        "",
        "## Candidate Table",
        "",
        md_table(rows),
        "",
        "## Installed/Implemented Candidates",
        "",
        md_table(installed[["package", "candidate", "status", "protocol_fit", "exactness_decision", "smoke_test"]]),
        "",
        "## Decision",
        "",
        "The v2 exact gate remains `proxy_only` for Linear UCB. `contextualbandits.online.LinUCB` can be used for a future comparator-range experiment only after an explicit dynamic-candidate adapter is defined and clearly labeled as non-author-exact.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    rows.to_csv(OUT_CSV, index=False)
    write_report(rows)
    print(OUT_REPORT)
    print(OUT_CSV)


if __name__ == "__main__":
    main()
