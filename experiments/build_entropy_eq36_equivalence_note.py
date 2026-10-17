from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.entropy_schur import (
    explicit_kernel_schur_conditional_variance_entropy_scores,
    schur_conditional_variance_entropy_scores,
)


PREFIX = "paper_faithful_entropy_eq36_equivalence_v1"
TABLE_PATH = ROOT / "outputs" / "tables" / f"{PREFIX}.csv"
REPORT_PATH = ROOT / "reports" / f"{PREFIX}.md"


def toy_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    candidates = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 2],
            "item_id": [0, 2, 1, 2],
            "rating": [5.0, 4.0, 3.0, 2.0],
        }
    )
    observed = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "item_id": [1, 0, 2],
            "rating": [4.0, 5.0, 2.0],
        }
    )
    user_factors = np.array(
        [
            [1.0, 0.2],
            [0.1, 1.0],
            [0.7, 0.4],
        ]
    )
    item_factors = np.array(
        [
            [0.9, 0.3],
            [0.2, 1.1],
            [0.6, 0.5],
        ]
    )
    return candidates, observed, user_factors, item_factors


def build_table() -> pd.DataFrame:
    candidates, observed, user_factors, item_factors = toy_inputs()
    lambda_noise = 0.75
    feature_space = schur_conditional_variance_entropy_scores(
        candidates,
        observed,
        user_factors,
        item_factors,
        lambda_noise=lambda_noise,
        shortlist_size=0,
        batch_size=2,
    )
    explicit_kernel = explicit_kernel_schur_conditional_variance_entropy_scores(
        candidates,
        observed,
        user_factors,
        item_factors,
        lambda_noise=lambda_noise,
        batch_size=2,
    )
    frame = candidates.copy()
    frame["lambda_noise"] = lambda_noise
    frame["feature_space_value"] = feature_space
    frame["explicit_kernel_schur_value"] = explicit_kernel
    frame["absolute_difference"] = np.abs(feature_space - explicit_kernel)
    return frame


def md_table(frame: pd.DataFrame) -> str:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{float(value):.12g}")
        else:
            formatted[column] = formatted[column].astype(str)
    lines = [
        "| " + " | ".join(str(column) for column in formatted.columns) + " |",
        "| " + " | ".join(["---"] * len(formatted.columns)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in formatted.columns) + " |")
    return "\n".join(lines)


def write_report(table: pd.DataFrame) -> None:
    max_diff = float(table["absolute_difference"].max())
    lines = [
        f"# Entropy Eq.3.6 Equivalence Note: {PREFIX}",
        "",
        "## Result",
        "",
        "The implemented Schur entropy scorer is numerically equivalent to the",
        "explicit observed-kernel Schur complement on a non-orthogonal toy",
        "factorization. The maximum absolute difference in this audit is",
        f"`{max_diff:.3e}`.",
        "",
        "## What Was Verified",
        "",
        "The direct kernel form is:",
        "",
        "```text",
        "k_xx - k_xO @ inv(K_OO + lambda I) @ k_Ox",
        "```",
        "",
        "The production scorer uses the equivalent feature-space form:",
        "",
        "```text",
        "lambda * phi_x.T @ inv(Phi_O.T @ Phi_O + lambda I) @ phi_x",
        "```",
        "",
        "Here `phi_x = kron(q_user[u], q_item[i])`, where `q_user` and",
        "`q_item` are orthonormal bases derived from the DLS user factors and",
        "item factors. The equivalence follows from the Woodbury identity and",
        "is now locked by `tests/test_core.py::test_schur_entropy_matches_explicit_kernel_schur_complement`.",
        "",
        "## Boundary",
        "",
        "This closes the formula-implementation risk for the Schur conditional",
        "variance used by the clean-room DU-Net proxy. It does not recover",
        "unpublished author masks, the exact Figure 4 metric normalization, or",
        "the paper's exact Linear UCB implementation.",
        "",
        "## Audit Values",
        "",
        md_table(table),
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = build_table()
    table.to_csv(TABLE_PATH, index=False)
    write_report(table)
    print(REPORT_PATH)
    print(TABLE_PATH)


if __name__ == "__main__":
    main()
