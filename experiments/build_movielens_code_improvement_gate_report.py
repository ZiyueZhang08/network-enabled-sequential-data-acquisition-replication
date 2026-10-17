"""Build the 3-seed MovieLens code-improvement gate report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

PREFIX = "movielens_100k_code_improvement_gate_3seeds"
PROTOCOL = "official_u1_rng_table1_init017"
CURRENT = "graph_late_lambda_half_start16"
LINEAR = "mf_linear_ucb_a002"
VARIANTS = [
    "backend_rank20_ep12",
    "backend_rank30_ep15",
    "dynamic_tune_lam_sig",
    "residual_dls_alpha_schedule",
    "entropy_rho010",
    "divcap_user2_item5",
]

SUMMARY_PATH = TABLE_DIR / f"{PREFIX}_summary.csv"
DECISION_PATH = TABLE_DIR / f"{PREFIX}_decision.csv"
REPORT_PATH = REPORT_DIR / "movielens_code_improvement_gate_3seeds.md"

MAX_REWARD_DROP_VS_CURRENT = 30.0


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def row_by_variant(summary: pd.DataFrame, variant: str) -> pd.Series | None:
    rows = summary[(summary["protocol"] == PROTOCOL) & (summary["variant"] == variant)]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_decision(summary: pd.DataFrame) -> pd.DataFrame:
    current = row_by_variant(summary, CURRENT)
    linear = row_by_variant(summary, LINEAR)
    if current is None or linear is None:
        raise ValueError(f"Need both {CURRENT} and {LINEAR} in {SUMMARY_PATH}")

    current_primary_gap = float(linear.linear_reward_norm_w_rmse - current.linear_reward_norm_w_rmse)
    current_reward = float(current.cumulative_reward)
    rows: list[dict[str, float | bool | str]] = []
    for variant in VARIANTS:
        row = row_by_variant(summary, variant)
        if row is None:
            rows.append(
                {
                    "variant": variant,
                    "status": "missing",
                    "primary_gap": float("nan"),
                    "primary_delta_vs_current": float("nan"),
                    "rmse_gap": float("nan"),
                    "mild_gap": float("nan"),
                    "medium_gap": float("nan"),
                    "reward_delta_vs_current": float("nan"),
                    "gate_pass": False,
                    "decision": "not run",
                }
            )
            continue
        primary_gap = float(linear.linear_reward_norm_w_rmse - row.linear_reward_norm_w_rmse)
        rmse_gap = float(linear.rmse - row.rmse)
        mild_gap = float(linear.mild_reward_w_rmse - row.mild_reward_w_rmse)
        medium_gap = float(linear.medium_reward_w_rmse - row.medium_reward_w_rmse)
        reward_delta = float(row.cumulative_reward - current_reward)
        gate_pass = bool(
            primary_gap > current_primary_gap
            and rmse_gap >= 0.0
            and mild_gap >= 0.0
            and medium_gap >= 0.0
            and reward_delta >= -MAX_REWARD_DROP_VS_CURRENT
        )
        rows.append(
            {
                "variant": variant,
                "status": "complete" if float(row.seed_count) >= 3 else "incomplete",
                "primary_gap": primary_gap,
                "primary_delta_vs_current": primary_gap - current_primary_gap,
                "rmse_gap": rmse_gap,
                "mild_gap": mild_gap,
                "medium_gap": medium_gap,
                "reward_delta_vs_current": reward_delta,
                "gate_pass": gate_pass,
                "decision": "promote to 20-seed confirmation" if gate_pass else "do not promote",
            }
        )
    return pd.DataFrame(rows)


def markdown_table(decision: pd.DataFrame) -> str:
    lines = [
        "| variant | primary gap | delta vs current | RMSE gap | mild gap | medium gap | reward delta vs current | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in decision.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {fmt(row.primary_gap)} | {fmt(row.primary_delta_vs_current)} | "
            f"{fmt(row.rmse_gap)} | {fmt(row.mild_gap)} | {fmt(row.medium_gap)} | "
            f"{fmt(row.reward_delta_vs_current, 2)} | {row.decision} |"
        )
    return "\n".join(lines)


def build_report(summary: pd.DataFrame, decision: pd.DataFrame) -> str:
    current = row_by_variant(summary, CURRENT)
    linear = row_by_variant(summary, LINEAR)
    assert current is not None and linear is not None
    current_gap = float(linear.linear_reward_norm_w_rmse - current.linear_reward_norm_w_rmse)
    promoted = decision[decision["gate_pass"]]
    if promoted.empty:
        verdict = (
            "No code-improvement variant passes the fixed 3-seed gate. This suggests the current bottleneck is not "
            "a simple local code/parameter improvement under this protocol."
        )
    else:
        winners = ", ".join(f"`{variant}`" for variant in promoted["variant"])
        verdict = f"Promote {winners} to 20-seed confirmation."
    return f"""# MovieLens Code-Improvement Gate

Generated from `{SUMMARY_PATH.relative_to(ROOT)}`.

## Protocol

- Split: official MovieLens `u1.base` / `u1.test`.
- Initial size: `1364`, aligned with the paper's reported MovieLens initial matrix density.
- Seeds: `0, 1, 2`.
- Current comparator DU-Net-style row: `{CURRENT}`.
- Linear comparator: `{LINEAR}`.

## Fixed Gate

A variant passes only if its primary Linear-minus-DU gap is greater than the current `{CURRENT}` gap (`{fmt(current_gap)}`), conventional RMSE gap is non-negative, mild and medium reward-weighted gaps are non-negative, and cumulative reward is not more than `{MAX_REWARD_DROP_VS_CURRENT:.0f}` below current DU-Net.

## Decision Table

{markdown_table(decision)}

## Verdict

{verdict}

Gap convention: positive error-metric gaps mean Linear UCB has higher error than the variant. Positive reward delta means the variant has higher cumulative reward than current DU-Net.

## Artifacts

- `{DECISION_PATH.relative_to(ROOT)}`
- `{REPORT_PATH.relative_to(ROOT)}`
"""


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")
    summary = pd.read_csv(SUMMARY_PATH)
    decision = build_decision(summary)
    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    decision.to_csv(DECISION_PATH, index=False)
    REPORT_PATH.write_text(build_report(summary, decision), encoding="utf-8")
    print(f"Wrote {DECISION_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
