"""Confirm the promoted entropy_rho010 code-improvement candidate at 20 seeds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

PROTOCOL = "official_u1_rng_table1_init017"
VARIANT = "entropy_rho010"
CURRENT = "graph_late_lambda_half_start16"
LINEAR = "mf_linear_ucb_a002"

VARIANT_METRICS = TABLE_DIR / "movielens_100k_entropy_rho010_confirm_20seeds_metrics.csv"
BASELINE_METRICS = TABLE_DIR / "movielens_100k_source_faithful_confirm_20seeds_metrics.csv"
PAIRED_PATH = TABLE_DIR / "movielens_100k_entropy_rho010_confirm_20seeds_paired_gaps.csv"
DECISION_PATH = TABLE_DIR / "movielens_100k_entropy_rho010_confirm_20seeds_decision.csv"
REPORT_PATH = REPORT_DIR / "movielens_entropy_rho010_confirm_20seeds.md"

MIN_SEEDS = 20
MIN_PRIMARY_WINS = 14
MAX_REWARD_DROP_VS_CURRENT = 30.0


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    filtered = metrics[metrics["protocol"] == PROTOCOL].copy()
    return (
        filtered.sort_values("cycle")
        .groupby(["protocol", "variant", "seed"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def ci95(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def build_paired(variant_final: pd.DataFrame, baseline_final: pd.DataFrame) -> pd.DataFrame:
    variant = variant_final[variant_final["variant"] == VARIANT]
    current = baseline_final[baseline_final["variant"] == CURRENT]
    linear = baseline_final[baseline_final["variant"] == LINEAR]
    paired = variant.merge(
        current,
        on=["protocol", "seed"],
        suffixes=("_variant", "_current"),
        how="inner",
    ).merge(
        linear,
        on=["protocol", "seed"],
        how="inner",
    )
    rows: list[dict[str, float | int | str]] = []
    for row in paired.itertuples(index=False):
        primary_gap = float(row.linear_reward_norm_w_rmse - row.linear_reward_norm_w_rmse_variant)
        current_gap = float(row.linear_reward_norm_w_rmse - row.linear_reward_norm_w_rmse_current)
        primary_delta = float(row.linear_reward_norm_w_rmse_current - row.linear_reward_norm_w_rmse_variant)
        rows.append(
            {
                "protocol": PROTOCOL,
                "variant": VARIANT,
                "seed": int(row.seed),
                "primary_gap": primary_gap,
                "current_primary_gap": current_gap,
                "primary_delta_vs_current": primary_delta,
                "primary_win_vs_linear": int(primary_gap > 0.0),
                "primary_win_vs_current": int(primary_delta > 0.0),
                "rmse_gap": float(row.rmse - row.rmse_variant),
                "mild_gap": float(row.mild_reward_w_rmse - row.mild_reward_w_rmse_variant),
                "medium_gap": float(row.medium_reward_w_rmse - row.medium_reward_w_rmse_variant),
                "reward_delta_vs_current": float(row.cumulative_reward_variant - row.cumulative_reward_current),
            }
        )
    return pd.DataFrame(rows)


def build_decision(paired: pd.DataFrame) -> pd.DataFrame:
    n = int(paired["seed"].nunique())
    primary = paired["primary_gap"].to_numpy(dtype=float)
    current_gap = paired["current_primary_gap"].to_numpy(dtype=float)
    primary_delta = paired["primary_delta_vs_current"].to_numpy(dtype=float)
    primary_gap = float(np.mean(primary)) if n else np.nan
    current_primary_gap = float(np.mean(current_gap)) if n else np.nan
    delta_vs_current = float(np.mean(primary_delta)) if n else np.nan
    wins_linear = int(paired["primary_win_vs_linear"].sum()) if n else 0
    wins_current = int(paired["primary_win_vs_current"].sum()) if n else 0
    rmse_gap = float(paired["rmse_gap"].mean()) if n else np.nan
    mild_gap = float(paired["mild_gap"].mean()) if n else np.nan
    medium_gap = float(paired["medium_gap"].mean()) if n else np.nan
    reward_delta = float(paired["reward_delta_vs_current"].mean()) if n else np.nan
    gate_pass = bool(
        n >= MIN_SEEDS
        and primary_gap > current_primary_gap
        and wins_linear >= MIN_PRIMARY_WINS
        and rmse_gap >= 0.0
        and mild_gap >= 0.0
        and medium_gap >= 0.0
        and reward_delta >= -MAX_REWARD_DROP_VS_CURRENT
    )
    decision = "promote as code-improved candidate" if gate_pass else "do not promote; keep as primary-tail diagnostic"
    return pd.DataFrame(
        [
            {
                "variant": VARIANT,
                "paired_seed_count": n,
                "primary_gap": primary_gap,
                "primary_gap_ci95": ci95(primary),
                "primary_seed_wins_vs_linear": wins_linear,
                "primary_win_rate_vs_linear": wins_linear / n if n else np.nan,
                "current_primary_gap": current_primary_gap,
                "primary_delta_vs_current": delta_vs_current,
                "primary_seed_wins_vs_current": wins_current,
                "rmse_gap": rmse_gap,
                "mild_gap": mild_gap,
                "medium_gap": medium_gap,
                "reward_delta_vs_current": reward_delta,
                "gate_pass": gate_pass,
                "decision": decision,
            }
        ]
    )


def build_report(decision: pd.DataFrame) -> str:
    row = decision.iloc[0]
    return f"""# MovieLens Entropy-Rho 20-Seed Confirmation

Generated from `{VARIANT_METRICS.relative_to(ROOT)}` and reused the unchanged 20-seed `{CURRENT}` / `{LINEAR}` baselines from `{BASELINE_METRICS.relative_to(ROOT)}`.

## Fixed Confirmation Rule

Promote `{VARIANT}` only if it beats the current `{CURRENT}` primary gap, wins the primary metric against Linear in at least `{MIN_PRIMARY_WINS}/{MIN_SEEDS}` paired seeds, has non-negative conventional RMSE, mild, and medium gaps versus Linear, and does not lose more than `{MAX_REWARD_DROP_VS_CURRENT:.0f}` cumulative reward points versus current DU-Net.

## Decision

| variant | primary gap | win vs Linear | current primary gap | delta vs current | win vs current | RMSE gap | mild gap | medium gap | reward delta vs current | decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `{row.variant}` | {fmt(row.primary_gap)} | {int(row.primary_seed_wins_vs_linear)}/{int(row.paired_seed_count)} | {fmt(row.current_primary_gap)} | {fmt(row.primary_delta_vs_current)} | {int(row.primary_seed_wins_vs_current)}/{int(row.paired_seed_count)} | {fmt(row.rmse_gap)} | {fmt(row.mild_gap)} | {fmt(row.medium_gap)} | {fmt(row.reward_delta_vs_current, 2)} | {row.decision} |

## Interpretation

`{VARIANT}` improves the primary reward-tail gap over current DU-Net, but the fixed confirmation gate requires conventional RMSE not to reverse. If the RMSE gap is negative, this row should be reported as a useful primary-tail/coherence diagnostic rather than promoted as the new main code-improved reproduction.

## Artifacts

- `{DECISION_PATH.relative_to(ROOT)}`
- `{PAIRED_PATH.relative_to(ROOT)}`
- `{REPORT_PATH.relative_to(ROOT)}`
"""


def main() -> None:
    if not VARIANT_METRICS.exists():
        raise FileNotFoundError(f"Missing variant metrics: {VARIANT_METRICS}")
    if not BASELINE_METRICS.exists():
        raise FileNotFoundError(f"Missing baseline metrics: {BASELINE_METRICS}")
    variant_final = final_rows(pd.read_csv(VARIANT_METRICS))
    baseline_final = final_rows(pd.read_csv(BASELINE_METRICS))
    paired = build_paired(variant_final, baseline_final)
    decision = build_decision(paired)
    PAIRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(PAIRED_PATH, index=False)
    decision.to_csv(DECISION_PATH, index=False)
    REPORT_PATH.write_text(build_report(decision), encoding="utf-8")
    print(f"Wrote {DECISION_PATH}")
    print(f"Wrote {PAIRED_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
