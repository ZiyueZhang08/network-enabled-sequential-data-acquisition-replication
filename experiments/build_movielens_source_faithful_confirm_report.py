"""Build the MovieLens source-faithful 20-seed confirmation report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

PREFIX = "movielens_100k_source_faithful_confirm_20seeds"
PROTOCOL = "official_u1_rng_table1_init017"
BASELINE = "mf_linear_ucb_a002"
VARIANT_ORDER = [
    "graph_late_lambda_half_start16_p000",
    "graph_late_lambda_half_start16_p001",
    "graph_late_lambda_half_start16",
]
PAIRED_COLUMNS = [
    "protocol",
    "variant",
    "seed",
    "primary_gap",
    "primary_win",
    "rmse_gap",
    "mild_gap",
    "medium_gap",
    "reward_gap",
    "du_primary",
    "linear_primary",
    "du_rmse",
    "linear_rmse",
    "du_reward",
    "linear_reward",
]

METRICS_PATH = TABLE_DIR / f"{PREFIX}_metrics.csv"
DECISION_PATH = TABLE_DIR / f"{PREFIX}_decision.csv"
PAIRED_PATH = TABLE_DIR / f"{PREFIX}_paired_seed_gaps.csv"
REPORT_PATH = REPORT_DIR / "movielens_source_faithful_confirm_20seeds.md"

MIN_SEEDS = 20
MIN_PRIMARY_WINS = 14
RMSE_CLOSE_TOLERANCE = 0.001
MAX_REWARD_LOSS = 150.0


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


def paired_gaps(final: pd.DataFrame) -> pd.DataFrame:
    baseline = final[final["variant"] == BASELINE]
    rows: list[dict[str, float | int | str]] = []
    for variant in VARIANT_ORDER:
        du = final[final["variant"] == variant]
        paired = du.merge(
            baseline,
            on=["protocol", "seed"],
            suffixes=("_du", "_linear"),
            how="inner",
        )
        for row in paired.itertuples(index=False):
            primary_gap = float(row.linear_reward_norm_w_rmse_linear - row.linear_reward_norm_w_rmse_du)
            rows.append(
                {
                    "protocol": PROTOCOL,
                    "variant": variant,
                    "seed": int(row.seed),
                    "primary_gap": primary_gap,
                    "primary_win": int(primary_gap > 0.0),
                    "rmse_gap": float(row.rmse_linear - row.rmse_du),
                    "mild_gap": float(row.mild_reward_w_rmse_linear - row.mild_reward_w_rmse_du),
                    "medium_gap": float(row.medium_reward_w_rmse_linear - row.medium_reward_w_rmse_du),
                    "reward_gap": float(row.cumulative_reward_du - row.cumulative_reward_linear),
                    "du_primary": float(row.linear_reward_norm_w_rmse_du),
                    "linear_primary": float(row.linear_reward_norm_w_rmse_linear),
                    "du_rmse": float(row.rmse_du),
                    "linear_rmse": float(row.rmse_linear),
                    "du_reward": float(row.cumulative_reward_du),
                    "linear_reward": float(row.cumulative_reward_linear),
                }
            )
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def variant_summary(gaps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool | str]] = []
    for variant in VARIANT_ORDER:
        group = gaps[gaps["variant"] == variant]
        n = int(group["seed"].nunique())
        primary = group["primary_gap"].to_numpy(dtype=float)
        wins = int(group["primary_win"].sum()) if n else 0
        primary_gap = float(np.mean(primary)) if n else np.nan
        rmse_gap = float(group["rmse_gap"].mean()) if n else np.nan
        mild_gap = float(group["mild_gap"].mean()) if n else np.nan
        medium_gap = float(group["medium_gap"].mean()) if n else np.nan
        reward_gap = float(group["reward_gap"].mean()) if n else np.nan
        rule_pass = bool(
            n >= MIN_SEEDS
            and primary_gap > 0.0
            and wins >= MIN_PRIMARY_WINS
            and rmse_gap >= -RMSE_CLOSE_TOLERANCE
            and mild_gap >= 0.0
            and medium_gap >= 0.0
            and reward_gap >= -MAX_REWARD_LOSS
        )
        rows.append(
            {
                "protocol": PROTOCOL,
                "variant": variant,
                "paired_seed_count": n,
                "primary_gap": primary_gap,
                "primary_gap_ci95": ci95(primary),
                "primary_seed_wins": wins,
                "primary_win_rate": wins / n if n else np.nan,
                "rmse_gap": rmse_gap,
                "mild_gap": mild_gap,
                "medium_gap": medium_gap,
                "reward_gap": reward_gap,
                "rule_pass": rule_pass,
                "decision": "",
            }
        )
    summary = pd.DataFrame(rows)
    decisions = assign_decisions(summary)
    summary["decision"] = summary["variant"].map(decisions)
    return summary


def assign_decisions(summary: pd.DataFrame) -> dict[str, str]:
    passed = {str(row.variant): bool(row.rule_pass) for row in summary.itertuples(index=False)}
    count_by_variant = {str(row.variant): int(row.paired_seed_count) for row in summary.itertuples(index=False)}
    if any(count < MIN_SEEDS for count in count_by_variant.values()):
        return {
            variant: f"insufficient seeds ({count_by_variant.get(variant, 0)}/{MIN_SEEDS})"
            for variant in VARIANT_ORDER
        }
    p000, p001, p005 = VARIANT_ORDER
    if passed[p000]:
        return {
            p000: "promote as main source-faithful result",
            p001: "secondary compromise; not needed for main claim" if passed[p001] else "not promoted",
            p005: "network-amplified secondary candidate" if passed[p005] else "not promoted",
        }
    if passed[p001]:
        return {
            p000: "most faithful diagnostic; gate failed",
            p001: "promote as fidelity/performance compromise",
            p005: "network-amplified secondary candidate" if passed[p005] else "not promoted",
        }
    if passed[p005]:
        return {
            p000: "most faithful diagnostic; gate failed",
            p001: "compromise gate failed",
            p005: "promote only as network-amplified DU-Net-style proxy",
        }
    return {
        p000: "source-faithful gate not confirmed",
        p001: "source-faithful gate not confirmed",
        p005: "source-faithful gate not confirmed",
    }


def decision_table(summary: pd.DataFrame) -> str:
    lines = [
        "| variant | primary gap | win rate | RMSE gap | mild gap | medium gap | reward gap | decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.variant}` | {fmt(row.primary_gap)} | "
            f"{int(row.primary_seed_wins)}/{int(row.paired_seed_count)} | "
            f"{fmt(row.rmse_gap)} | {fmt(row.mild_gap)} | {fmt(row.medium_gap)} | "
            f"{fmt(row.reward_gap, 2)} | {row.decision} |"
        )
    return "\n".join(lines)


def rule_text() -> str:
    return (
        f"Promotion rule, fixed before reading the 20-seed result: primary gap mean > 0; "
        f"primary paired-seed wins >= {MIN_PRIMARY_WINS}/{MIN_SEEDS}; conventional RMSE gap >= "
        f"`-{RMSE_CLOSE_TOLERANCE:.3f}`; mild and medium reward-weighted RMSE gaps >= 0; "
        f"mean cumulative reward gap >= `-{MAX_REWARD_LOSS:.0f}`."
    )


def build_report(summary: pd.DataFrame) -> str:
    if all(summary["paired_seed_count"] >= MIN_SEEDS):
        status = "complete"
    else:
        status = "incomplete"
    return f"""# MovieLens Source-Faithful 20-Seed Confirmation

Generated from `{METRICS_PATH.relative_to(ROOT)}`.

## Protocol

- Split: official MovieLens `u1.base` / `u1.test`.
- Initial size: `1364` ratings, aligned to the paper's reported MovieLens initial matrix density of approximately `0.086%`; under `u1.base`, this is `1.705%` of the 80,000 training ratings.
- Seeds: target `0..19`.
- Comparator: MF-centered Linear UCB proxy, `ucb_alpha=0.02`.
- Candidate set: `p=0`, `p=0.001`, and `p=0.005` DU-Net-style graph-late variants only.
- Status: `{status}`.

## Fixed Promotion Rule

{rule_text()}

Hierarchy: if `p=0` passes, it is the main source-faithful result; if `p=0` fails and `p=0.001` passes, `p=0.001` is the main fidelity/performance compromise; if only `p=0.005` passes, it is reported as a network-amplified DU-Net-style proxy, not the most source-faithful setting.

## Decision Table

{decision_table(summary)}

Gap convention: positive error-metric gaps mean Linear UCB has higher error than the DU-Net variant; positive reward gap means the DU-Net variant has higher cumulative reward.

## Reporting Structure

- Result A, exploratory best proxy: initial size `800`, primary gap `+0.0207`, `19/20` seed wins, and `79.5%` of the paper visual gap, with a conventional-RMSE caveat.
- Result B, source-faithful improvement gate: initial size `1364`; this confirmation table decides whether the more paper-like MovieLens protocol supports `p=0`, `p=0.001`, or only the network-amplified `p=0.005` proxy.

## Artifacts

- `{DECISION_PATH.relative_to(ROOT)}`
- `{PAIRED_PATH.relative_to(ROOT)}`
- `{REPORT_PATH.relative_to(ROOT)}`
"""


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")
    metrics = pd.read_csv(METRICS_PATH)
    final = final_rows(metrics)
    gaps = paired_gaps(final)
    summary = variant_summary(gaps)
    PAIRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(PAIRED_PATH, index=False)
    summary.to_csv(DECISION_PATH, index=False)
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")
    print(f"Wrote {DECISION_PATH}")
    print(f"Wrote {PAIRED_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
