"""Summarize the Phase 1 seed-15 micro-screen.

The Phase 1 gate is deliberately conservative: a candidate can repair seed 15
and still be rejected if it pays for that repair by hurting the stable seeds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

PREFIX = "movielens_100k_official_u1_phase1_seed15_micro_4seeds"
FROZEN = "graph_late_lambda_half_start16"
LINEAR = "mf_linear_ucb_a002"
MYOPIC = "myopic"
MICRO_CANDIDATES = [
    "graph_late_lambda_half_start15",
    "graph_late_lambda_half_start17",
    "graph_late_half_start16_frontier95_floor25",
]
DIAGNOSTIC_DU = "graph_base"
ORDER = [FROZEN, *MICRO_CANDIDATES, DIAGNOSTIC_DU, LINEAR, MYOPIC]
STABLE_SEEDS = [0, 1, 2]
SEED15 = 15


def ci95(values: pd.Series) -> float:
    clean = values.dropna()
    if len(clean) <= 1:
        return 0.0
    return float(1.96 * clean.std(ddof=1) / np.sqrt(len(clean)))


def variant_rank(variant: str) -> int:
    try:
        return ORDER.index(variant)
    except ValueError:
        return len(ORDER)


def markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    headers = list(frame.columns)
    rows = [[fmt(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def summarize_by_variant(final: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, group in final.groupby("variant", sort=False):
        row: dict[str, object] = {
            "variant": str(variant),
            "policy": str(group["policy"].iloc[0]),
            "seed_count": int(group["seed"].nunique()),
        }
        for col in [
            "cumulative_reward",
            "linear_reward_norm_w_rmse",
            "positive_reward_norm_w_rmse",
            "w_rmse",
            "rmse",
            "sigmoid_reward_w_rmse",
        ]:
            row[col] = float(group[col].mean())
            row[f"{col}_ci95"] = ci95(group[col])
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values("variant", key=lambda col: col.map(variant_rank)).reset_index(drop=True)


def build_gate_table(final: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    by_variant_seed = final.set_index(["variant", "seed"])
    by_variant = summary.set_index("variant")
    frozen_mean = float(by_variant.loc[FROZEN, "linear_reward_norm_w_rmse"])
    frozen_reward = float(by_variant.loc[FROZEN, "cumulative_reward"])
    frozen_rmse = float(by_variant.loc[FROZEN, "rmse"])
    frozen_linear_gap = float(by_variant.loc[LINEAR, "linear_reward_norm_w_rmse"] - frozen_mean)

    rows: list[dict[str, object]] = []
    for variant in [*MICRO_CANDIDATES, DIAGNOSTIC_DU]:
        primary_mean = float(by_variant.loc[variant, "linear_reward_norm_w_rmse"])
        reward_mean = float(by_variant.loc[variant, "cumulative_reward"])
        rmse_mean = float(by_variant.loc[variant, "rmse"])
        linear_gap = float(by_variant.loc[LINEAR, "linear_reward_norm_w_rmse"] - primary_mean)
        primary_delta = frozen_mean - primary_mean
        reward_delta = reward_mean - frozen_reward
        rmse_delta = frozen_rmse - rmse_mean
        stable_harms = 0
        stable_primary_deltas: list[float] = []
        for seed in STABLE_SEEDS:
            delta = float(
                by_variant_seed.loc[(FROZEN, seed), "linear_reward_norm_w_rmse"]
                - by_variant_seed.loc[(variant, seed), "linear_reward_norm_w_rmse"]
            )
            stable_primary_deltas.append(delta)
            if delta < -1e-12:
                stable_harms += 1
        seed15_delta = float(
            by_variant_seed.loc[(FROZEN, SEED15), "linear_reward_norm_w_rmse"]
            - by_variant_seed.loc[(variant, SEED15), "linear_reward_norm_w_rmse"]
        )
        seed15_linear_gap = float(
            by_variant_seed.loc[(LINEAR, SEED15), "linear_reward_norm_w_rmse"]
            - by_variant_seed.loc[(variant, SEED15), "linear_reward_norm_w_rmse"]
        )
        seed15_reward_delta = float(
            by_variant_seed.loc[(variant, SEED15), "cumulative_reward"]
            - by_variant_seed.loc[(FROZEN, SEED15), "cumulative_reward"]
        )
        frozen_by_seed = final.loc[final["variant"].eq(FROZEN)].set_index("seed")["linear_reward_norm_w_rmse"]
        candidate_by_seed = (
            final.loc[final["variant"].eq(variant)]
            .set_index("seed")["linear_reward_norm_w_rmse"]
            .reindex(frozen_by_seed.index)
        )
        primary_wins_vs_frozen = int((candidate_by_seed < frozen_by_seed).sum())
        primary_gate = primary_delta >= 0.003
        linear_gap_gate = (linear_gap - frozen_linear_gap) >= 0.003
        seed15_flip_gate = seed15_linear_gap > 0 and primary_delta >= 0.0
        strict_no_stable_harm = stable_harms == 0
        candidate_role = "micro_candidate" if variant in MICRO_CANDIDATES else "diagnostic_du"
        promote_20seed = (
            candidate_role == "micro_candidate"
            and strict_no_stable_harm
            and (primary_gate or linear_gap_gate or seed15_flip_gate)
        )
        if promote_20seed:
            decision = "promote"
        elif candidate_role != "micro_candidate":
            decision = "diagnostic_only"
        elif seed15_flip_gate:
            decision = "reject_borderline_seed15_tradeoff"
        else:
            decision = "reject"
        rows.append(
            {
                "variant": variant,
                "role": candidate_role,
                "primary_delta_vs_frozen": primary_delta,
                "linear_gap_delta_vs_frozen": linear_gap - frozen_linear_gap,
                "reward_delta_vs_frozen": reward_delta,
                "rmse_delta_vs_frozen": rmse_delta,
                "seed15_primary_delta_vs_frozen": seed15_delta,
                "seed15_linear_gap": seed15_linear_gap,
                "seed15_reward_delta_vs_frozen": seed15_reward_delta,
                "stable_seed_harm_count": stable_harms,
                "stable_seed_delta_min": min(stable_primary_deltas),
                "primary_wins_vs_frozen": primary_wins_vs_frozen,
                "meets_primary_003_gate": primary_gate,
                "meets_linear_gap_003_gate": linear_gap_gate,
                "seed15_flips_positive": seed15_flip_gate,
                "strict_no_stable_harm": strict_no_stable_harm,
                "promote_to_20seed": promote_20seed,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def build_seed_table(final: pd.DataFrame) -> pd.DataFrame:
    cols = ["variant", "seed", "cumulative_reward", "linear_reward_norm_w_rmse", "rmse", "w_rmse"]
    keep = [FROZEN, *MICRO_CANDIDATES, DIAGNOSTIC_DU, LINEAR, MYOPIC]
    return (
        final.loc[final["variant"].isin(keep), cols]
        .sort_values(["variant", "seed"], key=lambda col: col.map(variant_rank) if col.name == "variant" else col)
        .reset_index(drop=True)
    )


def write_report(summary: pd.DataFrame, gates: pd.DataFrame, seed_table: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "figure4a_phase1_seed15_micro_reflection.md"

    start17 = gates.loc[gates["variant"].eq("graph_late_lambda_half_start17")].iloc[0]
    start15 = gates.loc[gates["variant"].eq("graph_late_lambda_half_start15")].iloc[0]
    floor = gates.loc[gates["variant"].eq("graph_late_half_start16_frontier95_floor25")].iloc[0]
    lines = [
        "# Figure 4a Phase 1 Seed-15 Micro-Screen",
        "",
        "## Result",
        "",
        "- No Phase 1 micro-variant should be promoted to a 20-seed rerun.",
        "- `graph_late_lambda_half_start17` is the best micro-variant on the four-seed primary mean, "
        f"but its improvement is `{start17['primary_delta_vs_frozen']:.6f}`, just below the `0.003` gate.",
        "- `graph_late_lambda_half_start15` repairs seed 15 most strongly, "
        f"but it harms `{int(start15['stable_seed_harm_count'])}/3` stable seeds.",
        "- `graph_late_half_start16_frontier95_floor25` is effectively identical to the frozen baseline and "
        f"does not flip seed 15 (`seed15_linear_gap={floor['seed15_linear_gap']:.6f}`).",
        "",
        "## Four-Seed Matrix",
        "",
        markdown_table(
            summary[
                [
                    "variant",
                    "policy",
                    "seed_count",
                    "cumulative_reward",
                    "linear_reward_norm_w_rmse",
                    "positive_reward_norm_w_rmse",
                    "w_rmse",
                    "rmse",
                ]
            ]
        ),
        "",
        "## Promotion Gates",
        "",
        markdown_table(
            gates[
                [
                    "variant",
                    "primary_delta_vs_frozen",
                    "linear_gap_delta_vs_frozen",
                    "reward_delta_vs_frozen",
                    "seed15_primary_delta_vs_frozen",
                    "seed15_linear_gap",
                    "stable_seed_harm_count",
                    "promote_to_20seed",
                    "decision",
                ]
            ]
        ),
        "",
        "## Seed-Level Values",
        "",
        markdown_table(seed_table),
        "",
        "## Reflection",
        "",
        "- Seed 15 is not random noise in the narrow sense: earlier or slightly later lambda reduction can repair it.",
        "- The repair is not robust enough to replace the frozen schedule because the benefit concentrates on seed 15 "
        "and introduces losses on stable seeds.",
        "- This argues against a manual schedule tweak and for either freezing the current best or testing the paper-like "
        "adaptive tuning path, where the schedule can vary from observed data instead of being hand-picked.",
        "",
        "## Plan Adjustment",
        "",
        "- Do not run a 20-seed promotion for Phase 1 micro-variants.",
        "- Keep `graph_late_lambda_half_start16` as the frozen best.",
        "- Move next to Phase 4 narrow adaptive tuning on seeds `0,1,2,15`, with a stricter gate: it must beat the frozen "
        "four-seed mean by at least `0.003` and have `stable_seed_harm_count == 0`, or else local tuning stops and final "
        "reporting/source-recovery becomes the priority.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    metrics = pd.read_csv(TABLE_DIR / f"{PREFIX}_metrics.csv")
    final = metrics.sort_values("cycle").groupby(["protocol", "variant", "seed"], as_index=False).tail(1)
    summary = summarize_by_variant(final)
    gates = build_gate_table(final, summary)
    seed_table = build_seed_table(final)

    matrix_path = TABLE_DIR / "figure4a_phase1_seed15_micro_matrix.csv"
    gates_path = TABLE_DIR / "figure4a_phase1_seed15_micro_gates.csv"
    seed_path = TABLE_DIR / "figure4a_phase1_seed15_micro_seed_values.csv"
    summary.to_csv(matrix_path, index=False)
    gates.to_csv(gates_path, index=False)
    seed_table.to_csv(seed_path, index=False)
    report_path = write_report(summary, gates, seed_table)

    print(f"Wrote {matrix_path}")
    print(f"Wrote {gates_path}")
    print(f"Wrote {seed_path}")
    print(f"Wrote {report_path}")
    print(gates.to_string(index=False))


if __name__ == "__main__":
    main()
