"""Summarize the Phase 4 narrow adaptive-grid probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

BASE_PREFIX = "movielens_100k_official_u1_phase1_seed15_micro_4seeds"
ADAPTIVE_PREFIX = "movielens_100k_official_u1_phase4_adaptive_grid_4seeds"
FROZEN = "graph_late_lambda_half_start16"
ADAPTIVE = "graph_adaptive_grid_phase4_narrow"
LINEAR = "mf_linear_ucb_a002"
MYOPIC = "myopic"
ORDER = [FROZEN, ADAPTIVE, LINEAR, MYOPIC]
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


def final_rows() -> pd.DataFrame:
    base = pd.read_csv(TABLE_DIR / f"{BASE_PREFIX}_metrics.csv")
    adaptive = pd.read_csv(TABLE_DIR / f"{ADAPTIVE_PREFIX}_metrics.csv")
    base_final = base.loc[
        base["cycle"].eq(20) & base["variant"].isin([FROZEN, LINEAR, MYOPIC])
    ].copy()
    adaptive_final = adaptive.loc[adaptive["cycle"].eq(20)].copy()
    combined = pd.concat([base_final, adaptive_final], ignore_index=True)
    return combined.sort_values(["variant", "seed"], key=lambda col: col.map(variant_rank) if col.name == "variant" else col)


def summarize(final: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values("variant", key=lambda col: col.map(variant_rank)).reset_index(drop=True)


def build_gate(final: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    by_variant = summary.set_index("variant")
    by_variant_seed = final.set_index(["variant", "seed"])
    frozen_primary = float(by_variant.loc[FROZEN, "linear_reward_norm_w_rmse"])
    adaptive_primary = float(by_variant.loc[ADAPTIVE, "linear_reward_norm_w_rmse"])
    frozen_reward = float(by_variant.loc[FROZEN, "cumulative_reward"])
    adaptive_reward = float(by_variant.loc[ADAPTIVE, "cumulative_reward"])
    frozen_rmse = float(by_variant.loc[FROZEN, "rmse"])
    adaptive_rmse = float(by_variant.loc[ADAPTIVE, "rmse"])
    stable_deltas: list[float] = []
    for seed in STABLE_SEEDS:
        stable_deltas.append(
            float(
                by_variant_seed.loc[(FROZEN, seed), "linear_reward_norm_w_rmse"]
                - by_variant_seed.loc[(ADAPTIVE, seed), "linear_reward_norm_w_rmse"]
            )
        )
    seed15_delta = float(
        by_variant_seed.loc[(FROZEN, SEED15), "linear_reward_norm_w_rmse"]
        - by_variant_seed.loc[(ADAPTIVE, SEED15), "linear_reward_norm_w_rmse"]
    )
    seed15_linear_gap = float(
        by_variant_seed.loc[(LINEAR, SEED15), "linear_reward_norm_w_rmse"]
        - by_variant_seed.loc[(ADAPTIVE, SEED15), "linear_reward_norm_w_rmse"]
    )
    linear_gap = float(by_variant.loc[LINEAR, "linear_reward_norm_w_rmse"] - adaptive_primary)
    myopic_gap = float(by_variant.loc[MYOPIC, "linear_reward_norm_w_rmse"] - adaptive_primary)
    stable_harm_count = int(sum(delta < -1e-12 for delta in stable_deltas))
    primary_delta = frozen_primary - adaptive_primary
    promote = primary_delta >= 0.003 and stable_harm_count == 0
    return pd.DataFrame(
        [
            {
                "variant": ADAPTIVE,
                "primary_delta_vs_frozen": primary_delta,
                "reward_delta_vs_frozen": adaptive_reward - frozen_reward,
                "rmse_delta_vs_frozen": frozen_rmse - adaptive_rmse,
                "linear_gap_favoring_adaptive": linear_gap,
                "myopic_gap_favoring_adaptive": myopic_gap,
                "seed15_primary_delta_vs_frozen": seed15_delta,
                "seed15_linear_gap": seed15_linear_gap,
                "stable_seed_harm_count": stable_harm_count,
                "stable_seed_delta_min": min(stable_deltas),
                "stable_seed_delta_mean": float(np.mean(stable_deltas)),
                "meets_primary_003_gate": primary_delta >= 0.003,
                "strict_no_stable_harm": stable_harm_count == 0,
                "promote_to_20seed": promote,
                "decision": "promote" if promote else "reject_stop_local_tuning",
            }
        ]
    )


def adaptive_param_tail() -> pd.DataFrame:
    metrics = pd.read_csv(TABLE_DIR / f"{ADAPTIVE_PREFIX}_metrics.csv")
    cols = [
        "adaptive_lambda_entropy",
        "adaptive_diffusion_prob",
        "sigma",
        "cumulative_reward",
        "linear_reward_norm_w_rmse",
    ]
    return metrics.groupby("cycle", as_index=False)[cols].mean().tail(10).reset_index(drop=True)


def write_report(summary: pd.DataFrame, gate: pd.DataFrame, seed_table: pd.DataFrame, params: pd.DataFrame) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "figure4a_phase4_adaptive_reflection.md"
    row = gate.iloc[0]
    lines = [
        "# Figure 4a Phase 4 Narrow Adaptive Probe",
        "",
        "## Result",
        "",
        "- The narrow adaptive-grid probe failed the promotion gate.",
        f"- Primary delta vs frozen schedule: `{row['primary_delta_vs_frozen']:.6f}`; required `>= +0.003`.",
        f"- Reward delta vs frozen schedule: `{row['reward_delta_vs_frozen']:.2f}`.",
        f"- Stable-seed harm count: `{int(row['stable_seed_harm_count'])}/3`.",
        f"- Seed 15 does improve (`seed15_primary_delta_vs_frozen={row['seed15_primary_delta_vs_frozen']:.6f}`), "
        "but this repeats the Phase 1 pattern: seed-15 repair is paid for by stable-seed damage.",
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
        "## Gate",
        "",
        markdown_table(gate),
        "",
        "## Seed-Level Values",
        "",
        markdown_table(
            seed_table[
                ["variant", "seed", "cumulative_reward", "linear_reward_norm_w_rmse", "rmse", "w_rmse"]
            ]
        ),
        "",
        "## Adaptive Tail Parameters",
        "",
        markdown_table(params),
        "",
        "## Reflection",
        "",
        "- The adaptive path is more paper-like in spirit, but under the available local proxy it chooses lower-reward "
        "batches and worsens the reward-weighted objective.",
        "- Because Phase 1 and Phase 4 both found seed-15 fixes that do not generalize, the current evidence favors "
        "freezing `graph_late_lambda_half_start16` rather than opening more local tuning.",
        "- The next useful work is not another schedule sweep; it is final reporting plus metric/source reconciliation.",
        "",
        "## Plan Adjustment",
        "",
        "- Stop local tuning unless new source information appears.",
        "- Keep the current 20-seed best as the deliverable candidate.",
        "- Next: finalize the report package, mark proxy caveats clearly, and optionally run Phase 3 metric formula "
        "reconciliation from saved predictions before the final write-up.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    final = final_rows()
    summary = summarize(final)
    gate = build_gate(final, summary)
    params = adaptive_param_tail()

    summary_path = TABLE_DIR / "figure4a_phase4_adaptive_matrix.csv"
    gate_path = TABLE_DIR / "figure4a_phase4_adaptive_gate.csv"
    seed_path = TABLE_DIR / "figure4a_phase4_adaptive_seed_values.csv"
    params_path = TABLE_DIR / "figure4a_phase4_adaptive_tail_params.csv"
    summary.to_csv(summary_path, index=False)
    gate.to_csv(gate_path, index=False)
    final[["variant", "seed", "cumulative_reward", "linear_reward_norm_w_rmse", "rmse", "w_rmse"]].to_csv(
        seed_path, index=False
    )
    params.to_csv(params_path, index=False)
    report_path = write_report(summary, gate, final, params)

    print(f"Wrote {summary_path}")
    print(f"Wrote {gate_path}")
    print(f"Wrote {seed_path}")
    print(f"Wrote {params_path}")
    print(f"Wrote {report_path}")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
