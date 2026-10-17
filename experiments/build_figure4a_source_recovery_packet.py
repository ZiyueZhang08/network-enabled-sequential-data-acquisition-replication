"""Build the post-final-package Figure 4a source-recovery packet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

FINAL_SUMMARY = TABLE_DIR / "figure4a_final_package_summary.csv"
PHASE2_GATE = TABLE_DIR / "figure4a_observational_entropy_gate.csv"
PHASE3_GATE = TABLE_DIR / "figure4a_adaptive_grid_gate.csv"
NO_NETWORK_GATE = TABLE_DIR / "figure4a_no_network_fidelity_gate.csv"
EXP_GATE = TABLE_DIR / "figure4a_exp_fidelity_gate.csv"

OUT_BLOCKERS = TABLE_DIR / "figure4a_source_recovery_blockers.csv"
OUT_PACKET = REPORT_DIR / "figure4a_source_recovery_packet.md"

PAPER_GAP = 1.814 - 1.788


def fmt(value: object, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def blocker_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "blocker": "Exact Figure 4a w-RMSE definition",
                "why_it_matters": "Local reward-weighted w-RMSE gives DU-Net a positive gap, but conventional RMSE reverses the ordering.",
                "minimum_needed": "Weights, normalization denominator, test mask, clipping, aggregation over seeds/cycles.",
                "local_status": "metric proxy only",
            },
            {
                "priority": 2,
                "blocker": "MovieLens split and initialization protocol",
                "why_it_matters": "Figure 4a averages over 20 random initializations, but exact split/candidate/initial masks are not recovered.",
                "minimum_needed": "Train/test split files, initial observed sets, candidate masks, filtering, seed protocol.",
                "local_status": "close local split, not author exact",
            },
            {
                "priority": 3,
                "blocker": "DU-Net observational entropy and DLS implementation",
                "why_it_matters": "Local observational entropy variants improved conventional RMSE but hurt reward-weighted w-RMSE and reward.",
                "minimum_needed": "Entropy estimator, DLS smoothing rule, sigma schedule, network diffusion rule, feature normalization.",
                "local_status": "proxy mechanisms exhausted",
            },
            {
                "priority": 4,
                "blocker": "Per-cycle tuning protocol for lambda, sigma_t, p_t",
                "why_it_matters": "Local fixed, adaptive, guarded, and reward-floor adaptive variants did not close the gap.",
                "minimum_needed": "Inner train/acquisition/test construction, tuning grid, loss, tie-breaks, retuning frequency.",
                "local_status": "implemented approximations failed promotion gates",
            },
            {
                "priority": 5,
                "blocker": "Linear UCB feature/update protocol",
                "why_it_matters": "Main comparator is MF-centered Linear UCB proxy; alpha sensitivity is bounded but not source-exact.",
                "minimum_needed": "Feature map, alpha schedule, regularization, update frequency, reward transform.",
                "local_status": "proxy comparator labeled and frozen",
            },
        ]
    )


def markdown_blockers(frame: pd.DataFrame) -> str:
    lines = [
        "| priority | blocker | minimum needed | local status |",
        "|---:|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.priority} | {row.blocker} | {row.minimum_needed} | {row.local_status} |"
        )
    return "\n".join(lines)


def write_packet(blockers: pd.DataFrame) -> None:
    final = pd.read_csv(FINAL_SUMMARY)
    du = final[final["variant"] == "du_net"].iloc[0]
    linear = final[final["variant"] == "mf_linear_ucb_a002"].iloc[0]
    gap = float(linear.local_reward_weighted_w_rmse) - float(du.local_reward_weighted_w_rmse)
    phase2 = pd.read_csv(PHASE2_GATE).sort_values("w_rmse").iloc[0]
    phase3 = pd.read_csv(PHASE3_GATE).sort_values("w_rmse").iloc[0]
    no_network = pd.read_csv(NO_NETWORK_GATE).iloc[0]
    exp = pd.read_csv(EXP_GATE).iloc[0]

    email = f"""Subject: Request for DU-Net Figure 4a MovieLens reproduction details

Dear authors,

I am trying to reproduce the DU-Net MovieLens 100K result reported in Figure 4a
of "Network-Enabled Sequential Data Acquisition".

My current local reproduction is directionally positive under a transparent
reward-weighted proxy, but it is not an exact Figure 4a match:

- DU-Net local reward-weighted w-RMSE: {fmt(du.local_reward_weighted_w_rmse)}
- MF-centered Linear UCB proxy w-RMSE: {fmt(linear.local_reward_weighted_w_rmse)}
- Linear-minus-DU gap: {fmt(gap)}; paper visual gap is approximately {fmt(PAPER_GAP)}
- Conventional RMSE still favors the Linear UCB proxy: DU-Net {fmt(du.rmse)} vs Linear UCB {fmt(linear.rmse)}

I have also tried observational-entropy variants, paper-like adaptive tuning
approximations, a MovieLens no-network fidelity probe, and an exponential batch
sampling probe; none passed the promotion gates. The no-network probe changed
the primary metric by {fmt(no_network.primary_delta_vs_frozen)} and reward by
{fmt(no_network.reward_delta_vs_frozen, 2)}. The exponential-sampling probe
changed the primary metric by {fmt(exp.primary_delta_vs_frozen)} and reward by
{fmt(exp.reward_delta_vs_frozen, 2)}.

Would you be able to share any of the following minimum Figure 4a details?

1. Exact w-RMSE definition: weights, normalization, test mask, clipping, and
   seed/cycle aggregation.
2. MovieLens split/config: train/test split, initial observed set, candidate
   mask, filtering, and random initialization protocol.
3. DU-Net details: observational entropy estimator, DLS smoothing, network
   diffusion, SVD++ backend settings, and per-cycle tuning of lambda, sigma_t,
   and p_t.
4. Linear UCB details: feature map, alpha/regularization, update protocol, and
   reward transform.
5. Batch acquisition details: exact top-k vs exponential sampling formula,
   temperature, replacement policy, and tie handling.

Even the w-RMSE definition plus MovieLens split/config would be enough for me to
run a small seeds 0-2 validation before any 20-seed rerun.

Thank you for any guidance or artifacts you can share.
"""

    packet = f"""# Figure 4a Source-Recovery Packet

Generated by `experiments/build_figure4a_source_recovery_packet.py`.

## Current Progress

Local execution is complete. The final 20-seed package gives DU-Net
`{fmt(du.local_reward_weighted_w_rmse)}` versus MF-centered Linear UCB proxy
`{fmt(linear.local_reward_weighted_w_rmse)}` under the primary local
reward-weighted w-RMSE, for a Linear-minus-DU gap of `{fmt(gap)}`. That is
`{fmt(100 * gap / PAPER_GAP, 1)}%` of the paper visual gap.

Exact replication is not achieved. The result is directionally positive only
under the reward-weighted/high-rating proxy; conventional RMSE still favors the
Linear UCB proxy.

## Executed Local Branches

- Metric alignment: completed; not metric-blocked, but metric ambiguity remains.
- Observational entropy: no candidate promoted; best row `{phase2.variant}` with
  w-RMSE `{fmt(phase2.w_rmse)}`.
- Adaptive tuning: fixed, guarded, and reward-floor constrained adaptive variants
  did not promote; best adaptive row `{phase3.variant}` with w-RMSE
  `{fmt(phase3.w_rmse)}`.
- Comparator tightening: keep `mf_linear_ucb_a002` as MF-centered Linear UCB
  proxy, with final 20-seed primary gap `{fmt(gap)}`.
- No-network fidelity: rejected; primary delta
  `{fmt(no_network.primary_delta_vs_frozen)}`, reward delta
  `{fmt(no_network.reward_delta_vs_frozen, 2)}`.
- Exponential sampling fidelity: rejected; primary delta
  `{fmt(exp.primary_delta_vs_frozen)}`, reward delta
  `{fmt(exp.reward_delta_vs_frozen, 2)}`.

## Blocking Details To Recover

{markdown_blockers(blockers)}

## Search/Request Queries

- `"Network-Enabled Sequential Data Acquisition" DU-Net code`
- `"Network-Enabled Sequential Data Acquisition" Figure 4a`
- `"Network-Enabled Sequential Data Acquisition" GitHub`
- `"DU-Net" "MovieLens" "Linear UCB"`

The actionable next step is direct source recovery: author code, supplement,
exact metric definition, split files, or precise reproduction settings.

## Ready-To-Send Author Request

```text
{email}
```

## Linked Artifacts

- `reports/figure4a_final_package.md`
- `reports/figure4a_metric_alignment_audit.md`
- `reports/figure4a_observational_entropy_diagnostic.md`
- `reports/figure4a_adaptive_grid_diagnostic.md`
- `reports/figure4a_linear_ucb_comparator_diagnostic.md`
- `reports/figure4a_no_network_fidelity_probe.md`
- `reports/figure4a_exp_fidelity_probe.md`
- `outputs/tables/figure4a_source_recovery_blockers.csv`
"""
    OUT_PACKET.write_text(packet, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    blockers = blocker_rows()
    blockers.to_csv(OUT_BLOCKERS, index=False)
    write_packet(blockers)
    print(f"Wrote {OUT_BLOCKERS}")
    print(f"Wrote {OUT_PACKET}")
    print(blockers.to_string(index=False))


if __name__ == "__main__":
    main()
