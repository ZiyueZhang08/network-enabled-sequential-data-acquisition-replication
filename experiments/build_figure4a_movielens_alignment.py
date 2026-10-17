"""Build a MovieLens Figure 4a alignment report from PDF and local results."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_DIR = ROOT / "reports"

OUT_CSV = TABLE_DIR / "figure4a_movielens_alignment.csv"
OUT_REPORT = REPORT_DIR / "figure4a_movielens_alignment.md"

PAPER_PDF = ROOT / "Network-Enabled Sequential Data Acquisition (1).pdf"
PAPER_CROP = ROOT / "tmp" / "pdfs" / "figure4a_movielens_wrmse.png"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_by(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f"{value} not found in {key}")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "nan", "NaN"} else float("nan")


def current_rows() -> list[dict[str, object]]:
    optimized = read_rows(TABLE_DIR / "movielens_100k_svdpp_optimized_20seeds_final_summary.csv")
    baseline = read_rows(TABLE_DIR / "movielens_100k_svdpp_baselines_20seeds_final_summary.csv")
    mf_linear = read_rows(TABLE_DIR / "movielens_100k_svdpp_figure4a_linear_ucb_20seeds_final_summary.csv")

    du_net = row_by(optimized, "policy", "du_net")
    myopic = row_by(optimized, "policy", "myopic")
    raw_linear = row_by(baseline, "policy", "linear_ucb")
    best_mf_linear = min(mf_linear, key=lambda row: float(row["w_rmse"]))

    return [
        {
            "source": "current_20seed",
            "method": "DU-Net",
            "variant": "du_net",
            "w_rmse": as_float(du_net, "w_rmse"),
            "w_rmse_ci95": as_float(du_net, "w_rmse_ci95"),
            "cumulative_reward": as_float(du_net, "cumulative_reward"),
            "cumulative_reward_ci95": as_float(du_net, "cumulative_reward_ci95"),
            "note": "Current main MovieLens Figure-4a-style DU-Net result.",
        },
        {
            "source": "current_20seed",
            "method": "MF-centered Linear UCB",
            "variant": best_mf_linear["variant"],
            "w_rmse": as_float(best_mf_linear, "w_rmse"),
            "w_rmse_ci95": as_float(best_mf_linear, "w_rmse_ci95"),
            "cumulative_reward": as_float(best_mf_linear, "cumulative_reward"),
            "cumulative_reward_ci95": as_float(best_mf_linear, "cumulative_reward_ci95"),
            "note": "Best local Linear-UCB-style adapter from the new 20-seed Figure 4a comparison run.",
        },
        {
            "source": "current_20seed",
            "method": "Myopic",
            "variant": "myopic",
            "w_rmse": as_float(myopic, "w_rmse"),
            "w_rmse_ci95": as_float(myopic, "w_rmse_ci95"),
            "cumulative_reward": as_float(myopic, "cumulative_reward"),
            "cumulative_reward_ci95": as_float(myopic, "cumulative_reward_ci95"),
            "note": "Reward-only baseline from the current main MovieLens 20-seed result.",
        },
        {
            "source": "current_20seed",
            "method": "Raw Linear UCB",
            "variant": "linear_ucb",
            "w_rmse": as_float(raw_linear, "w_rmse"),
            "w_rmse_ci95": as_float(raw_linear, "w_rmse_ci95"),
            "cumulative_reward": as_float(raw_linear, "cumulative_reward"),
            "cumulative_reward_ci95": as_float(raw_linear, "cumulative_reward_ci95"),
            "note": "Legacy raw feature adapter; retained as a negative control, not the preferred paper-aligned comparison.",
        },
    ]


def paper_rows() -> list[dict[str, object]]:
    # Visual estimates from rendered PDF page 29, Figure 4a. The values are not
    # exact table data because the paper exposes this result as a plotted figure.
    estimates = [
        ("DU-Net", "du_net", 1.788, "Paper Figure 4a visual endpoint estimate at epoch 20."),
        ("Linear UCB", "linear_ucb", 1.814, "Paper Figure 4a visual endpoint estimate; best non-DU curve by w-RMSE."),
        ("Myopic", "myopic", 1.823, "Paper Figure 4a visual endpoint estimate for reward-only sampling."),
    ]
    return [
        {
            "source": "paper_figure4a_visual_estimate",
            "method": method,
            "variant": variant,
            "w_rmse": w_rmse,
            "w_rmse_ci95": "",
            "cumulative_reward": "",
            "cumulative_reward_ci95": "",
            "note": note,
        }
        for method, variant, w_rmse, note in estimates
    ]


def add_gap_fields(rows: list[dict[str, object]], du_net_by_source: dict[str, float]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        row = dict(row)
        du_net = du_net_by_source[str(row["source"])]
        w_rmse = float(row["w_rmse"])
        row["w_rmse_gap_vs_du_net"] = w_rmse - du_net
        row["w_rmse_ratio_vs_du_net"] = w_rmse / du_net if du_net else ""
        out.append(row)
    return out


def fmt(value: object, digits: int = 4) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, object]]) -> str:
    headers = ["source", "method", "variant", "w-RMSE", "gap vs DU-Net", "reward", "note"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source"]),
                    str(row["method"]),
                    str(row["variant"]),
                    fmt(row["w_rmse"], 4),
                    fmt(row["w_rmse_gap_vs_du_net"], 4),
                    fmt(row["cumulative_reward"], 2),
                    str(row["note"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_report(rows: list[dict[str, object]], path: Path, csv_path: Path) -> None:
    current_du = next(row for row in rows if row["source"] == "current_20seed" and row["method"] == "DU-Net")
    current_linear = next(
        row for row in rows if row["source"] == "current_20seed" and row["method"] == "MF-centered Linear UCB"
    )
    paper_du = next(row for row in rows if row["source"] == "paper_figure4a_visual_estimate" and row["method"] == "DU-Net")
    paper_linear = next(
        row for row in rows if row["source"] == "paper_figure4a_visual_estimate" and row["method"] == "Linear UCB"
    )
    report = f"""# Figure 4a MovieLens Alignment

Generated by `experiments/build_figure4a_movielens_alignment.py`.

## Paper Target

The original PDF is now available at `{PAPER_PDF.name}`. Figure 4a on PDF page
29 reports MovieLens w-RMSE under homogeneous response rates. A rendered crop is
available at `{PAPER_CROP.relative_to(ROOT)}`.

Visual endpoint estimates from Figure 4a:

- DU-Net: approximately {fmt(paper_du["w_rmse"], 3)} w-RMSE.
- Linear UCB: approximately {fmt(paper_linear["w_rmse"], 3)} w-RMSE.
- Paper DU-Net beats Linear UCB by roughly {fmt(paper_linear["w_rmse_gap_vs_du_net"], 3)} w-RMSE points.

These are visual estimates from a plotted figure, not exact author table data.
Absolute scale should be treated cautiously because the local reproduction uses
its own implementation of the paper metric and matrix-completion backend.

## Current Result

The current 20-seed MovieLens DU-Net result is {fmt(current_du["w_rmse"], 4)}
w-RMSE. The best local Linear-UCB-style adapter from the new 20-seed run is
`{current_linear["variant"]}` with {fmt(current_linear["w_rmse"], 4)} w-RMSE.
DU-Net is lower by {fmt(current_linear["w_rmse_gap_vs_du_net"], 4)} w-RMSE
points.

This matches the Figure 4a direction under the local reward-weighted w-RMSE
implementation: DU-Net remains better than the strongest Linear-UCB-style
comparator on that metric. A later paper-metric instrumentation pass shows that
this directional claim does not hold under a conventional test-set RMSE reading
of the PDF formula, where the MF-centered Linear UCB comparator is slightly
lower than DU-Net. The local gap is smaller than the visual paper gap, so this
is still a qualified alignment rather than an exact numerical match.

## Alignment Table

{markdown_table(rows)}

## Interpretation

- Absolute Figure 4a scale is not yet reproduced: the paper visual endpoint is
  around 1.79, while the local metric endpoint is around 0.83. This likely
  reflects remaining differences in exact preprocessing, noisy matrix
  completion, weighting, and DU-Net implementation details.
- Relative ordering is useful evidence for the user's stated goal only under
  the local reward-weighted w-RMSE implementation. Under conventional test-set
  RMSE, this local DU-Net proxy is not yet ahead of MF-centered Linear UCB.
- The same comparison is also consistent with the paper's MovieLens reward
  statement: local MF-centered Linear UCB has slightly higher cumulative reward
  than DU-Net, while DU-Net has better final w-RMSE.

CSV artifact: `{csv_path.relative_to(ROOT)}`.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = paper_rows() + current_rows()
    du_net_by_source = {
        "paper_figure4a_visual_estimate": next(
            float(row["w_rmse"]) for row in rows if row["source"] == "paper_figure4a_visual_estimate" and row["method"] == "DU-Net"
        ),
        "current_20seed": next(
            float(row["w_rmse"]) for row in rows if row["source"] == "current_20seed" and row["method"] == "DU-Net"
        ),
    }
    rows = add_gap_fields(rows, du_net_by_source)
    write_csv(rows, OUT_CSV)
    write_report(rows, OUT_REPORT, OUT_CSV)
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
