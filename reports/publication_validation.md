# Public export validation

Validation date: October 17, 2026.

Validation environment: Python 3.13.13; NumPy 2.5.1; pandas 2.3.3; Matplotlib 3.11.1; SciPy 1.18.0; h5py 3.16.0; pytest 9.1.1. These record the environment used for the checks, rather than a tested compatibility matrix for all versions.

## Scope

The source archive contains research results through July 12, 2026. This export adds a public README, paper citation metadata, external data setup instructions, publication exclusions, and this validation record. Existing algorithm code and experiment configurations are preserved. Two historical reports had local Windows image paths converted to repository-relative links.

## Checks performed

| Check | Outcome |
| --- | --- |
| Dataset-independent core tests, in the public export | `30 passed, 3 deselected` |
| Complete core test file, in a separate local copy with the supplied raw datasets restored | `33 passed` |
| Default synthetic example, in that separate local copy | Completed successfully; produced 4 policies × 21 cycle rows, summary tables, figures, and a report |
| Experiment configurations | All 139 JSON configurations parsed successfully |
| Public README and data README relative links | Checked against the exported files |
| Publication content | Raw data, processed masks, paper full text, agent state, caches, and large predictions excluded |

The three loader tests require external MovieLens / Flixster / Douban files and fail with file-not-found errors when run without those files. Download data before using the complete core-test command. No algorithm changes were made to hide or bypass these requirements.

Commands used from the repository root:

```bash
python -m pytest tests/test_core.py -k 'not LoaderTests' -q
python -m pytest tests/test_core.py -q  # separate local copy, with raw data restored
python experiments/run_main.py --config configs/mvp_synthetic.json
```

The full historical `tests/` suite was not run for this publication. Several audit builders rely on archive artifacts omitted from the public export. The 20-seed real-data experiments were not rerun for this publication; their tables and plots are historical outputs copied from the supplied archive.

## Fresh synthetic example

| Policy | Cumulative reward | RMSE | Local w-RMSE |
| --- | ---: | ---: | ---: |
| DU-Net | 1029.408517 | 0.888990 | 1.113103 |
| Entropy | 1004.229691 | 0.909210 | 1.110511 |
| Myopic | 1029.272587 | 1.022727 | 1.361347 |
| Random | 990.263585 | 0.881643 | 1.139041 |

These are synthetic workflow-check results, not evidence of exact paper replication. In this run DU-Net does not dominate every policy on every metric.

Fresh outputs were kept in the separate validation copy, so existing historical tables and reports in the public export were not replaced by the check run.

## Research limitation

Successful tests and execution establish that the checked code paths run in the validation environment. They do not recover the paper's author-specific masks, metric definition, comparator, embeddings, or tuning protocol. See the [final exactness gate](paper_exact_attempt_movielens_v2_final_exact_gate.md).
