# Network-Enabled Sequential Data Acquisition — Replication

An independent, paper-based reproduction and diagnostic study of **DU-Net** from:

> Junyu Cao and Yan Leng. *Network-Enabled Sequential Data Acquisition*. SSRN working paper, 2022; SSRN record revised May 2025.

**[Paper on SSRN](https://ssrn.com/abstract=4086999)** · **[DOI](https://doi.org/10.2139/ssrn.4086999)** · **[BibTeX](references.bib)** · **[Latest replication audit](reports/paper_exact_attempt_movielens_v2_final_exact_gate.md)**

**Status: partial reproduction; exact replication remains unresolved.** This is an independent implementation, rather than author-released code. The repository preserves runnable acquisition algorithms, experiment configurations, summary results, and explicit audits of the remaining differences from the paper.

中文说明：本项目围绕论文的 DU-Net 方法开展独立复现，包含 MovieLens、Flixster、Douban 实验、消融和 Figure 4 诊断。现有结果不能称为精确复现；具体通过项、失败项和缺失协议见下方说明及审计报告。

## Research question and connection to the paper

The paper studies how a platform should acquire user–item ratings under a limited budget while considering both immediate reward and the information needed for future recommendation tasks. DU-Net combines expected network-aware reward and information value, with dynamic locally smooth (DLS) user representations.

This project implements and investigates those components as follows:

| Paper component | Repository implementation | Qualification |
| --- | --- | --- |
| Sequential rating acquisition | [Acquisition loop](src/acquisition.py), [experiment runner](experiments/run_main.py) | Synthetic and real-data experiments |
| Low-rank preference estimation | [Matrix factorization](src/matrix_factorization.py) | Local MF / SVD++ backend |
| Dynamic locally smooth representations | [DLS](src/dls.py), [paper-oriented DLS](src/dls_paper.py), [embeddings](src/graph_embedding.py) | Several embedding backends are explicitly proxies |
| Reward with network effects | [Reward computation](src/reward.py) | Local network-reward conventions |
| Observational / conditional entropy | [Entropy](src/entropy.py), [Schur entropy](src/entropy_schur.py) | Paper-based implementation with backend caveats |
| MovieLens Figure 4 comparison | [Frozen v2 configuration](configs/paper_exact_attempt_movielens_v2.json), [final audit](reports/paper_exact_attempt_movielens_v2_final_exact_gate.md) | Exact metric, masks, comparator, and curve agreement unresolved |

The original paper's reported improvements are claims of the paper, not results established by this repository. Local proxy metrics and inverse-fit diagnostics must not be interpreted as recovered author protocols.

## Quick start

Use **Python 3.10 or newer** and run commands from the repository root.

```bash
git clone https://github.com/ZiyueZhang08/network-enabled-sequential-data-acquisition-replication.git
cd network-enabled-sequential-data-acquisition-replication
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

# End-to-end synthetic experiment: no external data required.
python experiments/run_main.py --config configs/mvp_synthetic.json
```

On Windows, activate with `.venv\Scripts\activate` instead. The synthetic runner writes:

- `outputs/tables/mvp_metrics.csv` and `outputs/tables/mvp_final_summary.csv`;
- `outputs/figures/mvp_{w_rmse,rmse,cumulative_reward}.png`;
- `reports/reproduction_report.md` (overwrites the historical report at that path).

The synthetic experiment checks the acquisition workflow; it is not a reproduction of the paper's real-data results.

## Data preparation and real-data experiments

Raw datasets are **not bundled**. Download them from their original sources; see [data preparation](data/README.md) for locations, expected paths, and usage terms. Original author split/mask artifacts have not been recovered.

After preparing MovieLens 100K:

```bash
# General MovieLens experiment.
python experiments/run_movielens.py \
  --config configs/movielens_100k_svdpp_optimized.json

# One-seed check of the frozen v2 protocol, using a separate output prefix.
python experiments/run_paper_faithful_movielens.py \
  --config configs/paper_exact_attempt_movielens_v2.json \
  --seeds 0 --output-prefix paper_exact_attempt_movielens_v2_local_check
```

To run the configured 20-seed protocol, omit `--seeds 0` and use a distinct output prefix. This is substantially more expensive than the synthetic example. The configuration filename describes an **attempt** at exact reproduction, not a successful exact match.

Flixster and Douban configurations are also provided in `configs/` and run through `experiments/run_movielens.py --config <configuration>`; that runner dispatches by the dataset name.

## Replication status

The supplied research archive records experiments through **July 12, 2026**. Its latest strict v2 audit reports **3 locally passed gates and 7 failed or source-blocked gates**:

| Locally passed | Still unresolved |
| --- | --- |
| Schur conditional entropy implementation, with source caveats | Exact w-RMSE weights and normalization |
| Full-pool softmax selection with temperature 1 | Author train / initial / candidate / test masks and seed protocol |
| Required final-cycle row-level prediction completeness in the original archive | Author Linear UCB features, parameters, and update protocol |
| | Exact DLS / ELPH backend and tuning protocol |
| | Figure 4a endpoint scale and Figure 4b reward-gap magnitude |

The row-level completeness result above is a **historical audit of the local archive**. Large row-level prediction files are excluded from this Git repository, so a fresh clone cannot independently recheck that gate without regenerating the predictions.

Recommended reading:

1. [Frozen protocol](reports/paper_exact_attempt_movielens_v2_protocol_lock.md)
2. [Final exactness gate](reports/paper_exact_attempt_movielens_v2_final_exact_gate.md)
3. [Metric reconstruction audit](reports/metric_reconstruction_audit_v4.md)
4. [Figure digitization precision](reports/figure4_digitization_precision_v2.md)
5. [Inverse mask-screening audit](reports/inverse_reconstruction_v2_mask_screening_report.md)
6. [Missing source artifacts and protocols](reports/paper_exact_attempt_movielens_v2_source_recovery_packet.md)

Older reports capture intermediate candidates and may describe different metrics, configurations, or partial results. The frozen v2 protocol and final gate above take precedence when describing current replication status. Author-request templates in the archive are historical drafts, not evidence that any request was sent or any source material was received.

## Repository structure

| Directory | Contents |
| --- | --- |
| `src/` | Data loaders, MF, DLS, entropy, reward, selection, metrics, tuning, and comparators |
| `experiments/` | Experiment runners, plotting, diagnostics, and report builders |
| `configs/` | Synthetic, MovieLens, Flixster, Douban, ablation, and frozen protocol configurations |
| `tests/` | Core behavior and historical audit / delivery regression tests |
| `data/README.md` | External dataset setup |
| `outputs/tables/` | Retained aggregate metrics and audit tables |
| `outputs/figures/` | Historical experiment figures |
| `reports/` | Interpretation, protocol locks, fidelity audits, and historical documentation |

The public export preserves the original code and relative experiment paths. It excludes raw data, processed masks, the paper PDF and full-text extraction, local agent state, caches, logs, and large prediction tables. See [excluded output inventory](reports/publication_exclusions.csv). The original ZIP remains the full local archive; the public repository is a curated export. Some historical report builders require omitted artifacts and cannot be rerun from summary files alone.

## Tests and validation

```bash
# Core tests independent of downloaded real datasets.
python -m pytest tests/test_core.py -k 'not LoaderTests'

# Include loader tests after obtaining MovieLens, Flixster, and Douban.
python -m pytest tests/test_core.py

# Historical audit / delivery tests; some require omitted archive artifacts.
python -m pytest tests
```

Unit tests validate implementation behavior, not agreement with the paper. See [publication validation](reports/publication_validation.md) for checks actually run for this export and their limits.

## Citation and attribution

Please cite the original paper when using its research ideas:

```bibtex
@misc{cao2022networkenabled,
  author       = {Cao, Junyu and Leng, Yan},
  title        = {Network-Enabled Sequential Data Acquisition},
  year         = {2022},
  howpublished = {SSRN working paper},
  doi          = {10.2139/ssrn.4086999},
  url          = {https://ssrn.com/abstract=4086999},
  note         = {SSRN record revised May 26, 2025}
}
```

Additional dataset and preprocessing references are in [references.bib](references.bib). GitHub citation metadata is provided in [CITATION.cff](CITATION.cff).

The original paper and datasets remain subject to their respective owners' terms. No blanket license for those third-party materials is granted by this repository. A project-wide software license has not been selected.
