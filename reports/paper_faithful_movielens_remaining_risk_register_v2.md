# Remaining Risk Register: paper_faithful_movielens_remaining_risk_register_v2

## Position

The current repository now has a strong clean-room reproduction path, but not an exact public-author replication. The controllable formula risk is closed for Schur Eq3.6 equivalence; the remaining blockers are source availability, exact metric normalization, exact masks, and the author Linear UCB protocol.

## Risk Register

| risk | status | evidence | next_action |
| --- | --- | --- | --- |
| Tau1 literal softmax selection | closed | tau1 literal mainline is in the P0 bundle and one-config Figure 4 table | Do not tune selection temperature for the main claim unless the paper source says so. |
| Schur / Eq3.6 formula implementation | closed_cleanroom | explicit observed-kernel Schur complement equivalence report plus unit test | Keep the explicit kernel helper as an audit path; production should stay on feature-space scoring. |
| Figure 4a ranking | strong_pattern_recovery | tau1 DU-Net is lower than both clean-room Linear UCB proxies under executable standard RMSE | Stop broad hyperparameter search; only source-backed metric/split changes are justified. |
| Figure 4b cumulative reward scale | partial | ranking direction matches the supplied visual extraction, but local reward gap is too large | Treat as partial recovery until exact reward definition and comparator protocol are available. |
| Exact Figure 4 metric normalization | unresolved | seed0 row-level audit is complete for tau1 DU-Net vs linucb_concat; direction matches but no metric candidate passes the scale/gap gate | Run remaining seeds only if exact metric recovery is still needed; otherwise keep metric exactness as unresolved. |
| Exact MovieLens split/masks | unresolved | local mask scan found no Monti/DU-Net Figure 4 masks; fallback is official MovieLens u1 split | Keep fallback label unless a public mask artifact or author split is recovered. |
| Author Linear UCB protocol | proxy_set | concat, MF-interaction, and bias-interaction proxies are implemented; author feature/update protocol unavailable | Report Linear UCB as a proxy set and keep strongest proxy caveat visible. |
| ELPH / exact author backend | open_source_dependent | local ELPH-style embeddings are dependency-free proxies and have not beaten the main spectral/user-feature backend | Do not promote ELPH proxy without source-backed backend details or a clean validation win. |

## Source Refresh

| source_item | status | source_url | evidence |
| --- | --- | --- | --- |
| target DU-Net paper public artifact search | not_found |  | Manual web refresh for exact phrases around DU-Net, Linear UCB, MovieLens, and network-enabled sequential data acquisition did not recover author code, masks, or Figure 4 point data. |
| Monti et al. RMGCNN paper | paper_available_no_masks | https://arxiv.org/abs/1704.06803 | The arXiv page exposes the paper/PDF/TeX source, while the paper text gives real-data graph construction and benchmark RMSEs but not MovieLens masks or seed mappings for this clean-room Figure 4 protocol. |
| MovieLens 100K official fallback | available | https://grouplens.org/datasets/movielens/100k/ | MovieLens 100K official u1.base/u1.test |
| local Monti mask scan | not_found | data/processed/movielens_monti_scan/scan_report.json | local exact 943x1681 candidate count: 0 |

## Hard Boundary

Do not call the result perfect exact replication unless the unresolved source-dependent gates are supplied or recovered. The strongest truthful label is: paper-faithful clean-room reproduction with closed tau1 selection and Eq3.6 formula risks, plus explicit unresolved metric/split/comparator caveats.

## Primary Artifacts

- P0 bundle: `reports/paper_faithful_movielens_tau1_literal_p0_reconstruction_v1.md`
- Eq3.6 note: `reports/paper_faithful_entropy_eq36_equivalence_v1.md`
- Risk table: `outputs/tables/paper_faithful_movielens_remaining_risk_register_v2.csv`
- Source table: `outputs/tables/paper_faithful_movielens_remaining_risk_register_v2_source_refresh.csv`
