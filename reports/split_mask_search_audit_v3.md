# Split/Mask Search Audit v3

Scope: exact-title public-source search plus local data/mask provenance for the MovieLens Figure 4 exact-reproduction gate.

## Result

No author MovieLens initial/candidate/test masks, seed protocol, or Figure 4 split script were recovered. Official MovieLens u1-u5 folds and related GCMC/Monti artifacts remain useful sensitivity context, but they do not close exact mask provenance.

## Evidence Table

| search_group | query_or_source | status | evidence | source_url | local_artifact | decision |
| --- | --- | --- | --- | --- | --- | --- |
| exact_title_public_code | "Network-Enabled Sequential Data Acquisition" GitHub / MovieLens / author / code | not_found | Web search returned no author repository, code bundle, split file, or mask artifact for the exact title. |  |  | Exact MovieLens masks remain source-blocked. |
| exact_title_preprint_or_supplement | OpenReview, arXiv, SSRN, PapersWithCode exact-title searches | not_found | Exact-title targeted searches did not expose ancillary data, code, or numeric Figure 4 tables. |  |  | Do not infer masks from visual Figure 4 curves. |
| gcmc_repository | riannevdberg/gc-mc README | available_partial | README lists MovieLens 100K commands on the official split, with and without features. | https://github.com/riannevdberg/gc-mc | data/raw/gcmc_README.md; data/raw/gcmc_data_utils.py | Supports u1 official-split provenance only; not DU-Net sequential initial/candidate/test masks. |
| gcmc_paper | Graph Convolutional Matrix Completion | context_only | Paper/repo are related matrix-completion context, not a source for Network-Enabled Sequential Data Acquisition masks. | https://arxiv.org/abs/1706.02263 | data/raw/gcmc_data_utils.py | Use as loader/preprocessing context, not exact mask evidence. |
| monti_rmgcnn_repository | fmonti/mgcnn README | available_partial | README states the repository has data for five datasets including MovieLens 100K, Douban, and Flixster. | https://github.com/fmonti/mgcnn | data/raw/gcmc/douban/training_test_dataset.mat; data/raw/gcmc/flixster/training_test_dataset.mat | Monti-style .mat masks found locally for Douban/Flixster, not MovieLens. |
| monti_rmgcnn_paper | Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks | context_only | Related recommender matrix-completion work; no recovered DU-Net MovieLens sequential mask artifact. | https://arxiv.org/abs/1704.06803 |  | Do not transfer masks from related datasets or papers. |
| local_movielens_official_folds | data/raw/ml-100k | available_not_exact | Official fold files present: u1.base, u1.test, u2.base, u2.test, u3.base, u3.test, u4.base, u4.test, u5.base, u5.test, ua.base, ua.test, ub.base, ub.test |  | data/raw/ml-100k | Use u1-u5 only for sensitivity, never as exact author masks without paper/source linkage. |
| local_monti_style_masks | data/raw/gcmc/**/training_test_dataset.mat | available_for_other_datasets_not_movielens | Local .mat masks present: data/raw/gcmc/douban/training_test_dataset.mat, data/raw/gcmc/flixster/training_test_dataset.mat |  | data/raw/gcmc | MovieLens training_test_dataset.mat is absent, so this cannot close the MovieLens mask gate. |
| prior_local_overlap | split_search_audit_v2 and gcmc_monti_split_provenance_audit | consistent_with_v3 | Prior local audits also found official-fold/GCMC/Monti provenance but no exact author MovieLens masks. |  | outputs/tables/split_search_audit_v2.csv; outputs/tables/gcmc_monti_split_provenance_audit.csv | Carry split/mask exactness as failed/source-blocked in v2. |

## Failed/Substitute Sources

| search_group | status | evidence | decision |
| --- | --- | --- | --- |
| exact_title_public_code | not_found | Web search returned no author repository, code bundle, split file, or mask artifact for the exact title. | Exact MovieLens masks remain source-blocked. |
| exact_title_preprint_or_supplement | not_found | Exact-title targeted searches did not expose ancillary data, code, or numeric Figure 4 tables. | Do not infer masks from visual Figure 4 curves. |
| local_movielens_official_folds | available_not_exact | Official fold files present: u1.base, u1.test, u2.base, u2.test, u3.base, u3.test, u4.base, u4.test, u5.base, u5.test, ua.base, ua.test, ub.base, ub.test | Use u1-u5 only for sensitivity, never as exact author masks without paper/source linkage. |
| local_monti_style_masks | available_for_other_datasets_not_movielens | Local .mat masks present: data/raw/gcmc/douban/training_test_dataset.mat, data/raw/gcmc/flixster/training_test_dataset.mat | MovieLens training_test_dataset.mat is absent, so this cannot close the MovieLens mask gate. |

## Decision

The v2 protocol may stay Table-1-shape aligned through the u1 fallback, but split/mask exactness must remain `source_blocked`. A perfect/exact reproduction claim requires author masks or an author-equivalent generation script.
