# Exact Replication Gate Audit v1

This is a boundary-setting audit, not a tuning report. It freezes what is closed, separates clean-room evidence from exact replication, and names the remaining source blockers.

## Gate Status

| gate | status | evidence | blocking_reason | next_action |
| --- | --- | --- | --- | --- |
| tau1_literal_mainline_freeze | closed_cleanroom | tau1 RMSE=1.070726; wins vs concat=20/20; wins vs mf_interaction=19/20 |  | Keep tau1 as mainline; keep tau0.3 robustness only. |
| row_level_metric_reconstruction_v3 | partial | Row-level schema is implemented, but completion table shows only seed0 tau1/concat currently available. | Full 20-seed row-level CSVs for all required methods are not generated. | Generate row-level predictions in bounded batches only when compute budget is explicitly allocated. |
| metric_definition_exactness | unresolved | Metric audit v2 finds direction under standard RMSE but no endpoint scale/gap candidate passes. | Paper formula, plotted Figure 4a scale, and local row-level metric candidates do not agree. | Do not call exact until a source-backed metric definition or author table is recovered. |
| figure4a_joint_endpoint_curve | partial | standard direction=True; w_rmse direction=False | Figure 4a standard RMSE proxy aligns in direction, but local w_rmse reverses the paper ordering. | Report both metric readings; do not select the convenient one as exact. |
| figure4b_same_config | partial | direction=True; gap_abs_error=82.294 | Ranking is paper-like, but reward gap magnitude is too large. | Treat lambda=0.3 and tau variants as diagnostics only; same-config gate remains partial. |
| split_mask_public_provenance | blocked_by_missing_source | split_search_audit_v2 records no exact author MovieLens masks or code; official u1-u5 folds are sensitivity-only. | Exact initial/candidate/test masks and seed mapping are not public in recovered sources. | Do not choose most paper-like fold as mainline. |
| linear_ucb_protocol | proxy_set_only | Local concat/MF/bias variants form a comparator range; contextualbandits/lrbandit are not installed and not author-confirmed. | Exact paper Linear UCB backend/protocol is not recovered. | Keep clean-room comparator caveat. |
| du_net_entropy_eq36 | closed_cleanroom | Toy equivalence report and tests show Schur conditional variance matches the explicit kernel Eq. 3.6 form to numerical precision. |  | Keep as closed unless paper reveals a different noise/backend convention. |
| all_candidate_shortlist | unresolved | Mainline uses paper_entropy_shortlist_size=20000 while final candidate size is about 76635. | All-candidate Schur is not run as a frozen diagnostic. | Run a small all-candidate diagnostic before any exact claim, not another tuning sweep. |
| lambda_noise_backend | unresolved | lambda_noise is tied to entropy_logdet_regularization=1.0 in the current executable path. | Paper does not expose enough backend/noise details to prove the literal convention. | Keep lambda_noise audit as source-blocked unless author code appears. |
| elph_backend | unresolved | Local graph embeddings include dependency-free ELPH-style proxies, not a recovered paper ELPH backend. | Exact ELPH implementation/parameters are not recovered. | Do not promote proxy ELPH into exact MovieLens Figure 4 claims. |

## Linear UCB Protocol

| candidate | implementation_status | paper_faithfulness | final_standard_rmse | final_w_rmse | final_cumulative_reward | evidence | decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| contextualbandits.online.LinUCB/default-disjoint | not_installed | candidate_external_protocol |  |  |  | Python package `contextualbandits` availability checked locally without adding dependencies. | Do not add dependency or promote until explicit author-code/package link is recovered. |
| lrbandit/OFUL paper-adjacent adapter | not_installed | candidate_external_protocol |  |  |  | Python package `lrbandit` availability checked locally without adding dependencies. | Do not add dependency or promote until explicit author-code/package link is recovered. |
| local linucb_concat | implemented_20seed | local_proxy_not_author_protocol | 1.1063891359597142 | 0.9655244892743132 | 1393.680246352064 | Included in tau1 mainline bundle summary. | Use as comparator range, not exact paper Linear UCB. |
| local linucb_mf_interaction | implemented_20seed | local_proxy_not_author_protocol | 1.0874219101410705 | 1.0482635496361028 | 1206.3700962567325 | Included in tau1 mainline bundle summary. | Use as comparator range, not exact paper Linear UCB. |
| local linucb_bias_interaction | implemented_20seed | local_proxy_not_author_protocol | 1.1303317140922473 | 0.9401492150791502 | 1521.645951182446 | Included in tau1 mainline bundle summary. | Use as comparator range, not exact paper Linear UCB. |

## Split Search

| search_group | query_or_source | status | evidence | local_artifact | decision |
| --- | --- | --- | --- | --- | --- |
| exact_paper_public_code | "Network-Enabled Sequential Data Acquisition" GitHub | not_found_public_web_or_local | Web search returned no author-code result for the exact title; local repo has PDF/Markdown/crops but no author code bundle. | Network-Enabled Sequential Data Acquisition (1).pdf | Do not claim exact paper masks from public code. |
| exact_paper_public_code | "Network-Enabled Sequential Data Acquisition" MovieLens split code | not_found_public_web_or_local | Exact-title and DU-Net/MovieLens searches did not recover split files or executable Figure 4 scripts. | reports/paper_url_dependency_audit.md | Keep author split/mask provenance unresolved. |
| grouplens_official_folds | MovieLens 100K u1-u5 official folds | available_not_exact | u1-u5 and ua/ub folds exist locally, but the paper does not state that Figure 4 uses one of these masks as the sequential acquisition test/candidate split. | data/raw/ml-100k/u1.base; data/raw/ml-100k/u1.test | Use only as sensitivity evidence, never as mainline by best-looking fold. |
| gcmc_loader | GCMC MovieLens official split loader | available_partial | Local GCMC provenance confirms a u1.base/u1.test loader pattern and a validation split, but not DU-Net's Figure 4 initial/candidate/test masks. | reports/gcmc_monti_split_provenance_audit.md | Provenance support only; exact mask still blocked. |
| monti_mat_files | Monti training_test_dataset.mat | available_for_flixster_douban_not_movielens | Local Monti-style .mat masks are present for Flixster and Douban, but not MovieLens. | data/raw/gcmc/flixster/training_test_dataset.mat; data/raw/gcmc/douban/training_test_dataset.mat | Do not infer MovieLens masks from other datasets. |
| local_split_distribution | official folds vs seeded random masks | distribution_similar_mask_not_exact | Prior overlap audit shows similar marginal distributions but different masks. | outputs/tables/movielens_split_overlap_audit.csv | Split alone is not the dominant scale fix, but exactness remains blocked. |

## Decision

The current result should be described as a strong clean-room recovery attempt with partial Figure 4 alignment. It should not be called exact or perfect until metric definition, split/mask provenance, Linear UCB protocol, all-candidate/noise backend, and ELPH backend gates are closed.

CSV artifacts: `outputs\tables\split_search_audit_v2.csv`, `outputs\tables\linear_ucb_protocol_audit_v1.csv`, `outputs\tables\exact_replication_gate_audit_v1.csv`.
