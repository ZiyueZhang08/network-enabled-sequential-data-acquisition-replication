# Paper Alignment Gap Matrix

This report aligns the reproduction plan and the paper's empirical claims with the current code/results state.

## Summary Judgment

The project is no longer an MVP. It is a **strong partial reproduction with paper-style 20-seed evidence**, especially on MovieLens and Flixster. Douban has moved from a near-tie to a diagnosed sparse-network case with positive network-observational candidates. The remaining distance to the paper is now mostly **faithfulness and completeness**, not basic execution.

Status counts: complete=2, mostly complete=3, partial=5, proxy-only=2.

## Gap Matrix

| paper requirement | status | gap | basic next | long-term next |
|---|---|---|---|---|
| Network-enabled sequential data acquisition problem | complete | No major structural gap; exact paper hyperparameters may still differ by dataset. | Keep this as frozen infrastructure and avoid unplanned refactors. | Only revisit if exact paper budget/split details are recovered from supplementary material. |
| Three real datasets: MovieLens, Flixster, Douban | complete | MovieLens uses a constructed similarity network, consistent with the plan; Flixster/Douban use GCMC social graphs. | Keep dataset loader tests as the guardrail. | Document any scale differences from the paper's exact preprocessing. |
| 20-cycle, batch-size-100, 20-seed main experiments | mostly complete | Main MovieLens/Flixster/Douban candidates are 20-seed and now synthesized in the final caveated report; a new full rerun should wait for primary-source or faithful-mechanism upgrades. | Treat current 20-seed summaries and final caveated synthesis as the frozen snapshot. | Rerun all final settings only after exact entropy/embedding/DLS/diffusion or author-code benchmark upgrades. |
| DU-Net combines reward and information value | complete proxy | Scalar mixture, batch composition, an optional reward-information joint gate, probability-weighted expected network log-det entropy, and reward-frontier entropy gating are implemented. Network-batch, stochastic entropy, weighted expected entropy, joint-gate, and reward-frontier variants change behavior but current information pressure still costs w-RMSE on Douban. | Report reward-information diagnostics alongside headline results; do not promote network-batch, stochastic entropy, weighted expected entropy, joint-gate, or reward-frontier variants without a stronger representation signal. | Move from local entropy-gate tuning to stronger structural representation or paper-specific benchmark definitions. |
| Network-amplified reward via diffusion | partial | Current Flixster DU-Net equals no-network; Douban no-DLS/no-network evidence suggests raw proxy diffusion is not the full paper mechanism. | Keep network ablations in the main report, not hidden as tuning failures. | Implement a more faithful independent-cascade expected reward approximation with calibrated degree normalization. |
| DLS dynamic locally smooth user representation | partial positive mechanism | DLS exists, but Douban shows sparse-graph noise; reliability shrinkage helps, while exact paper DLS architecture is not fully reproduced. | Use reliability-DLS and no-DLS as formal ablations. | Rebuild DLS around a stronger structural embedding and exact dynamic-neighborhood update rule. |
| Observational entropy / value of information | partial | Single-entry log-det, network-observational, first-order expected-network, stochastic subset, reliability-calibrated stochastic, probability-weighted expected network log-det, reward-frontier gated entropy, and network-batch approximations are implemented, but not the exact paper estimator. | Clearly label current entropy as proxy; do not promote expected, stochastic, weighted expected, or reward-frontier variants because current 3-seed pilots lose or match current netobs. | Stop local entropy-gate sweeps until structural embedding or the paper estimator itself is improved. |
| ELPH / higher-order graph representation | proxy only | Spectral, dependency-free Node2Vec/ELPH-style proxies, and a local torch SGNS Node2Vec path exist; true ELPH or validated third-party Node2Vec is still not integrated. | Keep spectral as frozen best candidate, report proxy status, and do not promote SGNS Node2Vec to 20 seeds. | Add a validated sparse Node2Vec package or external ELPH implementation only if dependencies are explicitly allowed, then rerun Douban embedding matrix. |
| 16 benchmark comparison | partial | Implemented baselines now cover the shared cross-dataset families, including the MovieLens graph_social/Thompson extension. The paper-specific audit recovers 14 explicit slots under the paper's 16-benchmark claim; all 14 explicit slots are now implemented or proxy-covered after adding OFUL, logistic, and finite-arm rows. Local source/footnote review plus public footnote-code comparison still cannot confirm the two paper-count slots. contextualbandits exposes plausible extra methods such as active-learning exploration and SoftMax chooser, but the paper text does not name them, so the benchmark claim is frozen as 14 covered/proxy-covered explicit slots plus 2 unresolved slots. Raw OFUL is weak; MF-centered OFUL repairs much of the collapse but still remains below myopic, graph_social, and DU-Net. Logistic UCB/TS need probability-scale exploration and finite-arm proxies are competitive only as local stress rows. | Use paper_benchmark_definition_audit, paper_benchmark_slot_recovery_audit, author_bandit_source_comparison, and final_benchmark_claim_freeze as the tracking artifacts. | Run final synthesis or final 20-seed rerun with the frozen 14+2 benchmark caveat unless original PDF, supplement, or author implementation becomes available. |
| Ablation studies: DLS, reward, entropy, network components | mostly complete | MovieLens/Flixster/Douban ablations exist, but exact paper ablation definitions may differ. | Promote Flixster/Douban 20-seed ablations into final result chapter. | Repeat ablations after faithful entropy/embedding upgrades. |
| Heterogeneous response robustness | mostly complete | Main three-dataset evidence exists; final unified response grid with all baselines remains open. | Report current 20-seed response-aware candidates. | Rerun heterogeneous response for full benchmark matrix. |
| Mechanism diagnosis: cold-start, centrality, graph density, reward-information tradeoff | partial and growing | Douban and Flixster now have diagnostics; Douban has focused graph-density, network-batch, stochastic-entropy, reliability-calibration, weighted expected-network, and reward-frontier tradeoff evidence; MovieLens mechanism diagnostics are less complete. | Use diagnostics to explain why Douban is hard and why Flixster network proxy is weak. | Run graph-density sensitivity and cold/warm diagnostics for final candidates across all datasets. |

## Basic Next Plan

1. Freeze the current 20-seed summary files and preserve them as the baseline snapshot.
2. Keep `reports/three_dataset_reproduction_summary.md` synchronized with the generated alignment matrix and remove stale 3-seed-only language.
3. Promote Flixster and Douban mechanism diagnostics into the result chapter.
4. Keep current Douban conclusions conservative: no-DLS is the strongest numerical sparse-graph fallback; netobs reliability-DLS is the most faithful positive candidate.

## Long-Term Plan

1. Upgrade the baseline matrix before claiming paper-level benchmark coverage.
2. Add a validated external Node2Vec/ELPH implementation only if dependencies are explicitly allowed later.
3. Rerun the full paper-quality matrix after the mechanism and benchmark definitions stabilize: main results, heterogeneous response, ablations, and baselines across MovieLens, Flixster, and Douban.

CSV artifact: `outputs\tables\paper_alignment_gap_matrix.csv`.
