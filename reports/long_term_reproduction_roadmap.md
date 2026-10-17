# Long-Term Reproduction Roadmap

This roadmap starts after the confirmed Douban mechanism diagnostics.

## Current Base

The current confirmed Douban interpretation is:

- `du_net_no_dls` is the best 20-seed Douban follow-up.
- `reliability_dls` nearly ties no-DLS and reduces centrality bias.
- weighted log-det entropy increases information-oriented selection but does not yet improve final w-RMSE.
- `network_observational` entropy with spectral embedding and reliability-DLS now modestly beats myopic in a narrow 20-seed confirmation.

## Next Implementation Package

1. Batch composition instead of scalar z-score mixing:
   - select a fixed share of each batch by reward, information, and diversity;
   - start with 70% reward / 30% information;
   - compare against scalar DU-Net, no-DLS, and reliability-DLS.
   - status: implemented as reward-frontier batch composition; the first Douban 3-seed pilot favored reliability-DLS 70/30, but the 20-seed confirmation did not beat myopic.

2. Sparse Node2Vec / ELPH embedding:
   - replace dense random-walk proxy with sparse co-occurrence accumulation;
   - add structural ELPH-style features: degree profile, common-neighbor profile, two-hop overlap, clustering;
   - rerun Flixster and Douban DLS ablations.
   - status: first dependency-light `sparse_node2vec` and `elph_struct` implementations are complete; neither has yet beaten spectral embedding on Douban.

3. Graph-density sensitivity:
   - edge dropout;
   - random rewiring;
   - densification by two-hop closure;
   - compare no-DLS, reliability-DLS, and full DLS.
   - status: implemented for Douban 3 seeds; rewiring hurts DLS, while reliability shrinkage stabilizes the original sparse graph.

4. Paper-quality rerun:
   - MovieLens 20 seeds;
   - Flixster 20 seeds;
   - Douban 20 seeds;
   - heterogeneous response 20 seeds;
   - ablation matrix 20 seeds;
   - baseline matrix 20 seeds;
   - mechanism diagnostics.

## Success Criteria

- Douban report explains not only whether DU-Net wins, but why DLS succeeds/fails under sparse graph structure.
- Entropy/information diagnostics should show selection behavior that is less myopic without sacrificing final w-RMSE.
- Final result chapter should clearly separate confirmed results, negative confirmations, and exploratory pilots.

## Immediate Paper-Quality Candidate

The `reliability_dls_frontier_batch_70r_30i` candidate has now been tested at 20 seeds in `configs/douban_svdpp_batch_composition_20seeds.json`.

Result: it increases entropy-oriented selection, but does not beat myopic on final w-RMSE.

The batch-composition candidate track is therefore closed for now. The active Douban candidate track has moved to network-observational entropy plus reliability-DLS.

## Network Entropy Update

The first `network_observational` pass is complete in `configs/douban_svdpp_network_entropy_embedding_3seeds.json`.

Result: top-k network-observational variants beat myopic in 3 seeds, and the spectral reliability-DLS version now also beats myopic in a narrow 20-seed confirmation.

Next immediate engineering step:

1. treat `netobs_reliability_spectral` as the new faithful Douban candidate, while noting that no-DLS remains the numerical w-RMSE winner;
2. improve sparse Node2Vec / ELPH quality before promoting them;
3. run Douban heterogeneous response and DLS/network ablations at 20 seeds for the confirmed candidates;
4. then rerun the final paper-quality Douban matrix with stronger baselines.

Completed after this note:

- A formal 20-seed mechanism matrix now exists at `outputs/tables/douban_svdpp_formal_mechanism_20seeds_final_summary.csv`.
- The full unified rerun exceeded local timeout, so the table is synthesized from matching completed 20-seed runs. This should be acceptable for reporting if the synthesis note is kept, but a single-pass rerun remains preferable for the final archive.
- Douban heterogeneous response, DLS/network ablation, and baseline matrix now include netobs reliability-DLS at 20 seeds. Netobs wins the baseline matrix; the first heterogeneous check favored reliability-DLS, but this is superseded by response-aware netobs.
- Dependency-free `sparse_node2vec_biased` and `elph_role` embeddings are implemented and tested. In the 3-seed Douban upgrade check they improve over the older proxies but still do not beat spectral embedding, so they are not promoted to 20 seeds.
- Response-aware netobs is implemented and confirmed at 20 seeds for Douban heterogeneous response. It supersedes reliability-DLS as the best alpha=0.02 candidate.
- `network_observational_expected` is implemented and tested as a first-order expected network log-det estimator. The 3-seed Douban pilot matches current netobs selections, so the next entropy step should be stochastic subset/log-det or batched network-observation entropy rather than promoting this identical variant to 20 seeds.
- `network_batch_composition` is implemented and tested. It raises Douban entropy percentile and lowers selected-user degree, but loses w-RMSE relative to netobs top-k. Keep it as mechanism evidence; the next gain must come from a stronger information signal or embedding, not more batch-penalty tuning.
- `network_observational_stochastic` is implemented and tested. At `p=0.01` it slightly loses to current netobs; at `p=0.05` it creates severe centrality bias. The next entropy step should calibrate neighbor inclusion by degree/reliability rather than increasing diffusion probability.
- `network_observational_stochastic_reliability` is implemented and tested. Reliability damping alone does not fix the `p=0.05` high-degree collapse, but adding degree normalization partly recovers w-RMSE. This closes the immediate stochastic-entropy branch: do not promote stochastic entropy to 20 seeds until a stronger structural embedding or network prior is available.
