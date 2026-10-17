# Embedding Fidelity Feasibility Note

This note records the current state of the graph-embedding fidelity gap after
the local SGNS Node2Vec feasibility check.

## Dependency Audit

| package | available | role |
|---|---:|---|
| `node2vec` | False | third-party Node2Vec package |
| `pecanpy` | False | fast sparse Node2Vec implementation |
| `gensim` | False | Word2Vec/SGNS backend often used by Node2Vec |
| `networkx` | True | graph utilities; not a Node2Vec trainer by itself |
| `karateclub` | False | graph embedding package with Node2Vec-style models |
| `torch` | True | local SGNS trainer backend |
| `torch_geometric` | False | PyG backend for ELPH/GraphSAGE-style models |
| `dgl` | False | DGL backend for graph neural models |
| `ogb` | False | Open Graph Benchmark utilities often used by ELPH code |
| `sklearn` | True | classical ML utilities |

## Current Local Implementations

`src/graph_embedding.py` now includes both sparse PPMI/SVD Node2Vec-style
proxies and a local `torch` SGNS trainer:

| method | reward | w-RMSE | entropy pct | selected degree | status |
|---|---:|---:|---:|---:|---|
| `spectral` | 2996.6666666666665 | 0.6625678339588474 | 0.5466543854315724 | 1.2251666666666667 | confirmed strongest Douban candidate |
| `sparse_node2vec_bfs` | 2983.333333333333 | 0.6633105063650949 | 0.5717085494975264 | 1.416 | close but below spectral |
| `elph_role` | 2987.0 | 0.663343426244691 | 0.5730784261108094 | 1.3426666666666665 | competitive but below spectral |
| `sparse_node2vec_sgns_bfs` | 2978.0 | 0.6642913414121945 | 0.5622767716320898 | 1.2701666666666669 | new fidelity check |
| `sparse_node2vec_sgns_dfs` | 2949.333333333333 | 0.6675941842740992 | 0.5948908581508741 | 1.267666666666667 | new negative check |

## Judgment

The environment does not currently provide a validated third-party Node2Vec or
ELPH stack (`node2vec`, `pecanpy`, `gensim`, `torch_geometric`, `dgl`, and `ogb`
are unavailable). `torch` is available, so the reproduction now includes a
dependency-light SGNS Node2Vec path. This closes the most realistic local
implementation branch without adding dependencies.

Empirically, SGNS-BFS is positive versus myopic but still weaker than the current
spectral and ELPH-role candidates on Douban. Therefore it should not be promoted
to 20 seeds.

## Next Decision

Stop local embedding-only sweeps unless a validated external implementation is
explicitly allowed later. The next executable reproduction work should move to
paper-specific benchmark definitions, while keeping `netobs_reliability_spectral`
as the current Douban faithful candidate.

CSV artifact: `outputs/tables/embedding_fidelity_audit.csv`.
