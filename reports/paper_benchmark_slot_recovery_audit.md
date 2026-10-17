# Paper Benchmark Slot Recovery Audit

This audit follows the queue item "Recover two unresolved paper benchmark
slots." It checks whether the two slots implied by the paper's 16-benchmark
claim can be recovered from the local full-text extraction and local artifacts.

## Local Source Evidence

The local paper text states in Section 4.2.1 that DU-Net is compared against 16
benchmarks, but the same section explicitly enumerates only 14 named benchmark
slots:

| family | explicitly named slots | count |
|---|---|---:|
| Sampling-based policies | Uniform random sampling, Myopic policy, Information-theoretic policy | 3 |
| Low-rank bandits | Explore Subspace Then Refine (ESTR) | 1 |
| Online linear bandits | LinUCB, OFUL, Linear Thompson Sampling, Logistic UCB, Logistic Thompson Sampling | 5 |
| Stochastic finite-arm bandits | Upper Confidence Bound, Thompson Sampling, Epsilon Greedy, Adaptive Greedy, Explore-Then-Exploit | 5 |
| Total explicitly named slots |  | 14 |

The relevant footnotes do not add additional benchmark names:

- Footnote 13 excludes the error-centric method by Boutilier et al. (2012).
- Footnotes 14 and 15 describe how ESTR was adapted.
- Footnote 16 points to the `contextualbandits` codebase for online linear
  bandits and unique-pull adaptation.
- Footnote 17 points to the low-rank-bandit codebase for OFUL adaptation.
- Footnote 18 states the finite-arm methods also remove selected user-item
  pairs from the acquisition set.

## Artifact Search

I also searched the local repository for likely hidden benchmark names such as
`Bootstrapped`, `Softmax`, `ExploreFirst`, `Partitioned`, `ActiveExplorer`, and
`contextualbandits`. No local paper/source artifact names either of the two
missing slots. The available local file set does not include the original PDF,
supplement, author code, or figure legends with the 16 benchmark labels.

## Decision

The two missing slots should remain unresolved rather than being inferred. The
current best interpretation is a count/name mismatch in the accessible local
paper extraction: the paper claims 16 benchmarks, but the locally available
Section 4.2.1 text and footnotes identify 14 explicit benchmark names.

This means the catalog coverage is now:

- 14 explicitly named slots implemented or proxy-covered.
- 2 paper-count slots unresolved.

The next useful execution branch is not another local baseline sweep. It is to
recover or compare against the author implementation, supplement, or original
PDF figure legends.
