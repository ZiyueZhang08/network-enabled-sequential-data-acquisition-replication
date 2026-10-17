# Author Bandit Source Comparison

This report closes queue item 18: "Recover or compare against exact author
bandit code / supplement / PDF labels."

## Search Scope

I checked the local workspace for original PDFs, supplement archives, HTML
exports, and author-code bundles. The workspace contains the local Markdown
paper extraction and current reproduction artifacts, but not the original PDF,
supplement, DU-Net author code, or figure-label source files.

I also checked public sources using the paper title, author names, author
emails, `DU-Net`, and `Data-Utility NETwork-amplified`. I did not find a public
DU-Net author repository, supplement, or original PDF with recoverable benchmark
labels. The McCombs faculty profile pages are public, but they did not expose a
paper artifact usable for benchmark recovery in this session.

## Footnote Code Sources

The local paper footnotes point to two external codebases:

- `https://github.com/david-cortes/contextualbandits`
- `https://github.com/kwangsungjun/lrbandit`

The `contextualbandits` README lists the same online/finite-arm algorithms that
the local paper text explicitly names:

- LinUCB.
- Linear Thompson Sampling.
- Logistic UCB and Thompson Sampling.
- Upper Confidence Bound.
- Thompson Sampling.
- Epsilon Greedy.
- Adaptive Greedy.
- Explore-Then-Exploit.

The same README also lists additional online methods under "Other", especially
active-learning exploration and SoftMax chooser. These are plausible
source-level candidates for the paper's 16-vs-14 count mismatch, but the paper
text does not name them as benchmark policies.

The `lrbandit` README identifies the repository as code for "Bilinear Bandits
with Low-rank Structure" and its replication command enumerates low-rank
variants such as `bltwostage-bm-sp_simple2`, `bltwostage-sp_simple2`,
`blonestage-sp_simple2`, and `bloful`. This confirms that the paper's low-rank
bandit / OFUL adaptation source is specialized and small-matrix-oriented, but it
does not recover the two missing DU-Net benchmark labels.

## Local Comparison

The local reproduction now covers the 14 explicitly named paper benchmark slots:

- Sampling: random, myopic, entropy.
- Low-rank bandit: local scalable ESTR proxy.
- Online linear bandits: LinUCB, OFUL, linear Thompson, Logistic UCB, Logistic
  Thompson.
- Finite-arm bandits: UCB, Thompson, Epsilon Greedy, Adaptive Greedy,
  Explore-Then-Exploit.

The current project already contains an active-learning proxy branch, but it is
not paper-confirmed as one of the two missing slots and it performed poorly in
the MovieLens 3-seed screen. The project does not have a separate benchmark row
for a contextualbandits-style SoftMax chooser; existing exponential sampling
temperature controls are acquisition-selection machinery rather than a confirmed
paper benchmark label.

## Decision

Do not rename the two unresolved slots to active learning / SoftMax from this
evidence alone. The public footnote source makes those names plausible, but the
paper text and available local artifacts do not confirm them.

Freeze the benchmark claim as:

- 14 explicitly named paper benchmark slots implemented or proxy-covered.
- 2 paper-count slots unresolved.
- Candidate explanation for the unresolved count: extra contextualbandits
  methods such as active-learning exploration and SoftMax chooser, pending
  confirmation from original PDF figure labels, supplement, or author code.

## Next

Proceed with final reporting or final rerun under the frozen benchmark caveat
unless an original PDF, supplement, or DU-Net author implementation becomes
available.
