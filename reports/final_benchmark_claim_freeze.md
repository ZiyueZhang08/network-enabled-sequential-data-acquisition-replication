# Final Benchmark Claim Freeze

This note records the fallback decision after local and public-source recovery
could not confirm the two missing benchmark names.

## Frozen Claim

The benchmark suite should be reported as:

- Paper target count: 16 benchmarks.
- Explicitly named benchmark slots recovered from local paper Section 4.2.1: 14.
- Explicit slots implemented or proxy-covered locally: 14.
- Paper-count slots still unresolved: 2.

This is the correct final wording unless a new source is supplied. It avoids
over-claiming exact paper benchmark reproduction while preserving the completed
coverage work.

## Why Not Fill the Two Slots Locally

The local paper text and footnotes name 14 policies. Public footnote source
inspection found that `contextualbandits` includes additional methods such as
active-learning exploration and SoftMax chooser, but the paper does not name
them in Section 4.2.1. The `lrbandit` source confirms a specialized low-rank
bandit / OFUL adaptation path, but it does not name extra DU-Net benchmarks.

Therefore, assigning names to the two unresolved slots would be inference, not
reproduction.

## Reporting Consequence

Final tables can include all implemented and proxy rows, but captions and
interpretation should say:

> We recover and cover the 14 benchmark policies explicitly named in the local
> paper text. The paper states a 16-benchmark comparison, but two count-implied
> benchmark labels remain unresolved without the original PDF figure labels,
> supplement, or author implementation.

## Next Execution Gate

The benchmark-definition branch is now stable enough for a final paper-quality
rerun or final synthesis, provided the rerun carries this caveat.
