# Historical Gensim threshold sweep

These files predate the PR 35 corrections: repeated-visit token counting,
integer-token negative-table assignment and the checked BLAS build. The stored
`corpus_frequency`, `keep_probability` and expected retained-center values used
walk-visit counts (issue 44). Gensim's internal subsampling used its own token counts.

The files remain unchanged. The rerun with token counts and the corrected build,
with a side-by-side comparison, is in
[`../issue44_token_counts/`](../issue44_token_counts/README.md).
