# Historical norm-frequency simulation

The `population` field in `summary.json` is a legacy label for exact one-step
KL on the **realized simple graphs**, with stationary reference noise. It is
not the KL of the DCSBM population proxy. The pooled slope can be regenerated
without training by `scripts/audit_realized_graph.py`.

The historical files remain unchanged. Reproducing that graph statistic does
not revalidate their Gensim training outputs; implementation corrections and
training controls are in PRs #35 and #36. See
[`analysis/issue20_realized_graph_audit.md`](../../analysis/issue20_realized_graph_audit.md)
for the distinction and diagnostic scope.

Its `frequency` axis and gauge direction used walk-visit counts, not token
counts (issue 44). The rerun with token counts and the corrected build is in
[`../issue44_token_counts/`](../issue44_token_counts/README.md).
