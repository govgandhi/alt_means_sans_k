# Realized-graph target audit

These are exact matrix and integer-allocation diagnostics for
[issue #20](https://github.com/govgandhi/alt_means_sans_k/issues/20).
No embeddings were trained. See the
[audit](../../analysis/issue20_realized_graph_audit.md) for derivations,
all five graph results, counterexamples, and remaining limitations.

## Reproduce

From the repository root, using Python 3.13.12:

    python3 -m venv .venv-issue20
    .venv-issue20/bin/python3 -m pip install -r analysis/realized_graph_audit_requirements.txt
    .venv-issue20/bin/python3 scripts/audit_realized_graph.py
    .venv-issue20/bin/python3 -m pytest tests/test_realized_graph_audit.py

The complete dependency set is pinned. The script also accepts
--output results/my_audit; changing that path changes the configuration
record in provenance. No submodule or data download is required.

## Contents

- checks.json: symbolic, K=1/K=3 population, and counterexample checks;
  independent reproduction of the historical realized-graph KL slope.
- graphs.json: all degrees, graph hashes, edge counts, target support,
  and adjacency-based identity checks.
- windows.json.gz: every node's KL, context marginal, and support for four
  windows on each of five graphs; graphwise means and slopes.
- allocations.json.gz: every node's intended and realized KL/marginal/support
  under the original beta=0/1, lambda=0/1 controls.
- window_kl.png: graphwise context-KL trends at fixed stationary noise.
- turnover_surface_qualified.pdf: the historical figure with three text labels
  corrected; all plotted vector paths and other text are verified unchanged.
- provenance.json: configuration, runtime, and source/artifact SHA-256 hashes.

The two large files use deterministic gzip compression to keep generated
data out of the text diff. Read either with Python's gzip.open and json.load;
no additional package is needed.

The historical pooled slope is reproduced at -0.9941285280913252.
At window 10 the graphwise slopes lie between -0.248835 and -0.223626.
Population identities pass within 5.69e-14. These results support the
population/realized distinction, not a causal explanation of trained norms.
The allocation audit shows that the existing mixing intervention can change
noise and support; historical training outputs have not been replaced.
