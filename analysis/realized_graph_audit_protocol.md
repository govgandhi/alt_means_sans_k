# Issue 20: exact targets and bounded diagnostics

Design recorded before running the new matrix/window diagnostics. This is a
retrospective audit of the manuscript and existing code, not a prospective
prediction study. The old KL slope and the allocation caveat are already known.
No embeddings will be trained and no density or learning-rate sweep will run.

## Questions and fixed checks

1. Verify the unrestricted ordered logistic target with positive center weights,
   conditional contexts R, and negative proposal q. Treat zero support explicitly;
   do not add pseudocounts or replace infinities with finite scores. Separately
   check the idealized target when positive-context collisions are skipped.
2. Check exact diagonal-inclusive DCSBM proxies with K=1 and K=3, eight nodes
   per block, propensities linearly spaced from 0.5 to 1.5 within each block,
   k=5, and alpha in {0.75, 1, 1.25}. Use B=(0.2) for K=1 and
   B=[[0.18,0.04,0.03],[0.04,0.16,0.05],[0.03,0.05,0.20]] for K=3.
   These are synthetic validation fixtures, not fitted or empirical parameters.
   Check normalized spectral radii, block context KL, score block structure,
   balanced center rows, and the context log-propensity coefficient. A check
   on population matrices is not a sample-graph convergence experiment.
3. Regenerate the existing five 600-node configuration graphs from seeds 0–4
   and attachment parameter 3, using the source generator unchanged. Recompute
   the stored one-step KL statistic from their adjacencies, then use uniform
   lag windows L in {1,2,5,10}. Hold each graph, uniform center exposure, and
   negative proposal pi fixed. Compute R_L=(T+...+T^L)/L exactly, without walks,
   integer pair allocation, optimization, or score truncation. Save all node
   degrees, KLs, positive marginals, and support counts. Also report KL against
   the changing positive marginal c R_L; this is a separate reference from pi.
   The figure shows per-graph KL slopes and means, not a trained norm curve.
4. Audit existing integer allocation at beta in {0,1}, lambda in {0,1}, across
   the same five graphs, using the original 240 events per node. Record intended
   versus realized marginal, support, and row KL. Preserve all historical files.
5. Verify explicit limits to interpretation: a two-node periodic window example;
   a connected aperiodic triangle-with-leaf example; adding an edge to a four-node
   star; and a reversible two-state chain whose balanced score norms reverse its
   KL ordering. These are mathematical counterexamples, not representative data.

## Interpretation

The three-level distinction is supported if population identities hold while
realized graphs retain degree-dependent target scores and KL. Weakening KL with
window size in these graphs supports a diffusion effect on targets only. A
nonmonotone window result or persistent degree slope is retained, not excluded.
No outcome licenses a claim that density or windows must remove a norm peak,
that either trained pipeline converged, or that the PNAS mechanism is resolved.

Changing center weights can alter the negative marginal and, at finite rank,
the weighted approximation objective. A later norm study must keep these effects
separate. Density needs its own model/regime: multiplying a population proxy by
a scalar leaves its transition matrix unchanged, and one-step sparse rows need
not approach dense proxy rows in KL merely because expected degree grows.

## Reproduction

Use Python 3.13.12 and an isolated environment:

```sh
python3 -m pip install -r analysis/realized_graph_audit_requirements.txt
python3 scripts/audit_realized_graph.py
python3 -m pytest tests/test_realized_graph_audit.py
```

The script defaults to `results/realized_graph_audit/` and accepts `--output`.
It records configuration, dependency versions, source hashes, graph hashes, and
node-level results. A repeated run should reproduce its JSON files byte for byte.
The existing graph generator and allocation functions are inputs, not edited.

Storage note added after the first run: the two large node-level JSON files
are gzip-compressed with a fixed timestamp. This changes storage only, not
the diagnostic design, values, or scope.

Figure correction added during PDF review: export a copy of the tracked
historical figure with three descriptive labels replacing causal wording.
Verify that all plotted vector paths and other text are unchanged. The original
figure and numeric results are retained; only the generator's descriptions and
plot labels change.
