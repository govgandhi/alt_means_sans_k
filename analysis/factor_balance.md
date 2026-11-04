# Factor balance and raw norms (issue 29)

## Answer and manuscript change

The proposed gradient-flow identity is correct for a differentiable loss that
depends only on `M = U V^T`, under simultaneous Euclidean flow with the same
rate for both factors. It preserves `D = U^T U - V^T V`; it does not attract an
unbalanced initialization to exact balance. If `D = 0`, then
`M M^T = (U U^T)^2`, so uniqueness of the positive-semidefinite square root gives
`U U^T = sqrt(M M^T)`, even for rank-deficient factors. Raw and canonical
balanced row norms then coincide.

This is established theory, not a new contribution of this paper. See Du, Hu
and Lee, *Algorithmic Regularization in Learning Deep Homogeneous Models:
Layers are Automatically Balanced*, NeurIPS 2018, Theorem 2.2 and its
matrix-factorization discussion:
https://papers.nips.cc/paper_files/paper/2018/file/fe131d7f5a6b38b23cc967316c13dae2-Paper.pdf
Their finite-step convergence results impose additional assumptions; they are
not invoked as a guarantee for SGNS.

For a simultaneous step, expansion gives

```
D_next - D = eta^2 (V^T G^T G V - U^T G G^T U),  G = gradient_M L.
```

This applies to a score-only sampled loss if both gradients use the pre-update
factors. It is not a formula for Gensim's sequential updates. Small random
`U` and zero `V` have relative Gram imbalance exactly one initially, even
though their absolute imbalance is small. Identical row norms do not imply
balanced factors, and balanced factors do not imply that scores have settled.

The main manuscript adds one short paragraph, crediting the existing identity
and qualifying the empirical agreement. The supplement gives the proof,
finite-step limitation, and measured diagnostic. We do not add a main-text
theorem claiming that the optimizer enforces balance.

## What was measured

`scripts/factor_balance.py` replays all twenty corrected native Gensim controls
from `results/subsampling_origin`: five graph seeds at four existing thresholds,
with the original five-epoch settings. No new condition or parameter is chosen.
Graph, walk, token-count, initialization and final factor hashes must match the
recorded runs before their measurements are accepted. Factors are measured
before any balancing transformation. The source results are unchanged.

The issue's requested diagnostic is `||D||_F / ||U^T U||_F`. The script also
reports the absolute imbalance, both Gram norms, `||r-r_bal||_2 / ||r||_2`, and
every signed node discrepancy `(r_i-r_bal_i)/r_i`, alongside both node norms.
All conditions and nodes are retained in `results/factor_balance/summary.json`;
the generated `table.tex` presents ranges across seeds, not confidence intervals.

Final relative Gram imbalance is 0.0786–0.163, while relative norm-vector
discrepancies are 0.55–1.15%. Individual errors reach 12.45%. Absolute imbalance
increases from 0.407–0.411 initially to about 225–910. Thus these finite-step
runs do not preserve the initial imbalance in absolute terms, despite close
aggregate norm agreement. We cannot claim that a tiny conserved initial
imbalance quantitatively explains the observed norm agreement.

This audit answers the requested diagnostic for the corrected native Gensim
controls. It does not certify the earlier uncorrected sweep cited in the issue,
the separate custom mini-batch pipeline, or subsequent learning schedules.
Their theoretical eligibility for the flow identity is distinct from their
empirical balance, which is not inferred here. Auditing already balanced
PR38 factors would not answer the native-balance question.

## Reproduce from a clean checkout

Build the pinned, patched runtime using
`analysis/negative_sampling_control_protocol.md` and
`analysis/negative_sampling_control_requirements.txt`, then run:

```sh
python3 scripts/factor_balance.py
python3 -m pytest -q tests/test_factor_balance.py
make -C paper/current main.pdf
```

`--output NEW_DIRECTORY` writes an independent replay; source-overlapping
destinations are rejected. Both artifacts are deterministic in the pinned
runtime. The script verifies the source configuration, dependency versions,
BLAS check, code hashes and completion manifest. It records its own hash and
the source manifests for provenance.

Tests check exact rank-deficient balance, zero-context initialization,
agreement with a direct full-score SVD, nonbalanced factors with identical
row norms, finite-step quadratic drift, and invalid denominators/output paths.
The flow and discrete identities were also checked entry by entry with SymPy
1.14.0 on symbolic rectangular factors; the dimension-independent derivation
is in the supplement. Reproduce the symbolic check with
`uv run scripts/verify_factor_balance_math.py` (its inline metadata pins SymPy),
or run it with `python3` in an environment containing SymPy 1.14.0.

Resolution: retain the identity with attribution and explicit assumptions;
describe raw/balanced norm agreement as measured rather than guaranteed.
No broader balance/convergence study is required for that bounded claim.
