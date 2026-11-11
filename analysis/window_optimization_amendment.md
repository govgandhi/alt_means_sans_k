# Complete the fixed-noise window comparison

This amendment continues the same investigation in draft PR #38. The initial
40 endpoints are retained as numerical diagnostics, not treated as completed
scientific evidence. The user's requested comparison is: same graph, exposure
and negative proposal; wider contexts; trained norms compared alongside context
KL. No new PR is created for this continuation.

## Diagnosis and intervention fixed before continuation

The original factorized objective and gradient pass independent checks, but the
one-sided initialization followed by unconstrained L-BFGS produces severe factor
scale imbalance. Saved endpoints also contain badly fitted extreme scores. We
must improve the actual loss/score fit, not merely rescale the displayed norms.

For U=Q_u R_u, V=Q_v R_v and R_u R_v^T=A S B^T, replace the factors by
U_b=Q_u A sqrt(S), V_b=Q_v B sqrt(S). Retain all 24 directions. This preserves
X=UV^T, loss and canonical balanced norms, and equalizes the factor Grams. Check
those identities at every restart. Start each L-BFGS block with fresh memory.

Keep the original five graphs, uniform exposure c=1/600, fixed q=pi, five
collision-allowing negatives, rank 24, exact windows 1/2/5/10, and both stored
initializations. Start from all 40 saved 500-iteration endpoints. No regularizer,
pseudocount, support clipping, new seed or target is introduced. This is a
retrospective optimization amendment, not a prospective prediction test.

## Fixed continuation and controls

Run up to twelve fresh-memory blocks of 250 L-BFGS iterations per endpoint.
Balance before every block. Per-block settings are maxfun 5000, maxls 40,
maxcor 10, gtol 1e-8, ftol 1e-12, float64 and one BLAS thread; two processes may run
independent graphs. Record actual iteration/function counts and all stop reasons.

For all ten window-ten endpoints, also run two 250-iteration blocks with fresh
memory but no balancing. Their matched balanced controls are the first two
blocks of the main continuation. Both controls reset memory identically; only
the factor coordinates differ. Budget caps match, not necessarily actual work.
The original endpoints remain the common control baseline.

After at least two blocks, stop a balanced continuation only after two consecutive
checkpoints satisfy all of: balanced factor-gradient infinity norm<=1e-7;
relative loss change<=1e-8; relative score RMS displacement<=1e-3; relative
balanced-node-norm RMS displacement<=1e-3; every shared degree-bin mean changes
by<=1e-3 relative to max(1,abs(previous mean)); and absolute upper-bin gap change
<=1e-3. These numerical choices target stability finer than reported contrasts;
they are not empirical constants or guarantees of global optimality. Otherwise
retain the fixed twelve-block endpoint and flag which gates failed. Do not select
starts by their result or extend the design according to a desired curve.

## Readouts and checks

Save every block's node norms/losses, context KL, degrees, scalar loss, canonical
factor gradients, score-space projected tangent gradient, numerical rank,
score singular values, positive/nonedge score quantiles, absolute and relative
score/norm displacements, factor scale imbalance, and degree-bin means. The
projected residual uses the current numerical-rank tangent space and is not a
minimum certificate. Report actual effort as well as the block index.

The primary figure places context KL and trained balanced norms against degree
side by side for every window, with paired upper-degree drops across all five
graphs. Use the original pooled degree-only quantile cuts. Average starts within
graph before graph-level SE; show all starts and their disagreement separately.
Show changes across the last blocks, not only the final graph-average curves.

With positive support everywhere a finite rank-constrained score minimizer exists,
but this nonconvex solver need not find the global one. With zero positives,
score-space coercivity fails; unrestricted infinite targets do not establish
non-attainment at rank 24. Retain score tails/norm drift and say when an equilibrium
interpretation remains unavailable. Finite-budget window sensitivity may still
strengthen the sparsity explanation; it cannot show that scalar KL uniquely
controls norms or explain the PNAS empirical curve on its own.

Unit checks independently verify balancing, objective preservation, tangent
projection and stopping rules. Recompute final losses/gradients, full-score SVD
norms and paired input hashes independently; repeat seed 0 in a fresh output.
Save complete outputs and deterministic source/input/runtime manifests. Both
controls and the full 40-condition continuation must complete before reporting.
