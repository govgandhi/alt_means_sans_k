# Context windows and late training history

Design fixed before either new scientific sweep. This is a retrospective
mechanism exploration informed by PRs #35, #36, and #38, not a prospective
prediction test. The user explicitly chose both context windows and training
history. Existing PRs, manuscript claims, and historical results remain unchanged.

## Questions and distinct experimental lanes

1. Can changing the late learning rate alter a peak that has already formed,
   with the complete earlier training state held fixed?
2. With equal center exposure and a fixed negative proposal, how do wider
   contexts affect balanced norms in bounded finite-rank expected-objective fits?

These are complementary studies of different specified objectives. Neither is
a reproduction of the PNAS empirical curve. The Gensim lane has rank 64, finite
walk contexts, empirical raw-token noise, skipped positive collisions, and
approximate/limited-range sigmoid updates. The expected-objective lane has rank
24, exact graph contexts, stationary noise, collision-allowing negatives, and
an exact smooth logistic loss. Do not interpret differences between lanes as
effects of a single controlled variable.

## A. Fork an already-peaked Gensim model

Use all five original 600-node configuration graphs, seeds 0–4, and the
unfiltered setting from PR #36. Keep its corpus, vocabulary, initialization,
dimension 64, window one, five negatives, corrected raw-token negative table,
single worker, and checked patched Gensim runtime.

Recreate the original first five epochs (0.025 decaying to 0.0001) and verify
the exact epoch-five factor hashes against PR #36. Fork the full model state,
including RNG, into two fifteen-epoch continuations:

- floor: constant learning rate 0.0001;
- reheat: constant learning rate 0.0025.

These rates come from the two earlier schedules; they are not selected from
new results. Save absolute epochs 5, 6, 7, 10, 15, and 20. The floor arm must
reproduce PR #36's saved epochs 10 and 20 and its token counts. An independent
toy test checks split versus uninterrupted training and callback noninterference.

Verify corpus, factors, RNG, vocabulary, and negative-table identity at the
fork, and matching effective/raw tokens across arms. Record cumulative
post-fork learning-rate times effective-token exposure: reheat has 25 times
the rate at equal tokens, so it is not matched optimization effort.

Readouts: node-level raw and canonical balanced norms; expected logistic loss
on the original corpus; score changes; idealized gradient residuals; positive
and negative mass outside Gensim's update range; epoch counts and pairing hashes.
The existing exact expected-pair evaluator includes skipped positive collisions.
It is an in-sample diagnostic, not Gensim's exact implemented update objective.

Primary comparison: within each graph, the change in the entire norm-degree
curve after reheating versus floor continuation, particularly the upper-degree
bins. Report every graph and both readouts. A changed/disappeared inherited peak
supports a causal effect of the late rate in this implementation. Persistence
does not establish a structural optimum. Neither result identifies a unique
freezing, clipping, or basin-selection mechanism.

## B. Exact expected-objective window fits

Use the same five graph seeds and generator, without sampling walks or allocating
integer pairs. Hold c_i=1/600, q_j=d_j/sum(d), k=5, and rank 24 fixed. Rank 24
matches the manuscript's original controlled grid. Use uniform lag windows
L in {1,2,5,10}, P_L=(T+...+T^L)/L, as in PR #38's diagnostic.

Minimize the collision-allowing expected logistic loss

    sum_ij c_i [P_L[i,j] log(1+exp(-X_ij)) + k q_j log(1+exp(X_ij))],
    X = U V^T.

Use float64, one BLAS thread, and two independently specified initializations
per graph (seed offsets 100000 and 200000). U is uniform on [-0.5/rank,0.5/rank];
V starts at zero, matching the original controlled-trainer convention. Reuse
each initialization across all windows. Never rebalance the zero score product
into U=V=0; that would create an artificial stationary point.

Run all 40 fits with SciPy L-BFGS-B: maxiter=500, maxfun=5000, maxls=40,
maxcor=10, gtol=1e-8, ftol=1e-12. These are numerical design choices, not
empirical parameters or convergence declarations. A timing-only objective/
gradient evaluation was used to bound computation before this protocol; no
scientific optimization outcome was inspected.

Record iteration 0,1,2,5,10,20,50,100,200,500 when reached, plus the final
endpoint. Save actual iteration/evaluation counts, termination status, loss,
factor-gradient maximum/RMS, score changes, all node norms and row losses, and
final factors. L-BFGS iterations are not SGD epochs or physical learning times.
Do not alter fitting budgets or select initializations according to outcomes.

Compute the exact unrestricted pairwise loss infimum as a lower bound. The
difference from a finite-rank fit includes rank restriction and cannot be
called an optimization gap. Preserve zero support: no pseudocounts or score
clipping. Window-one nonedges have -infinite free scores, so there is no finite
free-score target norm. Where P_L has full support, calculate the exact
full-rank free-score matrix and its balanced SVD norms as a distinct benchmark.
A truncated SVD is not the rank-constrained logistic optimum.

Verify gradients independently by finite differences including zero positives;
check loss against direct pair enumeration and the unrestricted bound. Cross-check
balanced readouts by full score SVD on a small fixture, and verify any finite
free-score benchmark reaches its analytic lower bound.

Primary comparisons: all graph-specific balanced norm-degree curves by window,
endpoint agreement/disagreement across starts, changes over the fixed iteration
budgets, and the finite full-rank benchmark where defined. KL is a descriptor,
not an assumed monotone predictor of these norms. Report raw norms secondarily.

An attenuated falling branch supports window sensitivity under this stated
objective. A persistent, reversed, or initialization-dependent branch is retained
and limits the hypothesis. Small residuals and agreement across starts strengthen
a reference fit, but do not prove a global optimum or a universal target norm.

## Shared reporting and stopping

Use outcome-independent log-degree quantile bins as in the preceding work
(eight requested quantiles; ties can reduce occupied bins). Plot individual
graphs, equally weighted graph means, and graph-level standard errors. For the
window lane, average initializations within each graph before graph-level
uncertainty; initialization replicates are not extra graphs. Show fixed
highest-minus-lowest and highest-minus-penultimate gaps alongside descriptive
interior maxima. Do not turn selected peak gaps into significance tests.

Both complete sweeps, including ambiguous results and stopping failures, are
the stopping condition. Do not expand into a PNAS reproduction, density study,
filtering/capacity sweep, or new manuscript mechanism claim in this exploration.
The exact beta=1/lambda=0 controlled-trainer trajectory requested in #22 remains
distinct from these tests. Save all node outcomes, input/source/runtime hashes,
and completed-run markers; use deterministic serialization and verify resumes.

Return: reproducible code and complete results, one figure per lane, a concise
joint findings report stating supported claims and exclusions, and a next-test
decision. The manuscript and PRs #35/#36 stay unchanged.
