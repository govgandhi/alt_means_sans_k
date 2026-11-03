# When do norms admit a structural interpretation?

SGNS can fit its learning objective very well without producing settled norm
geometry. Context structure affects the declining norm–degree relationship, but
a good loss alone does not make that relationship structurally interpretable.
The reported norms are balanced, so this limitation goes beyond factorization
ambiguity. This closes the bounded analysis of #20; the full PNAS inverted-U
mechanism is an explicit future question, not a requirement for closing this issue.

The requested comparison is now run: the same five graphs, equal center exposure,
stationary negative proposal, five negatives, rank 24 and two matched starts;
only the exact context window changes. This continues draft PR #38. PR #39 is
closed and consolidated into it. No further experiment is added to this analysis.

**Wider contexts attenuate the absolute norm decline from window 5 to 10, but
the original monotone window 1–10 result does not survive better optimization.**
The first 500-iteration endpoints remain archived as optimization diagnostics.

## Empirical result: window five versus window ten

The gap below is the highest degree-bin mean minus the preceding bin mean.
Negative values mean a decline. Starts are averaged within graph before computing
the mean and graph-level SE; they are not ten independent graph replicates.

| Window | Context KL gap ± SE | Balanced norm gap ± SE | Stable starts |
|---|---:|---:|---:|
| 5 | −0.2980 ± 0.0142 | −0.3258 ± 0.0155 | 7/10 |
| 10 | −0.1303 ± 0.0063 | −0.2173 ± 0.0123 | 10/10 |

From window 5 to 10, the upper-degree norm decline weakens in every graph/start.
The lowest-to-highest bin difference also shrinks in magnitude, from −0.7227 to
−0.4837. This is not solely a change in overall norm scale: the upper decline
relative to the preceding bin falls from 12.31% to 10.03%, and the lowest-to-highest
relative decline falls from 23.74% to 19.88%. These ratios are formed within graph
after averaging starts. The two starts closely agree, although three window-five
fits miss the strict two-consecutive-checkpoint stopping requirement. Numerical
stability is not a certificate of global optimality.

Context KL's **absolute** degree contrast weakens too, supporting a contribution
from context mixing. Its relative upper decline instead rises from 28.56% to
31.93%, because its overall level falls. Thus the two profiles do not follow a
common scale-free law. Window ten has positive context probability for every
pair and still has declining norms: exact zero-support sparsity is not necessary.
Window five retains 318, 608, 282, 426 and 414 zero entries in the five 600×600
matrices. Widening changes both support and the distribution of positive weights;
this experiment does not isolate those two changes.

The short windows remain optimization diagnostics. Their mean upper gaps are
+0.0915 ± 0.1050 at window one and −0.9492 ± 0.0562 at window two, with no stable
starts in either case. Window one's uppermost-bin gap becomes
positive in six of ten starts, but eight still have a negative lowest-to-highest
gap. This is an upper-bin rebound, not a reversal of the whole relation. All
window-one and window-two fits retain norm drift and initialization disagreement.
The [all-node plot](../results/window_training_continuation/report/all_final_nodes.png)
shows their large low-degree outliers; mean curves alone conceal that dispersion.

## Mathematical boundary and near-optimal fits

Proposition 11 in [the manuscript](../paper/current/main.tex) gives the boundary
for a finite unrestricted score matrix with fixed nonnegative positive-pair
weights and strictly positive negative-pair weights. If all positive-pair weights
are positive too, strict convexity and coercivity give a finite unique optimum;
loss convergence then implies score and balanced-norm convergence. If any positive
weight is zero, an unrestricted minimizing sequence instead sends those scores
to minus infinity. Its total squared balanced center norm equals the nuclear norm
of the score matrix and diverges. This need not happen to every individual node.
The result is a direct calculation, not a literature novelty claim or a guarantee
about the rank-24 optimizer. Positive weights guarantee existence under rank at
most 24, but do not guarantee uniqueness or algorithmic convergence.

For window one, second starts on graphs 3 and 4 have losses only 1.303×10⁻⁶ and
7.327×10⁻⁷ above the unrestricted loss infimum. Since the unrestricted infimum is
at most the rank-24 infimum, these also bound their objective suboptimality above.
Yet their final highest-bin means change by 5.01% and 1.13% over the last block.
Near-optimal objective value does not identify a settled radial geometry here.
Zero-positive pairs permit unbounded negative score tails; the observed finite
trajectory does not prove divergence or non-attainment at rank 24.

Population degree cancellation concerns the idealized finite score matrix when
all block-window probabilities are positive. Realized contexts change that target;
rank constraints, factorization and optimization determine the resulting norms.
Balancing fixes factorization gauge but does not itself settle the score geometry.
An explanation of the inverted-U must account for these additional stages. These
equal-exposure synthetic fits test its declining component, not the full empirical
PNAS pipeline.
The separate training-history result remains as reported in the
[earlier findings](window_training_exploration_findings.md).

## Why the continuation is credible, and its limits

The [amendment](window_optimization_amendment.md) fixed all 40 continuations and
ten conditioning controls before execution. Score-preserving factor balancing
changes numerical coordinates, not the scores, objective or canonical norms at
restart. At matched two-block budgets, balancing subsequently improves both loss
and canonical gradient in all ten window-ten controls. Actual iterations are 500
in each arm. The full continuation retains every start, outcome and stop reason.

Independent checks reconstruct targets and final losses, gradients and norms
using full score-matrix SVDs, and audit stopping gates and paired controls. A fresh
seed-zero repeat reproduces all ten conditions and 30 scientific files exactly.
Only final factors are saved; intermediate matrix displacements receive algebraic
and recorded-scalar checks, not independent reconstruction of every matrix.

The current evidence supports context-dependent attenuation and training-dependent
norm geometry. It does not establish a universal monotone window law, a KL-only
mechanism, converged short-window norms, or an explanation of the empirical U shape.
Issue #20's conceptual objection is addressed by the three-level distinction,
the positive/zero-support proposition, and the controlled window-five/ten result.
PR #38 holds the consolidated changes for review; #21/#22 retain their own scope.
Further PNAS mechanism work is deferred, and no additional experiment is required
to close this bounded investigation.

Evidence: [full comparison](../results/window_training_continuation/report/kl_and_trained_norms.png),
[wide windows](../results/window_training_continuation/report/wide_window_comparison.png),
[controls](../results/window_training_continuation/report/conditioning_controls.png),
[all generated tables](../results/window_training_continuation/report/tables.md),
[reproduction instructions](window_comparison_reproduction.md).
