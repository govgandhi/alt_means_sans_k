# Does an accurate objective determine norm--degree shape?

**Not in general, even after balancing and removing overall scale.** The exact
counterexample establishes this for a fixed four-node objective. The existing
larger fits separately support a reproducible window-five/ten shape difference;
they do not establish a unique rank-24 geometry.

## Exact result

On a triangle with one pendant node, fix one-step contexts, uniform exposure,
stationary negative sampling, five collision-allowing negatives and dimension
four. For **any positive loss tolerance**, the proof constructs rising, falling
and peaked balanced profiles within that tolerance of the same unrestricted
infimum. They have identical overall RMS at each tolerance and distinct exact
RMS-normalized shapes. Their RMS can grow as the tolerance shrinks; their losses
need not be exactly equal. This is what the objective permits, not what training
necessarily selects.

The reproducible calculation checks three prescribed profiles at each of
`1e-2`, `1e-6`, `1e-10`, retaining all nine score matrices, direct losses and
full-SVD radii. At tolerance `1e-6`, the three gaps are approximately `4.286e-7`.
The proof includes zero-positive diagonals and one common diagonal scale across
profiles; neither factor rescaling nor different overall RMS explains the result.
This is not a claim about the rank-24 optimizer on the larger graphs or a claim
of literature novelty.

## Existing fits after removing scale

All 40 final balanced fits and 24,000 node readouts are retained. Norms and losses
are reconstructed from saved factors after checking original source, graph,
target, runtime and factor hashes. Each fit is normalized by its own node RMS
**before** averaging starts within graph; SE uses five independent graphs.

| Window | Normalized upper-bin gap ± graph SE | Range of paired-start normalized node RMS differences | Stable fits |
|---|---:|---:|---:|
| 1 | +0.007446 ± 0.006926 | 0.984–1.187 | 0/10 |
| 2 | −0.064479 ± 0.005671 | 0.487–0.886 | 0/10 |
| 5 | −0.112947 ± 0.005360 | 0.00000244–0.004512 | 7/10 |
| 10 | −0.093384 ± 0.005299 | 0.00000193–0.0000554 | 10/10 |

Window five to ten weakens the normalized upper decline in all ten matched
comparisons. The normalized lowest-to-highest gap also weakens, from −0.250524 to
−0.207838. These are shape changes, not only uniform contraction. The upper-gap
change in every graph/start exceeds the between-start upper-gap discrepancy at
either window; this empirical comparison is not a bound over all possible fits.

Normalization reverses the window-two/five ordering seen in absolute upper gaps.
There is **no monotone all-window law** on either interpretation. Short-window
disagreement includes substantial low-degree outliers; the complete node plot
uses log scales and omits no nodes. Both objective gaps appear beside every
start comparison. The large short-window disagreements do not constitute two
certified tiny-gap solutions at the tolerances achieved by the best single fits.
Recorded intermediate drift remains diagnostic because intermediate factors
were not saved.

## Supplementary bound and what remains unresolved

Only window ten has a finite unrestricted target on these graphs. The general
bound permits RMS radius errors of **127.5–226.4**, against target RMS values of
**2.83–2.84**. All ten normalized bounds reduce to the trivial geometric bound
`sqrt(2)`. They certify no useful accuracy; they do not demonstrate instability.
Directly calculated normalized errors to the known unrestricted target are
0.0505–0.0537, despite much smaller differences between starts. A unique
unrestricted target does not establish uniqueness under rank 24, and its loss
gap cannot distinguish approximation error from incomplete optimization.

The meaningful addition is the exact shape counterexample plus the surviving
window-five/ten result after scale removal. Explaining the empirical PNAS
inverted-U remains a separate question. No new training or smoothing sweep was
performed, and this work remains in PR #38.

Evidence: [composite figure](../results/norm_identifiability/norm_identifiability.png),
[all nodes](../results/norm_identifiability/normalized_all_nodes.png),
[every condition and pair](../results/norm_identifiability/empirical/tables.md),
[exact constructions](../results/norm_identifiability/exact/exact.json),
[supporting proof](../paper/current/norm_geometry_supplement.tex),
[scope](norm_identifiability_protocol.md),
[reproduction](window_comparison_reproduction.md#normalized-profile-addition).
