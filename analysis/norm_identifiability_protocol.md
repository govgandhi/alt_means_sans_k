# Scope of the PR38 norm-profile addition

This is a retrospective mathematical and read-only analysis of the existing
fixed-noise fits, authorized on 2026-11-02. It is not a new training experiment
or a preregistration of unseen outcomes. The exploratory feasibility check used
one window-ten endpoint and the exact four-node construction; all conditions are
included in the completed analysis below.

Question: with the graph, sampling objective, dimension and balanced norm
convention fixed, how much does a small objective gap constrain norm--degree
shape? Separate what the objective permits from what the optimizer selects.

- Construct rising, falling and peaked profiles on the triangle with one pendant
  node, at illustrative loss tolerances `1e-2`, `1e-6`, `1e-10`. Keep uniform
  center exposure, stationary noise and five collision-allowing negatives fixed.
  Use four dimensions and the same RMS across the three profiles at each
  tolerance. RMS may grow as the tolerance shrinks. Report actual loss gaps,
  full score matrices and SVD-based radii. No optimizer is involved.
- Reconstruct all 40 final balanced fits: five graphs, windows 1/2/5/10, two
  starts. The ten raw-coordinate conditioning controls answer a different
  question and remain in their original report. Verify frozen inputs, actual
  final losses and full-SVD canonical norms. Preserve every node and condition.
- Normalize each fit's radii by its own node-level RMS before averaging starts
  within a graph. Use the original common degree-bin boundaries. Report both
  scales, paired-start differences, exact unrestricted loss gaps, and recorded
  last-block changes. Intermediate factors are unavailable; checkpoint changes
  remain recorded diagnostics, not independently reconstructed certificates.
- Compare window-five and window-ten normalized upper-bin contrasts in every
  graph/start. Across-graph uncertainty uses five graphs, not ten independent
  starts. Between-start agreement is an empirical check, not an uncertainty
  bound over every near-optimal solution.
- For full-support conditions only, compute the exact unrestricted target and a
  conservative loss-to-radius bound. Put the derivation and condition-level
  values in supporting material. A bound no better than the trivial normalized
  profile bound is reported as uninformative, not as evidence of instability.

An exact counterexample establishes nonidentification for its stated objective.
Disagreement in the larger fits establishes ambiguity at a stated accuracy only
if both fits have sufficiently small certified objective gaps at that accuracy;
otherwise it remains optimization uncertainty. Agreement does not prove unique
rank-constrained geometry. Full-support uniqueness concerns unrestricted scores,
not arbitrary rank-constrained optima. None of these additions identifies the
mechanism of the empirical PNAS inverted-U.

Stop after the proof, exact example, existing-fit audit, composite figure and
concise manuscript update. Do not add smoothing or training sweeps, or open
another PR. Preserve all earlier frozen sources and results.
