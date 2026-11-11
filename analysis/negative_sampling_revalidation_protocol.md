# Corrected negative-sampling revalidation

Design recorded before any corrected target training. This is retrospective:
the original target outcomes are already known. It is not a new prospective test.
The original protocol, predictions and results remain immutable historical records.

## Questions and fixed design

1. Does the degree–norm pattern persist under fixed original-token negative noise
   after correcting repeated-visit counts, integer-token table assignment and BLAS?
2. Does the original two-feature prediction procedure still meet its four criteria?

Use every condition from `results/prospective_validation/protocol.json`: five
configuration calibration graphs, five direct BA graphs, five Holme–Kim graphs,
five walk/optimizer repetitions on the one NetScience component, and all three
thresholds. Graph construction, walks, random seeds, window, dimension, learning
rates, epochs, nominal negative count and exponent remain unchanged. NetScience
has 379 nodes; the synthetic graphs have 600. No parameters are chosen from the
corrected outcomes. This requires 30 calibration and 90 target model fits.

Count every token occurrence, including repeated visits within a walk, before
computing external retention probabilities. Generate one retained corpus per
condition. Clone the same untrained template for both arms:

- `retained`: negatives follow correctly assigned retained-token counts.
- `raw`: negatives follow correctly assigned original-token counts, fixed across
  thresholds in node coordinates up to cumulative-table rounding.

Only the negative table differs within a pair. Verify identical vocabulary,
initial arrays, optimizer RNG, learning schedule and positive corpus. Internal
subsampling remains disabled. The corrected BLAS build is mandatory for both.
Negative collisions with the positive output are skipped without resampling by
Gensim; equal nominal proposals do not mean equal accepted negative updates.

## Calibration and freeze

Fit a separate predictor for each arm on its corrected calibration outcomes only.
Use the original log positive-pair count and pair-marginal context KL, the original
ExtraTrees hyperparameters, and both original responses. The KL feature is not KL
against either arm's negative sampler. Report leave-one-calibration-seed-out scores
without modifying the model. Do not select the better arm after evaluation.

Write and hash all target features and both arms' predictions before any corrected
target embedding is trained. Record protocol, code, dependency versions, numerical
extension, source data, corpus and initialization hashes. Verify them again during
evaluation. Store all corrected artifacts in a separate output directory. Existing
completed runs are verified; partial runs resume only with the same plan and hashes.

## Readouts and decisions

Use the original six log-degree quantile bins (ties can reduce occupied bins).
Report each graph/repetition separately and aggregate families exactly as in the
original protocol for its nine family-by-threshold curves. Report raw norm as
primary, balanced norm as secondary; show every condition, including failures.
Record peak bin, endpoint differences, interior maximum, node-wise paired RMSE
and rank correlation. No equivalence or significance threshold is invented.
Plots show individual graph/repetition curves, not node-based standard errors.
NetScience repetitions quantify stochastic variation on one network.

Evaluate the original four primary criteria separately for both sampler arms:

- median absolute predicted peak error at most one occupied bin;
- at least two thirds of nine predicted peaks within one bin;
- pooled predicted/observed node Spearman correlation at least 0.5;
- observed peak degree nondecreasing with weaker thinning in at least two families.

Retain all per-family correlations and absolute errors so pooled success cannot
hide local failure. An interior maximum is a descriptive shape, not a statistical
test of unimodality. A shifted or lost peak limits robustness even if the predictive
criteria pass. Conversely, preserved patterns do not guarantee predictive transport.

Persistence licenses the claim that changing the noise distribution is unnecessary
for those specific patterns in these tested conditions. Passing prediction criteria
supports revalidation of this prediction procedure, not proof that its features
fully explain the mechanism. Changes relative to the historical pipeline combine
counting, mapping and arithmetic corrections and must not be attributed to one alone.
The historical calibration control separately isolates mapping and runtime effects.
The internal-subsampling threshold sweep and other saved experiments are not rerun
by this protocol; their old artifacts must be marked as requiring revalidation.

## Reproduction

Use Python 3.13.12 and the pinned runtime/build requirements from the calibration
control. Install into a virtual environment, then run:

```sh
python3 scripts/build_control_gensim.py --install
python3 scripts/check_control_gensim.py
python3 -m pytest
python3 scripts/revalidate_negative_sampling.py
```

The last command defaults to `results/negative_sampling_revalidation`. Use
`--output /path/to/fresh/output` for a fresh full rerun, or `--stage calibrate`,
`--stage predict`, `--stage evaluate` for separate stages. `--stage verify` checks
a completed result without training. Each stage records its inputs before use.
