# Post hoc predictor comparisons (issue 40)

This analysis answers the three questions in issue #40. It is a retrospective
comparison using existing calibration and target records, not a new prospective
test or new embedding training. The historical protocol and frozen predictions
remain unchanged, including the original four pass/fail criteria.

## Design and reproduction

Run from a clean checkout using the pinned environment documented in
`analysis/negative_sampling_control_protocol.md`:

```sh
python3 scripts/prospective_baselines.py
python3 -m pytest -q tests/test_prospective_baselines.py
```

`--output PATH` writes a separate reproduction. The default destination is
`results/prospective_baselines/`; source directories cannot be overwritten.
The script verifies dependency versions, the original freeze manifest and
outcome hash, and every corrected completion-manifest entry before fitting.

Original inputs are `results/prospective_validation/{calibration,outcomes,
frozen_predictions}.jsonl`. Corrected inputs are the corresponding files in
`results/negative_sampling_revalidation/`, separately for retained-count and
raw-count negative sampling. Corrected evaluations were already retrospective.

All regressors use the original frozen ExtraTrees parameters and random seed.
Only calibration outcomes enter fitting. Compare log pair count plus context
KL with log degree plus log threshold, and the pair-only, KL-only and degree-only
ablations. No settings or features are selected based on these target scores.
The original predictor is evaluated from its saved predictions; a refit checks
exact reproducibility. Raw center norm remains primary; balanced norm is secondary.

The nine within-curve Spearman correlations use individual nodes within each
family/threshold group, pooling five replicates. Report their median and range,
not a confidence interval. Also retain all 45 separate replicate correlations
for both responses, alongside all curve peaks, errors and correlations. These
groups are not independent: thresholds share graphs and NetScience's replicates
share one observed network. No node-level significance test is justified here.

`summary.json` retains all comparisons, source hashes and package versions.
The same run generates `README.md` with all ablations and `table.tex` plus
`balanced_table.tex` for the supplement, preventing hand-transcribed table values.
`paper/current/supp.tex` gives the methods, comparisons and interpretation;
the main text cites its sections and tables through automatic cross-document
references. `make main.pdf` in `paper/current` builds both documents in order.
Rerunning with identical inputs produces identical outputs.

## Answers to the issue

1. **Add the post hoc baseline and within-curve comparison.** The two-input
   predictor improves node-norm ranking relative to this degree/threshold
   comparator. Both pass the aggregate criteria, and the baseline localizes
   more peaks in the corrected retained-noise comparison. The result does not
   prove a causal mechanism or information statistically independent of degree.
   Degree alone, unlike degree plus threshold, fails the pooled-correlation
   criterion. In the corrected raw-noise arm, pairs alone also pass all four.
2. **Separate criterion four in interpretation.** It evaluates observed peak
   ordering, permits ties, and therefore has the same result for every predictor.
   Keep the frozen record intact; distinguish three predictive criteria from
   this response property in the manuscript. The pooled correlation remains a
   valid declared metric, but mixes within-curve and between-condition variation.
3. **Prespecify comparators in the next genuinely new test.** Before inspecting
   natural-text or mobility outcomes, freeze a frequency/degree baseline and
   include threshold when it varies, along with the pair/KL ablations. Specify
   within-target ranking metrics, peak errors and independent-replicate
   uncertainty, and the rule for comparing predictors. Do not choose new pass
   cutoffs retrospectively from these targets. This is a future design requirement,
   not a claim that the next protocol has already been registered.

The manuscript addition is deliberately short. The full results include both
corrected noise arms, secondary responses and failures; original artifacts and
the user's personal draft are preserved.
