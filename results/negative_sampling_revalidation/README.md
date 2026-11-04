# Corrected negative-sampling revalidation

Retrospective revalidation on the originally held-out targets. The old target results were already known; all corrected predictions were frozen before corrected target training. No target outcomes were used to fit or tune a model.

Verified 30 calibration model fits and 90 target fits: 9,000 calibration node-condition rows and 23,685 target rows, each containing both arms. All 15 calibration pairs and 45 target pairs are present, with unique keys and unchanged frozen features/predictions.

## Prediction criteria

The same two features, ExtraTrees settings and four primary criteria were used separately for each arm. The historical column describes the original bug-affected pipeline; differences from it combine multiple corrections.

| Criterion | Required | Historical | Corrected retained noise | Corrected fixed raw noise |
|---|---:|---:|---:|---:|
| Median absolute peak error | ≤1 bin | 0 | 0 | 0 |
| Peaks within one bin | ≥6/9 | 8/9 | 8/9 | 8/9 |
| Pooled node Spearman | ≥0.5 | 0.880555 | 0.874989 | 0.872642 |
| Families with nondecreasing peaks | ≥2/3 | 3 | 3 | 3 |

All four primary criteria passed: **retained: True**; **raw: True**.

## Sampler robustness

| Scope | Response | Same peak bin | Interior peaks: retained / raw | Interior peaks lost / gained under raw noise |
|---|---|---:|---:|---:|
| calibration | raw_norm | 15/15 | 10 / 10 | 0 / 0 |
| calibration | balanced_norm | 15/15 | 10 / 10 | 0 / 0 |
| targets | raw_norm | 39/45 | 35 / 35 | 0 / 0 |
| targets | balanced_norm | 39/45 | 33 / 31 | 2 / 0 |
| target_family_curves | raw_norm | 8/9 | 7 / 7 | 0 / 0 |
| target_family_curves | balanced_norm | 8/9 | 7 / 7 | 0 / 0 |

Individual runs with changed peaks (all exceptions):

| Family | Seed | Threshold | Response | Retained / raw peak | Interior: retained / raw |
|---|---:|---:|---|---:|---|
| Direct BA | 104 | 0.003 | balanced_norm | 3 / 4 | True / False |
| NetScience | 301 | 0.001 | balanced_norm | 4 / 5 | True / True |
| NetScience | 302 | 0.001 | raw_norm | 4 / 5 | True / True |
| NetScience | 302 | 0.001 | balanced_norm | 4 / 5 | True / True |
| NetScience | 302 | 0.003 | raw_norm | 4 / 5 | True / True |
| NetScience | 303 | 0.001 | raw_norm | 4 / 5 | True / True |
| NetScience | 303 | 0.001 | balanced_norm | 4 / 5 | True / True |
| NetScience | 303 | 0.003 | raw_norm | 4 / 5 | True / True |
| NetScience | 304 | 0.001 | raw_norm | 4 / 3 | True / True |
| NetScience | 304 | 0.001 | balanced_norm | 4 / 3 | True / True |
| NetScience | 305 | 0.001 | raw_norm | 3 / 5 | True / True |
| Clustered power law | 203 | 0.003 | balanced_norm | 3 / 4 | True / False |

Peak-bin equality is descriptive, not equivalence of the embeddings. Interior maxima are not significance tests of unimodality. Ties in degree quantiles reduce the number of occupied bins. NetScience's five runs repeat walks/optimization on one observed network.

## Every family and threshold

Peak indices below are one-based occupied-bin indices. Absolute norm errors and within-condition rank correlations expose failures hidden by pooled criteria. Full individual-run curves, endpoints, bin sizes and paired sampler differences are in summary.json.

| Family | Threshold | Noise | Observed / predicted peak | Interior pooled peak | Individual interior peaks | Node RMSE | Node Spearman |
|---|---:|---|---:|---|---:|---:|---:|
| Direct BA | 0.0003 | retained | 1 / 2 | False | 0/5 | 0.264221 | 0.796708 |
| Direct BA | 0.0003 | raw | 1 / 2 | False | 0/5 | 0.253505 | 0.790010 |
| Direct BA | 0.001 | retained | 3 / 3 | True | 5/5 | 0.278764 | 0.934091 |
| Direct BA | 0.001 | raw | 3 / 3 | True | 5/5 | 0.282339 | 0.933990 |
| Direct BA | 0.003 | retained | 3 / 3 | True | 5/5 | 0.237814 | 0.917418 |
| Direct BA | 0.003 | raw | 3 / 3 | True | 5/5 | 0.242118 | 0.916046 |
| Clustered power law | 0.0003 | retained | 2 / 2 | True | 1/5 | 0.309933 | 0.756808 |
| Clustered power law | 0.0003 | raw | 2 / 2 | True | 1/5 | 0.304186 | 0.731043 |
| Clustered power law | 0.001 | retained | 4 / 4 | True | 5/5 | 0.281912 | 0.903156 |
| Clustered power law | 0.001 | raw | 4 / 4 | True | 5/5 | 0.291619 | 0.901016 |
| Clustered power law | 0.003 | retained | 5 / 4 | False | 4/5 | 0.269848 | 0.886482 |
| Clustered power law | 0.003 | raw | 5 / 4 | False | 4/5 | 0.273765 | 0.883590 |
| NetScience | 0.0003 | retained | 3 / 1 | True | 5/5 | 0.435081 | 0.835173 |
| NetScience | 0.0003 | raw | 3 / 1 | True | 5/5 | 0.455468 | 0.805591 |
| NetScience | 0.001 | retained | 4 / 4 | True | 5/5 | 0.550726 | 0.829798 |
| NetScience | 0.001 | raw | 5 / 4 | True | 5/5 | 0.560721 | 0.817481 |
| NetScience | 0.003 | retained | 5 / 5 | True | 5/5 | 0.496581 | 0.548267 |
| NetScience | 0.003 | raw | 5 / 5 | True | 5/5 | 0.492503 | 0.542714 |

Family pooling can hide different shapes across graphs. In the clustered family at t=0.0003, the pooled raw-norm peak is interior while only one of five individual graphs has an interior peak. At t=0.003, the pooled peak is at the upper boundary while four of five individual graphs have interior peaks. Both samplers show these classifications.

## NetScience absolute-scale error

The predictor continues to overestimate NetScience norms and misses its strongest-thinning peak by two bins under both samplers. Passing pooled criteria therefore does not imply accurate absolute norms. Node rows below pool five stochastic repetitions of the same 379-node graph.

| Threshold | Noise | Node rows | Observed mean | Predicted mean | Observed minus predicted |
|---:|---|---:|---:|---:|---:|
| 0.0003 | retained | 1895 | 3.812914 | 4.223745 | -0.410832 |
| 0.0003 | raw | 1895 | 3.788004 | 4.215769 | -0.427765 |
| 0.001 | retained | 1895 | 4.161190 | 4.682989 | -0.521800 |
| 0.001 | raw | 1895 | 4.139157 | 4.670415 | -0.531257 |
| 0.003 | retained | 1895 | 4.570215 | 4.950786 | -0.380571 |
| 0.003 | raw | 1895 | 4.566021 | 4.938275 | -0.372254 |

## Evidence boundary

Both arms use corrected repeated-visit counts, correctly assigned negative weights and the checked BLAS build. Raw-noise probabilities are verified fixed across thresholds. The exact same retained corpus, initialization and optimizer state are used within each pair. Equal nominal negative proposals do not imply equal accepted updates because Gensim skips collisions with positive output tokens.

Passing the prediction criteria supports this fixed prediction procedure on these original targets after correction. It does not establish a complete exposure/context mechanism, a universal response curve, equal embeddings under different samplers, or new prospective generalization. Any lost/shifted peak limits the stronger claim that all shapes are invariant.

The original prospective results, calibration-only control and first unpatched control remain historical artifacts. Other experiments using the old Gensim wrappers, including the internal-subsampling threshold sweep, require their own revalidation. No manuscript claim is automatically promoted by this report.

## Reproduce

See analysis/negative_sampling_revalidation_protocol.md for the locked design and setup. Run scripts/revalidate_negative_sampling.py into a fresh directory, run scripts/check_revalidation_repeat.py, then run this reporting script with --training-log. All numerical summaries are recomputed from node records and checked against saved summaries before this report is written.

Training log: 120 completed fits; no numerical warnings or exceptions.

A fresh-process NetScience rerun (seed 301, all three thresholds, both arms) reproduced every field of 1,137 node-condition rows exactly. See repeat_check.json and scripts/check_revalidation_repeat.py.
