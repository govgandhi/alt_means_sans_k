# Subsampling-origin diagnostic (issue 21)

Corrected Gensim, five configuration graphs, five epochs each. All four
settings share each graph's raw walks, initialization and negative CDF.
This is a retrospective, fixed-epoch test; update budgets are not matched.

## Primary outcome: raw input norm

An interior peak is a descriptive maximum of binned means, not a test of
unimodality. Peak gaps are selected from the observed curve.

| sample | graph seed | occupied bins | peak bin | peak degree | peak − low | peak − high | interior |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0 | 6 | 5 | 7.8157 | 0.719479 | 0.273929 | True |
| 0 | 1 | 6 | 5 | 7.8055 | 0.697541 | 0.213899 | True |
| 0 | 2 | 6 | 5 | 7.7368 | 0.642887 | 0.237519 | True |
| 0 | 3 | 6 | 5 | 7.4363 | 0.631765 | 0.194733 | True |
| 0 | 4 | 6 | 5 | 7.4540 | 0.654971 | 0.196855 | True |
| 0.0003 | 0 | 6 | 1 | 2.9779 | 0.000000 | 0.788812 | False |
| 0.0003 | 1 | 6 | 1 | 2.9818 | 0.000000 | 0.766030 | False |
| 0.0003 | 2 | 6 | 1 | 2.9953 | 0.000000 | 0.719406 | False |
| 0.0003 | 3 | 6 | 1 | 2.9898 | 0.000000 | 0.757363 | False |
| 0.0003 | 4 | 6 | 1 | 2.9781 | 0.000000 | 0.714794 | False |
| 0.001 | 0 | 6 | 5 | 7.8157 | 0.397767 | 0.599471 | True |
| 0.001 | 1 | 6 | 4 | 6.0000 | 0.446681 | 0.542280 | True |
| 0.001 | 2 | 6 | 4 | 6.0000 | 0.376761 | 0.531202 | True |
| 0.001 | 3 | 6 | 4 | 6.0000 | 0.503385 | 0.605167 | True |
| 0.001 | 4 | 6 | 5 | 7.4540 | 0.380580 | 0.453684 | True |
| 0.003 | 0 | 6 | 5 | 7.8157 | 0.614509 | 0.343615 | True |
| 0.003 | 1 | 6 | 4 | 6.0000 | 0.564719 | 0.167458 | True |
| 0.003 | 2 | 6 | 5 | 7.7368 | 0.476879 | 0.226150 | True |
| 0.003 | 3 | 6 | 4 | 6.0000 | 0.597298 | 0.280300 | True |
| 0.003 | 4 | 6 | 5 | 7.4540 | 0.603312 | 0.231042 | True |

## Pooled curves (comparison with the original sweep)

| sample | response | peak degree | peak − high | interior |
|---:|:---|---:|---:|:---:|
| 0 | raw_norm | 7.7311 | 0.243978 | True |
| 0 | balanced_norm | 7.7311 | 0.174168 | True |
| 0.0003 | raw_norm | 2.9847 | 0.758533 | False |
| 0.0003 | balanced_norm | 2.9847 | 0.758723 | False |
| 0.001 | raw_norm | 6.0000 | 0.527165 | True |
| 0.001 | balanced_norm | 6.0000 | 0.494545 | True |
| 0.003 | raw_norm | 7.7311 | 0.256983 | True |
| 0.003 | balanced_norm | 7.7311 | 0.197619 | True |

## Training accounting

Effective tokens are Gensim's retained-token accounting, not positive
pairs or gradient updates. Raw tokens include repeated epoch passes.

| sample | seed | raw tokens | effective tokens | downsampled nodes |
|---:|---:|---:|---:|---:|
| 0 | 0 | 4500000 | 4500000 | 0 |
| 0.0003 | 0 | 4500000 | 2594212 | 595 |
| 0.001 | 0 | 4500000 | 3877573 | 75 |
| 0.003 | 0 | 4500000 | 4346059 | 10 |
| 0 | 1 | 4500000 | 4500000 | 0 |
| 0.0003 | 1 | 4500000 | 2600305 | 592 |
| 0.001 | 1 | 4500000 | 3893624 | 76 |
| 0.003 | 1 | 4500000 | 4380735 | 8 |
| 0 | 2 | 4500000 | 4500000 | 0 |
| 0.0003 | 2 | 4500000 | 2598874 | 592 |
| 0.001 | 2 | 4500000 | 3869472 | 71 |
| 0.003 | 2 | 4500000 | 4353114 | 10 |
| 0 | 3 | 4500000 | 4500000 | 0 |
| 0.0003 | 3 | 4500000 | 2605376 | 596 |
| 0.001 | 3 | 4500000 | 3899543 | 66 |
| 0.003 | 3 | 4500000 | 4378047 | 8 |
| 0 | 4 | 4500000 | 4500000 | 0 |
| 0.0003 | 4 | 4500000 | 2601444 | 591 |
| 0.001 | 4 | 4500000 | 3878268 | 65 |
| 0.003 | 4 | 4500000 | 4381099 | 12 |

## Evidence and limitations

`outcomes.csv` contains every node and both norm readouts. `summary.json`
contains all individual and pooled curves, bin counts, and graph means/SEs.
Standard errors average graph means equally within common degree bins;
they do not treat individual nodes as independent replicates. Colored
curves use each graph's quantiles; black curves use common boundaries.
`checks.json`, `plan.json` and `completion.json` record the pairing, runtime,
source and output hashes. The plan precedes training.

Filtering also changes context formation and the training budget. This
test can establish that filtering is unnecessary for turnover in these
settings if unfiltered turnover persists. It cannot establish the cause,
convergence, generality, or an exposure-only effect. No manuscript claims
or frozen historical outputs were modified.

Reproduce with `python3 scripts/check_subsampling_origin.py --output /fresh/path`
after following `analysis/subsampling_origin_protocol.md`.
