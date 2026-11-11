# Training trajectories: issue 21

Forty continuous runs: five graphs × four filtering settings × two schedules.
All 20 baseline continuations reproduce the corrected epoch-five input/output
hashes and effective-token counts exactly. Initialization is checkpoint zero.

The baseline schedule anneals over five epochs, then stays at 0.0001.
The other schedule uses a constant 0.0025.
Epoch count does not redefine either schedule.

## Interior-peak counts and pooled peak degrees

Cells show interior maxima / 5 graphs, followed by pooled peak degree.
A boundary maximum is not an inverted-U. These are descriptive quantile-bin
maxima, not statistical tests. Eight requested bins give six occupied bins.

### raw_norm

| schedule | sample | epoch 1 | epoch 2 | epoch 5 | epoch 10 | epoch 20 |
|:---|---:|:---|:---|:---|:---|:---|
| baseline_then_floor | 0 | 0/5; 16.04 | 5/5; 7.73 | 5/5; 7.73 | 5/5; 7.73 | 5/5; 7.73 |
| baseline_then_floor | 0.0003 | 5/5; 4.00 | 2/5; 2.98 | 0/5; 2.98 | 0/5; 2.98 | 0/5; 2.98 |
| baseline_then_floor | 0.001 | 5/5; 7.73 | 5/5; 6.00 | 5/5; 6.00 | 5/5; 6.00 | 5/5; 6.00 |
| baseline_then_floor | 0.003 | 0/5; 16.04 | 5/5; 7.73 | 5/5; 7.73 | 5/5; 7.73 | 5/5; 7.73 |
| constant | 0 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 |
| constant | 0.0003 | 0/5; 16.04 | 2/5; 2.98 | 5/5; 4.00 | 4/5; 4.00 | 0/5; 2.98 |
| constant | 0.001 | 4/5; 7.73 | 0/5; 16.04 | 0/5; 16.04 | 5/5; 7.73 | 5/5; 7.73 |
| constant | 0.003 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 |

### balanced_norm

| schedule | sample | epoch 1 | epoch 2 | epoch 5 | epoch 10 | epoch 20 |
|:---|---:|:---|:---|:---|:---|:---|
| baseline_then_floor | 0 | 0/5; 16.04 | 5/5; 7.73 | 5/5; 7.73 | 5/5; 7.73 | 5/5; 7.73 |
| baseline_then_floor | 0.0003 | 5/5; 4.00 | 2/5; 2.98 | 0/5; 2.98 | 0/5; 2.98 | 0/5; 2.98 |
| baseline_then_floor | 0.001 | 5/5; 7.73 | 5/5; 6.00 | 5/5; 6.00 | 5/5; 6.00 | 5/5; 6.00 |
| baseline_then_floor | 0.003 | 0/5; 16.04 | 5/5; 7.73 | 5/5; 7.73 | 5/5; 7.73 | 5/5; 7.73 |
| constant | 0 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 |
| constant | 0.0003 | 0/5; 16.04 | 2/5; 2.98 | 5/5; 4.00 | 4/5; 4.00 | 0/5; 2.98 |
| constant | 0.001 | 4/5; 7.73 | 0/5; 16.04 | 0/5; 16.04 | 5/5; 7.73 | 5/5; 7.73 |
| constant | 0.003 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 | 0/5; 16.04 |

## Fixed expected-objective diagnostic

Loss is normalized per expected positive pair; compare changes within
each filtering condition. Values are graph means ± standard errors.

| schedule | sample | loss epoch 1 | loss epoch 5 | loss epoch 20 | mean reduction 5→20 |
|:---|---:|:---|:---|:---|---:|
| baseline_then_floor | 0 | 0.427690 ± 0.004584 | 0.386216 ± 0.004436 | 0.384713 ± 0.004437 | 0.389% |
| baseline_then_floor | 0.0003 | 0.979740 ± 0.003039 | 0.905719 ± 0.002768 | 0.902961 ± 0.002775 | 0.305% |
| baseline_then_floor | 0.001 | 0.646452 ± 0.004665 | 0.588265 ± 0.004539 | 0.585957 ± 0.004531 | 0.392% |
| baseline_then_floor | 0.003 | 0.493163 ± 0.007862 | 0.446009 ± 0.007532 | 0.444363 ± 0.007504 | 0.369% |
| constant | 0 | 2.235162 ± 0.010420 | 0.417886 ± 0.004664 | 0.389260 ± 0.004501 | 6.852% |
| constant | 0.0003 | 2.691527 ± 0.004415 | 1.201045 ± 0.004604 | 0.908205 ± 0.002848 | 24.381% |
| constant | 0.001 | 2.652605 ± 0.000925 | 0.673653 ± 0.004653 | 0.592298 ± 0.004538 | 12.079% |
| constant | 0.003 | 2.490193 ± 0.005889 | 0.498659 ± 0.008722 | 0.454198 ± 0.007642 | 8.912% |

## Numerical limits and files

Maximum expected mass outside Gensim's open (-6,6) update interval: positive 2.030488%, negative 89.155421%.
The stable logistic loss and its gradient are idealized diagnostics:
Gensim uses approximate sigmoid arithmetic and skips updates outside (-6,6).
Independent retention and uniform integer draws are assumed for expectation.
A stable curve, small gradient, or flat loss is not proof of convergence.
These are in-sample diagnostics, not held-out predictive performance.

`runs/` contains every node's norms, loss and score changes at checkpoints,
plus actual effective-token accounting. `graphs/` contains expected positive
exposure and hashes; expected pairs are not measured realized training pairs.
Score changes span unequal checkpoint intervals: 1, 1, 3, 5 and 10 epochs.
They are displacements over those intervals, not rates or speeds.
The idealized gradient is neither a Gensim update nor a gauge-invariant
distance to an optimum.
`summary.json` preserves all graph curves, bins/counts and progress values.
Norm plots show graph means ± SE, with individual epoch-20 curves in grey.
The loss plots show all graph trajectories; degree loss plots show bin means.
Degree-bin losses weight nodes equally after normalizing each node by its
expected positive-pair count. Global loss weights by expected exposure.

Neither the training budget across filtering settings nor embedding dimension
is matched to the separate controlled trainer in this experiment. Generality,
convergence and the cause of degree-dependent optimization remain unproven.
