# Issue 21: training duration and learning-rate history

Status: design fixed before trajectory outcomes. Retrospective mechanism diagnostic,
not a prospective prediction test. The earlier no-filter control is already known.

## Question and design

Does the degree–norm turnover persist, disappear, or change as training progresses,
and does this depend on the learning-rate history? Use all five graph/walk/optimizer
seeds and all four thresholds from `analysis/subsampling_origin_protocol.md`.
Keep its graphs, walks, initialization, corrected raw-token noise distribution,
dimension 64, window one, five negatives and single worker. All three PR 35 fixes
remain active. The earlier baseline outputs are immutable comparison inputs.

Train 40 continuous trajectories: five seeds × four thresholds × two schedules.
Record checkpoints at initialization and after 1, 2, 5, 10 and 20 epochs. Do not
restart between checkpoints or stretch the schedule according to a checkpoint.

- `baseline_then_floor`: the original linear 0.025 to 0.0001 schedule over five
  epochs, then constant 0.0001 through epoch 20. Checkpoint five must reproduce
  the earlier control's input/output hashes and per-node norms exactly.
- `constant`: learning rate 0.0025 throughout. This value is the existing final
  learning rate in `simulate_turnover_surface.Config`; it is not selected from
  the new results. This arm varies training duration at fixed step size.

The first arm provides an exact continuation; the second checks sensitivity to
learning-rate history. This is not a comprehensive learning-rate sweep. The
post-five-epoch floor is small, so a stable first-arm curve is weak evidence of
stationarity. Checkpoints and the twenty-epoch limit are diagnostic design choices,
not estimates of convergence or natural thresholds. They include the existing
five-epoch result, earlier stages, and two successive doublings thereafter.

## Fixed expected-objective diagnostic

Use each graph's original finite walk corpus. For two positions a<b in a walk,
the probability they become adjacent after independent token retention is
r[a] r[b] product(1-r[c]) over intervening positions. Sum both directed positive
pairs, including repeated-node self-pairs. For sample=0, r=1. Otherwise use
r=(sample_int+1)/2^32, matching the inclusive retention comparison under a uniform
32-bit draw. This integrates over filtering rather than drawing new evaluation
samples at each checkpoint.

Compute negative proposal probabilities from the corrected integer CDF, including
bisect_left endpoint allocation and the modulo distribution of a uniform 32-bit
integer. If P[i,j] is the expected positive-pair count and m[i]=sum_j P[i,j], the
negative weight is k q[j] (m[i]-P[i,j]), because Gensim skips a proposal equal to
the positive output without resampling. Evaluate stable logistic losses on U V^T,
normalized globally by total positive mass and per node by m[i]. This is an
in-sample objective diagnostic, not a generalization test. Compare loss changes
within a filtering condition; different thresholds define different objectives.

This expected objective assumes independent uniform random draws. It does not
replay the correlated PRNG or Gensim's approximate sigmoid arithmetic. Gensim also
skips updates when scores are <=-6 or >=6. Record positive- and negative-weighted
fractions outside this interval at every checkpoint. A plateau therefore does
not establish a logistic optimum. Record per-node score-row RMS changes between
saved checkpoints, in addition to losses and norms. Record the RMS parameter
gradient of this idealized normalized objective as a further diagnostic, without
using it as a convergence threshold or treating it as gauge invariant.
Differences between bin losses
are not distances to a known optimum; shared output vectors couple the groups.

## Readouts and integrity

Raw input norm is primary, balanced SVD norm secondary. Use the earlier requested
eight log-degree quantiles (ties reduce the occupied bins). Preserve all individual
graph curves, bin counts, peaks, endpoint gaps, per-node expected losses and score
changes. Plot equally weighted graph means and standard errors on common,
outcome-independent degree bins. Selected maxima and peak gaps are descriptive,
not significance tests. Include all schedules, thresholds and seeds.

Save a source/runtime/input-hashed plan before training. Verify raw walk counts,
initialization/RNG/noise pairing and the epoch-five baseline bridge. Diagnostic
callbacks must not consume the model RNG or mutate its parameters. Record effective
tokens by epoch separately from expected positive pairs; neither is a measured
realized positive-pair count. This study does not claim matched update budgets
between filtering settings. Save completed runs individually so an interruption
can resume with the identical plan; verify source and output hashes on reuse.

## Interpretation fixed before outcomes

- Turnover emerging or moving during training supports dependence on training
  progress; it does not identify why high-degree nodes behave differently.
- Persistence across both schedules and later checkpoints supports robustness in
  these settings; it does not establish convergence or a structural explanation.
- Lost, shallow or schedule-dependent turnover limits the manuscript's claim.
- Disagreement between raw and balanced readouts limits gauge robustness.
- Objective improvements alongside stable curves show that curve stability is
  not objective convergence. Flat objective or norm trajectories alone are not
  sufficient to claim convergence either.

Capacity matching (24 versus 64 dimensions), exact accepted-update matching and
changes in the conditional context distribution remain separate unrun controls.
No existing manuscript claim should be strengthened before this full sweep ends.

## Reproduction

Use Python 3.13.12 and the existing pinned PR 35 runtime/build requirements, with
its patched Gensim extension and passing BLAS check. Then run:

```sh
python3 -m pytest
python3 scripts/training_trajectory/run.py
python3 scripts/training_trajectory/report.py
```

Default outputs: `results/training_trajectory`. Use `--output /fresh/path` on both
commands for a fresh run. Completed invocations verify artifacts. The old
`results/subsampling_origin` remains the reference and is not overwritten.
