# Issue 21: learning-rate history changes the conclusion

Forty trajectories are complete: five graphs × four filtering settings × two
predetermined learning-rate schedules. The corrected five-epoch baseline is
reproduced exactly in all twenty continuation runs. No parameter was selected
using these trajectory outcomes.

The most direct result is **schedule dependence at a fixed twenty-epoch budget**.
With filtering off, the original decay schedule produces interior binned maxima
on all five graphs by epoch two, and all five retain them through epoch twenty.
The constant-0.0025 alternative has no interior maximum at any recorded checkpoint;
its epoch-twenty curves increase across all six occupied bins on all five graphs.
Raw and balanced norms agree on these interior/boundary classifications.
The weak-filter setting, sample=0.003, shows the same contrast.

| Filtering off, epoch 20 | Original schedule + continuation | Constant 0.0025 |
|---|---:|---:|
| Graphs with interior raw-norm maxima | 5/5 | 0/5 |
| Graphs with interior balanced-norm maxima | 5/5 | 0/5 |
| Pooled peak degree, both readouts | 7.73 | 16.04 (highest bin) |
| Mean expected logistic loss per positive pair | 0.384713 | 0.389260 |

The two schedules have identical effective/raw token counts at every epoch within
each graph/filter pair. Different learning-rate histories therefore change the
curve in this fixed implementation without changing those token counts. They do
not yield equal optimization progress: loss continues improving in both schedules.
The similar final scalar losses do not establish equivalent embeddings or optima.

All forty expected-loss trajectories decrease at every recorded interval from
epoch one onward. From five to twenty epochs, average reductions range from
0.30–0.39% under the original continuation and 6.85–24.38% under constant rate,
depending on filtering. The first arm continues at a very small rate, making
persistence there particularly weak evidence of stationarity.

The other conditions also limit simple stories. Under constant rate, strong
filtering produces transient interior maxima that all disappear by epoch twenty.
Intermediate filtering has shallow early maxima on four graphs, no interior
maximum at epochs two/five, and interior maxima on all five by epochs ten/twenty.
These are binned maxima, not identified dynamical phases.

## Answers to the issue's three questions

1. **No-filter baseline:** completed with graph-level standard errors. Filtering
   is unnecessary for turnover under the original schedule.
2. **Epoch/learning-rate test:** completed for two fixed schedules through twenty
   epochs. It demonstrates schedule and duration dependence, but does not identify
   convergence, high-degree saturation, or the reason the controlled trainer differs.
   Capacity and actual positive-pair budget matching remain unrun.
3. **Discussion wording:** qualified in `paper/current/main.tex`. The square-root
   retained-token law is preserved as mathematics; it is no longer presented as
   establishing the origin of the Gensim turnover. The historical budget-matched
   figure is identified as uncorrected and its node-based error bars are labeled.

## Strongest supported claim and boundary

In these finite-training experiments, subsampling is unnecessary for turnover
under the original learning-rate schedule, but the turnover is sensitive to
learning-rate history. Increasing the number of epochs does not settle the
mechanism independently of the schedule.

It is still unsupported that either schedule has converged; that high-degree
nodes reaching an optimum causes the peak; or that the curve is determined by
graph structure and filtering alone. Balanced-norm agreement rules out neither
optimization dependence nor implementation effects.

The expected logistic loss integrates independent retention and uniform proposal
draws on the same corpus; it is an in-sample diagnostic. Gensim uses approximate
sigmoids and skips updates outside (-6,6). The maximum outside-window weighted
mass is 2.03% for positives and 89.16% for negatives across these runs. Large
negative mass there is not itself a bug or a measured loss contribution; the
smooth diagnostic should not be equated to the implemented update rule.

## Evidence

- Fixed design: `analysis/training_trajectory_protocol.md`
- Complete numerical tables: `results/training_trajectory/README.md`
- All node outcomes and epoch counts: `results/training_trajectory/runs/`
- Main comparison: `results/training_trajectory/unfiltered_schedule_comparison.pdf`
- Full reproducible verification: `python3 scripts/training_trajectory/verify.py --repeat`
- Fresh-platform reproduction after installing the pinned, patched environment:
  `python3 scripts/training_trajectory/reproduce.py --output /fresh/path`.
  This regenerates the baseline on that platform before testing exact continuations;
  the archived macOS binary need not match a newly compiled binary byte for byte.

The verifier independently rebuilds all 480 graph/readout/checkpoint curves,
checks all twenty baseline bridges against the saved node outcomes, and can rerun
the complete unfiltered constant-rate seed-zero trajectory in a fresh process.
Issue 21 remains open for the mechanism and pipeline-matching questions. The PR
adds evidence and narrows the manuscript's claims; it does not declare convergence.
