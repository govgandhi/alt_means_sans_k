# Issue 21: the first control is complete

The corrected fixed-epoch experiment finds an interior maximum of the binned
raw-norm curve on **all five graphs with filtering disabled**. The balanced-norm
readout also has an interior maximum on all five. Subsampling is therefore not
necessary for this descriptive turnover in the tested configuration graphs.

This does not establish the cause of turnover. In particular, five epochs is a
training schedule, not evidence of convergence. The original explanation needs
to distinguish the origin of the turnover from changes caused by subsampling.

## What was compared

Twenty Gensim fits: five graphs × filtering off and the three existing thresholds.
Each graph uses the same raw walks, vocabulary, initial vectors and negative
sampling distribution across settings. All three corrections from PR 35 are
active. The protocol was written before these corrected outcomes were obtained.
The analysis is retrospective, since the older buggy runs were already known.

| Filtering setting | Graphs with an interior raw-norm peak | Pooled peak degree | Pooled peak minus highest-degree-bin norm |
|---|---:|---:|---:|
| Off | 5/5 | 7.7311 | 0.243978 |
| Strong, sample = 0.0003 | 0/5 | 2.9847 (lowest bin) | 0.758533 |
| Intermediate, sample = 0.001 | 5/5 | 6.0000 | 0.527165 |
| Weak, sample = 0.003 | 5/5 | 7.7311 | 0.256983 |

With filtering off, the five individual raw-norm peak-to-highest-bin drops range
from 0.194733 to 0.273929. The corresponding balanced-norm drops range from
0.129584 to 0.210897. Every unfiltered model accounts for all 4,500,000 training
tokens (900,000 raw tokens, visited for five epochs); no node is downsampled.

The original request for eight quantile bins yields six occupied bins because
degrees repeat. These are maxima of binned curves, not formal tests of unimodality.
The peak is chosen from the same curve; the reported drop is descriptive and is
not a prespecified significance contrast. Five graph/walk/optimizer seeds do not
establish generality across graph families, dimensions or training schedules.

## What this supports

A defensible statement is: “In these finite-training Gensim experiments, the
nonmonotonic degree–norm relation persists without token subsampling. Subsampling
changes the location and shape of the relation.”

The experiment does not support “subsampling creates the turnover” as a necessary
mechanism. It also does not establish that the paper's exposure/context explanation
fully accounts for the unfiltered curve. Fixed epochs leave different effective
training budgets, and filtering changes contexts as well as token exposure.

The next unresolved part of issue 21 is a prespecified training-duration and
learning-rate comparison. It should track complete degree-bin trajectories and
avoid equating additional epochs with convergence. That experiment has not been
run here. Issue 21 should remain open pending that distinction or an explicit
narrowing of the manuscript's claims.

## Verification and reproduction

All 32 repository tests pass, including four comparisons of the new runner against
the corrected original trainer, a check of graph-level standard errors, and
protection of historical output paths. A separate direct calculation from all
12,000 saved node rows reproduces all 40 graph-by-setting-by-readout curves.
All 20 pairing checks pass. Reinvoking the script verifies source/runtime/output
hashes and leaves completed results unchanged.

- Design: `analysis/subsampling_origin_protocol.md`
- Standalone command: `python3 scripts/check_subsampling_origin.py`
- All outcomes, checks and curves: `results/subsampling_origin/`
- Numerical report: `results/subsampling_origin/README.md`
- Figure: `results/subsampling_origin/curves.png`

The control is on `codex/subsampling-origin`, based on PR 35. The follow-up in
`analysis/training_trajectory_findings.md` qualifies this result: the unfiltered
turnover depends on learning-rate history. Historical outputs are preserved.
