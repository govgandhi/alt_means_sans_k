# Context windows and training history: archived first pass

**Window results below are superseded by the
[continued comparison](window_comparison_findings.md).** Better optimization
changes the initial monotone window trend. The training-history results remain
valid. PR #39 is closed and this evidence is consolidated into draft PR #38.

The complete frozen exploration contains five paired training forks and forty
expected-objective fits. It strengthens the evidence for finite-training history
dependence, while leaving the converged window mechanism unresolved. PRs #35 and
#36 remain unchanged for YY's review.

## Training history: reheating did not erase the inherited peak

All five original unfiltered models were reconstructed exactly at epoch five.
Each full training state, including its RNG, was copied into continuations at
0.0001 and 0.0025 for fifteen further epochs. Every low-rate continuation matches
PR #36's uninterrupted factors exactly. Both arms have identical token counts and
random-stream endpoints; the higher rate has 25 times the rate-times-token exposure.

| Balanced-norm readout | Interior maxima | Highest minus preceding degree bin, mean ± graph SE |
|---|---:|---:|
| Shared epoch-five state | 5/5 | −0.1760 ± 0.0119 |
| Continue at 0.0001 to epoch twenty | 5/5 | −0.1583 ± 0.0116 |
| Reheat at 0.0025 to epoch twenty | 5/5 | −0.1681 ± 0.0111 |
| 0.0025 from initialization, epoch twenty (PR #36 reference) | 0/5 | +0.5392 ± 0.0258 |

Raw norms give the same peak classifications. Reheating changes the embeddings,
but preserves the interior maximum at every saved checkpoint on every graph.
Thus a small final learning rate alone does not account for the observed peak:
within this budget, the preceding training history matters even when the final
rate is the same. This does not establish permanent trapping, separate optima,
hysteresis, or a unique early-time mechanism.

Reheating is not evidence of improved optimization. Mean idealized expected loss
is 0.386216 at the fork, 0.384713 after floor continuation, and 0.386902 after
reheating. The latter is higher than the starting loss on all five graphs.
Gensim's updates use an approximate sigmoid and skip scores outside (−6,6);
90.70% of negative weighted mass is outside that range after reheating, versus
87.55% after floor continuation. Neither number is a loss contribution. The
smooth expected loss is a diagnostic, not an exact description of those updates.

## Context windows: a consistent endpoint trend, without convergence

The second lane fixes equal center exposure, stationary negative noise, rank 24,
and collision-allowing logistic loss. It averages exact walk lags 1 through L;
there is no sampled corpus or integer pair allocation. Every graph uses two
matched initializations across windows.

| Window | Upper-bin balanced-norm gap, mean ± graph SE | Iteration-cap stops |
|---|---:|---:|
| 1 | −1.2563 ± 0.1241 | 10/10 |
| 2 | −0.5964 ± 0.0264 | 10/10 |
| 5 | −0.3760 ± 0.0170 | 10/10 |
| 10 | −0.2166 ± 0.0141 | 10/10 |

The upper-bin drop weakens with wider windows in each of the ten matched
initialization/graph sequences. **All forty fits hit the 500-iteration limit**;
none reports successful convergence. Final factor-gradient maxima
range from 0.0000537 to 0.0861. The two starts' losses differ by as much as 0.706
within a graph/window. Short-window node norms have very large dispersion and
initialization disagreement, visible in the all-node plot; their binned means
must not be mistaken for well-estimated equilibrium norms.

The defensible result is window sensitivity of these bounded optimization
endpoints. It does not establish attenuation of a converged rank-24 target.
Gradient magnitudes depend on factor gauge, different windows define different
losses, and checkpoint score displacements span unequal iteration intervals.
The unrestricted loss gap includes rank restriction, not just optimization error.

A separate exact benchmark is available at window ten: every context entry is
positive, so the full-rank free-score matrix is finite. Its balanced upper-bin
gap is −0.1460 ± 0.0082, negative on all five graphs. Wider windows therefore do
not eliminate all degree dependence in this exact finite-graph benchmark.
This is not the rank-24 optimum. At window one the nonedge scores are −infinity,
so no corresponding finite whole-matrix target norm exists.

## Implication and next experiment

For the PNAS connection, the useful hypothesis is sensitivity to **training
history**, rather than a claim that lowering the final rate creates the U shape.
The present experiment neither reproduces the empirical PNAS pipeline nor
explains its curve. The two lanes also use different dimensions, negative-pair
conventions and context construction; differences between lanes are not a
single-variable intervention.

The optimizer-conditioning and budget check proposed after this first pass has
now been run; see the [revised evidence](window_comparison_findings.md).
The exact β=1, λ=0 controlled-SGD trajectory requested in
#22 also remains unrun. Issues #20–#22 should remain open for these mechanism
questions; this exploration does not justify closing them.

## Evidence and checks

- Fixed design: [protocol](window_training_exploration_protocol.md).
- [Complete generated tables](../results/window_training_exploration/report/tables.md),
  [training figure](../results/window_training_exploration/report/schedule_fork.png),
  [window figure](../results/window_training_exploration/report/expected_windows.png),
  [all nodes and starts](../results/window_training_exploration/report/all_window_nodes.png),
  and [full-rank benchmark](../results/window_training_exploration/report/full_rank_benchmark.png).
- [Reproduction instructions](../results/window_training_exploration/README.md).
  All node outcomes, final factors, stop reasons and source/runtime hashes are saved.
- Independent checks reconstruct graph windows, corpus pair counts and negative
  proposals; recompute losses/gradients; and verify balanced norms by full score
  SVD. Fresh seed-zero reruns reproduce both training arms and all eight window
  fits byte for byte. A separate check rebuilds report bins and graph-level SEs.

The design, complete outcomes and limitations belong together. No parameter,
initialization, graph, or optimizer budget was selected after inspecting these
new scientific outcomes. Maxima and graph-level SEs are descriptive, not formal
hypothesis tests or uncertainty over all possible graphs and training choices.
