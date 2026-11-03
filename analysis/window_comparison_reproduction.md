# Reproduce the continued window comparison

Use Python 3.13.12 and the pinned packages below. The window lane does not need
the patched Gensim extension used by the separate training-history experiment.
All commands run from the repository root, without system-wide installation.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r analysis/window_training_exploration_requirements.txt
.venv/bin/python scripts/window_training/reproduce_continuation.py --output /tmp/fresh-window-comparison
```

The wrapper regenerates all original 500-iteration window fits, then the 40
balanced continuations and ten raw-coordinate controls, plots, tables and an
independent audit. It takes no archived factors as inputs. The full wrapper was
not rerun in one invocation here; its original fits and continuation were each
executed and independently checked, including fresh seed-zero repeats. Different
platforms may differ in floating-point rounding; local source/runtime manifests
are recorded, and bitwise identity across platforms is not promised.

To inspect or continue the committed endpoints in their recorded runtime:

```sh
python3 scripts/window_training/continue_windows.py
python3 scripts/window_training/report_continuation.py
python3 scripts/window_training/verify_continuation.py > results/window_training_continuation/report/verification.json
```

Completed conditions are verified and reused. To repeat all ten seed-zero
continuations from their original endpoints in a fresh temporary directory:

```sh
python3 scripts/window_training/verify_continuation.py --repeat --seeds 0 > results/window_training_continuation/report/verification.json
```

For a fresh continuation only, pass `--output /tmp/fresh-continuation` to the
runner, report and verifier. That mode uses the committed original endpoints;
the first wrapper regenerates those endpoints too. `--seeds` permits resumable
subsets, but all five are required for a complete comparison.

The numerical design is in [the frozen amendment](window_optimization_amendment.md);
all parameters and paths live in the runners' configuration dataclasses.
`results/window_training_continuation/plan.json` pins the source, reference and
runtime; `completion.json` hashes raw outcomes and final factors. Do not put
additional files beside those immutable raw files: documentation belongs in
`analysis/`, derived reports/audits in `report/`. The report manifest hashes only
its explicit generated artifacts, avoiding a provenance cycle with verification.

The verifier checks every condition and start, target hashes, original endpoints,
final objective/gradient/full-SVD norms, stopping rules and matched control caps.
It also reconstructs report aggregates independently. Intermediate snapshots save
node readouts and diagnostics but not factor matrices, so their complete matrix
displacements cannot be reconstructed independently. All final factors are saved.

Run the window/continuation tests with the pinned environment. The earlier
symbolic/PDF audit has separate dependencies in
`analysis/realized_graph_audit_requirements.txt`; it does not change these runs.

The final proposition has a separate symbolic check, with no training or new
empirical data. In an environment containing the pinned SymPy version above, run:

```sh
python3 scripts/check_sgns_norm_boundary.py
```

It verifies derivatives, stationary scores, tail limits, and an exact two-node
SGNS illustration whose balanced norms grow while loss approaches its infimum.
This algebraic example is not one of the experimental simple graphs.
