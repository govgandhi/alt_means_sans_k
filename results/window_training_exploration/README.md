# Window and training-history exploration

Read the concise [findings](../../analysis/window_training_exploration_findings.md)
and the [design frozen before execution](../../analysis/window_training_exploration_protocol.md).
Both lanes are complete: five paired schedule forks (60 saved checkpoints) and
40 expected-window fits. Every expected-window fit reached the fixed iteration
cap; these are unconverged endpoints. Existing PR #36 runs provide the separately
labeled constant-from-start reference. No manuscript claim is changed here.

## Artifacts

- `schedule/`: complete paired states, node outcomes, all checkpoint factors,
  epoch-five and uninterrupted-trajectory bridges, and completion hashes.
- `windows/`: exact graph/target metadata, all node checkpoints, final factors,
  optimizer stops, finite free-score benchmarks, and completion hashes.
- `report/summary.json`: every graph's bin means, initialization comparisons,
  optimization progress and graph-level uncertainty. Two starts are averaged
  within each graph; they are not ten independent graph replicates.
- `report/tables.md`: generated endpoint and initialization tables.
- `report/schedule_fork.{png,pdf}` and `report/expected_windows.{png,pdf}`:
  main comparisons; faint curves show every graph.
- `report/all_window_nodes.{png,pdf}`: every node and both starts, with log axes.
- `report/full_rank_benchmark.{png,pdf}`: distinct exact full-rank window-ten
  benchmark versus the bounded rank-24 endpoints.
- `verification.json`: independent audit results and repeat-artifact count.

The `.json.gz` and `.npy.gz` files use deterministic gzip timestamps. Saved NPZ
checkpoints use fixed ZIP metadata. Input/source hashes and complete artifact
manifests reject stale or changed resumes. Plot PDFs omit creation timestamps.

## From a fresh checkout

Use Python 3.13.12 for exact archived-runtime reproduction. Build tools and all
runtime packages are pinned. Installation is confined to a virtual environment;
a working C compiler is needed for the small Gensim extension build.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r analysis/window_training_exploration_requirements.txt
.venv/bin/python -m pip install -r analysis/negative_sampling_control_build_requirements.txt
.venv/bin/python scripts/build_control_gensim.py --install
.venv/bin/python scripts/check_control_gensim.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/window_training/reproduce.py --output /tmp/window-reproduction
```

The wrapper rebuilds the earlier corrected baselines and trajectories locally
before checking the new schedule forks. A freshly compiled platform need not
match the archived macOS extension hash. Local prefix and uninterrupted bridges
remain exact. Cross-platform bitwise identity of floating-point results is not
promised; each run records its own runtime and source hashes. The complete wrapper
includes the earlier reference sweeps and takes longer than this exploration alone.

With the same checked runtime and archived references, run only the new lanes:

```sh
python3 scripts/window_training/schedule_fork.py
python3 scripts/window_training/expected_window.py
python3 scripts/window_training/report.py
python3 scripts/window_training/verify.py
```

Use `--output /fresh/path/schedule` and `--output /fresh/path/windows` to generate
fresh lane outputs; report and verify take their parent via `--output /fresh/path`.
The schedule runner also accepts `--reference /rebuilt/training_trajectory`.
Completed default invocations verify/reuse the immutable raw artifacts.

`python3 scripts/window_training/verify.py --repeat` reconstructs seed zero from
scratch for both schedule arms and all eight window fits, compares every saved
scientific artifact, and leaves the originals untouched. The recorded check found
42 byte-identical scientific artifacts. The verifier independently derives
logistic loss/gradients, reconstructs pair weights and graph kernels, uses full
score-matrix SVDs for balanced norms, checks the report's binning/SEs, and verifies
all source/data manifests. It never calls a fitting routine except with `--repeat`.
Use `--compare-repeat /existing/repeat/output` to check a retained fresh repeat
without training again. Verification emits JSON to stdout (training logs go to
stderr), so adding `> results/window_training_exploration/verification.json`
records the audit and per-file repeat hashes reproducibly.
