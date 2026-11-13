#!/usr/bin/env bash
# R3 reproducibility checks for c958c528 (read-only against the worktree).
# Everything writes under $R3; the worktree is only read. A fresh clone is used
# for steps that execute repository code, so nothing can write into the worktree.
set -euo pipefail
R3=${R3:-<scratch>/round/r3}
WT=$R3/../wt
C=$R3/clone
SHA=c958c528eada4cc55864e61265c9ff4c3eae7a98
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
PY=$R3/venv/bin/python

# 0. Fresh clone at the reviewed commit.
[ -d "$C" ] || { git clone --quiet --no-checkout <repo> "$C"; git -C "$C" checkout --quiet $SHA; }

# 1. README environment (Python 3.13.12 + pinned requirements + patched Gensim).
[ -x "$PY" ] || uv venv --python 3.13.12 "$R3/venv"
uv pip install --python "$PY" -r "$WT/analysis/negative_sampling_control_requirements.txt" \
  -r "$WT/analysis/negative_sampling_control_build_requirements.txt"
"$PY" "$WT/scripts/build_control_gensim.py" --work-dir "$R3/gensim-build" --install
"$PY" "$WT/scripts/check_control_gensim.py"            # passes; extension sha differs from the recorded one

# 2. Tests: README env alone (collection error), make test (10 errors), union env (136 pass).
(cd "$C" && "$PY" -m pytest -p no:cacheprovider -q) || true
(cd "$C" && make test) || true
uv pip install --python "$PY" PyMuPDF==1.26.7 sympy==1.14.0 mpmath==1.3.0
(cd "$C" && "$PY" -m pytest -p no:cacheprovider -q)

# 3. Hash manifests.
python3 -I "$R3/check_manifests.py" "$WT"

# 4. Gate failures on a reader machine.
(cd "$C" && "$PY" scripts/factor_balance.py --output "$R3/fb_out") || true            # Patched BLAS validation differs
(cd "$C" && "$PY" scripts/training_trajectory/run.py --output "$R3/regen/traj_fresh") || true  # Numerical extension differs
"$PY" "$R3/replay_factor_balance.py" "$C" "$R3/regen/factor_balance_replay.json"     # 20/20 bit-identical

# 5. Regenerate artefacts from committed data.
(cd "$C" && "$PY" scripts/prospective_baselines.py --output "$R3/regen/baselines")    # Tables S1-S2 byte-identical
mkdir -p "$R3/regen/fig6" && cp "$C/results/prospective_validation/outcomes.jsonl" "$R3/regen/fig6/"
(cd "$C" && "$PY" scripts/prospective_norm_turnover.py plot --output-dir "$R3/regen/fig6")
cp -R "$C/results/training_trajectory" "$R3/regen/traj" && (cd "$C" && "$PY" scripts/training_trajectory/report.py --output "$R3/regen/traj")
mkdir -p "$R3/regen/cont" && cp -R "$C/results/window_training_continuation/." "$R3/regen/cont/" && mv "$R3/regen/cont/report" "$R3/regen/cont_report_committed"
(cd "$C" && "$PY" scripts/window_training/report_continuation.py --output "$R3/regen/cont")
"$PY" "$R3/compare_images.py" "$C/results/prospective_validation/prospective_validation.pdf" "$R3/regen/fig6/prospective_validation.pdf"
"$PY" "$R3/compare_images.py" "$C/results/training_trajectory/unfiltered_schedule_comparison.pdf" "$R3/regen/traj/unfiltered_schedule_comparison.pdf"
"$PY" "$R3/compare_images.py" "$R3/regen/cont_report_committed/wide_window_comparison.pdf" "$R3/regen/cont/report/wide_window_comparison.pdf"

# 6. Cheap reruns: controlled trainer (Fig. 1 data, ~75 s) and Gensim sweep (Fig. 4, ~45 s).
(cd "$C/scripts" && "$PY" "$C/scripts/simulate_turnover_surface.py" --output-dir "$R3/regen/turnover")
python3 -I "$R3/compare_json.py" "$WT/results/turnover_surface/summary.json" "$R3/regen/turnover/summary.json"
"$PY" -I "$R3/error_bars.py" fig6 "$R3/regen/fig6/outcomes.jsonl"
"$PY" -I "$R3/error_bars.py" fig1 "$R3/regen/turnover/node_results.csv"
(cd "$C" && "$PY" scripts/simulate_gensim_threshold_sweep.py --output-dir "$R3/regen/gensim_corrected")
# Historical Fig. 4: 5902c3c script versions + stock (defective) Gensim 4.4.0 wheel.
[ -x "$R3/venv_hist/bin/python" ] || { uv venv --python 3.13.12 "$R3/venv_hist"; uv pip install --python "$R3/venv_hist/bin/python" -r "$WT/analysis/negative_sampling_control_requirements.txt"; }
mkdir -p "$R3/hist_src/scripts"
for f in simulate_gensim_threshold_sweep.py simulate_norm_frequency.py simulate_exact_center_exposure.py simulate_turnover_surface.py; do
  git -C "$WT" show 5902c3c:scripts/$f > "$R3/hist_src/scripts/$f"; done
(cd "$R3/hist_src/scripts" && "$R3/venv_hist/bin/python" simulate_gensim_threshold_sweep.py --output-dir "$R3/regen/gensim_hist" \
  --controlled-summary "$WT/results/turnover_surface/summary.json")
python3 -I "$R3/compare_json.py" "$WT/results/gensim_threshold_sweep/summary.json" "$R3/regen/gensim_hist/summary.json"

# 7. README flagship recipe: corrected revalidation (120 fits, ~4 min), repeat check, report.
(cd "$C" && "$PY" scripts/revalidate_negative_sampling.py --output "$R3/regen/fresh-revalidation" > "$R3/regen/revalidation-training.txt" 2>&1)
(cd "$C" && "$PY" scripts/check_revalidation_repeat.py --output "$R3/regen/fresh-revalidation")
(cd "$C" && "$PY" scripts/report_negative_sampling_revalidation.py --output "$R3/regen/fresh-revalidation" \
  --training-log "$R3/regen/revalidation-training.txt")
"$PY" "$R3/compare_images.py" "$C/results/negative_sampling_revalidation/predictor_revalidation.png" \
  "$R3/regen/fresh-revalidation/predictor_revalidation.png"

# 8. Paper build from the fresh clone.
(cd "$C/paper/current" && make main.pdf)
