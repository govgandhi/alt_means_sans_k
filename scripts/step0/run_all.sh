#!/usr/bin/env bash
# Rebuild step 0 (Fig. 1) from scratch: fetch data, build corpora, train all runs, theory, checks, figures.
# Usage: scripts/step0/run_all.sh [PYTHON]   (default .venv/bin/python, the pinned patched environment)
# JOBS sets parallel single-threaded runs (default 8). Training is seeded and single-threaded per run,
# so outputs are reproducible; runs/ files are overwritten.
set -euo pipefail
PY="${1:-.venv/bin/python}"
JOBS="${JOBS:-8}"
DATA=data/derived/step0
RUNS=results/step0/runs
PNAS=data/external/pnas_murray2023/extracted

"$PY" scripts/step0/fetch_data.py
"$PY" -I scripts/step0/corpora.py \
  --edgelist "$PNAS/derived/Derived/Network/precedence_flows_edgelist.csv" \
  --abstracts data/external/kddcup2003_hepth/extracted --out "$DATA"
"$PY" -I scripts/step0/population_corpus.py --data "$DATA"
"$PY" -I scripts/step0/ba_corpus.py --out "$DATA"
"$PY" -I scripts/step0/blogcatalog_corpus.py \
  --mat data/external/blogcatalog/downloads/blogcatalog.mat --out "$DATA"

jobs=$(mktemp)
for k in 0 1 2 3 4; do
  for c in A B; do echo "real $c $k"; echo "null $c $k"; done
  for c in A B SW; do echo "ba $c $k"; echo "blog $c $k"; done
  for c in A B A50 B50; do echo "pop $c $k"; done
  for c in A B SW; do echo "text $c $k"; done
done > "$jobs"
xargs -P "$JOBS" -L 1 "$PY" -I scripts/step0/train.py --data "$DATA" --out "$RUNS" < "$jobs"
rm -f "$jobs"

"$PY" -I scripts/step0/theory.py --runs "$RUNS"
"$PY" -I scripts/step0/theory.py --runs "$RUNS" --corpus ba --out results/step0/theory_ba.json
"$PY" -I scripts/step0/checks.py --data "$DATA" --runs "$RUNS" --replay-dir "$(mktemp -d)"
FACTORS="$PNAS/derived/Derived/Embeddings/precedence/factors/d300_ws1_precedence_w2vfactors.csv"
SIZES="$PNAS/additional/Additional/2008-2019_inst_sizes.txt"
"$PY" -I scripts/step0/anchor_pnas.py --factors "$FACTORS" --sizes "$SIZES" \
  --gml "$PNAS/derived/Derived/Network/co_occur_network.gml" --out results/step0/anchor_pnas.png
"$PY" -I scripts/step0/figure.py --runs "$RUNS" --anchor-factors "$FACTORS" --anchor-sizes "$SIZES" \
  --out results/step0/step0_problem.png
"$PY" -I scripts/step0/figure_paper.py --runs "$RUNS" --anchor-factors "$FACTORS" --anchor-sizes "$SIZES" \
  --out results/step0/fig1_problem.pdf
echo "step 0 rebuilt; copy results/step0/fig1_problem.pdf to the manuscript's figures/ folder"
