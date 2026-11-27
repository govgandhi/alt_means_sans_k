#!/usr/bin/env bash
# Regenerate every run behind analysis/norm_front_findings.md (seed 0, N=10,000).
# Outputs go to results/norm_front/runs/ (gitignored). The large-d graph runs take longest.
set -euo pipefail
cd "$(dirname "$0")"
r() { uv run --quiet run_one.py "$@"; }

# BA expected graph (iid tokens proportional to degree)
r A_rep --setting A
r A_fresh --setting A --fresh
r B_rep --setting B
r B_fresh --setting B --fresh
r B_shuffle --setting B --shuffle
r SW_rep --setting SW --every 5
r SW_fresh --setting SW --fresh --every 5
r B20_fresh --setting B --fresh --over '{"epochs": 20}' --every 5
r B50_rep --setting B --over '{"epochs": 50}' --every 5

# Actual BA graph (uniform walks started in proportion to degree)
for s in A B; do
  r G_${s}_rep --setting $s --graph
  r G_${s}_fresh --setting $s --graph --fresh_walks
done
r G_SW_rep --setting SW --graph --every 5
r G_SW_fresh --setting SW --graph --fresh_walks --every 5
r G_B_ep20 --setting B --graph --over '{"epochs": 20}' --every 5
r G_A_w1 --setting A --graph --over '{"window": 1, "shrink": false}'
r G_A_d300 --setting A --graph --over '{"dim": 300}'
for d in 30 100 1000; do
  r G_B_d$d --setting B --graph --over "{\"dim\": $d}"
done

uv run --quiet fig.py
uv run --quiet compare.py B_rep A_rep SW_rep
uv run --quiet overlap.py B_rep 50
uv run --quiet overlap.py B_fresh 50
