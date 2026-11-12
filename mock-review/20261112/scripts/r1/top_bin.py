"""Scratch check (R1): how much of the corrected no-subsampling 'turnover' is a top-bin effect?

Reads results/subsampling_origin/outcomes.csv (read-only). For each graph at
threshold 0, bins log degree into 8 requested quantile bins (as in the paper),
reports occupied bins, top-bin degree range and share of nodes, and the
node-level Spearman correlation of norm with degree below and above the
pooled peak. Also fits a LOWESS-free check: mean norm by integer degree.
"""

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

rows = list(csv.DictReader(open(sys.argv[1])))
for thr in ["0.0", "0.0003", "0.001", "0.003"]:
    print(f"threshold {thr}")
    for seed in sorted({r["seed"] for r in rows}):
        local = [r for r in rows if r["seed"] == seed and r["threshold"] == thr]
        d = np.array([float(r["degree"]) for r in local])
        y = np.array([float(r["raw_norm"]) for r in local])
        logd = np.log(d)
        edges = np.unique(np.quantile(logd, np.linspace(0, 1, 9)))
        lab = np.clip(np.digitize(logd, edges[1:-1], right=True), 0, len(edges) - 2)
        occ = sorted(set(lab))
        means = [y[lab == b].mean() for b in occ]
        peak = int(np.argmax(means))
        top = lab == occ[-1]
        below = d <= np.exp(edges[peak + 1]) if peak + 1 < len(edges) else d > 0
        rho_hi = spearmanr(d[d >= d[top].min()], y[d >= d[top].min()])[0]
        print(
            f"  seed {seed}: bins={len(occ)} peak_bin={peak + 1} top-bin deg {d[top].min():.0f}-{d[top].max():.0f}"
            f" share={top.mean():.3f}; mean norm by bin={np.round(means, 2).tolist()}; rho(norm,deg) within top bin={rho_hi:.2f}"
        )
