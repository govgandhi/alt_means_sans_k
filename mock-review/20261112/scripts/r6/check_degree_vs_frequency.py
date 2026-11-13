"""R6 check: does graph degree coincide with realized walk frequency in the corrected corpora?
Reads results/subsampling_origin/outcomes.csv (tokens = every occurrence, corrected counts).
Run: r6/.venv/bin/python r6/check_degree_vs_frequency.py
"""
import csv
import numpy as np
from scipy.stats import spearmanr
P = "<scratch>/round/wt/results/subsampling_origin/outcomes.csv"
rows = [r for r in csv.DictReader(open(P)) if r["threshold"] == "0.0"]
for s in range(5):
    sel = [r for r in rows if int(r["seed"]) == s]
    d = np.array([float(r["degree"]) for r in sel]); t = np.array([float(r["tokens"]) for r in sel])
    ratio = t / (d / d.sum() * t.sum())
    print(f"seed {s}: spearman(degree,tokens)={spearmanr(d, t).statistic:.4f}; tokens/expected: min {ratio.min():.3f} max {ratio.max():.3f}; CV within degree-3 class {np.std(t[d == 3]) / np.mean(t[d == 3]):.3f}")
