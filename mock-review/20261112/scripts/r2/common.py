"""Shared helpers for R2 checks (read-only on the worktree)."""
import json, math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

WT = Path("<scratch>/round/wt")
PV = WT / "results/prospective_validation"
THRESHOLDS = (3e-4, 1e-3, 3e-3)
FAMILIES = ("barabasi_albert", "powerlaw_cluster", "netscience")

def load_jsonl(p):
    with open(p) as h:
        return [json.loads(l) for l in h if l.strip()]

def binned_curve(degree, values, bins=6):
    # exact copy of scripts/prospective_norm_turnover.py:binned_curve
    log_degree = np.log(degree)
    edges = np.unique(np.quantile(log_degree, np.linspace(0.0, 1.0, bins + 1)))
    labels = np.clip(np.digitize(log_degree, edges[1:-1], right=True), 0, len(edges) - 2)
    x, y, e, n = [], [], [], []
    for label in range(len(edges) - 1):
        ch = labels == label
        if not np.any(ch):
            continue
        x.append(float(np.exp(np.mean(log_degree[ch]))))
        y.append(float(np.mean(values[ch])))
        e.append(float(np.std(values[ch]) / math.sqrt(np.sum(ch))))
        n.append(int(np.sum(ch)))
    return np.asarray(x), np.asarray(y), np.asarray(e), np.asarray(n)

def peak(degree, values, bins=6):
    x, y, _, _ = binned_curve(degree, values, bins)
    p = int(np.argmax(y))
    return p, x, y

def sp(a, b):
    return float(spearmanr(a, b).statistic)
