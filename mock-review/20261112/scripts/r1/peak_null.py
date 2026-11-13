"""Scratch check (R1): how informative are the frozen peak criteria?

Reads the historical prospective outcomes (read-only) and computes
(1) number of occupied quantile bins per family/threshold,
(2) peak-bin errors of trivial predictors that use no node-level information,
(3) within-curve Spearman ceilings for degree-only predictors versus the
    frozen exposure--KL predictor and raw realized pair counts.
"""

import json
import math
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

PATH = sys.argv[1]
rows = [json.loads(line) for line in open(PATH)]

groups = defaultdict(list)
for r in rows:
    groups[(r["family"], r["threshold"])].append(r)


def binned(degree, values, bins=6):
    logd = np.log(degree)
    edges = np.unique(np.quantile(logd, np.linspace(0, 1, bins + 1)))
    labels = np.clip(np.digitize(logd, edges[1:-1], right=True), 0, len(edges) - 2)
    xs, ys = [], []
    for lab in range(len(edges) - 1):
        m = labels == lab
        if not m.any():
            continue
        xs.append(float(np.exp(logd[m].mean())))
        ys.append(float(values[m].mean()))
    return np.array(xs), np.array(ys)


thr_index = {0.0003: 0, 0.001: 1, 0.003: 2}
out = []
for (fam, thr), local in sorted(groups.items()):
    d = np.array([r["degree"] for r in local], float)
    obs = np.array([r["raw_norm"] for r in local], float)
    pred = np.array([r["predicted_raw_norm"] for r in local], float)
    pairs = np.array([r["realized_positive_pairs"] for r in local], float)
    kl = np.array([r["context_kl"] for r in local], float)
    xs, ys = binned(d, obs)
    _, yp = binned(d, pred)
    nb = len(ys)
    a_bin = int(np.argmax(ys))
    p_bin = int(np.argmax(yp))
    # degree oracle: replace each node's norm by the in-sample mean norm of its degree
    by_deg = defaultdict(list)
    for di, oi in zip(d, obs):
        by_deg[di].append(oi)
    oracle = np.array([np.mean(by_deg[di]) for di in d])
    out.append(
        dict(
            family=fam,
            threshold=thr,
            n_nodes=len(local),
            n_bins=nb,
            observed_bin=a_bin,
            frozen_bin=p_bin,
            frozen_err=abs(a_bin - p_bin),
            const_mid_err=abs(a_bin - (nb - 1) // 2),
            thr_index_err=abs(a_bin - min(nb - 1, thr_index[thr] + 1)),
            rho_frozen=spearmanr(obs, pred)[0],
            rho_pairs=spearmanr(obs, pairs)[0],
            rho_negkl=spearmanr(obs, -kl)[0],
            rho_degree_oracle=spearmanr(obs, oracle)[0],
            distinct_degrees=len(set(d)),
        )
    )

hdr = list(out[0].keys())
print("\t".join(hdr))
for o in out:
    print(
        "\t".join(f"{o[k]:.3f}" if isinstance(o[k], float) else str(o[k]) for k in hdr)
    )

for key in ["frozen_err", "const_mid_err", "thr_index_err"]:
    errs = [o[key] for o in out]
    print(
        key,
        "median",
        float(np.median(errs)),
        "within1",
        sum(e <= 1 for e in errs),
        "/",
        len(errs),
        "exact",
        sum(e == 0 for e in errs),
    )
for key in ["rho_frozen", "rho_pairs", "rho_negkl", "rho_degree_oracle"]:
    vals = [o[key] for o in out]
    print(
        key, "median %.3f range [%.3f, %.3f]" % (np.median(vals), min(vals), max(vals))
    )
