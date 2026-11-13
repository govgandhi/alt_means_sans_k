"""Corrected internal-subsampling sweep (results/subsampling_origin/outcomes.csv): per-graph peak movement, margins, pooled vs per-graph."""
from common import *
import csv, collections
rows = list(csv.DictReader(open(WT / "results/subsampling_origin/outcomes.csv")))
for r in rows:
    for k in ("seed", "node"): r[k] = int(r[k])
    for k in ("threshold", "degree", "raw_norm", "balanced_norm"): r[k] = float(r[k])
def curve(sel, bins=8):
    d = np.array([r["degree"] for r in sel]); a = np.array([r["raw_norm"] for r in sel])
    return binned_curve(d, a, bins)
print("per-graph raw-norm peak bin (graph-specific 8 requested quantile bins; occupied bins), peak deg, top-two margin")
pk = collections.defaultdict(dict)
for t in (0.0, 0.0003, 0.001, 0.003):
    line = []
    for s in range(5):
        sel = [r for r in rows if r["seed"] == s and r["threshold"] == t]
        x, y, e, n = curve(sel)
        o = np.argsort(y)[::-1]; pk[t][s] = int(o[0])
        line.append(f"s{s}:b{o[0]}({x[o[0]]:.2f}) m={y[o[0]]-y[o[1]]:.3f}")
    x, y, e, n = curve([r for r in rows if r["threshold"] == t])
    print(f" t={t:g}: " + "  ".join(line) + f" | pooled peak deg {x[np.argmax(y)]:.2f}, pooled drop to last {y.max()-y[-1]:.3f}")
for a, b in ((0.0003, 0.001), (0.001, 0.003), (0.003, 0.0)):
    mv = [pk[b][s] - pk[a][s] for s in range(5)]
    print(f"peak-bin change {a:g} -> {b:g} per graph: {mv}  (graphs moving up: {sum(m > 0 for m in mv)}/5)")
