"""R6 check: how many nodes carry the 'decreasing branch', and does it survive finer tail bins?

Reads results/subsampling_origin/outcomes.csv (corrected Gensim, five-epoch endpoints,
read-only). Prints pooled (5 graphs) mean raw/balanced norms in fixed degree bins
that resolve the upper tail, with node counts and per-graph SD of bin means,
plus the degree range covered by the paper's top quantile bin.
Run: r6/.venv/bin/python r6/check_tail_resolution.py
"""
import csv
import numpy as np
P = "<scratch>/round/wt/results/subsampling_origin/outcomes.csv"
rows = list(csv.DictReader(open(P)))
edges = [(1, 3), (4, 4), (5, 5), (6, 6), (7, 8), (9, 11), (12, 16), (17, 25), (26, 40), (41, 1000)]
for t in ("0.0", "0.0003", "0.001", "0.003"):
    sel = [r for r in rows if r["threshold"] == t]
    deg = np.array([float(r["degree"]) for r in sel]); seed = np.array([int(r["seed"]) for r in sel])
    out = []
    for resp in ("raw_norm", "balanced_norm"):
        val = np.array([float(r[resp]) for r in sel])
        cells = []
        for lo, hi in edges:
            m = (deg >= lo) & (deg <= hi)
            per_graph = [val[m & (seed == s)].mean() for s in range(5) if (m & (seed == s)).any()]
            cells.append(f"[{lo}-{hi if hi < 1000 else 'max'}] n={m.sum()} {val[m].mean():.2f}±{np.std(per_graph, ddof=1)/np.sqrt(len(per_graph)):.2f}")
        out.append(resp + ": " + " | ".join(cells))
    # paper's top quantile bin (8 requested log-degree quantiles, pooled)
    ld = np.log(deg); e = np.unique(np.quantile(ld, np.linspace(0, 1, 9)))
    lab = np.clip(np.digitize(ld, e[1:-1], right=True), 0, len(e) - 2)
    top = lab == lab.max()
    print(f"t={t}: paper top bin covers degrees {int(deg[top].min())}-{int(deg[top].max())}, {top.sum()} of {len(deg)} node rows ({top.mean():.1%})")
    for o in out:
        print("   " + o)
