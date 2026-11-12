"""Compare plotted node-pooled SE with replicate-level SE for Fig. 6 curves."""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(line) for line in open(sys.argv[1])]
groups = defaultdict(list)
for r in rows:
    groups[(r["family"], r["threshold"])].append(r)
ratios = []
for key, local in sorted(groups.items()):
    deg = np.array([float(r["degree"]) for r in local])
    y = np.array([float(r["raw_norm"]) for r in local])
    rep = np.array([r["replicate"] for r in local])
    ld = np.log(deg)
    edges = np.unique(np.quantile(ld, np.linspace(0, 1, 7)))
    lab = np.clip(np.digitize(ld, edges[1:-1], right=True), 0, len(edges) - 2)
    out = []
    for b in sorted(set(lab.tolist())):
        sel = lab == b
        node_se = y[sel].std() / np.sqrt(sel.sum())
        rep_means = [y[sel & (rep == k)].mean() for k in sorted(set(rep.tolist())) if np.any(sel & (rep == k))]
        rep_se = np.std(rep_means, ddof=1) / np.sqrt(len(rep_means))
        out.append((round(float(node_se), 4), round(float(rep_se), 4)))
        ratios.append(rep_se / node_se)
    print(key, "(node SE, replicate SE) per bin:", out)
r = np.array(ratios)
print("replicate/node SE ratio: median %.2f, min %.2f, max %.2f" % (np.median(r), r.min(), r.max()))
