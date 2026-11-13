"""Count occupied quantile degree bins per prospective family/threshold."""
import json
import sys
from collections import defaultdict

import numpy as np

rows = [json.loads(line) for line in open(sys.argv[1])]
print(len(rows), sorted(rows[0].keys()))
groups = defaultdict(list)
for r in rows:
    groups[(r["family"], r["threshold"])].append(r)
for key, local in sorted(groups.items()):
    deg = np.array([float(r["degree"]) for r in local])
    ld = np.log(deg)
    edges = np.unique(np.quantile(ld, np.linspace(0, 1, 7)))
    lab = np.clip(np.digitize(ld, edges[1:-1], right=True), 0, len(edges) - 2)
    occ = sorted(set(lab.tolist()))
    centers = [float(np.exp(ld[lab == b].mean())) for b in occ]
    counts = [int((lab == b).sum()) for b in occ]
    reps = sorted({r.get("replicate") for r in local})
    print(key, "nodes", len(local), "occupied", len(occ), "centers", [round(c, 2) for c in centers], "counts", counts, "reps", reps)
