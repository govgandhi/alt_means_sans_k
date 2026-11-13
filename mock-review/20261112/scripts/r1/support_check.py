"""Scratch check (R1): are the prospective targets inside the calibration support?

Reads calibration.jsonl and outcomes.jsonl (read-only) and reports degree,
log-pair and KL ranges per family, plus the share of target nodes whose
features fall outside the calibration range.
"""

import json
import sys
from collections import defaultdict

import numpy as np

cal_path, out_path = sys.argv[1], sys.argv[2]
cal = [json.loads(line) for line in open(cal_path)]
out = [json.loads(line) for line in open(out_path)]

print("calibration keys:", sorted(cal[0].keys()))


def stats(rows, label):
    d = np.array([r["degree"] for r in rows], float)
    lp = np.log(np.array([r["realized_positive_pairs"] for r in rows], float))
    kl = np.array([r["context_kl"] for r in rows], float)
    print(
        f"{label:28s} n={len(rows):5d} deg[min,med,max]=[{d.min():.0f},{np.median(d):.0f},{d.max():.0f}]"
        f" share deg<3={np.mean(d < 3):.3f} logpairs[{lp.min():.2f},{lp.max():.2f}] KL[{kl.min():.2f},{kl.max():.2f}]"
    )
    return d, lp, kl


by_thr_cal = defaultdict(list)
for r in cal:
    by_thr_cal[r["threshold"]].append(r)
by_ft = defaultdict(list)
for r in out:
    by_ft[(r["family"], r["threshold"])].append(r)

ranges = {}
for thr, rows in sorted(by_thr_cal.items()):
    d, lp, kl = stats(rows, f"calibration t={thr}")
    ranges[thr] = (lp.min(), lp.max(), kl.min(), kl.max(), d.min())

for (fam, thr), rows in sorted(by_ft.items()):
    d, lp, kl = stats(rows, f"{fam[:12]} t={thr}")
    lo_lp, hi_lp, lo_kl, hi_kl, dmin = ranges[thr]
    outside = (lp < lo_lp) | (lp > hi_lp) | (kl < lo_kl) | (kl > hi_kl)
    print(
        f"    outside calibration feature box at same t: {outside.mean():.3f}; below calibration min degree: {np.mean(d < dmin):.3f}"
    )
