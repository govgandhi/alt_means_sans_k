"""Summarize the penalty ablation: top-bin gap and share of the full shift per variant.

share = (gap_variant - gap_GF) / (gap_all - gap_GF), with GF the eps = 0 flow; the same
for the top-bin absolute balanced norm (log ratio to GF). 5 seeds, tau = 0.0628.
Output: results/norm_theory/t2_ablation.json
Run: .venv/bin/python -I scripts/norm_theory/t2_ablation_summary.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
M = ROOT / "results/norm_theory"
EDGES = [1, 3, 5, 7, 10, 14, 10**9]
VARIANTS = ["all", "center", "context", "positive", "negative", "nocross"]


def stats(v, d):
    lab = np.digitize(d, EDGES) - 1
    p = np.array([(v / v.mean())[lab == b].mean() for b in range(6)])
    return p[-1] - p[-2], np.log(v[lab == 5].mean())


def main() -> None:
    out = {}
    for eps in ("0.0025", "0.000625"):
        g = [np.load(M / f"t2_abl_all_eps0.0_s{s}.npz") for s in range(5)]
        g_stats = [stats(z["balanced_norm"], z["degree"]) for z in g]
        res = {"gf_gap": float(np.mean([x[0] for x in g_stats]))}
        full = None
        for v in VARIANTS:
            paths = [M / f"t2_abl_{v}_eps{eps}_s{s}.npz" for s in range(5)]
            if not all(p.exists() for p in paths):
                continue
            st = [
                stats(np.load(p)["balanced_norm"], np.load(p)["degree"]) for p in paths
            ]
            dgap = np.array([a[0] - b[0] for a, b in zip(st, g_stats)])
            dtop = np.array([a[1] - b[1] for a, b in zip(st, g_stats)])
            if v == "all":
                full = (dgap, dtop)
            res[v] = dict(
                gap=float(np.mean([a[0] for a in st])),
                gap_shift=float(dgap.mean()),
                top_log_shift=float(dtop.mean()),
                share_gap=float((dgap / full[0]).mean()) if full else None,
                share_top=float((dtop / full[1]).mean()) if full else None,
            )
        res["cross_share_gap"] = (
            1 - res["nocross"]["share_gap"]
            if "nocross" in res and full is not None
            else None
        )
        out[eps] = res
        print(eps, json.dumps(res, indent=None))
    (M / "t2_ablation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
