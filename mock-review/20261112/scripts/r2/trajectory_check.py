"""Check training-trajectory claims from stored per-graph curves; measure margins of the 'interior max' calls."""
from common import *
import collections
d = json.load(open(WT / "results/training_trajectory/summary.json"))
pg = d["per_graph"]
tab = collections.defaultdict(list)
for r in pg:
    tab[(r["schedule"], r["threshold"], r["epoch"], r["response"])].append(r)
for sched in ("baseline_then_floor", "constant"):
    for t in sorted({r["threshold"] for r in pg}):
        line = []
        for ep in sorted({r["epoch"] for r in pg}):
            rr = tab[(sched, t, ep, "raw_norm")]
            if not rr: continue
            line.append(f"e{ep}:{sum(x['interior_peak'] for x in rr)}/{len(rr)}")
        print(sched, t, " ".join(line))
# monotonicity at epoch 20 constant, t=0: are all bins increasing? smallest step
print("\nconstant, no filter, epoch 20: per-graph bin means and min consecutive increment")
for r in tab[("constant", 0.0, 20, "raw_norm")]:
    m = np.array(r["mean_norm"]); print("  seed", r["seed"], np.round(m, 3), "min step %.4f" % np.min(np.diff(m)), "bins", len(m))
print("baseline, no filter, epoch 20: peak-minus-highest per graph (raw):", [round(r["peak_minus_highest"], 3) for r in tab[("baseline_then_floor", 0.0, 20, "raw_norm")]])
print("baseline, no filter, epoch 2: peak-minus-highest per graph (raw):", [round(r["peak_minus_highest"], 3) for r in tab[("baseline_then_floor", 0.0, 2, "raw_norm")]])
print("baseline, no filter, epoch 1:", [(r["interior_peak"], round(r["peak_minus_highest"], 3)) for r in tab[("baseline_then_floor", 0.0, 1, "raw_norm")]])
# balanced agreement
dis = 0; tot = 0
for (s, t, e, resp), rr in tab.items():
    if resp != "raw_norm": continue
    bb = {x["seed"]: x for x in tab[(s, t, e, "balanced_norm")]}
    for x in rr:
        tot += 1; dis += x["interior_peak"] != bb[x["seed"]]["interior_peak"]
print(f"\nraw vs balanced interior-peak disagreements: {dis}/{tot}")
# thresholded classification: how many interior-peak calls have a peak-minus-endpoint margin < 0.02?
small = [(r["schedule"], r["threshold"], r["epoch"], r["seed"], round(min(r["peak_minus_lowest"], r["peak_minus_highest"]), 4)) for r in pg if r["response"] == "raw_norm" and r["interior_peak"] and min(r["peak_minus_lowest"], r["peak_minus_highest"]) < 0.02]
print("interior-peak calls with min(peak-low, peak-high) < 0.02:", len(small), small[:12])
print("\nraw/balanced disagreements (schedule, t, epoch, seed, raw_interior, raw margin, bal margin):")
for (s, t, e, resp), rr in sorted(tab.items()):
    if resp != "raw_norm": continue
    bb = {x["seed"]: x for x in tab[(s, t, e, "balanced_norm")]}
    for x in rr:
        y = bb[x["seed"]]
        if x["interior_peak"] != y["interior_peak"]:
            print("  ", s, t, e, x["seed"], x["interior_peak"], round(min(x["peak_minus_lowest"], x["peak_minus_highest"]), 4), round(min(y["peak_minus_lowest"], y["peak_minus_highest"]), 4))
