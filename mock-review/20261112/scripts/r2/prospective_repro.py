"""Reproduce headline prospective numbers from stored outcomes; probe pooling and peak criteria."""
from common import *
import itertools
rows = load_jsonl(PV / "outcomes.jsonl")
fro = load_jsonl(PV / "frozen_predictions.jsonl")
summ = json.load(open(PV / "summary.json"))
# 0. frozen predictions identical to the predictions in outcomes?
fk = {(r["family"], r["replicate"], r["threshold"], r["node"]): r for r in fro}
mism = sum(1 for r in rows if fk[(r["family"], r["replicate"], r["threshold"], r["node"])]["predicted_raw_norm"] != r["predicted_raw_norm"])
print("rows", len(rows), "frozen", len(fro), "prediction mismatches outcomes-vs-frozen:", mism)

A = np.array([r["raw_norm"] for r in rows]); P = np.array([r["predicted_raw_norm"] for r in rows])
print("pooled Spearman raw:", round(sp(A, P), 4), " stored:", summ["criteria"]["pooled_spearman"])
Ab = np.array([r["balanced_norm"] for r in rows]); Pb = np.array([r["predicted_balanced_norm"] for r in rows])
print("pooled Spearman balanced (not reported):", round(sp(Ab, Pb), 4))

def curve_rows(fam, t):
    return [r for r in rows if r["family"] == fam and math.isclose(r["threshold"], t)]

errs = []; within = []; tab = []
for fam in FAMILIES:
    for t in THRESHOLDS:
        loc = curve_rows(fam, t)
        d = np.array([r["degree"] for r in loc]); a = np.array([r["raw_norm"] for r in loc]); p = np.array([r["predicted_raw_norm"] for r in loc])
        xa, ya, ea, na = binned_curve(d, a); xp, yp, _, _ = binned_curve(d, p)
        pa, pp = int(np.argmax(ya)), int(np.argmax(yp))
        errs.append(abs(pa - pp)); within.append(sp(a, p))
        # margin between top observed bin and runner-up in SE units
        order = np.argsort(ya)[::-1]
        gap = ya[order[0]] - ya[order[1]]; se = math.hypot(ea[order[0]], ea[order[1]])
        tab.append((fam, t, len(xa), pa, pp, abs(pa - pp), round(within[-1], 3), int(order[1]), round(gap, 3), round(gap / se, 2)))
print("\nfamily threshold nbins obs_peak pred_peak err within_rho runnerup gap gap/SE(node-based)")
for x in tab: print(*x)
errs = np.array(errs)
print("median abs peak error", np.median(errs), " within one:", int(np.sum(errs <= 1)), "/9")
w = np.array(within); print("within-curve rho median", round(float(np.median(w)), 3), "range", round(w.min(), 3), round(w.max(), 3))

# 1. Pooled Spearman decomposition: center within curve / within curve x replicate
import collections
def center(groups, v):
    out = v.copy()
    for g in set(groups):
        m = np.array([gg == g for gg in groups]); out[m] = v[m] - v[m].mean()
    return out
gc = [(r["family"], r["threshold"]) for r in rows]
gr = [(r["family"], r["threshold"], r["replicate"]) for r in rows]
print("\npooled rho after centering both within curve:", round(sp(center(gc, A), center(gc, P)), 4))
print("pooled rho after centering within curve x replicate:", round(sp(center(gr, A), center(gr, P)), 4))
# between-curve: correlation of 9 curve means and 45 group means
cm = collections.defaultdict(list)
for r in rows: cm[(r["family"], r["threshold"])].append((r["raw_norm"], r["predicted_raw_norm"]))
ma = np.array([np.mean([x[0] for x in v]) for v in cm.values()]); mp = np.array([np.mean([x[1] for x in v]) for v in cm.values()])
print("9 curve means obs:", np.round(ma, 3)); print("9 curve means pred:", np.round(mp, 3))
print("between-curve Spearman of means (n=9):", round(sp(ma, mp), 3), " pred-obs mean bias by curve:", np.round(mp - ma, 3))
# fraction of pooled rank variance explained by curve
from scipy.stats import rankdata
ra, rp = rankdata(A), rankdata(P)
def r2_between(rk):
    m = np.zeros_like(rk)
    for g in set(gc):
        s = np.array([gg == g for gg in gc]); m[s] = rk[s].mean()
    return np.var(m) / np.var(rk)
print("share of rank variance between curves: observed", round(r2_between(ra), 3), "predicted", round(r2_between(rp), 3))
# a predictor that knows only the curve (threshold x family) mean from... calibration threshold means
cal = load_jsonl(PV / "calibration.jsonl")
calmean = {t: np.mean([r["raw_norm"] for r in cal if math.isclose(r["threshold"], t)]) for t in THRESHOLDS}
Pthr = np.array([calmean[min(THRESHOLDS, key=lambda t: abs(t - r["threshold"]))] for r in rows])
print("calibration threshold means:", {t: round(v, 3) for t, v in calmean.items()})
print("pooled rho of threshold-only constant predictor (calibration mean per threshold; ties):", round(sp(A, Pthr), 4))
json.dump({"errors": errs.tolist(), "within": within}, open("prospective_repro_out.json", "w"))
