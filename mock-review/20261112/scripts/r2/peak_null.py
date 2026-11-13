"""How discriminating are the frozen peak criteria? Trivial and chance predictors; argmax stability."""
from common import *
from itertools import product
rows = load_jsonl(PV / "outcomes.jsonl")
obs = {}; nb = {}
for fam in FAMILIES:
    for t in THRESHOLDS:
        loc = [r for r in rows if r["family"] == fam and math.isclose(r["threshold"], t)]
        d = np.array([r["degree"] for r in loc]); a = np.array([r["raw_norm"] for r in loc])
        x, y, _, _ = binned_curve(d, a); obs[(fam, t)] = int(np.argmax(y)); nb[(fam, t)] = len(x)
keys = list(obs)
print("observed peak bins:", [obs[k] for k in keys], "bins per curve:", [nb[k] for k in keys])
def crit(pred):
    e = np.array([abs(obs[k] - pred[k]) for k in keys]); return np.median(e), int(np.sum(e <= 1))
print("\nConstant-bin predictors (same bin index in every curve, clipped to available bins):")
for b in range(6):
    pred = {k: min(b, nb[k] - 1) for k in keys}; med, w = crit(pred)
    print(f"  always bin {b}: median err {med}, within one {w}/9, passes crit1&2: {med <= 1 and w >= 6}")
# constant 'second-highest-bin' and 'middle bin'
pred = {k: nb[k] // 2 for k in keys}; print("  middle bin (B//2):", crit(pred))
pred = {k: nb[k] - 2 for k in keys}; print("  second-to-last bin:", crit(pred))
# exact enumeration under uniform random bin guess, independent per curve
p_within = [len([b for b in range(nb[k]) if abs(b - obs[k]) <= 1]) / nb[k] for k in keys]
print("\nP(within one) per curve under uniform guess:", np.round(p_within, 3), "expected count", round(sum(p_within), 2))
# exact distribution of errors under uniform guessing
dist = {0: 1.0}
probs = []
pass12 = 0.0
for combo in product(*[range(nb[k]) for k in keys]):
    pass
# enumerate efficiently: 4^3*5^3*6^3 = 64*125*216 = 1.7M combos -> fine with numpy
grids = np.array(np.meshgrid(*[np.arange(nb[k]) for k in keys], indexing="ij")).reshape(len(keys), -1)
E = np.abs(grids - np.array([obs[k] for k in keys])[:, None])
med = np.median(E, axis=0); w = (E <= 1).sum(axis=0)
print("uniform random guess: P(median<=1)=%.3f P(>=6/9 within one)=%.3f P(both)=%.3f P(median==0)=%.4f P(>=8/9 within)=%.4f" % (
    np.mean(med <= 1), np.mean(w >= 6), np.mean((med <= 1) & (w >= 6)), np.mean(med == 0), np.mean(w >= 8)))
# calibration-transfer baseline: predict target peak at bin closest (in log degree) to calibration observed peak degree for that threshold
cal = load_jsonl(PV / "calibration.jsonl")
calpk = {}
for t in THRESHOLDS:
    loc = [r for r in cal if math.isclose(r["threshold"], t)]
    d = np.array([r["degree"] for r in loc]); a = np.array([r["raw_norm"] for r in loc])
    x, y, _, _ = binned_curve(d, a); calpk[t] = float(x[np.argmax(y)])
    print("calibration t=%g bins %s means %s peak deg %.2f" % (t, np.round(x, 2), np.round(y, 3), calpk[t]))
pred = {}
for fam in FAMILIES:
    for t in THRESHOLDS:
        loc = [r for r in rows if r["family"] == fam and math.isclose(r["threshold"], t)]
        d = np.array([r["degree"] for r in loc]); x, _, _, _ = binned_curve(d, np.zeros_like(d))
        pred[(fam, t)] = int(np.argmin(np.abs(np.log(x) - np.log(calpk[t]))))
print("calibration-peak transfer (no model, no target info except degree bins): pred bins", [pred[k] for k in keys], "->", crit(pred))
