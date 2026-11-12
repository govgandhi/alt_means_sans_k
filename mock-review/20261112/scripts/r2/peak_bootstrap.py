"""Replicate-cluster bootstrap of the frozen criteria and binning sensitivity (raw norm, original arm)."""
from common import *
rows = load_jsonl(PV / "outcomes.jsonl")
REPS = {f: sorted({r["replicate"] for r in rows if r["family"] == f}) for f in FAMILIES}
by = {}
for r in rows:
    by.setdefault((r["family"], r["threshold"], r["replicate"]), []).append(r)
arr = {k: (np.array([r["degree"] for r in v]), np.array([r["raw_norm"] for r in v]), np.array([r["predicted_raw_norm"] for r in v])) for k, v in by.items()}
def stats(sample, bins=6, binfun=None):
    errs, within, A, P = [], [], [], []
    for fam in FAMILIES:
        for t in THRESHOLDS:
            parts = [arr[(fam, t, rep)] for rep in sample[fam]]
            d = np.concatenate([p[0] for p in parts]); a = np.concatenate([p[1] for p in parts]); p_ = np.concatenate([p[2] for p in parts])
            if binfun is None:
                _, ya, _, _ = binned_curve(d, a, bins); _, yp, _, _ = binned_curve(d, p_, bins)
            else:
                ya, yp = binfun(d, a), binfun(d, p_)
            errs.append(abs(int(np.argmax(ya)) - int(np.argmax(yp)))); within.append(sp(a, p_)); A.append(a); P.append(p_)
    errs = np.array(errs)
    return float(np.median(errs)), int(np.sum(errs <= 1)), int(np.sum(errs == 0)), float(np.median(within)), sp(np.concatenate(A), np.concatenate(P)), errs
full = {f: REPS[f] for f in FAMILIES}
print("full:", stats(full)[:5])
rng = np.random.default_rng(1)
B = 2000
res = []
for b in range(B):
    s = {f: list(rng.choice(REPS[f], size=len(REPS[f]), replace=True)) for f in FAMILIES}
    res.append(stats(s))
med = np.array([r[0] for r in res]); w1 = np.array([r[1] for r in res]); ex = np.array([r[2] for r in res]); wm = np.array([r[3] for r in res]); pool = np.array([r[4] for r in res])
E = np.array([r[5] for r in res])
print(f"bootstrap B={B} (resample replicates within family, thresholds kept together)")
print("  median peak error: P(=0)=%.3f P(<=1)=%.3f" % (np.mean(med == 0), np.mean(med <= 1)))
print("  peaks within one: mean %.2f, 2.5-97.5%%: %s, P(>=8)=%.3f P(>=6)=%.3f" % (w1.mean(), np.percentile(w1, [2.5, 97.5]), np.mean(w1 >= 8), np.mean(w1 >= 6)))
print("  exact hits: mean %.2f, 95%%: %s" % (ex.mean(), np.percentile(ex, [2.5, 97.5])))
print("  within-curve median rho: %.3f [%.3f, %.3f]" % (np.median(wm), *np.percentile(wm, [2.5, 97.5])))
print("  pooled rho: %.3f [%.3f, %.3f]" % (np.median(pool), *np.percentile(pool, [2.5, 97.5])))
names = [(f[:2], t) for f in FAMILIES for t in THRESHOLDS]
print("  per-curve P(error==reported):", [(n, round(float(np.mean(E[:, i] == stats(full)[5][i])), 2)) for i, n in enumerate(names)])
# leave-one-replicate-out
print("\nleave-one-replicate-out (per family):")
for f in FAMILIES:
    for rep in REPS[f]:
        s = dict(full); s[f] = [x for x in REPS[f] if x != rep]
        st = stats(s); print("  drop", f, rep, "median", st[0], "within1", st[1], "exact", st[2], "errs", st[5].tolist())
# binning sensitivity
print("\nbinning sensitivity (quantile bins requested):")
for nbins in (3, 4, 5, 6, 7, 8, 10, 12):
    st = stats(full, bins=nbins); print("  bins=%d: median %.1f within1 %d/9 exact %d/9 errs %s" % (nbins, st[0], st[1], st[2], st[5].tolist()))
def eqwidth(k):
    def f(d, v):
        ld = np.log(d); edges = np.linspace(ld.min(), ld.max(), k + 1)
        lab = np.clip(np.digitize(ld, edges[1:-1], right=True), 0, k - 1)
        return np.array([v[lab == i].mean() for i in range(k) if np.any(lab == i)])
    return f
for k in (4, 6, 8):
    st = stats(full, binfun=eqwidth(k)); print("  equal-width log bins k=%d: median %.1f within1 %d/9 exact %d/9 errs %s" % (k, st[0], st[1], st[2], st[5].tolist()))
