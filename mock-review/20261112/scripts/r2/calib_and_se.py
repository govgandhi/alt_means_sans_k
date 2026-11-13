"""(1) Reproduce calibration leave-one-seed-out R^2/Spearman; within-threshold version.
(2) Seed-level vs node-level SE for Fig. 6 binned curves. (3) Per-replicate peak errors (45 curves)."""
from common import *
from sklearn.ensemble import ExtraTreesRegressor
PAR = dict(n_estimators=400, min_samples_leaf=40, max_features=2, random_state=20260813, n_jobs=1)
cal = load_jsonl(PV / "calibration.jsonl")
X = np.column_stack([np.log([r["realized_positive_pairs"] for r in cal]), [r["context_kl"] for r in cal]])
y = np.array([r["raw_norm"] for r in cal]); seed = np.array([r["replicate"] for r in cal]); thr = np.array([r["threshold"] for r in cal])
pred = np.zeros_like(y)
for s in range(5):
    m = ExtraTreesRegressor(**PAR).fit(X[seed != s], y[seed != s]); pred[seed == s] = m.predict(X[seed == s])
res = y - pred
print("calibration LOSO raw: R2 %.4f Spearman %.4f (claimed 0.828 / 0.932)" % (1 - np.sum(res**2) / np.sum((y - y.mean())**2), sp(y, pred)))
for t in THRESHOLDS:
    s_ = np.isclose(thr, t); print("   within t=%g: R2 %.3f Spearman %.3f" % (t, 1 - np.sum(res[s_]**2) / np.sum((y[s_] - y[s_].mean())**2), sp(y[s_], pred[s_])))
# (2) SEs
rows = load_jsonl(PV / "outcomes.jsonl")
print("\nFig. 6 error bars: node-level SE (as coded) vs SE across 5 replicate means (as captioned)")
for fam in FAMILIES:
    for t in THRESHOLDS:
        loc = [r for r in rows if r["family"] == fam and math.isclose(r["threshold"], t)]
        d = np.array([r["degree"] for r in loc]); a = np.array([r["raw_norm"] for r in loc]); rep = np.array([r["replicate"] for r in loc])
        x, ybin, enode, n = binned_curve(d, a)
        # replicate-level: same pooled edges, mean within replicate then SD/sqrt(5)
        ld = np.log(d); edges = np.unique(np.quantile(ld, np.linspace(0, 1, 7)))
        lab = np.clip(np.digitize(ld, edges[1:-1], right=True), 0, len(edges) - 2)
        eseed = []
        for b in sorted(set(lab)):
            ms = [a[(lab == b) & (rep == r)].mean() for r in sorted(set(rep)) if np.any((lab == b) & (rep == r))]
            eseed.append(np.std(ms, ddof=1) / np.sqrt(len(ms)))
        eseed = np.array(eseed)
        print(f"  {fam[:4]} {t:g}: node SE {np.round(enode,3)} seed SE {np.round(eseed,3)} ratio {np.round(eseed/enode,2)}")
# (3) per-replicate peak errors using pooled-curve bin edges
print("\nPer-replicate peak errors (pooled bin edges), raw norm:")
allerr = []
for fam in FAMILIES:
    for t in THRESHOLDS:
        loc = [r for r in rows if r["family"] == fam and math.isclose(r["threshold"], t)]
        d = np.array([r["degree"] for r in loc]); a = np.array([r["raw_norm"] for r in loc]); p = np.array([r["predicted_raw_norm"] for r in loc]); rep = np.array([r["replicate"] for r in loc])
        ld = np.log(d); edges = np.unique(np.quantile(ld, np.linspace(0, 1, 7)))
        lab = np.clip(np.digitize(ld, edges[1:-1], right=True), 0, len(edges) - 2)
        errs = []; obs_pk = []
        for r_ in sorted(set(rep)):
            s_ = rep == r_
            ya = [a[s_ & (lab == b)].mean() for b in range(len(edges) - 1) if np.any(s_ & (lab == b))]
            yp = [p[s_ & (lab == b)].mean() for b in range(len(edges) - 1) if np.any(s_ & (lab == b))]
            errs.append(abs(int(np.argmax(ya)) - int(np.argmax(yp)))); obs_pk.append(int(np.argmax(ya)))
        allerr += errs
        print(f"  {fam[:4]} {t:g}: observed peak bins by replicate {obs_pk}; errors {errs}")
allerr = np.array(allerr)
print("45 replicate curves: median error %.1f, within one %d/45 (%.2f), exact %d/45" % (np.median(allerr), np.sum(allerr <= 1), np.mean(allerr <= 1), np.sum(allerr == 0)))
