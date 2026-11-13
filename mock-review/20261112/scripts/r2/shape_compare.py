"""Binned-curve (shape) accuracy: frozen exposure-KL vs post hoc degree+threshold vs calibration-curve transfer."""
from common import *
from sklearn.ensemble import ExtraTreesRegressor
PAR = dict(n_estimators=400, min_samples_leaf=40, max_features=2, random_state=20260813, n_jobs=1)
cal = load_jsonl(PV / "calibration.jsonl"); tg = load_jsonl(PV / "outcomes.jsonl")
Xc = np.column_stack([np.log([r["degree"] for r in cal]), np.log([r["threshold"] for r in cal])])
Xt = np.column_stack([np.log([r["degree"] for r in tg]), np.log([r["threshold"] for r in tg])])
dt = ExtraTreesRegressor(**PAR).fit(Xc, [r["raw_norm"] for r in cal]).predict(Xt)
A = np.array([r["raw_norm"] for r in tg]); P = np.array([r["predicted_raw_norm"] for r in tg])
deg = np.array([r["degree"] for r in tg])
groups = {}
for i, r in enumerate(tg): groups.setdefault((r["family"], r["threshold"]), []).append(i)
print("curve | bins | centered binned RMSE: pairsKL, deg+thr | binned Pearson: pairsKL, deg+thr | level bias pairsKL, deg+thr")
agg = {"pk": [], "dt": []}
for key, ix in groups.items():
    ix = np.array(ix)
    _, yo, _, _ = binned_curve(deg[ix], A[ix]); _, yp, _, _ = binned_curve(deg[ix], P[ix]); _, yd, _, _ = binned_curve(deg[ix], dt[ix])
    c = lambda y: y - y.mean()
    r1 = np.sqrt(np.mean((c(yp) - c(yo)) ** 2)); r2 = np.sqrt(np.mean((c(yd) - c(yo)) ** 2))
    p1 = np.corrcoef(yp, yo)[0, 1]; p2 = np.corrcoef(yd, yo)[0, 1]
    agg["pk"].append(r1); agg["dt"].append(r2)
    print(f"{key[0][:4]} {key[1]:g} | {len(yo)} | {r1:.3f}, {r2:.3f} | {p1:.2f}, {p2:.2f} | {yp.mean()-yo.mean():+.3f}, {yd.mean()-yo.mean():+.3f}")
print("median centered binned RMSE: pairsKL %.3f  deg+thr %.3f; pairsKL better in %d/9 curves" % (np.median(agg["pk"]), np.median(agg["dt"]), sum(a < b for a, b in zip(agg["pk"], agg["dt"]))))
