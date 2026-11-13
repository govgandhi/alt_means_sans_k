"""A predictor with no node-level information beyond its degree-bin index passes all four frozen criteria?"""
from common import *
rows = load_jsonl(PV / "outcomes.jsonl"); cal = load_jsonl(PV / "calibration.jsonl")
calmean = {t: np.mean([r["raw_norm"] for r in cal if math.isclose(r["threshold"], t)]) for t in THRESHOLDS}
A = np.array([r["raw_norm"] for r in rows])
def evaluate(P):
    errs = []
    for fam in FAMILIES:
        for t in THRESHOLDS:
            ix = np.array([i for i, r in enumerate(rows) if r["family"] == fam and math.isclose(r["threshold"], t)])
            d = np.array([rows[i]["degree"] for i in ix])
            _, ya, _, _ = binned_curve(d, A[ix]); _, yp, _, _ = binned_curve(d, P[ix])
            errs.append(abs(int(np.argmax(ya)) - int(np.argmax(yp))))
    errs = np.array(errs)
    return float(np.median(errs)), int(np.sum(errs <= 1)), round(sp(A, P), 3)
# bin index of each node within its curve (target degrees known before training)
binidx = np.zeros(len(rows), int)
for fam in FAMILIES:
    for t in THRESHOLDS:
        ix = np.array([i for i, r in enumerate(rows) if r["family"] == fam and math.isclose(r["threshold"], t)])
        ld = np.log([rows[i]["degree"] for i in ix]); edges = np.unique(np.quantile(ld, np.linspace(0, 1, 7)))
        binidx[ix] = np.clip(np.digitize(ld, edges[1:-1], right=True), 0, len(edges) - 2)
base = np.array([calmean[min(THRESHOLDS, key=lambda t: abs(t - r["threshold"]))] for r in rows])
for b in (1, 2, 3):
    for eps in (0.01, 0.05):
        P = base + eps * (binidx == b)
        med, w, rho = evaluate(P)
        print(f"null predictor: calibration mean per threshold + {eps} bump at bin {b}: median err {med}, within one {w}/9, pooled rho {rho}; passes 1-3: {med <= 1 and w >= 6 and rho >= 0.5} (criterion 4 is predictor-independent and passed)")
