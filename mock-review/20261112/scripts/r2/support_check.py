"""Covariate shift: are target (log pairs, KL) inside the calibration support, per threshold? Degree-range overlap."""
from common import *
cal = load_jsonl(PV / "calibration.jsonl"); tg = load_jsonl(PV / "outcomes.jsonl")
for t in THRESHOLDS:
    c = [r for r in cal if math.isclose(r["threshold"], t)]
    lp = np.log([r["realized_positive_pairs"] for r in c]); kl = np.array([r["context_kl"] for r in c])
    q = lambda v, p: np.percentile(v, p)
    for fam in FAMILIES:
        z = [r for r in tg if r["family"] == fam and math.isclose(r["threshold"], t)]
        zl = np.log([r["realized_positive_pairs"] for r in z]); zk = np.array([r["context_kl"] for r in z])
        outside_box = np.mean((zl < lp.min()) | (zl > lp.max()) | (zk < kl.min()) | (zk > kl.max()))
        # sparse-region proxy: fewer than 40 calibration rows within a small box around the point (leaf size)
        dl, dk = 0.05 * (lp.max() - lp.min()), 0.05 * (kl.max() - kl.min())
        sparse = np.mean([np.sum((np.abs(lp - a) < dl) & (np.abs(kl - b) < dk)) < 40 for a, b in zip(zl, zk)])
        print(f"t={t:g} {fam[:4]}: outside calibration box {100*outside_box:.1f}% ; in sparse calibration region (<40 cal rows in 5% box) {100*sparse:.1f}%")
cd = np.array([r["degree"] for r in cal if math.isclose(r["threshold"], 3e-4)])
print("calibration degree: min %d, P(d<=2) %.3f, max %d" % (cd.min(), np.mean(cd <= 2), cd.max()))
nd = np.array([r["degree"] for r in tg if r["family"] == "netscience" and math.isclose(r["threshold"], 3e-4)])
print("NetScience degree: min %d, P(d<=2) %.3f, max %d" % (nd.min(), np.mean(nd <= 2), nd.max()))
