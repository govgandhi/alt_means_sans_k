"""Selection-corrected interior-maximum check: choose the peak bin on one part of the data, estimate the
peak-minus-endpoints contrast on the other part. (a) prospective observed curves: 2-vs-3 replicate splits;
(b) corrected Gensim no-filter/threshold curves per graph: random 50/50 node splits within bins."""
from common import *
from itertools import combinations
import csv
rows = load_jsonl(PV / "outcomes.jsonl")
print("(a) prospective observed raw-norm curves (pooled-curve bin edges), 10 splits of 5 replicates into 2 (select) + 3 (estimate)")
for fam in FAMILIES:
    for t in THRESHOLDS:
        loc = [r for r in rows if r["family"] == fam and math.isclose(r["threshold"], t)]
        d = np.array([r["degree"] for r in loc]); a = np.array([r["raw_norm"] for r in loc]); rep = np.array([r["replicate"] for r in loc])
        ld = np.log(d); edges = np.unique(np.quantile(ld, np.linspace(0, 1, 7))); lab = np.clip(np.digitize(ld, edges[1:-1], right=True), 0, len(edges) - 2)
        B = len(edges) - 1; reps = sorted(set(rep)); hits = 0; contrasts = []
        for sel in combinations(reps, 2):
            s = np.isin(rep, sel)
            ys = np.array([a[s & (lab == b)].mean() for b in range(B)]); ye = np.array([a[~s & (lab == b)].mean() for b in range(B)])
            p = int(np.argmax(ys))
            if 0 < p < B - 1:
                c = ye[p] - max(ye[0], ye[-1]); contrasts.append(c); hits += c > 0
            else:
                contrasts.append(float("nan"))
        cs = np.array(contrasts)
        print(f"   {fam[:4]} t={t:g}: interior peak selected in {np.sum(~np.isnan(cs))}/10 splits; held-out contrast >0 in {hits}/10; median held-out contrast {np.nanmedian(cs) if np.any(~np.isnan(cs)) else float('nan'):+.3f}")
print("\n(b) corrected Gensim (results/subsampling_origin/outcomes.csv), per graph, 200 random node splits within bins (8 requested quantile bins)")
R = list(csv.DictReader(open(WT / "results/subsampling_origin/outcomes.csv")))
rng = np.random.default_rng(7)
for t in ("0.0", "0.0003", "0.001", "0.003"):
    out = []
    for s in "01234":
        sel = [r for r in R if r["seed"] == s and r["threshold"] == t]
        d = np.array([float(r["degree"]) for r in sel]); a = np.array([float(r["raw_norm"]) for r in sel])
        ld = np.log(d); edges = np.unique(np.quantile(ld, np.linspace(0, 1, 9))); lab = np.clip(np.digitize(ld, edges[1:-1], right=True), 0, len(edges) - 2)
        occ = [b for b in range(len(edges) - 1) if np.any(lab == b)]
        pos = 0; inter = 0
        for _ in range(200):
            half = rng.random(len(a)) < 0.5
            ys = np.array([a[half & (lab == b)].mean() for b in occ]); ye = np.array([a[~half & (lab == b)].mean() for b in occ])
            p = int(np.argmax(ys))
            if 0 < p < len(occ) - 1:
                inter += 1; pos += (ye[p] - max(ye[0], ye[-1])) > 0
        out.append(f"s{s}: interior {inter}/200, held-out>ends {pos}/200")
    print(f"   t={t}: " + "; ".join(out))
