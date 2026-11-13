from common import *
rows = load_jsonl(PV / "outcomes.jsonl")
import collections
for fam in FAMILIES:
    loc = [r for r in rows if r["family"] == fam and math.isclose(r["threshold"], 3e-4)]
    d = np.array([r["degree"] for r in loc])
    x, y, e, n = binned_curve(d, np.zeros_like(d))
    c = collections.Counter(d.astype(int))
    print(fam, "nodes(pooled)", len(d), "bins", len(x), "bin centers", np.round(x, 2), "counts", n)
    print("   min/max deg", d.min(), d.max(), "frac deg<=5", np.mean(d <= 5).round(3), "top degree counts", sorted(c.items())[:8])
