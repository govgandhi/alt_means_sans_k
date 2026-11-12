"""Refit ExtraTrees baselines (frozen settings) on calibration; evaluate on original targets.
Adds: oracle degree-function ceiling, within-degree-class ranking, partial rank correlation given degree."""
from common import *
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
print("sklearn", sklearn.__version__)
PAR = dict(n_estimators=400, min_samples_leaf=40, max_features=2, random_state=20260813, n_jobs=1)
cal = load_jsonl(PV / "calibration.jsonl"); tg = load_jsonl(PV / "outcomes.jsonl")
def X(rows, kind):
    cols = {"pairs_kl": ("realized_positive_pairs", "context_kl"), "degree_threshold": ("degree", "threshold"),
            "pairs": ("realized_positive_pairs",), "kl": ("context_kl",), "degree": ("degree",),
            "degree_threshold_pairs": ("degree", "threshold", "realized_positive_pairs"),
            "tokens": ("retained_tokens",), "degree_threshold_kl": ("degree", "threshold", "context_kl")}[kind]
    return np.column_stack([np.asarray([r[c] for r in rows], float) if c == "context_kl" else np.log([r[c] for r in rows]) for c in cols])
y = np.array([r["raw_norm"] for r in cal]); A = np.array([r["raw_norm"] for r in tg])
preds = {}
for kind in ("pairs_kl", "degree_threshold", "degree", "degree_threshold_pairs", "degree_threshold_kl", "tokens"):
    par = dict(PAR); par["max_features"] = min(2, X(cal, kind).shape[1]) if kind != "degree_threshold_pairs" and kind != "degree_threshold_kl" else 2
    m = ExtraTreesRegressor(**par).fit(X(cal, kind), y); preds[kind] = m.predict(X(tg, kind))
saved = np.array([r["predicted_raw_norm"] for r in tg])
print("refit pairs_kl max |diff| vs frozen:", float(np.max(np.abs(preds["pairs_kl"] - saved))))
groups = {}
for i, r in enumerate(tg): groups.setdefault((r["family"], r["threshold"]), []).append(i)
deg = np.array([r["degree"] for r in tg]); rep = np.array([r["replicate"] for r in tg])
def within(p):
    return [sp(A[ix], p[ix]) for ix in groups.values()]
for k, p in preds.items():
    w = within(p); print(f"{k:24s} pooled {sp(A, p):.3f} within median {np.median(w):.3f} range [{min(w):.3f},{max(w):.3f}]")
# oracle degree function, cross-fitted across replicates (in each curve: mean observed norm of same exact degree in the other 4 replicates)
orc = np.zeros_like(A); orc_in = np.zeros_like(A)
for key, ix in groups.items():
    ix = np.array(ix)
    for i_rep in set(rep[ix]):
        tr = ix[rep[ix] != i_rep]; te = ix[rep[ix] == i_rep]
        mp = {}
        for d_ in set(deg[tr]): mp[d_] = A[tr][deg[tr] == d_].mean()
        # nearest-degree fallback
        dk = np.array(sorted(mp))
        orc[te] = [mp.get(d_, mp[dk[np.argmin(np.abs(np.log(dk) - np.log(d_)))]]) for d_ in deg[te]]
    for d_ in set(deg[ix]):
        s = ix[deg[ix] == d_]; orc_in[s] = A[s].mean()
w = within(orc); print(f"{'ORACLE degree (cross-fit)':24s} within median {np.median(w):.3f} range [{min(w):.3f},{max(w):.3f}]")
w = within(orc_in); print(f"{'ORACLE degree (in-sample)':24s} within median {np.median(w):.3f} range [{min(w):.3f},{max(w):.3f}]")
# Within exact-degree classes: does pairs+KL rank nodes of the SAME degree?
print("\nSpearman(pred, obs) within exact-degree classes (classes with >=50 node-replicates), pairs+KL frozen predictions:")
out = []
for key, ix in groups.items():
    ix = np.array(ix); vals = []; ns = []
    for d_ in sorted(set(deg[ix])):
        s = ix[deg[ix] == d_]
        if len(s) >= 50: vals.append(sp(A[s], saved[s])); ns.append(len(s))
    out.append((key, np.round(vals[:4], 2).tolist(), ns[:4], round(float(np.average(vals, weights=ns)), 3)))
for o in out: print("  ", o)
# partial Spearman given log degree (rank residuals on a smooth function of degree within curve)
from scipy.stats import rankdata
def partial_rank(a, p, d):
    ra, rp = rankdata(a), rankdata(p)
    # remove per-degree-class mean rank (nonparametric control for degree)
    for d_ in set(d):
        s = d == d_; ra[s] -= ra[s].mean(); rp[s] -= rp[s].mean()
    return float(np.corrcoef(ra, rp)[0, 1])
pr = [partial_rank(A[np.array(ix)], saved[np.array(ix)], deg[np.array(ix)]) for ix in groups.values()]
print("\nwithin-curve rank corr after removing exact-degree class means: median %.3f range [%.3f, %.3f]" % (np.median(pr), min(pr), max(pr)))
# share of within-curve observed-rank variance explained by exact degree class
sh = []
for ix in groups.values():
    ix = np.array(ix); ra = rankdata(A[ix]); m = np.zeros_like(ra)
    for d_ in set(deg[ix]): s = deg[ix] == d_; m[s] = ra[s].mean()
    sh.append(np.var(m) / np.var(ra))
print("share of within-curve observed rank variance between exact-degree classes: median %.3f range [%.3f, %.3f]" % (np.median(sh), min(sh), max(sh)))
