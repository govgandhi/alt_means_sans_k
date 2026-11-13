"""Does the frozen predictor track realized-corpus sampling noise?
(a) NetScience: same graph across 5 walk/optimizer replicates -> within-node, across-replicate co-variation.
(b) All families: regenerate graphs (networkx 3.6.1, same seeds as scripts/prospective_norm_turnover.py),
    compute structural one-step KL D(T_i||pi); split empirical KL into structural + residual; which predicts norm within degree class?"""
from common import *
import networkx as nx
rows = load_jsonl(PV / "outcomes.jsonl")
# (a)
print("(a) NetScience within-node deviations across 5 replicates")
for t in THRESHOLDS:
    loc = [r for r in rows if r["family"] == "netscience" and math.isclose(r["threshold"], t)]
    by = {}
    for r in loc: by.setdefault(r["node"], []).append(r)
    da, dp, dk, dn = [], [], [], []
    for node, rr in by.items():
        a = np.array([x["raw_norm"] for x in rr]); p = np.array([x["predicted_raw_norm"] for x in rr]); k = np.array([x["context_kl"] for x in rr]); n_ = np.log([x["realized_positive_pairs"] for x in rr])
        da += list(a - a.mean()); dp += list(p - p.mean()); dk += list(k - k.mean()); dn += list(n_ - n_.mean())
    da, dp, dk, dn = map(np.array, (da, dp, dk, dn))
    print(f"   t={t:g}: Spearman(within-node obs dev, pred dev) = {sp(da, dp):.3f}; obs dev vs KL dev {sp(da, dk):.3f}; vs log-pairs dev {sp(da, dn):.3f}; "
          f"share of obs-norm variance that is within-node across replicates: {np.var(da) / np.var([x['raw_norm'] for x in loc]):.3f}")
# (b)
def largest_simple_component(g):
    g = nx.Graph(g); g.remove_edges_from(nx.selfloop_edges(g))
    c = max(nx.connected_components(g), key=len); g = g.subgraph(c).copy()
    return nx.convert_node_labels_to_integers(g, ordering="sorted")
def graph(fam, rep):
    if fam == "barabasi_albert": g = nx.barabasi_albert_graph(600, 3, seed=rep)
    elif fam == "powerlaw_cluster": g = nx.powerlaw_cluster_graph(600, 3, 0.35, seed=rep)
    else: g = largest_simple_component(nx.read_gml(WT / "are_angles_important/netscience/netscience.gml"))
    return largest_simple_component(g)
print("\n(b) structural vs residual KL, networkx", nx.__version__)
ok = True; out = []
for fam in FAMILIES:
    for rep in sorted({r["replicate"] for r in rows if r["family"] == fam}):
        g = graph(fam, rep); deg = np.array([g.degree(i) for i in range(g.number_of_nodes())], float)
        vol = deg.sum()
        skl = np.array([math.log(vol / deg[i]) - np.mean([math.log(deg[j]) for j in g.neighbors(i)]) for i in range(len(deg))])
        for t in THRESHOLDS:
            loc = sorted([r for r in rows if r["family"] == fam and r["replicate"] == rep and math.isclose(r["threshold"], t)], key=lambda r: r["node"])
            sd = np.array([r["degree"] for r in loc])
            if not np.array_equal(sd, deg): ok = False
            for r in loc: out.append((fam, t, rep, r["degree"], skl[r["node"]], r["context_kl"], r["raw_norm"], r["predicted_raw_norm"]))
print("   regenerated degrees match stored degrees for all graphs:", ok)
import collections
res = collections.defaultdict(list)
for fam in FAMILIES:
    for t in THRESHOLDS:
        z = [o for o in out if o[0] == fam and math.isclose(o[1], t)]
        d = np.array([o[3] for o in z]); s = np.array([o[4] for o in z]); e = np.array([o[5] for o in z]); a = np.array([o[6] for o in z]); p = np.array([o[7] for o in z])
        # within exact-degree classes: regress empirical KL on structural KL (linear) -> residual
        r_s, r_e, r_res, r2 = [], [], [], []
        for dv in sorted(set(d)):
            m = d == dv
            if m.sum() < 50: continue
            beta = np.polyfit(s[m], e[m], 1); resid = e[m] - np.polyval(beta, s[m])
            r2.append(1 - np.var(resid) / np.var(e[m]))
            r_s.append(sp(a[m], s[m])); r_e.append(sp(a[m], e[m])); r_res.append(sp(a[m], resid))
        print(f"   {fam[:4]} t={t:g}: within-degree-class: R2(emp KL ~ struct KL) median {np.median(r2):.2f}; Spearman(norm, struct KL) {np.median(r_s):+.2f}; (norm, emp KL) {np.median(r_e):+.2f}; (norm, residual KL) {np.median(r_res):+.2f}")
