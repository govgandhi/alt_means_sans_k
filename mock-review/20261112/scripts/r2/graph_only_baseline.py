"""Graph-only comparator (no corpus statistics): log degree, log threshold, structural KL D(T_i||pi) from the graph.
Same ExtraTrees settings, calibration-only fit. Compare with frozen pairs+KL on all reported metrics."""
from common import *
import networkx as nx
from sklearn.ensemble import ExtraTreesRegressor
PAR = dict(n_estimators=400, min_samples_leaf=40, max_features=3, random_state=20260813, n_jobs=1)
def config_graph(n, m, seed):  # copy of scripts/simulate_exact_center_exposure.py:configuration_graph
    t = nx.barabasi_albert_graph(n, m, seed=seed); ds = [d for _, d in t.degree()]
    g = nx.Graph(nx.configuration_model(ds, seed=seed)); g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        comps = [sorted(c) for c in nx.connected_components(g)]
        for a, b in zip(comps, comps[1:]): g.add_edge(a[0], b[0])
    return nx.convert_node_labels_to_integers(g, ordering="sorted")
def lsc(g):
    g = nx.Graph(g); g.remove_edges_from(nx.selfloop_edges(g)); c = max(nx.connected_components(g), key=len)
    return nx.convert_node_labels_to_integers(g.subgraph(c).copy(), ordering="sorted")
def target_graph(fam, rep):
    if fam == "barabasi_albert": return lsc(nx.barabasi_albert_graph(600, 3, seed=rep))
    if fam == "powerlaw_cluster": return lsc(nx.powerlaw_cluster_graph(600, 3, 0.35, seed=rep))
    return lsc(lsc(nx.read_gml(WT / "are_angles_important/netscience/netscience.gml")))
def skl(g):
    deg = np.array([g.degree(i) for i in range(g.number_of_nodes())], float); vol = deg.sum()
    return deg, np.array([math.log(vol / deg[i]) - np.mean([math.log(deg[j]) for j in g.neighbors(i)]) for i in range(len(deg))])
cal = load_jsonl(PV / "calibration.jsonl"); tg = load_jsonl(PV / "outcomes.jsonl")
cache = {}
for s in range(5):
    deg, k = skl(config_graph(600, 3, s)); cache[("cal", s)] = (deg, k)
ok = all(np.array_equal(cache[("cal", r["replicate"])][0][[r["node"]]], [r["degree"]]) for r in cal[::97])
print("calibration graph degrees reproduced (spot check):", ok)
for fam in FAMILIES:
    for rep in sorted({r["replicate"] for r in tg if r["family"] == fam}):
        cache[(fam, rep)] = skl(target_graph(fam, rep))
def feats(rows, key):
    return np.column_stack([np.log([r["degree"] for r in rows]), np.log([r["threshold"] for r in rows]), [cache[key(r)][1][r["node"]] for r in rows]])
Xc = feats(cal, lambda r: ("cal", r["replicate"])); Xt = feats(tg, lambda r: (r["family"], r["replicate"]))
G = ExtraTreesRegressor(**PAR).fit(Xc, [r["raw_norm"] for r in cal]).predict(Xt)
A = np.array([r["raw_norm"] for r in tg]); P = np.array([r["predicted_raw_norm"] for r in tg])
groups = {}
for i, r in enumerate(tg): groups.setdefault((r["family"], r["threshold"]), []).append(np.int64(i))
def metrics(pred):
    errs, within = [], []
    for key, ix in groups.items():
        ix = np.array(ix); d = np.array([tg[i]["degree"] for i in ix])
        _, ya, _, _ = binned_curve(d, A[ix]); _, yp, _, _ = binned_curve(d, pred[ix])
        errs.append(abs(int(np.argmax(ya)) - int(np.argmax(yp)))); within.append(sp(A[ix], pred[ix]))
    e = np.array(errs); return np.median(e), int(np.sum(e <= 1)), round(sp(A, pred), 3), round(float(np.median(within)), 3), round(min(within), 3), round(max(within), 3), e.tolist()
print("frozen pairs+KL                : med err, within1, pooled, within median [min,max], errs =", metrics(P))
print("graph-only deg+thr+structural KL:", metrics(G))
