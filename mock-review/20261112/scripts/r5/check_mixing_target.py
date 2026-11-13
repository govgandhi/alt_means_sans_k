"""R5: free (unrestricted) SGNS target under the paper's context-mixing intervention (main.tex Eq. 70).
For lambda>0 the mixed rows have full support, so Prop. 11 gives a unique finite minimizer whose balanced
norms are computable without training. Equal exposure (beta=0): c_i = 1/n, q = sum_i c_i T_i, k = 5.
Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy,networkx python check_mixing_target.py
"""
import networkx as nx, numpy as np
def configuration_graph(nodes, attachment, seed):
    t = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    g = nx.Graph(nx.configuration_model([d for _, d in t.degree()], seed=seed)); g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        cs = [sorted(c) for c in nx.connected_components(g)]
        for a, b in zip(cs, cs[1:]): g.add_edge(a[0], b[0])
    return nx.convert_node_labels_to_integers(g, ordering="sorted")
def radii(X, rank=None):
    P, S, _ = np.linalg.svd(X)
    if rank: P, S = P[:, :rank], S[:rank]
    return np.sqrt((P ** 2) @ S)
def spearman(x, y):
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
edges = [3, 4, 5, 6, 8, 11, 16, 25, 1000]
for seed in range(2):
    A = nx.to_numpy_array(configuration_graph(600, 3, seed)); d = A.sum(1); n = len(d)
    T = A / d[:, None]
    for beta in (0.0, 1.0):
        c = d ** beta / (d ** beta).sum(); q = c @ T
        for lam in (1 / 3, 2 / 3, 1.0):
            Tl = (1 - lam) * T + lam * q[None, :]
            X = np.log(Tl / (5 * q[None, :]))
            for rank in (None, 24):
                r = radii(X, rank); rr = r / np.sqrt(np.mean(r ** 2))
                bins = [rr[(d >= lo) & (d < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]
                print(f"seed={seed} beta={beta:.0f} lambda={lam:.2f} rank={rank or 'full':>4}: Spearman={spearman(r, d):+.3f} "
                      f"RMS={np.sqrt(np.mean(r**2)):.3f} normalized bins={np.round(bins, 3)}")
