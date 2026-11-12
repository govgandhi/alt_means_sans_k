"""R5: at the unique optimum of full-support objectives, is the balanced norm increasing in context KL
(the sign assumed by the response model, main.tex l.940-953)?  Intended mixed targets (lambda=1/3, 2/3, beta=0)
and window-10 fixed-noise target (c uniform, q = pi), configuration graph seed 0.
Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy,networkx python check_kl_vs_target_norm.py
"""
import networkx as nx, numpy as np
def configuration_graph(nodes, attachment, seed):
    t = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    g = nx.Graph(nx.configuration_model([d for _, d in t.degree()], seed=seed)); g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        cs = [sorted(c) for c in nx.connected_components(g)]
        for a, b in zip(cs, cs[1:]): g.add_edge(a[0], b[0])
    return nx.convert_node_labels_to_integers(g, ordering="sorted")
def radii(X):
    P, S, _ = np.linalg.svd(X); return np.sqrt((P ** 2) @ S)
def sp(x, y): return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])
A = nx.to_numpy_array(configuration_graph(600, 3, 0)); d = A.sum(1); n = len(d); T = A / d[:, None]; pi = d / d.sum()
q = np.ones(n) / n @ T
for lam in (1/3, 2/3):
    P = (1 - lam) * T + lam * q[None, :]
    KL = np.sum(P * np.log(P / q[None, :]), axis=1)
    r = radii(np.log(P / (5 * q[None, :])))
    print(f"mixed lambda={lam:.2f}: Spearman(KL,deg)={sp(KL, d):+.3f}  Spearman(norm,deg)={sp(r, d):+.3f}  Spearman(norm,KL)={sp(r, KL):+.3f}")
Tr = np.eye(n); S = np.zeros((n, n))
for L in range(1, 11):
    Tr = Tr @ T; S += Tr
P = S / 10
KL = np.sum(P * np.log(P / pi[None, :]), axis=1)
r = radii(np.log(P / (5 * pi[None, :])))
print(f"window 10, q=pi: Spearman(KL,deg)={sp(KL, d):+.3f}  Spearman(norm,deg)={sp(r, d):+.3f}  Spearman(norm,KL)={sp(r, KL):+.3f}")
