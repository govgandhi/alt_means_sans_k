"""R5: realized positive support after the paper's largest-remainder pair allocation (copied from
scripts/simulate_exact_center_exposure.py::exact_allocate and simulate_turnover_surface.py::allocate_pairs).
Question: does context mixing (lambda>0) give the full positive support assumed by Prop. 11's attained case?
Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy,networkx python check_allocation_support.py
"""
import networkx as nx, numpy as np
def configuration_graph(nodes, attachment, seed):
    t = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    g = nx.Graph(nx.configuration_model([d for _, d in t.degree()], seed=seed)); g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        cs = [sorted(c) for c in nx.connected_components(g)]
        for a, b in zip(cs, cs[1:]): g.add_edge(a[0], b[0])
    return nx.convert_node_labels_to_integers(g, ordering="sorted")
def exact_allocate(weights, total):
    expected = weights / weights.sum() * total
    allocated = np.floor(expected).astype(np.int64)
    remainder = int(total - allocated.sum())
    if remainder:
        order = np.argsort(expected - allocated, kind="stable"); allocated[order[-remainder:]] += 1
    return allocated
A = nx.to_numpy_array(configuration_graph(600, 3, 0)); d = A.sum(1); n = len(d); T = A / d[:, None]; pi = d / d.sum()
events = 240
for beta in (0.0, 1.0):
    cc = exact_allocate(pi ** beta, n * events); c = cc / cc.sum(); q = c @ T
    for lam in (0.0, 1/3, 2/3, 1.0):
        Tl = (1 - lam) * T + lam * q[None, :]
        C = np.array([exact_allocate(Tl[i], int(cc[i])) for i in range(n)])
        nonedge = (A == 0)
        pos = C > 0
        frac_full = pos.mean(); frac_nonedge = pos[nonedge].mean()
        # among nonedge pairs that received a count, mean context degree vs overall mean degree
        deg_recv = d[np.where(pos & nonedge)[1]].mean() if (pos & nonedge).any() else float('nan')
        print(f"beta={beta:.0f} lambda={lam:.2f}: realized positive support={frac_full:.3f} of n^2; nonedge support={frac_nonedge:.3f}; "
              f"mean degree of nonedge contexts that received pairs={deg_recv:.1f} (graph mean {d.mean():.1f}); "
              f"zero-positive pairs remain -> Prop.11 non-attained case: {not pos.all()}")
