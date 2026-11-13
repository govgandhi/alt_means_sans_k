"""R5: balanced norms of the *realized-graph* SGNS target (no training).

Question: does the free (unrestricted) window-L target on a sparse configuration
graph already carry a decreasing norm-degree relation, and does the large-window
linearization  X ~ -log k 11^T + vol D^-1/2 Gamma Phi Gamma^T D^-1/2  predict it?
The graph generator is copied from scripts/simulate_norm_frequency.py (read-only).

Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy,scipy,networkx python check_realized_target.py
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def configuration_graph(nodes: int, attachment: int, seed: int) -> nx.Graph:
    template = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    degree_sequence = [degree for _, degree in template.degree()]
    multigraph = nx.configuration_model(degree_sequence, seed=seed)
    graph = nx.Graph(multigraph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if not nx.is_connected(graph):
        components = [sorted(c) for c in nx.connected_components(graph)]
        for left, right in zip(components, components[1:], strict=False):
            graph.add_edge(left[0], right[0])
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


def balanced_norms(X: np.ndarray, rank: int | None = None) -> np.ndarray:
    P, S, _ = np.linalg.svd(X)
    if rank is not None:
        P, S = P[:, :rank], S[:rank]
    return np.sqrt((P**2) @ S)


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def binned(d, r, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        out.append(r[m].mean() if m.any() else np.nan)
    return np.array(out)


k = 5
edges = np.array([3, 4, 5, 6, 8, 11, 16, 25, 1000])
for seed in range(3):
    G = configuration_graph(600, 3, seed)
    A = nx.to_numpy_array(G)
    d = A.sum(1)
    vol = d.sum()
    n = len(d)
    T = A / d[:, None]
    pi = d / vol
    Tr = np.eye(n)
    Ssum = np.zeros((n, n))
    for L in range(1, 11):
        Tr = Tr @ T
        Ssum += Tr
        if L not in (2, 5, 10):
            continue
        P = Ssum / L
        support = (P > 0).mean()
        if not (P > 0).all():
            print(
                f"seed={seed} L={L}: positive support fraction={support:.4f} -> free target infinite; skipped"
            )
            continue
        X = np.log(P / (k * pi[None, :]))  # free score; center share cancels
        for rank in (None, 24):
            r = balanced_norms(X, rank)
            rr = r / np.sqrt(np.mean(r**2))
            print(
                f"seed={seed} L={L} rank={'full' if rank is None else rank}: Spearman(norm,deg)={spearman(r, d):+.3f}; "
                f"RMS-normalized bin means={np.round(binned(d, rr, edges), 4)}"
            )
        # linearization check: symmetric congruence factor of the fluctuation part
        Sn = np.diag(d**-0.5) @ A @ np.diag(d**-0.5)
        lam, Gam = np.linalg.eigh(Sn)
        phi = np.array([np.mean([l**j for j in range(1, L + 1)]) for l in lam])
        E = vol * np.diag(d**-0.5) @ (Gam * phi) @ Gam.T @ np.diag(d**-0.5) - 1.0
        lin_err = np.abs(np.log1p(E) - E).max()
        # congruence factor row norms ~ vol/d_i * sum_k Gam_ik^2 |phi_k| (trivial vector removed)
        mask = ~np.isclose(lam, 1.0)
        cong = vol / d * ((Gam[:, mask] ** 2) @ np.abs(phi[mask]))
        print(
            f"   linearization max|log(1+E)-E|={lin_err:.3f}; Spearman(cong.-factor sq. norm, d)={spearman(cong, d):+.3f}; "
            f"slope log(cong) vs log d={np.polyfit(np.log(d), np.log(cong), 1)[0]:+.3f}"
        )
