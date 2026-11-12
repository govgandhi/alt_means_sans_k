"""R5: Propositions prop:loss-norm-boundary and prop:normalized-shapes (main.tex l.691-747).

(1) Reproduce the triangle+pendant construction from the main-text constants.
(2) Show its freedom lives in the zero-positive SELF-pair scores M_ii and needs d = n.
(3) Window two on the same graph has full support -> unique balanced profile (construction fails).
(4) What does gradient descent select?  Factorized GD from Gensim-like init on the window-one
    objective, at d = 4 and d = 2, tracked over iterations; compared with the min-nuclear-norm
    max-margin direction on the zero-positive set (implicit-bias benchmark).
(5) The construction generalizes to the n=600 experimental graph at window one (full rank only).

Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy,scipy,cvxpy,networkx python check_props.py
"""

from __future__ import annotations

import cvxpy as cp
import networkx as nx
import numpy as np
from scipy.special import expit

rng = np.random.default_rng(3)


def report(tag, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f" :: {detail}" if detail else ""))


def info(tag, detail):
    print(f"[INFO] {tag} :: {detail}")


def psd_sqrt_diag(M):
    w, Y = np.linalg.eigh(M @ M.T)
    return np.diag((Y * np.sqrt(np.clip(w, 0, None))) @ Y.T)


def loss(M, Pp, Pn):
    return float(np.sum(Pp * np.logaddexp(0, -M) + Pn * np.logaddexp(0, M)))


# ---------------------------------------------------------------- weights (main.tex l.736)
E = [(0, 1), (0, 2), (1, 2), (0, 3)]
A = np.zeros((4, 4))
for i, j in E:
    A[i, j] = A[j, i] = 1
d = A.sum(1)
report("degrees (3,2,2,1)", list(d) == [3, 2, 2, 1])
Pp = A / (4 * d[:, None])
Pn = np.tile(5 * d / 32, (4, 1))
zero = Pp == 0
B0 = Pn[zero].sum()
report("zero-positive negative mass B0 = 35/16", np.isclose(B0, 35 / 16))
xedge = np.log(8 / (5 * np.outer(d, d)))
report(
    "edge optimum log[8/(5 d_i d_j)]",
    np.allclose(np.log(Pp[A > 0] / Pn[A > 0]), xedge[A > 0]),
)
Linf = float(
    np.sum((Pp * np.logaddexp(0, -xedge) + Pn * np.logaddexp(0, xedge))[A > 0])
)

profiles = {
    "rising": np.array([3, 2, 2, 1.0]),
    "falling": np.array([1, 2, 2, 3.0]),
    "peaked": np.array([1, 3, 3, 1.0]),
}
for delta in [1e-2, 1e-6, 1e-10]:
    s = max(0.0, np.log(B0 / delta))
    base = np.where(A > 0, xedge, -s)
    np.fill_diagonal(base, 0)
    R = np.abs(base).sum(1)
    al = {k: h**2 / (0.25 * np.sum(h**2)) for k, h in profiles.items()}
    D = max(np.max(np.maximum(s, R + 1) / a) for a in al.values())
    rows = []
    for name, a in al.items():
        M = base.copy()
        np.fill_diagonal(M, -D * a)
        neg_def = np.linalg.eigvalsh(M).max() < 0
        r2 = psd_sqrt_diag(M)
        gap = loss(M, Pp, Pn) - Linf
        rho = np.sqrt(r2.mean())
        rows.append(
            (
                name,
                neg_def,
                np.allclose(r2, D * a),
                np.isclose(rho, np.sqrt(D)),
                0 < gap <= delta,
                np.linalg.matrix_rank(M),
                np.round(np.sqrt(r2) / rho, 4),
            )
        )
        # balanced factors U=(-M)^{1/2}, V=-U
        w, Y = np.linalg.eigh(-M)
        U = (Y * np.sqrt(w)) @ Y.T
        V = -U
        assert np.allclose(U @ V.T, M) and np.allclose(U.T @ U, V.T @ V)
    ok = all(r[1] and r[2] and r[3] and r[4] for r in rows)
    report(
        f"construction delta={delta:g}: neg.def., r_i^2=D alpha_i, rho=sqrt(D), 0<gap<=delta",
        ok,
        f"D={D:.3f}, s={s:.3f}, ranks={[r[5] for r in rows]}, profiles={[list(r[6]) for r in rows]}",
    )
    # where does the norm live?  fraction of sum r_i^2 carried by -M_ii = all of it (since (MM^T)^1/2 = -M)
    info(
        f"delta={delta:g}",
        f"self-pair scores u_i.v_i = M_ii = {np.round(-D * al['peaked'], 2)} (peaked); "
        f"edge scores in [{xedge[A > 0].min():.2f},{xedge[A > 0].max():.2f}]; V=-U (antiparallel center/context)",
    )

# ---------------------------------------------------------------- (3) window two
T = A / d[:, None]
P2 = (T + T @ T) / 2
Pp2 = P2 / 4
report("window two: full positive support on this graph", (Pp2 > 0).all())
M2 = np.log(Pp2 / Pn)
r2 = np.sqrt(psd_sqrt_diag(M2))
info(
    "window two unique balanced profile (normalized, by node deg 3,2,2,1)",
    f"{np.round(r2 / np.sqrt(np.mean(r2**2)), 4)}",
)


# ---------------------------------------------------------------- (4) what GD selects
def gd(dim, iters, step=2.0, seed=0, checkpoints=()):
    r = np.random.default_rng(seed)
    U = r.uniform(-0.5 / dim, 0.5 / dim, size=(4, dim))
    V = np.zeros((4, dim))
    out = {}
    for t in range(1, iters + 1):
        M = U @ V.T
        G = -Pp * expit(-M) + Pn * expit(M)
        U, V = U - step * G @ V, V - step * G.T @ U
        if t in checkpoints:
            M = U @ V.T
            rr = np.sqrt(psd_sqrt_diag(M))
            out[t] = (
                loss(M, Pp, Pn) - Linf,
                np.round(rr / np.sqrt(np.mean(rr**2)), 4),
                np.round(np.diag(M), 2),
            )
    return out


cps = (10**3, 10**4, 10**5, 10**6)
for dim in (4, 2):
    for seed in (0, 1):
        res = gd(dim, 10**6, seed=seed, checkpoints=cps)
        for t, (gap, prof, diag) in res.items():
            info(
                f"GD d={dim} seed={seed} iter={t:>7d}",
                f"gap={gap:.2e} normalized radii={prof} diag(M)={diag}",
            )

# min-nuclear-norm max-margin direction on the zero-positive set (edges fixed at 0)
Dl = cp.Variable((4, 4))
cons = [Dl[i, j] == 0 for i in range(4) for j in range(4) if A[i, j] > 0]
cons += [Dl[i, j] <= -1 for i in range(4) for j in range(4) if zero[i, j]]
prob = cp.Problem(cp.Minimize(cp.normNuc(Dl)), cons)
prob.solve()
Dv = Dl.value
rr = np.sqrt(psd_sqrt_diag(Dv))
info(
    "min-nuclear-norm margin direction",
    f"||Delta||_*={prob.value:.4f}; normalized radii={np.round(rr / np.sqrt(np.mean(rr**2)), 4)}; "
    f"Delta=\n{np.round(Dv, 3)}",
)


# ---------------------------------------------------------------- (5) generalization to the experimental graph
def configuration_graph(nodes, attachment, seed):
    template = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    multigraph = nx.configuration_model([dd for _, dd in template.degree()], seed=seed)
    graph = nx.Graph(multigraph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if not nx.is_connected(graph):
        comps = [sorted(c) for c in nx.connected_components(graph)]
        for l_, r_ in zip(comps, comps[1:]):
            graph.add_edge(l_[0], r_[0])
    return nx.convert_node_labels_to_integers(graph, ordering="sorted")


Gx = configuration_graph(600, 3, 0)
Ax = nx.to_numpy_array(Gx)
dx = Ax.sum(1)
n = len(dx)
vol = dx.sum()
Ppx = Ax / (n * dx[:, None])
Pnx = np.tile(5 * dx / vol / n, (n, 1))
xe = np.log(vol / (5 * np.outer(dx, dx)))
delta = 1e-6
B0x = Pnx[Ppx == 0].sum()
s = np.log(B0x / delta)
base = np.where(Ax > 0, xe, -s)
np.fill_diagonal(base, 0)
Rx = np.abs(base).sum(1)
Linfx = float(np.sum((Ppx * np.logaddexp(0, -xe) + Pnx * np.logaddexp(0, xe))[Ax > 0]))
logd = np.log(dx)
targets = {
    "rising": dx**0.5,
    "falling": dx**-0.5,
    "peaked": np.exp(-((logd - np.log(8)) ** 2)),
}
al = {k: h**2 / np.mean(h**2) for k, h in targets.items()}
Dg = max(np.max(np.maximum(s, Rx + 1) / a) for a in al.values())
for name, a in al.items():
    M = base.copy()
    np.fill_diagonal(M, -Dg * a)
    nd = np.linalg.eigvalsh(M).max() < 0
    r = np.sqrt(psd_sqrt_diag(M))
    rt = r / np.sqrt(np.mean(r**2))
    gap = loss(M, Ppx, Pnx) - Linfx
    corr = np.corrcoef(rt, np.sqrt(a))[0, 1]
    report(
        f"n=600 config graph, window one, full rank: '{name}' profile within 1e-6 of infimum",
        nd and 0 < gap <= delta * (1 + 1e-9) and corr > 0.999999,
        f"gap={gap:.2e}, corr(normalized radius, target)={corr:.7f}",
    )
