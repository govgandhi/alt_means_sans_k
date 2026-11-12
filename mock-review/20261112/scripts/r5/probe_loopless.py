"""Probe: how far is the loopless window-T pair matrix from the rank-K multiplicative form?"""
import numpy as np
rng = np.random.default_rng(1)
B = np.array([[0.9, 0.2, 0.1], [0.2, 0.7, 0.15], [0.1, 0.15, 0.8]])
def run(sizes, theta_kind, L):
    K = 3; g = np.repeat(np.arange(K), sizes); n = len(g)
    Z = np.zeros((n, K)); Z[np.arange(n), g] = 1
    th = (rng.pareto(2.5, n) + 0.3) if theta_kind == "pareto" else rng.uniform(0.5, 1.5, n)
    th = th / th.max()
    Om = np.outer(th, th) * (Z @ B @ Z.T)
    def J_of(Om):
        d = Om.sum(1); T = Om / d[:, None]; pi = d / d.sum()
        P = np.zeros_like(T); Tr = np.eye(n)
        for r in range(L):
            Tr = Tr @ T; P += Tr / L
        return pi[:, None] * P
    J = J_of(Om); J0 = J_of(Om - np.diag(np.diag(Om)))
    sv = np.linalg.svd(J, compute_uv=False); sv0 = np.linalg.svd(J0, compute_uv=False)
    E = np.linalg.norm(J0 - J, 2)
    # slope + direction spread of rank-K balanced factor
    P_, S_, Qt = np.linalg.svd(J0); U = P_[:, :3] * np.sqrt(S_[:3])
    x = np.log(th); y = np.log(np.linalg.norm(U, axis=1))
    xs = np.concatenate([x[g == a] - x[g == a].mean() for a in range(3)])
    ys = np.concatenate([y[g == a] - y[g == a].mean() for a in range(3)])
    dirs = U / np.linalg.norm(U, axis=1)[:, None]
    spread = max(np.abs(dirs[g == a] - dirs[g == a].mean(0)).max() for a in range(3))
    s = Z.T @ th
    print(f"n={n:5d} theta={theta_kind:7s} L={L:2d} sv(J)[:4]={sv[:4]/sv[0]} sv(J0)[2:5]/sv0[0]={sv0[2:5]/sv0[0]} "
          f"||J0-J||/sv(J)[2]={E/sv[2]:.3f} slope={xs@ys/(xs@xs):.4f} dirspread={spread:.3e} max th/s={np.max(th/s[g]):.3f}")
for kind in ["uniform", "pareto"]:
    for L in [1, 3, 10]:
        for sc in [1, 4, 16]:
            run([10*sc, 15*sc, 12*sc], kind, L)
