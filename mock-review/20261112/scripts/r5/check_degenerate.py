"""R5: degenerate block structures where 'direction depends only on community' is not injective.
K=2 bipartite DCSBM (B_aa=0), uniform window two: F = (W+W^2)/2 is rank one, so the stationary pair matrix
has rank one and both communities share one direction; the alpha=1 SGNS score has rank one with
community information only in context rows.
Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy python check_degenerate.py
"""
import numpy as np
rng = np.random.default_rng(2)
g = np.repeat([0, 1], [30, 40]); n = len(g); Z = np.eye(2)[g]
th = rng.uniform(0.2, 1, n); B = np.array([[0, 0.6], [0.6, 0]])
Om = np.outer(th, th) * (Z @ B @ Z.T); d = Om.sum(1); T = Om / d[:, None]; pi = d / d.sum()
P = (T + T @ T) / 2; J = pi[:, None] * P
sv = np.linalg.svd(J, compute_uv=False)
U = np.linalg.svd(J)[0][:, :1] * np.sqrt(sv[0])
dirs = np.sign(U[:, 0])
print(f"[INFO] bipartite, window 2: singular values of J = {np.round(sv[:3] / sv[0], 6)} -> rank {int((sv > 1e-12 * sv[0]).sum())}")
print(f"[{'PASS' if len(set(dirs)) == 1 else 'FAIL'}] both communities share a single direction (direction carries no community label)")
X = np.log(J / (5 * np.outer(pi, pi)))
svx = np.linalg.svd(X, compute_uv=False)
print(f"[INFO] alpha=1 SGNS score rank = {int((svx > 1e-10 * svx[0]).sum())}; center rows constant across ALL nodes: "
      f"{np.ptp(np.linalg.svd(X)[0][:, 0]) < 1e-10}")
