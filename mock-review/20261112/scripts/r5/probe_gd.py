import numpy as np
from scipy.special import expit
A = np.zeros((4, 4))
for i, j in [(0, 1), (0, 2), (1, 2), (0, 3)]:
    A[i, j] = A[j, i] = 1
d = A.sum(1); Pp = A / (4 * d[:, None]); Pn = np.tile(5 * d / 32, (4, 1))
xedge = np.log(8 / (5 * np.outer(d, d)))
Linf = float(np.sum((Pp * np.logaddexp(0, -xedge) + Pn * np.logaddexp(0, xedge))[A > 0]))
def L(M): return float(np.sum(Pp * np.logaddexp(0, -M) + Pn * np.logaddexp(0, M)))
for step in [2.0, 0.5]:
    r = np.random.default_rng(0)
    U = r.uniform(-0.5/4, 0.5/4, size=(4, 4)); V = np.zeros((4, 4))
    for t in range(1, 200001):
        M = U @ V.T
        G = -Pp * expit(-M) + Pn * expit(M)
        gU, gV = G @ V, G.T @ U
        U, V = U - step * gU, V - step * gV
        if t in (10, 100, 1000, 10000, 100000, 200000):
            print(step, t, f"gap={L(U@V.T)-Linf:.3e} |grad|={np.sqrt((gU**2).sum()+(gV**2).sum()):.2e} |U|={np.linalg.norm(U):.2f} edges={np.round((U@V.T)[A>0][:4],3)} target={np.round(xedge[A>0][:4],3)}")
