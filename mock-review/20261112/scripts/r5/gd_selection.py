"""R5: which normalized shape does factorized GD select on the window-one triangle+pendant objective?
Safeguarded GD (step halves if the loss increases), Gensim-like init (small U, zero V).
Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy,scipy python gd_selection.py
"""
import numpy as np
from scipy.special import expit
A = np.zeros((4, 4))
for i, j in [(0, 1), (0, 2), (1, 2), (0, 3)]:
    A[i, j] = A[j, i] = 1
d = A.sum(1); Pp = A / (4 * d[:, None]); Pn = np.tile(5 * d / 32, (4, 1))
xedge = np.log(8 / (5 * np.outer(d, d)))
Linf = float(np.sum((Pp * np.logaddexp(0, -xedge) + Pn * np.logaddexp(0, xedge))[A > 0]))
def L(M): return float(np.sum(Pp * np.logaddexp(0, -M) + Pn * np.logaddexp(0, M)))
def radii(M):
    w, Y = np.linalg.eigh(M @ M.T); r = np.sqrt(np.diag((Y * np.sqrt(np.clip(w, 0, None))) @ Y.T))
    return r / np.sqrt(np.mean(r ** 2))
for dim in (4, 3, 2):
    for seed in (0, 1, 2):
        r = np.random.default_rng(seed)
        U = r.uniform(-0.5 / dim, 0.5 / dim, size=(4, dim)); V = np.zeros((4, dim))
        step = 0.5; Lc = L(U @ V.T); line = []
        for t in range(1, 3_000_001):
            M = U @ V.T
            G = -Pp * expit(-M) + Pn * expit(M)
            Un, Vn = U - step * G @ V, V - step * G.T @ U
            Ln = L(Un @ Vn.T)
            if Ln > Lc + 1e-15:
                step *= 0.5
                continue
            U, V, Lc = Un, Vn, Ln
            if t in (10**4, 10**5, 10**6, 3*10**6):
                line.append(f"t={t:.0e}: gap={Lc-Linf:.1e} r~={np.round(radii(U@V.T),3)}")
        print(f"d={dim} seed={seed} final step={step:.3g}\n   " + "\n   ".join(line))
