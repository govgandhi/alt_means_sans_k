# /// script
# dependencies = ["numpy", "scipy", "networkx", "numba"]
# ///
"""Does the learned non-population part of W C^T align with the corpus's sampling noise?

D = N - n q^T: empirical window-1 pair counts minus their rank-1 expectation (the SGNS
gradient at the population optimum for exponent 1). Compare top-k left singular subspaces of
D and of the learned residual (W C^T minus its leading mode)."""
import sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds, LinearOperator
sys.path.insert(0, str(Path(__file__).parent))
from sgns import ba_corpus, binned, OUT

name, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50
z = np.load(OUT / f"{name}.npz")
W, C = z["W"].astype(np.float64), z["C"].astype(np.float64)
walks, deg, p = ba_corpus(0)
N = len(deg)
a, b = walks[:, :-1].ravel(), walks[:, 1:].ravel()
P = sp.coo_matrix((np.ones(2 * len(a)), (np.r_[a, b], np.r_[b, a])), shape=(N, N)).tocsr()
row = np.asarray(P.sum(1)).ravel()
q = np.bincount(walks.ravel(), minlength=N) / walks.size
D = LinearOperator((N, N), matvec=lambda v: P @ v - row * (q @ v),
                   rmatvec=lambda v: P.T @ v - q * (row @ v), dtype=float)
Ud, sd, _ = svds(D, k=k)
Ud = Ud[:, np.argsort(-sd)]
Qw, Rw = np.linalg.qr(W); Qc, Rc = np.linalg.qr(C)
U, s, Vt = np.linalg.svd(Rw @ Rc.T)
Um = Qw @ U  # left singular vectors of M = W C^T
pop = Um[:, 0]
Ur = Um[:, 1:k + 1]
print(f"{name}: leading mode vs uniform-ish |corr(pop, 1)| = {abs(pop.sum())/np.sqrt(N):.3f}")
print(f"top-{k} subspace overlap learned-residual vs D: {np.linalg.norm(Ud.T @ Ur)**2 / k:.3f}"
      f"  (random expectation {k/N:.4f})")
for kk in (5, 10, 20):
    print(f"  top-{kk}: {np.linalg.norm(Ud[:, :kk].T @ Um[:, 1:kk+1])**2 / kk:.3f}")
# which rows carry the leading noise modes?  energy of row i in top-k modes of D
e_D = (Ud ** 2).sum(1)
e_R = (Ur ** 2).sum(1)
x, yD = binned(deg, e_D); _, yR = binned(deg, e_R)
print("deg      ", np.round(x, 1))
print("D top-k row energy x N/k ", np.round(yD * N / k, 2))
print("learned  row energy x N/k", np.round(yR * N / k, 2))
