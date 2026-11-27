# /// script
# dependencies = ["numpy", "scipy", "networkx", "numba"]
# ///
"""Rank-d truncated SVD of NetMF's window-1 target on the actual BA graph: no SGD at all."""
import sys
from pathlib import Path
import numpy as np, networkx as nx, scipy.sparse as sp
from scipy.sparse.linalg import svds
sys.path.insert(0, str(Path(__file__).parent))
from sgns import binned
g = nx.barabasi_albert_graph(10_000, 5, seed=0)
A = nx.to_scipy_sparse_array(g, nodelist=range(10_000), format="coo").astype(float)
deg = np.asarray(A.sum(1)).ravel(); vol = deg.sum()
vals = np.log(np.maximum(vol * A.data / (5 * deg[A.row] * deg[A.col]), 1.0))
M = sp.csr_matrix((vals, (A.row, A.col)), shape=A.shape)
print("full-row norm of target (sqrt sum sq)", np.round(binned(deg, np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()))[1], 2))
U, s, Vt = svds(M, k=600)
o = np.argsort(-s); U, s = U[:, o], s[o]
for d in (30, 100, 300, 600):
    bal = np.linalg.norm(U[:, :d] * np.sqrt(s[:d]), axis=1)
    x, y = binned(deg, bal)
    print(f"d={d:4d} bal", np.round(y, 2))
print("deg      ", np.round(x, 0))
