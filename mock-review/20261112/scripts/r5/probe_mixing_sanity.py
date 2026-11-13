import networkx as nx, numpy as np
from check_mixing_target import configuration_graph, spearman
A = nx.to_numpy_array(configuration_graph(600, 3, 0)); d = A.sum(1); n = len(d); T = A / d[:, None]
c = np.ones(n) / n; q = c @ T
for lam in (1/3, 2/3):
    X = np.log(((1 - lam) * T + lam * q[None, :]) / (5 * q[None, :]))
    w, Y = np.linalg.eigh(X @ X.T); r = np.sqrt(np.diag((Y * np.sqrt(np.clip(w, 0, None))) @ Y.T))
    Xc = X - np.log(lam / 5)  # remove the constant rank-one part
    wc, Yc = np.linalg.eigh(Xc @ Xc.T); rc = np.sqrt(np.diag((Yc * np.sqrt(np.clip(wc, 0, None))) @ Yc.T))
    print(f"lambda={lam:.2f}: eigh-balanced Spearman={spearman(r, d):+.3f}; row-norm Spearman={spearman(np.linalg.norm(X, axis=1), d):+.3f}; "
          f"edge-bump part balanced Spearman={spearman(rc, d):+.3f}; constant part = log(lam/k) = {np.log(lam/5):.3f}")
