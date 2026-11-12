"""R5: deterministic link between Gram imbalance D = U^TU - V^TV and raw-vs-balanced norms.
(UU^T)^2 - MM^T = U D U^T, and ||A^{1/2}-B^{1/2}||_op <= ||A-B||_op^{1/2} for PSD A,B, so
max_i |r_raw,i^2 - r_bal,i^2| <= ||UU^T - (MM^T)^{1/2}||_op <= ||U D U^T||_op^{1/2}.
Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy python check_imbalance_bound.py
"""
import numpy as np
rng = np.random.default_rng(9)
def psd_sqrt(S):
    w, Y = np.linalg.eigh(S); return (Y * np.sqrt(np.clip(w, 0, None))) @ Y.T
worst = 0.0; worst_diag = 0.0
for _ in range(3000):
    n, d = rng.integers(5, 40), rng.integers(2, 8)
    U = rng.normal(size=(n, d)); V = U @ (np.eye(d) + rng.normal(size=(d, d)) * rng.uniform(0, 0.5)) + rng.normal(size=(n, d)) * rng.uniform(0, 0.3)
    M = U @ V.T; D = U.T @ U - V.T @ V
    lhs = np.linalg.norm(U @ U.T - psd_sqrt(M @ M.T), 2)
    rhs = np.sqrt(np.linalg.norm(U @ D @ U.T, 2))
    diag = np.abs((U ** 2).sum(1) - np.diag(psd_sqrt(M @ M.T))).max()
    worst = max(worst, lhs / rhs); worst_diag = max(worst_diag, diag / rhs)
print(f"[{'PASS' if worst <= 1 + 1e-9 else 'FAIL'}] ||UU^T-(MM^T)^1/2||_op <= ||UDU^T||_op^1/2 : max ratio={worst:.4f}; "
      f"max_i |r_raw^2-r_bal^2| / bound = {worst_diag:.4f}")
