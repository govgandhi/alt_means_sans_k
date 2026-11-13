"""R5: numerical check of the full-support loss-to-radius bound (norm_geometry_supplement.tex Sec. 2)."""
import numpy as np
rng = np.random.default_rng(5)
def ell(x, a, b): return a * np.logaddexp(0, -x) + b * np.logaddexp(0, x)
psi = lambda t: t - 1 + np.exp(-t)
worst = 0.0
for _ in range(200000):
    a, b = np.exp(rng.uniform(-12, 2, 2)); z = rng.normal() * rng.uniform(0.01, 30)
    xs = np.log(a / b); h = a * b / (a + b)
    lhs = ell(xs + z, a, b) - ell(xs, a, b); rhs = h * psi(abs(z))
    worst = max(worst, rhs / max(lhs, 1e-300))
print(f"[{'PASS' if worst <= 1 + 1e-9 else 'FAIL'}] scalar curvature bound l(x*+z)-l(x*) >= h psi(|z|): max rhs/lhs={worst:.6f}")
t = np.linspace(0, 50, 100001)
print(f"[{'PASS' if np.all(psi(t) >= t**2/(t+2) - 1e-12) else 'FAIL'}] psi(t) >= t^2/(t+2)")
worst = 0.0
for _ in range(2000):
    n = rng.integers(2, 8); A = rng.normal(size=(n, n)); A = A + A.T; B = A + rng.normal(size=(n, n)) * rng.uniform(0.01, 3); B = B + B.T
    def absm(S):
        w, Y = np.linalg.eigh(S); return (Y * np.abs(w)) @ Y.T
    worst = max(worst, np.linalg.norm(absm(A) - absm(B)) / np.linalg.norm(A - B))
print(f"[{'PASS' if worst <= 1 + 1e-9 else 'FAIL'}] || |A|-|B| ||_F <= ||A-B||_F (Hermitian): max ratio={worst:.4f}")
# end-to-end on a small full-support problem: random X near X*, check RMS(r-r*) <= e
n = 6; a = rng.uniform(0.01, 1, (n, n)); b = rng.uniform(0.01, 1, (n, n)); Xs = np.log(a / b)
h = (a * b / (a + b)).min()
def radii(X):
    w, Y = np.linalg.eigh(X @ X.T); return np.sqrt(np.clip(np.diag((Y * np.sqrt(np.clip(w, 0, None))) @ Y.T), 0, None))
ok = True; ratios = []
for _ in range(5000):
    X = Xs + rng.normal(size=(n, n)) * rng.uniform(0.001, 2)
    Delta = np.sum(ell(X, a, b) - ell(Xs, a, b)); c = Delta / h; Bnd = (c + np.sqrt(c * c + 8 * c)) / 2
    e = (2 / n) ** 0.25 * np.sqrt(Bnd); err = np.sqrt(np.mean((radii(X) - radii(Xs)) ** 2))
    ok &= err <= e + 1e-12; ratios.append(err / e)
print(f"[{'PASS' if ok else 'FAIL'}] end-to-end RMS(r-r*) <= (2/n)^(1/4) sqrt(B): median tightness err/e={np.median(ratios):.3f}, max={np.max(ratios):.3f}")
