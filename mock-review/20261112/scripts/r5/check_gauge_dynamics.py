"""R5: gauge, balancing and balance-dynamics checks (main.tex Sec. VII; factor_balance.tex).

Symbolic (SymPy): flow conservation; unequal-rate invariant; weight-decay contraction;
single-step drift; scalar contraction under finite steps.
Numeric (NumPy): balanced-factor uniqueness up to O(h); Gensim-style per-pair update obeys
the simultaneous-step eta^2 drift formula; SGD-noise drift; weight decay drives balance.

Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with sympy,numpy,scipy python check_gauge_dynamics.py
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.special import expit

rng = np.random.default_rng(7)


def report(tag, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f" :: {detail}" if detail else ""))


def info(tag, detail):
    print(f"[INFO] {tag} :: {detail}")


# ------------------------------------------------------------------ symbolic
u = sp.Matrix(3, 2, sp.symbols("u:6", real=True))
v = sp.Matrix(3, 2, sp.symbols("v:6", real=True))
G = sp.Matrix(3, 3, sp.symbols("g:9", real=True))
eta, etu, etv, lam = sp.symbols("eta eta_u eta_v lambda", positive=True)
du, dv = -G * v, -G.T * u
flow = (u.T * du + du.T * u) - (v.T * dv + dv.T * v)
report(
    "SymPy: d/dt(U^TU - V^TV) = 0 under equal-rate flow",
    flow.applyfunc(sp.expand) == sp.zeros(2),
)
# unequal learning rates: Udot = -eta_u G V, Vdot = -eta_v G^T U  => U^TU/eta_u - V^TV/eta_v conserved
du2, dv2 = -etu * G * v, -etv * G.T * u
flow2 = (u.T * du2 + du2.T * u) / etu - (v.T * dv2 + dv2.T * v) / etv
report(
    "SymPy: unequal rates conserve U^TU/eta_u - V^TV/eta_v (not U^TU - V^TV)",
    flow2.applyfunc(sp.expand) == sp.zeros(2),
)
flow2b = (u.T * du2 + du2.T * u) - (v.T * dv2 + dv2.T * v)
report(
    "SymPy: ...and U^TU - V^TV is NOT conserved for eta_u != eta_v",
    flow2b.applyfunc(sp.expand) != sp.zeros(2),
)
# weight decay lambda(|U|^2+|V|^2)/2:  dD/dt = -2 lambda D
du3, dv3 = -G * v - lam * u, -G.T * u - lam * v
D = u.T * u - v.T * v
flow3 = (u.T * du3 + du3.T * u) - (v.T * dv3 + dv3.T * v)
report(
    "SymPy: equal weight decay gives dD/dt = -2 lambda D (balance is attracting)",
    (flow3 + 2 * lam * D).applyfunc(sp.expand) == sp.zeros(2),
)
# single simultaneous step drift
nu, nv = u + eta * du, v + eta * dv
step = (nu.T * nu - nv.T * nv) - D
quad = eta**2 * (v.T * G.T * G * v - u.T * G * G.T * u)
report(
    "SymPy: one simultaneous step changes D by eta^2 (V^TG^TGV - U^TGG^TU)",
    (step - quad).applyfunc(sp.expand) == sp.zeros(2),
)
# scalar case: D_next = D (1 - eta^2 g^2): finite steps contract imbalance
a, b, g = sp.symbols("a b g", real=True)
an, bn = a - eta * g * b, b - eta * g * a
report(
    "SymPy: scalar factorization, D_next = (1 - eta^2 g^2) D",
    sp.expand((an**2 - bn**2) - (1 - eta**2 * g**2) * (a**2 - b**2)) == 0,
)

# ------------------------------------------------------------------ balanced uniqueness
n, h = 12, 4
M = rng.normal(size=(n, h)) @ rng.normal(size=(h, n))
P, S, Qt = np.linalg.svd(M)
Ub, Vb = P[:, :h] * np.sqrt(S[:h]), Qt[:h].T * np.sqrt(S[:h])
O = np.linalg.qr(rng.normal(size=(h, h)))[0]
report(
    "balanced factors: (U O, V O) balanced and same product",
    np.allclose((Ub @ O).T @ (Ub @ O), (Vb @ O).T @ (Vb @ O))
    and np.allclose((Ub @ O) @ (Vb @ O).T, M),
)
w, Y = np.linalg.eigh(M @ M.T)
sq = (Y * np.sqrt(np.clip(w, 0, None))) @ Y.T
report(
    "balanced squared row norms = diag((MM^T)^{1/2})",
    np.allclose((Ub**2).sum(1), np.diag(sq)),
)
# repeated singular values: SVD non-unique but norms identified
Mr = Ub[:, :2] @ Vb[:, :2].T + 0  # generic
Q1 = np.linalg.qr(rng.normal(size=(n, 3)))[0]
Q2 = np.linalg.qr(rng.normal(size=(n, 3)))[0]
Mrep = Q1 @ np.diag([2.0, 2.0, 1.0]) @ Q2.T
P2, S2, Qt2 = np.linalg.svd(Mrep)
U_a = P2[:, :3] * np.sqrt(S2[:3])
R2 = np.eye(3)
t = 0.7
R2[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
U_b = (P2[:, :3] @ R2) * np.sqrt(
    S2[:3]
)  # alternative singular basis within the repeated block
report(
    "repeated singular values: alternative SVD bases give equal balanced row norms",
    np.allclose((U_a**2).sum(1), (U_b**2).sum(1)),
)

# ------------------------------------------------------------------ Gensim-style per-pair update
d, nv_ = 8, 30
Uin = rng.normal(scale=0.3, size=(nv_, d))
Vout = rng.normal(scale=0.3, size=(nv_, d))
alpha_lr, kneg = 0.025, 5
inp, outp = 3, 11
negs = rng.choice([j for j in range(nv_) if j != outp], size=kneg, replace=False)
D0 = Uin.T @ Uin - Vout.T @ Vout
# gensim: l1 = syn0[input]; for target in [pos]+negs: f = l1.syn1neg[t]; g=(label-sigma)*alpha;
# work += g*syn1neg[t]; syn1neg[t] += g*l1 ; finally syn0[input] += work
U1, V1 = Uin.copy(), Vout.copy()
l1 = U1[inp].copy()
work = np.zeros(d)
for t_, label in [(outp, 1.0)] + [(j, 0.0) for j in negs]:
    f = l1 @ V1[t_]
    gg = (label - expit(f)) * alpha_lr
    work += gg * V1[t_]
    V1[t_] += gg * l1
U1[inp] += work
D1 = U1.T @ U1 - V1.T @ V1
# simultaneous-step formula with G = dL/dM on the sampled pair loss (eta absorbed: G_sc = -(label-sigma))
Gm = np.zeros((nv_, nv_))
for t_, label in [(outp, 1.0)] + [(j, 0.0) for j in negs]:
    Gm[inp, t_] = -(label - expit(Uin[inp] @ Vout[t_]))
pred = alpha_lr**2 * (Vout.T @ Gm.T @ Gm @ Vout - Uin.T @ Gm @ Gm.T @ Uin)
report(
    "Gensim per-pair update (distinct targets) is a simultaneous step: drift = eta^2 formula",
    np.allclose(D1 - D0, pred),
    f"max |err|={np.abs(D1 - D0 - pred).max():.2e}",
)
info(
    "factor_balance.tex l.27",
    "claims the formula does not apply 'verbatim to Gensim's sequential updates'; "
    "within one (input, k+1 targets) update it applies exactly; repeated negatives break it",
)


# ------------------------------------------------------------------ dynamics on a small SGNS problem
def loss_grad(Uf, Vf, Pp, Pn):
    Xs = Uf @ Vf.T
    Gs = -Pp * expit(-Xs) + Pn * expit(Xs)
    return Gs


nn = 10
Pp = rng.random((nn, nn)) + 0.05
Pp /= Pp.sum()
Pn = 5 * np.outer(Pp.sum(1), Pp.sum(0))
U0 = rng.normal(scale=0.01, size=(nn, 4))
V0 = np.zeros((nn, 4))  # Gensim-like init: small centers, zero contexts


def run(step, iters, wd=0.0, eta_u=None, eta_v=None, noise=0.0):
    Uf, Vf = U0.copy(), V0.copy()
    eta_u = step if eta_u is None else eta_u
    eta_v = step if eta_v is None else eta_v
    for _ in range(iters):
        Gs = loss_grad(Uf, Vf, Pp, Pn)
        if noise:
            Gs = Gs + noise * rng.normal(size=Gs.shape) * np.abs(Gs).mean()
        Uf, Vf = Uf - eta_u * (Gs @ Vf + wd * Uf), Vf - eta_v * (Gs.T @ Uf + wd * Vf)
    Dm = Uf.T @ Uf - Vf.T @ Vf
    return Uf, Vf, np.linalg.norm(Dm) / np.linalg.norm(Uf.T @ Uf)


D_init = np.linalg.norm(U0.T @ U0 - V0.T @ V0)
for label, kw in [
    ("small-step GD (flow proxy)", dict(step=0.5, iters=40000)),
    ("large-step GD (eta=4)", dict(step=4.0, iters=5000)),
    ("GD + weight decay 1e-3", dict(step=0.5, iters=40000, wd=1e-3)),
    ("unequal rates (eta_v = 4 eta_u)", dict(step=0.5, iters=40000, eta_v=2.0)),
    ("noisy gradients (SGD proxy)", dict(step=0.5, iters=40000, noise=3.0)),
]:
    Uf, Vf, rel = run(**kw)
    Dabs = np.linalg.norm(Uf.T @ Uf - Vf.T @ Vf)
    info(
        f"dynamics: {label}",
        f"|D| {D_init:.2e} -> {Dabs:.2e}; relative imbalance {rel:.3e}",
    )
