"""R5 numerical checks of the population results in main.tex Secs. II-V.

Run: PYTHONDONTWRITEBYTECODE=1 uv run --with numpy,scipy python check_population.py
Reads nothing from the worktree; writes nothing. Prints PASS/INFO lines.
"""

from __future__ import annotations

import numpy as np

rng = np.random.default_rng(20261112)
np.set_printoptions(precision=4, suppress=True)


def report(tag: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f" :: {detail}" if detail else ""))


def info(tag: str, detail: str) -> None:
    print(f"[INFO] {tag} :: {detail}")


def build(sizes, B, theta=None):
    K = len(sizes)
    g = np.repeat(np.arange(K), sizes)
    n = len(g)
    Z = np.zeros((n, K))
    Z[np.arange(n), g] = 1
    if theta is None:
        theta = rng.pareto(2.5, n) + 0.3  # heavy-tailed propensities
        theta = theta / theta.max()
    Omega = np.outer(theta, theta) * (Z @ B @ Z.T)
    s = Z.T @ theta
    kappa = B @ s
    delta = Omega.sum(1)
    return dict(
        g=g, n=n, Z=Z, theta=theta, Omega=Omega, s=s, kappa=kappa, delta=delta, B=B, K=K
    )


def within_block_slope(x, y, g):
    """Pooled within-block OLS slope of y on x (both demeaned per block)."""
    xs, ys = [], []
    for a in np.unique(g):
        m = g == a
        xs.append(x[m] - x[m].mean())
        ys.append(y[m] - y[m].mean())
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    return float(xs @ ys / (xs @ xs))


def balanced(M, h):
    P, S, Qt = np.linalg.svd(M)
    return P[:, :h] * np.sqrt(S[:h]), Qt[:h].T * np.sqrt(S[:h]), S


def max_within_block_row_spread(U, g):
    out = 0.0
    for a in np.unique(g):
        R = U[g == a]
        out = max(out, float(np.abs(R - R.mean(0)).max()))
    return out


# ---------------------------------------------------------------- Sec. II
B = np.array([[0.9, 0.2, 0.1], [0.2, 0.7, 0.15], [0.1, 0.15, 0.8]])
m = build([40, 60, 50], B)
g, n, Z, th, Om, s, ka, de = (m[k] for k in "g n Z theta Omega s kappa delta".split())
report("II delta_i = theta_i kappa_{g_i}", np.allclose(de, th * ka[g]))
report("II vol = s^T B s", np.isclose(de.sum(), s @ B @ s))
Dm12 = np.diag(de**-0.5)
diag_pert = Dm12 @ np.diag(np.diag(Om)) @ Dm12
eta = th * np.diag(B)[g] / ka[g]
report(
    "II diag perturbation norm = max theta_i B_aa/kappa_a",
    np.isclose(np.linalg.norm(diag_pert, 2), eta.max()),
    f"{eta.max():.4g}",
)
report(
    "II bound <= max theta_i/s_a",
    eta.max() <= (th / s[g]).max() + 1e-15,
    f"{eta.max():.4g} <= {(th / s[g]).max():.4g}",
)
report(
    "II relative degree change = eta_i",
    np.allclose((de - (de - np.diag(Om))) / de, eta),
)

# identifiability gauge: theta_i -> c_a theta_i, B_ab -> B_ab/(c_a c_b)
c = np.array([2.0, 0.5, 3.0])
th2 = th * c[g]
B2 = B / np.outer(c, c)
m2 = build([40, 60, 50], B2, theta=th2)
report("II Omega invariant under (theta,B) block gauge", np.allclose(m2["Omega"], Om))
report("II theta_i/s_a invariant under gauge", np.allclose(th2 / m2["s"][g], th / s[g]))
info(
    "II gauge",
    "theta_i theta_j B <= 1 and 'bounded propensities, growing s_a' are NOT gauge invariant;"
    " max_i theta_i/s_{g_i} -> 0 is the invariant condition",
)

# ---------------------------------------------------------------- Sec. III lemma
K, h_true = 4, 3
sizes = [15, 20, 25, 10]
gg = np.repeat(np.arange(K), sizes)
nn = len(gg)
ZZ = np.zeros((nn, K))
ZZ[np.arange(nn), gg] = 1
xi = rng.uniform(0.2, 3, nn)
ze = rng.uniform(0.2, 3, nn)
A = rng.normal(size=(K, h_true)) @ rng.normal(size=(h_true, K))  # rank 3, nonsymmetric
M = np.diag(xi) @ ZZ @ A @ ZZ.T @ np.diag(ze)
h = np.linalg.matrix_rank(M)
U, V, _ = balanced(M, h)
R = rng.normal(size=(h, h))
U2, V2 = U @ R, V @ np.linalg.inv(R).T
for tag, UU, VV in [("balanced", U, V), ("random invertible gauge", U2, V2)]:
    report(
        f"III lemma rays ({tag}) u_i/xi_i block-constant",
        max_within_block_row_spread(UU / xi[:, None], gg) < 1e-9,
    )
    report(
        f"III lemma rays ({tag}) v_i/zeta_i block-constant",
        max_within_block_row_spread(VV / ze[:, None], gg) < 1e-9,
    )
# minimality is needed: append a cancelling column
N = rng.normal(size=(nn, 1))
U3, V3 = np.hstack([U, N]), np.hstack([V, np.zeros((nn, 1))])
report("III non-minimal factorization still reproduces M", np.allclose(U3 @ V3.T, M))
report(
    "III ...but its rows are not rays (minimality is necessary)",
    max_within_block_row_spread(U3 / xi[:, None], gg) > 1e-3,
)
# balanced but non-minimal (wider) factors keep ray structure
O = np.linalg.qr(rng.normal(size=(h + 3, h)))[0]
U4, V4 = U @ O.T, V @ O.T
report("III wider balanced factors: balance holds", np.allclose(U4.T @ U4, V4.T @ V4))
report(
    "III wider balanced factors keep rays (balance => col(U)=col(M))",
    max_within_block_row_spread(U4 / xi[:, None], gg) < 1e-9,
)
# degenerate: zero row and proportional rows
A0 = A.copy()
A0[1] = 0
A0[3] = 2.5 * A0[2]
M0 = np.diag(xi) @ ZZ @ A0 @ ZZ.T @ np.diag(ze)
h0 = np.linalg.matrix_rank(M0)
U0, V0, _ = balanced(M0, h0)
report(
    "III zero row of A gives zero community vector",
    np.abs(U0[gg == 1]).max() < 1e-9,
    f"rank={h0}",
)
d2 = U0[gg == 2][0] / np.linalg.norm(U0[gg == 2][0])
d3 = U0[gg == 3][0] / np.linalg.norm(U0[gg == 3][0])
report("III proportional rows of A give identical directions", np.allclose(d2, d3))

# ---------------------------------------------------------------- Sec. IV spectral
S = Dm12 @ Om @ Dm12
lam, vec = np.linalg.eigh(S)
idx = np.argsort(-np.abs(lam))[:3]
Us, lam_k = vec[:, idx], lam[idx]
report("IV rank(S) = K", np.linalg.matrix_rank(S) == 3)
report("IV ||u_i||^2 = theta_i/s_a", np.allclose((Us**2).sum(1), th / s[g]))
report(
    "IV directions community-only",
    max_within_block_row_spread(Us / np.linalg.norm(Us, axis=1)[:, None], g) < 1e-9,
)
report("IV trivial eigenvalue is 1", np.isclose(lam.max(), 1.0))
X = Us * np.sqrt(np.abs(lam_k))
Xn = X / np.linalg.norm(X, axis=1)[:, None]
cosX = Xn @ Xn.T
pred = B / np.sqrt(np.outer(np.diag(B), np.diag(B)))
report(
    "IV weighted cosine = B_ab/sqrt(B_aa B_bb) when B>0 (PSD)",
    np.linalg.eigvalsh(B).min() > 0 and np.allclose(cosX, pred[g][:, g]),
)
# indefinite B
Bi = np.array([[0.1, 0.8, 0.3], [0.8, 0.1, 0.2], [0.3, 0.2, 0.6]])
mi = build([40, 60, 50], Bi)
Si = np.diag(mi["delta"] ** -0.5) @ mi["Omega"] @ np.diag(mi["delta"] ** -0.5)
li, vi = np.linalg.eigh(Si)
ii = np.argsort(-np.abs(li))[:3]
Xi = vi[:, ii] * np.sqrt(np.abs(li[ii]))
Xin = Xi / np.linalg.norm(Xi, axis=1)[:, None]
report(
    "IV indefinite B: weighted rows still rays",
    max_within_block_row_spread(Xin, mi["g"]) < 1e-9,
    f"eig(B)={np.linalg.eigvalsh(Bi)}",
)
cos_i = Xin @ Xin.T
report(
    "IV indefinite B: cosine != B_ab/sqrt(B_aa B_bb) (formula needs PSD)",
    not np.allclose(
        cos_i, (Bi / np.sqrt(np.outer(np.diag(Bi), np.diag(Bi))))[mi["g"]][:, mi["g"]]
    ),
)
# repeated eigenvalues: symmetric planted partition with equal block masses
Bp = 0.6 * np.eye(3) + 0.1 * (np.ones((3, 3)) - np.eye(3))
thp = np.tile(
    rng.uniform(0.2, 1.0, 50), 3
)  # identical propensity multiset per block -> equal s, kappa
mp = build([50, 50, 50], Bp, theta=thp)
Sp = np.diag(mp["delta"] ** -0.5) @ mp["Omega"] @ np.diag(mp["delta"] ** -0.5)
lp, vp = np.linalg.eigh(Sp)
ip = np.argsort(-np.abs(lp))[:3]
info("IV repeated-eigenvalue case: top eigenvalues", f"{lp[ip]}")
Up = vp[:, ip]
report(
    "IV repeated eigenvalue, all K vectors: norms identified",
    np.allclose((Up**2).sum(1), mp["theta"] / mp["s"][mp["g"]]),
)
# truncate to d=2 (drop trivial, keep one of the degenerate pair) under random rotations of the eigenspace
deg = Up[:, 1:3]
norms_by_choice = []
for _ in range(3):
    t = rng.uniform(0, np.pi)
    Rt = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    keep = np.hstack([Up[:, :1], (deg @ Rt)[:, :1]])
    ratio = (keep**2).sum(1) / (mp["theta"] / mp["s"][mp["g"]])
    norms_by_choice.append([ratio[mp["g"] == a].mean() for a in range(3)])
norms_by_choice = np.array(norms_by_choice)
report(
    "IV truncation at a repeated eigenvalue: block calibration ||y_a|| depends on basis choice",
    np.ptp(norms_by_choice, axis=0).max() > 1e-3,
    f"||y_a||^2 per choice=\n{norms_by_choice}",
)
report(
    "IV ...but within-block radius still proportional to sqrt(theta)",
    all(
        np.ptp(
            (
                (np.hstack([Up[:, :1], (deg @ np.eye(2))[:, :1]]) ** 2).sum(1)
                / (mp["theta"] / mp["s"][mp["g"]])
            )[mp["g"] == a]
        )
        < 1e-9
        for a in range(3)
    ),
)
# random-walk (right) eigenvectors are block-constant (exponent 0)
phi = np.diag(de**-0.5) @ Us
report(
    "IV right eigenvectors of T = D^-1 Omega are block-constant (exponent 0; omitted from Table I)",
    max_within_block_row_spread(phi, g) < 1e-9,
)

# ---------------------------------------------------------------- Sec. V raw pairs
T = np.diag(1 / de) @ Om
W = (B * s[None, :]) / ka[:, None]
report("V W row-stochastic", np.allclose(W.sum(1), 1))
ok = True
Tr, Wr = np.eye(n), np.eye(3)
for r in range(1, 8):
    Tr, Wr = Tr @ T, Wr @ W
    ok &= np.allclose(Tr, (th / s[g])[None, :] * Wr[g][:, g])
report("V Lemma lag-endpoint for r=1..7", ok)
pi = de / de.sum()
report("V stationary distribution", np.allclose(pi @ T, pi))


def pair_matrix(T, pi, omega):
    P = np.zeros_like(T)
    Tr = np.eye(len(T))
    for w in omega:
        Tr = Tr @ T
        P += w * Tr
    return pi[:, None] * P, P


for label, omega in [
    ("window 3 uniform", np.ones(3) / 3),
    ("walk length 5, window 3, omega_r ~ (5-r)", np.array([4, 3, 2]) / 9),
    ("window 10 uniform", np.ones(10) / 10),
]:
    J, P = pair_matrix(T, pi, omega)
    F = sum(w * np.linalg.matrix_power(W, r + 1) for r, w in enumerate(omega))
    G = ka[:, None] * F / (de.sum() * s[None, :])
    report(
        f"V Thm raw pairs J = Theta Z G Z^T Theta [{label}]",
        np.allclose(J, np.outer(th, th) * G[g][:, g]),
    )
    report(
        f"V J and G symmetric [{label}]", np.allclose(J, J.T) and np.allclose(G, G.T)
    )
    UJ, VJ, SJ = balanced(J, 3)
    slope = within_block_slope(np.log(th), np.log(np.linalg.norm(UJ, axis=1)), g)
    report(
        f"V radial exponent of minimal factor = 1 [{label}]",
        abs(slope - 1) < 1e-9,
        f"slope={slope:.6f}",
    )
    report(
        f"V community-only direction [{label}]",
        max_within_block_row_spread(UJ / np.linalg.norm(UJ, axis=1)[:, None], g) < 1e-9,
    )

# loopless walk: exact structure breaks; measure deviation as blocks grow
info(
    "V loopless",
    "slope of log||u_i|| on log theta_i in rank-K balanced factor of loopless J (window 3)",
)
for scale in [1, 4, 16]:
    mm = build([10 * scale, 15 * scale, 12 * scale], B)
    Om0 = mm["Omega"] - np.diag(np.diag(mm["Omega"]))
    d0 = Om0.sum(1)
    T0 = np.diag(1 / d0) @ Om0
    J0, _ = pair_matrix(T0, d0 / d0.sum(), np.ones(3) / 3)
    U0, _, S0 = balanced(J0, 3)
    sl = within_block_slope(
        np.log(mm["theta"]), np.log(np.linalg.norm(U0, axis=1)), mm["g"]
    )
    dirs = U0 / np.linalg.norm(U0, axis=1)[:, None]
    info(
        f"V loopless n={mm['n']}",
        f"slope={sl:.4f}, sigma_(K+1)/sigma_K={S0[3] / S0[2]:.2e}, dir spread={max_within_block_row_spread(dirs, mm['g']):.2e},"
        f" max theta/s={(mm['theta'] / mm['s'][mm['g']]).max():.3f}",
    )

# uniform-start finite walks (DeepWalk default) break the radial form
ell, omega_len = 5, 1
P0 = np.ones(n) / n
Jun = np.zeros((n, n))
Tt = np.eye(n)
for t in range(ell - 1):
    pt = P0 @ Tt  # marginal at position t
    Jun += pt[:, None] * T  # forward pair (w_t, w_{t+1})
    Jun += (pt[:, None] * T).T  # same pair seen from the later token as center
    Tt = Tt @ T
Jun /= Jun.sum()
sv = np.linalg.svd(Jun, compute_uv=False)
info(
    "V uniform starts, length-5 walks, window 1",
    f"rank-K tail sigma_4/sigma_3={sv[3] / sv[2]:.3e}, sigma_7/sigma_3={sv[6] / sv[2]:.2e} (rank up to 2K)",
)
Uu, _, _ = balanced(Jun, 6)
sl = within_block_slope(np.log(th), np.log(np.linalg.norm(Uu, axis=1)), g)
info(
    "V uniform starts",
    f"within-block radial slope of balanced rank-2K factor = {sl:.3f} (stationary start: 1)",
)

# ---------------------------------------------------------------- Sec. VI SGNS
k = 5
omega = np.ones(3) / 3
J, P = pair_matrix(T, pi, omega)
F = sum(w * np.linalg.matrix_power(W, r + 1) for r, w in enumerate(omega))
for alpha in [0.5, 0.75, 1.0, 1.25]:
    q = de**alpha / (de**alpha).sum()
    X = np.log(J / (k * pi[:, None] * q[None, :]))
    H = (
        np.log(F)
        - np.log(s)[None, :]
        - alpha * np.log(ka)[None, :]
        + np.log((de**alpha).sum())
        - np.log(k)
    )
    report(
        f"VI Thm SGNS score formula alpha={alpha}",
        np.allclose(X, H[g][:, g] + (1 - alpha) * np.log(th)[None, :]),
    )
    rk = np.linalg.matrix_rank(X, tol=1e-9 * np.abs(X).max())
    report(f"VI rank(X*) <= K alpha={alpha}", rk <= 3, f"rank={rk}")
    Ux, Vx, _ = balanced(X, rk)
    report(
        f"VI balanced center rows block-constant alpha={alpha}",
        max_within_block_row_spread(Ux, g) < 1e-8,
    )
    if alpha != 1:
        # context rows affine in log theta: residual of fit v_j = e_b + log(theta_j) w
        Dsg = np.hstack([Z, np.log(th)[:, None]])
        coef, *_ = np.linalg.lstsq(Dsg, Vx, rcond=None)
        res = np.abs(Dsg @ coef - Vx).max()
        report(f"VI context rows affine in log theta alpha={alpha}", res < 1e-8)
        # context norms: quadratic in log theta with vertex b_a; report vertex vs data range
        w = coef[-1]
        for a in range(3):
            e = coef[a]
            vert = -(e @ w) / (w @ w)
            lo, hi = np.log(th[g == a]).min(), np.log(th[g == a]).max()
            info(
                f"VI alpha={alpha} block {a} context-norm parabola",
                f"vertex log(theta)={vert:.3f}, data range=[{lo:.3f},{hi:.3f}] -> "
                f"{'NON-MONOTONE within block' if lo < vert < hi else 'monotone on range'}",
            )
    else:
        report(
            "VI alpha=1 context rows block-constant",
            max_within_block_row_spread(Vx, g) < 1e-8,
        )

# NetMF identity and the note.tex double-count concern
vol = de.sum()
netmf = np.log(vol * P @ np.diag(1 / de)) - np.log(k)
X1 = np.log(J / (k * np.outer(pi, pi)))
report(
    "VI NetMF target equals alpha=1 SGNS score (no double count)",
    np.allclose(netmf, X1),
)
report(
    "VI J/(pi_i pi_j) is block-constant BEFORE the log",
    max_within_block_row_spread(np.exp(X1), g) < 1e-8
    and max_within_block_row_spread(np.exp(X1).T, g) < 1e-8,
)
C = P @ np.diag(1 / de)
dbl = np.log(C) - np.log(pi)[:, None] - np.log(pi)[None, :]
resid = dbl - (-np.log(th)[:, None] - np.log(th)[None, :])
report(
    "VI note.tex double-counted matrix = block-constant - log theta_i - log theta_j",
    max_within_block_row_spread(resid, g) < 1e-8
    and max_within_block_row_spread(resid.T, g) < 1e-8,
)

# symmetric (tied / both orientations) score
for alpha in [0.75, 1.0]:
    q = de**alpha / (de**alpha).sum()
    Xs = np.log(2 * J / (k * (np.outer(pi, q) + np.outer(q, pi))))
    G = ka[:, None] * F / (vol * s[None, :])
    Za = (de**alpha).sum()
    Cab = np.log(2 * G * vol * Za / k)
    a_, b_ = g[:, None], g[None, :]
    pred = Cab[a_, b_] - np.log(
        th[None, :] ** (alpha - 1) * ka[a_] * ka[b_] ** alpha
        + th[:, None] ** (alpha - 1) * ka[b_] * ka[a_] ** alpha
    )
    report(f"VI symmetric score formula alpha={alpha}", np.allclose(Xs, pred))
    sv = np.linalg.svd(Xs, compute_uv=False)
    rk = int((sv > 1e-10 * sv[0]).sum())
    info(
        f"VI symmetric score alpha={alpha}",
        f"numerical rank={rk} of n={n}; sigma_(K+1)/sigma_1={sv[3] / sv[0]:.2e}",
    )

# loopless window-one remark (alpha = 1)
Om0 = Om - np.diag(np.diag(Om))
d0 = Om0.sum(1)
vol0 = d0.sum()
J0 = Om0 / vol0
X0 = np.where(
    J0 > 0,
    np.log(np.where(J0 > 0, J0, 1) / (k * np.outer(d0 / vol0, d0 / vol0))),
    -np.inf,
)
H1w1 = np.log(vol * B / (k * np.outer(ka, ka)))
pred0 = (
    H1w1[g][:, g]
    + np.log(vol0 / vol)
    - np.log(1 - eta)[:, None]
    - np.log(1 - eta)[None, :]
)
off = ~np.eye(n, dtype=bool)
report(
    "VI Remark (loopless, window 1) off-diagonal score formula",
    np.allclose(X0[off], pred0[off]),
)

# F_ab = 0: bipartite blocks at window one
Bb = np.array([[0.0, 0.5], [0.5, 0.0]])
mb = build([20, 30], Bb)
Tb = np.diag(1 / mb["delta"]) @ mb["Omega"]
Jb = mb["delta"][:, None] / mb["delta"].sum() * Tb
info(
    "VI F_ab=0 case",
    f"bipartite window 1: fraction of zero positive pairs={np.mean(Jb == 0):.3f}; "
    f"window 2 F={np.round(0.5 * (np.array([[0, 1], [1, 0]]) + np.eye(2)), 3).tolist()} positive",
)

# ---------------------------------------------------------------- Sec. X KL
KLpop = np.array([np.sum(F[a] * np.log(F[a] * vol / (s * ka))) for a in range(3)])
KLdirect = np.array([np.sum(P[i] * np.log(P[i] / pi)) for i in range(n)])
report(
    "X population context KL formula and block constancy",
    np.allclose(KLdirect, KLpop[g]),
)
# realized-graph KL formula
Ar = (rng.random((30, 30)) < 0.2).astype(float)
Ar = np.triu(Ar, 1)
Ar = Ar + Ar.T
dr = Ar.sum(1)
keep = dr > 0
Ar = Ar[keep][:, keep]
dr = Ar.sum(1)
Tr_ = Ar / dr[:, None]
pir = dr / dr.sum()
KLr = np.array(
    [
        np.sum(Tr_[i][Tr_[i] > 0] * np.log(Tr_[i][Tr_[i] > 0] / pir[Tr_[i] > 0]))
        for i in range(len(dr))
    ]
)
KLf = np.log(dr.sum() / dr) - (Ar @ np.log(dr)) / dr
report("X realized-graph KL formula", np.allclose(KLr, KLf))
