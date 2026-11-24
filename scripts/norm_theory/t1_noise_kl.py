"""T1: is the reused-corpus hump on the expected-BA corpus a fit to sampling noise?

Expected-BA corpus (scripts/step0/ba_corpus.py): every token is an iid draw from
q proportional to degree, so the population context law of every node is q, the
population KL is zero, and the population skip-gram optimum has flat norms.
A finite corpus gives node i an empirical context law C_i with m_i draws, and
    E[KL(C_i || q)] = H(q) - log m + (1/m) sum_j E[X_j log X_j],  X_j ~ Bin(m, q_j)
(exact for unweighted draws; Poisson approximation for the expectation). It falls
with m: about H(q) - log m while m << exp(H(q)), and (V - 1)/(2m) once m >> V.
Shrinking windows give weighted draws; we use m_eff = (sum w)^2 / sum w^2.
Model for the reused-corpus excess over the fresh-corpus (noise-free) run:
    norm_reused^2 - norm_fresh^2 = b * KL_i * (1 - exp(-n_i tau / c)),
the noise target (proportional to KL_i, Oyama-type) reached at a rate set by the
node's update count n_i times the summed learning rate tau.
Readouts: per-node KL vs committed norms (global, within count bins, and the
idiosyncratic part KL_i - E[KL | n_i]); the same for the fresh-corpus runs
(control); and the fitted model on the 50-epoch animation runs, with its peak path.
Output: results/norm_theory/t1.json
Run: .venv/bin/python -I scripts/norm_theory/t1_noise_kl.py
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.special import gammaln
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/derived/step0"
SETTING = {"B": (1, False), "A": (5, True), "SW": (10, True)}


def context_kl(walks, counts, window, shrink):
    V = len(counts)
    q = counts / counts.sum()
    M = sp.csr_matrix((V, V))
    for t in range(1, window + 1):
        w = (window - t + 1) / window if shrink else 1.0
        a, b = walks[:, :-t].ravel(), walks[:, t:].ravel()
        M = (
            M
            + sp.coo_matrix(
                (np.full(2 * len(a), w), (np.r_[a, b], np.r_[b, a])), shape=(V, V)
            ).tocsr()
        )
    rs = np.asarray(M.sum(1)).ravel()
    R = (sp.diags(1 / rs) @ M).tocoo()
    kl = np.bincount(R.row, weights=R.data * np.log(R.data / q[R.col]), minlength=V)
    return kl


def poisson_xlogx(lam, kmax=None):
    """E[X log X] for X ~ Poisson(lam), vectorized over lam (truncated sum)."""
    lam = np.asarray(lam, float)
    kmax = int(max(30, lam.max() + 12 * np.sqrt(lam.max() + 1)))
    k = np.arange(1, kmax + 1)
    logpmf = k[None, :] * np.log(lam[:, None]) - lam[:, None] - gammaln(k + 1)[None, :]
    return (np.exp(logpmf) * k * np.log(k)).sum(1)


def expected_kl(m, q):
    H = -(q * np.log(q)).sum()
    out = []
    for mm in np.atleast_1d(m):
        out.append(H - np.log(mm) + poisson_xlogx(mm * q).sum() / mm)
    return np.array(out)


def m_eff_per_token(window, shrink):
    w = np.array(
        [(window - t + 1) / window if shrink else 1.0 for t in range(1, window + 1)]
    )
    return 2 * w.sum() ** 2 / (w**2).sum()


def binned_spearman(x, y, n, min_size=30):
    bins = np.floor(np.log10(n) * 4)
    rho, wt = [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= min_size:
            rho.append(spearmanr(x[m], y[m]).statistic)
            wt.append(m.sum())
    return float(np.average(rho, weights=wt))


def readout(norm, n, kl, kl_expected):
    resid = kl - kl_expected
    return dict(
        spearman_norm_kl=float(spearmanr(norm, kl).statistic),
        spearman_norm_count=float(spearmanr(norm, n).statistic),
        within_bin_norm_kl=binned_spearman(norm, kl, n),
        within_bin_norm_kl_resid=binned_spearman(norm, resid, n),
    )


def profile(n, y):
    edges = 10 ** np.arange(
        np.floor(np.log10(n.min()) * 4) / 4, np.log10(n.max()) + 0.25, 0.25
    )
    idx = np.digitize(n, edges)
    keep = [b for b in np.unique(idx) if (idx == b).sum() >= 10]
    x = np.array([np.exp(np.log(n[idx == b]).mean()) for b in keep])
    m = np.array([y[idx == b].mean() for b in keep])
    return x, m / y.mean()


def tau_linear(e, E, a0=0.025, a1=1e-4):
    return a0 * e - (a0 - a1) * e**2 / (2 * E)


def main() -> None:
    out = {"setting": {}, "theory_curve": {}}
    per_seed = {c: [] for c in SETTING}
    kl_store = {}
    for seed in range(5):
        z = np.load(DATA / f"ba_walks_s{seed}.npz")
        walks = z["walks"].astype(np.int64)
        V = len(z["strength"])
        counts = np.bincount(walks.ravel(), minlength=V).astype(float)
        q = counts / counts.sum()
        for c, (w, shrink) in SETTING.items():
            kl = context_kl(walks, counts, w, shrink)
            kexp = expected_kl(m_eff_per_token(w, shrink) * counts, q)
            kl_store[(seed, c)] = (counts, kl, kexp)
            r = np.load(ROOT / f"results/step0/runs/ba_{c}_s{seed}.npz")
            idx = np.array([int(t[1:]) for t in r["tokens"]])
            row = readout(r["balanced_norm"], counts[idx], kl[idx], kexp[idx])
            row["kl_vs_expected_pearson"] = float(np.corrcoef(kl, kexp)[0, 1])
            row["kl_mean_realized"] = float(kl.mean())
            row["kl_mean_expected"] = float(kexp.mean())
            fz = ROOT / f"results/norm_tests/fresh/ba_{c}_fresh_s{seed}.npz"
            if fz.exists():
                f = np.load(fz)
                fidx = np.array([int(t[1:]) for t in f["tokens"]])
                row["fresh"] = readout(
                    f["balanced_norm"][-1], counts[fidx], kl[fidx], kexp[fidx]
                )
            per_seed[c].append(row)
            print(
                seed,
                c,
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in row.items()
                    if k != "fresh"
                },
                flush=True,
            )
    for c, rows in per_seed.items():
        keys = [k for k in rows[0] if k != "fresh"]
        out["setting"][c] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
        if "fresh" in rows[0]:
            out["setting"][c]["fresh"] = {
                k: float(np.mean([r["fresh"][k] for r in rows]))
                for k in rows[0]["fresh"]
            }
    # theory curve E[KL | n] for each window, seed 0 q
    counts0, _, _ = kl_store[(0, "B")]
    q0 = counts0 / counts0.sum()
    grid = np.geomspace(counts0.min(), counts0.max(), 25)
    for c, (w, shrink) in SETTING.items():
        out["theory_curve"][c] = dict(
            n=grid.tolist(),
            kl=expected_kl(m_eff_per_token(w, shrink) * grid, q0).tolist(),
        )
    # speed x noise model on the 50-epoch animation (setting B, seed 0)
    zr = np.load(ROOT / "results/norm_tests/anim/ba_B50_reused_s0.npz")
    zf = np.load(ROOT / "results/norm_tests/anim/ba_B50_fresh_s0.npz")
    assert np.array_equal(zr["tokens"], zf["tokens"])
    counts, kl, kexp = kl_store[(0, "B")]
    idx = np.array([int(t[1:]) for t in zr["tokens"]])
    n, k_i = counts[idx], kl[idx]
    E = zr["balanced_norm"].shape[0]
    epochs = np.arange(1, E + 1)
    taus = tau_linear(epochs, E)
    D = zr["balanced_norm"] ** 2 - zf["balanced_norm"] ** 2  # (E, nodes)
    use = epochs >= 2  # epoch 1 shares the corpus in both arms

    def model(p, t):
        b, logc = p
        return b * k_i[None, :] * (1 - np.exp(-n[None, :] * t[:, None] / np.exp(logc)))

    fit = least_squares(
        lambda p: (model(p, taus[use]) - D[use]).ravel(), x0=[1.0, np.log(100.0)]
    )
    pred = model(fit.x, taus)
    ss = ((D[use] - pred[use]) ** 2).sum()
    st = ((D[use] - D[use].mean()) ** 2).sum()
    peaks = []
    for e in range(E):
        xr, mr = profile(n, zr["balanced_norm"][e])
        xp, mp = profile(
            n, np.sqrt(np.maximum(zf["balanced_norm"][e] ** 2 + pred[e], 0))
        )
        xf, mf = profile(n, zf["balanced_norm"][e])
        peaks.append(
            dict(
                epoch=int(e + 1),
                tau=float(taus[e]),
                observed_peak=float(xr[np.argmax(mr)]),
                predicted_peak=float(xp[np.argmax(mp)]),
                fresh_peak=float(xf[np.argmax(mf)]),
                observed_range=float(np.ptp(mr)),
                predicted_range=float(np.ptp(mp)),
                fresh_range=float(np.ptp(mf)),
            )
        )
    # same model without the KL factor (pure speed) for comparison
    fit0 = least_squares(
        lambda p: (
            p[0] * (1 - np.exp(-n[None, :] * taus[use][:, None] / np.exp(p[1])))
            - D[use]
        ).ravel(),
        x0=[1.0, np.log(100.0)],
    )
    r0 = fit0.fun
    out["anim_fit"] = dict(
        b=float(fit.x[0]),
        c=float(np.exp(fit.x[1])),
        r2=float(1 - ss / st),
        r2_without_kl=float(1 - (r0**2).sum() / st),
        within_epoch_spearman_resid=[
            float(binned_spearman(D[e], k_i - kexp[idx], n)) for e in range(E)
        ],
        peaks=peaks,
    )
    print(
        "fit",
        out["anim_fit"]["b"],
        out["anim_fit"]["c"],
        out["anim_fit"]["r2"],
        out["anim_fit"]["r2_without_kl"],
    )
    for pk in peaks:
        if pk["epoch"] in (1, 2, 3, 5, 10, 20, 30, 50):
            print(pk)
    (ROOT / "results/norm_theory/t1.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
