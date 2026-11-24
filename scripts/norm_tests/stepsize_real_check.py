"""Matched-tau step-size runs on BlogCatalog and hep-th text: shape and a per-node test.

Inputs: results/norm_tests/stepsize_real/<corpus>_E{5,25,100}.npz (one-hop setting,
constant rate 0.06275/E for E epochs, so all three end at the same summed rate).
1. Binned balanced-norm profile of each run (Fig. 1 binning), peak, top-bin gap.
2. Per-node shrinkage under large steps, log of (mean-normalized) E5 over E100 norms,
   against predictors computed from the small-step (E100) model only:
     degree or count;
     uncertainty  u_i = sum_j a_ij / sum_j (P_ij + N_ij), a = (P + N) s(1 - s);
     fisher       F_i = sum_j a_ij + sum_j a_ji (row and column Fisher mass);
     radial       r_i = ||w_i||^2 / <s_ij^2>_a, the local linear-response size of the
                  first-order SGD penalty
                  (eps/4) sum (P + N) s(1 - s)(||w||^2 + ||c||^2).
   P = window-1 pair counts in the corpus; N_ij = 5 n_i q_j with n_i = sum_j P_ij and
   q the negative law (exponent 1). Prediction: shrinkage falls with r_i (more negative
   shrinkage for larger r_i), within frequency bins as well as globally.
Output: results/norm_tests/stepsize_real.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.special import expit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/norm_tests"))
from analyze import fig1_curve, rnd  # noqa: E402

R = ROOT / "results/norm_tests/stepsize_real"
DATA = ROOT / "data/derived/step0"


def pair_counts(corpus: str, tokens: list[str]) -> sp.csr_matrix:
    """Window-1 ordered pair counts (both directions) over the model vocabulary."""
    vocab = {t: i for i, t in enumerate(tokens)}
    V = len(tokens)
    rows, cols = [], []
    if corpus == "blog":
        walks = np.load(DATA / "blog_walks_s0.npz")["walks"]
        idx = np.full(walks.max() + 1, -1)
        for t, i in vocab.items():
            idx[int(t[1:])] = i
        ids = idx[walks]
        a, b = ids[:, :-1].ravel(), ids[:, 1:].ravel()
        keep = (a >= 0) & (b >= 0)
        rows, cols = [a[keep], b[keep]], [b[keep], a[keep]]
    else:
        with open(DATA / "hepth_tokens.txt") as f:
            for line in f:
                ids = np.array([vocab[w] for w in line.split() if w in vocab], np.int64)
                if len(ids) > 1:
                    rows += [ids[:-1], ids[1:]]
                    cols += [ids[1:], ids[:-1]]
    r, c = np.concatenate(rows), np.concatenate(cols)
    M = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(V, V)).tocsr()
    M.sum_duplicates()
    return M


def predictors(W, C, P, q, k=5, chunk=1000):
    """Per-node uncertainty u, radial size, and Fisher mass F (row + column)."""
    n = np.asarray(P.sum(1)).ravel()
    u = np.empty(len(W))
    s2 = np.empty(len(W))
    F_row = np.empty(len(W))
    F_col = np.zeros(len(W))
    for s in range(0, len(W), chunk):
        S = W[s : s + chunk] @ C.T
        sig = expit(S)
        Pd = P[s : s + chunk].toarray()
        A = (Pd + k * n[s : s + chunk, None] * q[None, :]) * sig * (1 - sig)
        tot = Pd.sum(1) + k * n[s : s + chunk]
        u[s : s + chunk] = A.sum(1) / tot
        s2[s : s + chunk] = (A * S**2).sum(1) / A.sum(1)
        F_row[s : s + chunk] = A.sum(1)
        F_col += A.sum(0)
    radial = (W**2).sum(1) / s2
    return u, radial, F_row + F_col


def within_bins(y, x, deg, min_n=30):
    bins = np.floor(np.log10(deg) * 4)
    rho, w = [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= min_n:
            rho.append(spearmanr(y[m], x[m]).statistic)
            w.append(m.sum())
    return float(np.average(rho, weights=w))


def main() -> None:
    out = {}
    for corpus in ("blog", "text"):
        runs = {E: R / f"{corpus}_E{E}.npz" for E in (5, 25, 100)}
        if not all(p.exists() for p in runs.values()):
            print(corpus, "incomplete")
            continue
        z = {E: np.load(p) for E, p in runs.items()}
        tokens = list(z[100]["tokens"])
        assert all(list(z[E]["tokens"]) == tokens for E in z)
        x = z[100]["x"]
        entry = {"profiles": {}}
        for E in z:
            b = z[E]["balanced_norm"][-1].astype(np.float64)
            pos, m, _ = fig1_curve([x], [b])
            entry["profiles"][f"E{E}"] = dict(
                rate=float(z[E]["tau"][0]),
                tau=float(z[E]["tau"][-1]),
                x=rnd(pos, 1),
                y=rnd(m),
                peak=float(pos[int(np.argmax(m))]),
                top_gap=float(m[-1] - m[-2]),
            )
        W = z[100]["W"].astype(np.float64)
        C = z[100]["C"].astype(np.float64)
        counts = z[100]["counts"]
        q = counts / counts.sum()
        P = pair_counts(corpus, tokens)
        u, radial, fisher = predictors(W, C, P, q)
        norm = {E: z[E]["balanced_norm"][-1].astype(np.float64) for E in z}
        for big in (5, 25):
            shrink = np.log(
                (norm[big] / norm[big].mean()) / (norm[100] / norm[100].mean())
            )
            entry[f"shrink_E{big}_vs_E100"] = {
                name: dict(
                    spearman=float(spearmanr(shrink, v).statistic),
                    within_bins=within_bins(shrink, v, x),
                )
                for name, v in (
                    ("frequency", x),
                    ("uncertainty", u),
                    ("radial", radial),
                    ("fisher", fisher),
                )
            }
        entry["radial_vs_frequency_spearman"] = float(spearmanr(radial, x).statistic)
        out[corpus] = entry
        print(
            corpus,
            json.dumps({k: v for k, v in entry.items() if k != "profiles"}, indent=1),
        )
        for E, pr in entry["profiles"].items():
            print(
                corpus,
                E,
                "rate %.5f peak %.0f gap %.3f"
                % (pr["rate"], pr["peak"], pr["top_gap"]),
            )
    (ROOT / "results/norm_tests/stepsize_real.json").write_text(
        json.dumps(out, indent=1)
    )


if __name__ == "__main__":
    main()
