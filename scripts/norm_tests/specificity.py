"""Test 5: at fixed degree, does norm track context specificity (local hubs)?

BlogCatalog. Context distribution of node i at window w with Gensim's shrinking window:
C_i = sum_t a_t (P^t)_i / sum_t a_t, a_t = (w - t + 1) / w (fixed window 1: a_1 = 1),
P the walk transition matrix. Specificity = KL(C_i || pi), pi the stationary law
(start law of the walks). Subsampling (settings A, SW) is ignored here.
Readouts per run: Spearman(norm, KL) within quarter-decade degree bins (mean over bins,
node-weighted), and OLS of standardized log balanced norm on standardized log degree
and KL. Uses committed Fig. 1(c) runs and the per-epoch animation runs.
Output: results/norm_tests/specificity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data/external/blogcatalog/downloads/blogcatalog.mat"
WINDOW = {"A": 5, "B": 1, "SW": 10, "A50": 5, "B50": 1}


def kl_by_window() -> tuple[dict[int, np.ndarray], np.ndarray]:
    A = sp.csr_matrix(scipy.io.loadmat(MAT)["network"], dtype=np.float64)
    k = np.asarray(A.sum(1)).ravel()
    pi = k / k.sum()
    P = sp.diags(1 / k) @ A
    out, X = {}, None
    acc = {w: np.zeros(A.shape, dtype=np.float32) for w in (5, 10)}
    for t in range(1, 11):
        X = (
            P.toarray().astype(np.float32)
            if X is None
            else (P.T @ X.T).T.astype(np.float32)
        )  # X = P^t (rows)
        if t == 1:
            out[1] = kl(X, pi)
        for w in acc:
            if t <= w:
                acc[w] += np.float32((w - t + 1) / w) * X
        print(f"P^{t}", flush=True)
    for w, M in acc.items():
        out[w] = kl(M / M.sum(1, keepdims=True), pi)
    return out, k


def kl(C: np.ndarray, pi: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(C > 0, C * np.log(C / pi[None, :].astype(np.float32)), 0.0)
    return terms.sum(1).astype(np.float64)


def readout(norm, degree, spec) -> dict:
    logd = np.log10(degree)
    bins = np.floor(logd * 4)
    rho, weight = [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= 30 and np.ptp(spec[m]) > 0:
            rho.append(spearmanr(norm[m], spec[m]).statistic)
            weight.append(m.sum())
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    X = np.c_[np.ones(len(norm)), z(np.log(degree)), z(spec)]
    coef, *_ = np.linalg.lstsq(X, z(np.log(norm)), rcond=None)
    return dict(
        within_bin_spearman=float(np.average(rho, weights=weight)),
        bins=len(rho),
        beta_log_degree=float(coef[1]),
        beta_kl=float(coef[2]),
        spearman_degree_kl=float(spearmanr(degree, spec).statistic),
    )


def main() -> None:
    spec, k = kl_by_window()
    result = {"kl_vs_degree": {}, "committed": {}, "epochs": {}}
    for w, v in spec.items():
        result["kl_vs_degree"][w] = float(spearmanr(k, v).statistic)
    for col in ("A", "B", "SW"):
        rows = []
        for s in range(5):
            z = np.load(ROOT / f"results/step0/runs/blog_{col}_s{s}.npz")
            idx = np.array([int(t[1:]) for t in z["tokens"]])
            rows.append(readout(z["balanced_norm"], k[idx], spec[WINDOW[col]][idx]))
        result["committed"][col] = {
            key: float(np.mean([r[key] for r in rows])) for key in rows[0]
        } | {"within_bin_spearman_per_seed": [r["within_bin_spearman"] for r in rows]}
    for col in ("B50", "A50"):
        path = ROOT / f"results/norm_tests/anim/blog_{col}_reused_s0.npz"
        if not path.exists():
            continue
        z = np.load(path)
        idx = np.array([int(t[1:]) for t in z["tokens"]])
        result["epochs"][col] = [
            readout(b, k[idx], spec[WINDOW[col]][idx]) for b in z["balanced_norm"]
        ]
    out = ROOT / "results/norm_tests/specificity.json"
    out.write_text(json.dumps(result, indent=1))
    json.dump(result["kl_vs_degree"] | result["committed"], sys.stdout, indent=1)


if __name__ == "__main__":
    main()
