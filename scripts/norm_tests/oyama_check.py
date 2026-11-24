"""Does the norm-KL relation of Oyama, Yokoi & Shimodaira (EMNLP 2023) hold in our runs?

Factors from oyama_train.py (BlogCatalog walks or hep-th text; checkpoints 1-50 epochs).
Per checkpoint, with q the corpus unigram and r the negative-sampling law
(Oyama Eqs. 4-5):
  model KL    KL(p_hat(.|w) || q), p_hat(c|w) = r(c) exp(u_w.v_c) / Z_w  (exact)
  quadratic   (1/2)(u_w - u_bar)^T G (u_w - u_bar), u_bar = sum_w q(w) u_w,
              G = sum_c q(c)(v_c - v_bar)(v_c - v_bar)^T          (their approximation)
  empirical   KL(C_i || pi): BlogCatalog walk contexts (specificity.py) or hep-th
              window co-occurrence; subsampling ignored in both
  balanced    balanced left norm of (W, C); raw ||u_w||
Readouts: Spearman of each norm readout with empirical KL (global and within
quarter-decade degree bins), Pearson of quadratic vs model KL, and the binned
norm-degree profile of each readout.
Output: results/norm_tests/oyama.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/norm_tests"))
from analyze import fig1_curve, rnd  # noqa: E402
from simulate_norm_frequency import balanced_left_norm  # noqa: E402
from specificity import kl_by_window  # noqa: E402


def model_kl(W, C, base, ref, chunk=1024):
    """KL(p_hat(.|w) || ref), p_hat(c|w) proportional to base(c) exp(u_w.v_c)."""
    out = np.empty(len(W))
    logb, logr = np.log(base), np.log(ref)
    for s in range(0, len(W), chunk):
        L = W[s : s + chunk] @ C.T + logb[None, :]
        L -= L.max(1, keepdims=True)
        logP = L - np.log(np.exp(L).sum(1, keepdims=True))
        out[s : s + chunk] = (np.exp(logP) * (logP - logr[None, :])).sum(1)
    return out


def quadratic(W, C, q):
    vbar = q @ C
    G = (C - vbar).T @ ((C - vbar) * q[:, None])
    D = W - q @ W
    return 0.5 * np.einsum("ij,jk,ik->i", D, G, D)


def within_bin(norm, degree, spec):
    bins = np.floor(np.log10(degree) * 4)
    rho, w = [], []
    for b in np.unique(bins):
        m = bins == b
        if m.sum() >= 30:
            rho.append(spearmanr(norm[m], spec[m]).statistic)
            w.append(m.sum())
    return float(np.average(rho, weights=w))


def text_kl(tokens, window, shrink):
    """KL(C_i || pi) from hep-th window co-occurrence over the model vocabulary.

    Out-of-vocabulary tokens are dropped before windowing, as in Gensim; with a
    shrinking window, offset t has weight (w - t + 1) / w. Subsampling is ignored.
    """
    vocab = {t: i for i, t in enumerate(tokens)}
    V = len(tokens)
    rows, cols, vals = [], [], []
    with open(ROOT / "data/derived/step0/hepth_tokens.txt") as f:
        for line in f:
            ids = np.array([vocab[w] for w in line.split() if w in vocab], np.int64)
            for t in range(1, window + 1):
                if len(ids) <= t:
                    break
                wt = (window - t + 1) / window if shrink else 1.0
                a, b = ids[:-t], ids[t:]
                rows += [a, b]
                cols += [b, a]
                vals += [np.full(len(a), wt)] * 2
    M = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(V, V),
    ).tocsr()
    M.sum_duplicates()
    pi = np.asarray(M.sum(0)).ravel()
    pi /= pi.sum()
    rs = np.asarray(M.sum(1)).ravel()
    M = sp.diags(1 / rs) @ M
    M = M.tocoo()
    terms = M.data * np.log(M.data / pi[M.col])
    return np.bincount(M.row, weights=terms, minlength=V)


def main() -> None:
    blog = None
    result = {}
    for path in sorted((ROOT / "results/norm_tests/oyama").glob("*.npz")):
        name = path.stem
        z = np.load(path)
        st = json.loads(str(z["settings"]))
        counts = z["counts"]
        q = counts / counts.sum()
        ns = counts ** st["ns_exponent"]
        ns /= ns.sum()
        if name.startswith("blog"):
            if blog is None:
                blog = kl_by_window()
            spec, k = blog
            idx = np.array([int(t[1:]) for t in z["tokens"]])
            deg, kl_emp = k[idx], spec[st["window"]][idx]
        else:
            deg = counts
            kl_emp = text_kl(list(z["tokens"]), st["window"], st["shrink_windows"])
        rows = []
        for e, W, C in zip(z["epochs"], z["W"], z["C"]):
            W, C = W.astype(np.float64), C.astype(np.float64)
            readouts = {
                "raw": np.linalg.norm(W, axis=1),
                "balanced": balanced_left_norm(W, C),
                "model_kl": model_kl(W, C, ns, q),
                "quadratic": quadratic(W, C, q),
            }
            row = {"epoch": int(e)}
            for key, v in readouts.items():
                x = np.sqrt(np.maximum(v, 0)) if key in ("model_kl", "quadratic") else v
                pos, m, _ = fig1_curve([deg], [x])
                row[key] = dict(
                    spearman_kl=float(spearmanr(v, kl_emp).statistic),
                    within_bin_kl=within_bin(v, deg, kl_emp),
                    spearman_degree=float(spearmanr(v, deg).statistic),
                    profile_x=rnd(pos, 1),
                    profile=rnd(m),
                    peak=float(pos[int(np.argmax(m))]),
                )
            row["quadratic_vs_model_kl_pearson"] = float(
                pearsonr(readouts["quadratic"], readouts["model_kl"]).statistic
            )
            row["model_kl_vs_empirical_pearson"] = float(
                pearsonr(readouts["model_kl"], kl_emp).statistic
            )
            row["model_kl_mean"] = float(readouts["model_kl"].mean())
            row["empirical_kl_mean"] = float(kl_emp.mean())
            rows.append(row)
            print(
                name,
                e,
                {
                    n: round(row[n]["within_bin_kl"], 3)
                    for n in ("raw", "balanced", "model_kl", "quadratic")
                },
                round(row["quadratic_vs_model_kl_pearson"], 3),
                round(row["model_kl_vs_empirical_pearson"], 3),
                flush=True,
            )
        result[name] = rows
    (ROOT / "results/norm_tests/oyama.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
