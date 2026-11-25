"""Does a corpus-only 'context genericity' predict which items large steps shorten?

At the SGNS optimum a pair's predicted probability is sigma* = P/(P + N), so its
contribution to the first-order SGD penalty, (P + N) sigma(1 - sigma), equals
P r / (1 + r) with r = N/P = k q_j / P(j|i) = k exp(-PMI_ij). Per update of item i,
the penalty is therefore
    g_i = E_{j ~ P(.|i)} [ r_ij / (1 + r_ij) ],
the average over its contexts of how close each pair is to chance. One-hop on a graph:
r_ij = k d_i d_j / vol. Prediction: shrinkage under large steps (log of mean-normalized
large-step over small-step balanced norm) falls with g_i within frequency bins, and with
F_i = n_i g_i globally; KL (the mean PMI) is the wrong functional and should do worse.
Uses only corpus statistics; no trained model enters the predictor.
Output: results/norm_tests/genericity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/norm_tests"))
import pr36_common  # noqa: E402, F401
from analyze import PR36_EDGES, interp_snap  # noqa: E402
from pr36_common import setup  # noqa: E402
from stepsize_real_check import pair_counts, within_bins  # noqa: E402

K = 5


def stats(P, q):
    """P: sparse or dense positive counts (rows = items), q: negative law."""
    P = sp.csr_matrix(P)
    n = np.asarray(P.sum(1)).ravel()
    Pc = P.tocoo()
    cond = Pc.data / n[Pc.row]
    r = K * q[Pc.col] / cond
    g = np.bincount(Pc.row, weights=Pc.data * r / (1 + r), minlength=P.shape[0]) / n
    pi = np.asarray(P.sum(0)).ravel()
    pi = pi / pi.sum()
    kl = np.bincount(
        Pc.row, weights=cond * np.log(cond / pi[Pc.col]), minlength=P.shape[0]
    )
    return n, g, n * g, kl


def readout(shrink, x, preds):
    return {
        name: dict(
            spearman=float(spearmanr(shrink, v).statistic),
            within_bins=within_bins(shrink, v, x, min_n=20),
        )
        for name, v in preds.items()
    }


def pr36_within(shrink, deg, v):
    lab = np.digitize(deg, PR36_EDGES) - 1
    rho, w = [], []
    for b in np.unique(lab):
        m = lab == b
        if m.sum() >= 20:
            rho.append(spearmanr(shrink[m], v[m]).statistic)
            w.append(m.sum())
    return float(np.average(rho, weights=w))


def main() -> None:
    out = {"pr36": {}}
    R = ROOT / "results/norm_tests/clock"
    for rate in ("0.025", "0.0025"):
        rows = []
        for seed in range(5):
            s = setup(seed)
            n, g, F, kl = stats(s["positive"], s["proposal"])
            gf = np.load(R / f"replica_gf-exact_gf_s{seed}.npz")
            z = np.load(R / f"gensim_{rate}_s{seed}.npz")
            a = interp_snap(z, "balanced_norm", 0.0628)
            b = interp_snap(gf, "balanced_norm", 0.0628)
            shrink = np.log((a / a.mean()) / (b / b.mean()))
            deg = s["degree"]
            preds = {"degree": deg, "kl": kl, "genericity": g, "fisher_corpus": F}
            rows.append(
                {
                    k: dict(
                        spearman=float(spearmanr(shrink, v).statistic),
                        within_bins=pr36_within(shrink, deg, v),
                    )
                    for k, v in preds.items()
                }
            )
            if seed == 0:
                out["pr36"]["genericity_vs_degree_spearman"] = float(
                    spearmanr(g, deg).statistic
                )
        out["pr36"][rate] = {
            k: {
                m: float(np.mean([r[k][m] for r in rows]))
                for m in ("spearman", "within_bins")
            }
            for k in rows[0]
        }
    for corpus in ("blog", "text"):
        z = {
            E: np.load(ROOT / f"results/norm_tests/stepsize_real/{corpus}_E{E}.npz")
            for E in (5, 25, 100)
        }
        tokens = list(z[100]["tokens"])
        counts = z[100]["counts"]
        q = counts / counts.sum()
        P = pair_counts(corpus, tokens)
        n, g, F, kl = stats(P, q)
        x = z[100]["x"]
        norm = {E: z[E]["balanced_norm"][-1].astype(np.float64) for E in z}
        out[corpus] = {}
        for big in (5, 25):
            shrink = np.log(
                (norm[big] / norm[big].mean()) / (norm[100] / norm[100].mean())
            )
            out[corpus][f"E{big}"] = readout(
                shrink,
                x,
                {"frequency": x, "kl": kl, "genericity": g, "fisher_corpus": F},
            )
    (ROOT / "results/norm_tests/genericity.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        for kk, vv in v.items():
            if isinstance(vv, dict):
                print(
                    k,
                    kk,
                    {
                        n_: (round(m["spearman"], 2), round(m["within_bins"], 2))
                        for n_, m in vv.items()
                    },
                )


if __name__ == "__main__":
    main()
