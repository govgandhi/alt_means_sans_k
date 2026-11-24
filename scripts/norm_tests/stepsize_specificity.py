"""Is the large-step shrinkage specificity-dependent at fixed degree?

PR #36 graphs (5 seeds), window 1. Balanced norms at matched summed learning rate tau
from the clock runs: Gensim constant rates 0.025 and 0.000625, and full-batch gradient
flow (replica_gf-exact). Each profile is divided by its node mean. Shrinkage of node i
is log(norm_step / norm_GF). Context specificity KL_i = KL(P_i || proposal) with P_i the
realized positive-pair row and proposal the Gensim negative law.
Candidate mechanism (Smith et al. 2021, per-example gradient-variance term): large
steps shrink context-diverse (low KL) nodes most. Prediction: shrinkage correlates
positively with KL within degree bins.
Output: results/norm_tests/stepsize_specificity.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/norm_tests"))
import pr36_common  # noqa: E402, F401  (sets sys.path)
from analyze import PR36_EDGES, interp_snap  # noqa: E402
from pr36_common import setup  # noqa: E402

R = ROOT / "results/norm_tests/clock"
RUNS = {"0.025": "gensim_0.025", "0.000625": "gensim_0.000625"}
GF = "replica_gf-exact_gf"
TAUS = (0.0628, 0.1)


def main() -> None:
    out = {}
    pooled = {(r, t): [] for r in RUNS for t in TAUS}
    for seed in range(5):
        s = setup(seed)
        P = s["positive"] / s["positive"].sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = np.where(P > 0, P * np.log(P / s["proposal"][None, :]), 0).sum(1)
        deg = s["degree"]
        gf = np.load(R / f"{GF}_s{seed}.npz")
        for r, name in RUNS.items():
            z = np.load(R / f"{name}_s{seed}.npz")
            assert np.array_equal(z["degree"], gf["degree"])
            for t in TAUS:
                a, b = (
                    interp_snap(z, "balanced_norm", t),
                    interp_snap(gf, "balanced_norm", t),
                )
                if a is None or b is None:
                    continue
                shrink = np.log((a / a.mean()) / (b / b.mean()))
                pooled[(r, t)].append((shrink, deg, kl, seed))
    for (r, t), items in pooled.items():
        if not items:
            continue
        per_seed = []
        for shrink, deg, kl, seed in items:
            lab = np.digitize(deg, PR36_EDGES) - 1
            rho, w = [], []
            for b in np.unique(lab):
                m = lab == b
                if m.sum() >= 20 and np.ptp(kl[m]) > 0:
                    rho.append(spearmanr(shrink[m], kl[m]).statistic)
                    w.append(m.sum())
            z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
            X = np.c_[np.ones(len(deg)), z(np.log(deg)), z(kl)]
            coef, *_ = np.linalg.lstsq(X, shrink, rcond=None)
            per_seed.append(
                dict(
                    within_bin_spearman_kl=float(np.average(rho, weights=w)),
                    spearman_degree=float(spearmanr(shrink, deg).statistic),
                    spearman_kl=float(spearmanr(shrink, kl).statistic),
                    beta_log_degree=float(coef[1]),
                    beta_kl=float(coef[2]),
                    kl_degree_spearman=float(spearmanr(kl, deg).statistic),
                )
            )
        keys = per_seed[0].keys()
        out[f"rate{r}_tau{t}"] = {
            k: dict(
                mean=float(np.mean([p[k] for p in per_seed])),
                se=float(
                    np.std([p[k] for p in per_seed], ddof=1) / np.sqrt(len(per_seed))
                ),
            )
            for k in keys
        }
        print(r, t, {k: round(v["mean"], 3) for k, v in out[f"rate{r}_tau{t}"].items()})
    (ROOT / "results/norm_tests/stepsize_specificity.json").write_text(
        json.dumps(out, indent=1)
    )


if __name__ == "__main__":
    main()
