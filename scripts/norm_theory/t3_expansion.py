"""T3: why does Oyama's quadratic norm-KL relation fail at window 1 on graphs?

For node w and context c let delta_wc = <u_w - u_bar, v_c> with u_bar the q-weighted
mean center vector, and K_w(t) = log E_r[exp(t delta_wc)] its cumulant generating
function under the model's base law r (negative-sampling law; r = q at exponent 1).
The model's conditional is p(c|w) proportional to r(c) exp(delta_wc), so
    KL(p(.|w) || r) = K_w'(1) - K_w(1) = kappa2/2 + kappa3/3 + kappa4/8 + ...,
and Oyama et al.'s Eq. 11 keeps kappa2/2 = (1/2) Var_r(delta) (their quadratic form
with G = Cov_r(v)). The expansion is accurate when |kappa3/3| << kappa2/2, i.e. when
delta's spread is small or nearly Gaussian. We also check their Eq. 10 assumption that
the unigram equals the model's conditional at u_bar: KL(q || p(.|u_bar)).
Factors: results/norm_tests/oyama/*.npz (BlogCatalog, hep-th) at epochs 5 and 50.
Output: results/norm_theory/t3.json
Run: .venv/bin/python -I scripts/norm_theory/t3_expansion.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RUNS = ["blog_B50_s0", "blog_B50w5_s0", "blog_A50_s0", "text_B50_s0", "text_A50_s0"]


def stats(W, C, q, r, chunk=512):
    ubar = q @ W
    D = W - ubar
    out = {k: np.empty(len(W)) for k in ("k2", "k3", "k4", "kl", "sd")}
    for s in range(0, len(W), chunk):
        X = D[s : s + chunk] @ C.T  # delta
        mu = X @ r
        Y = X - mu[:, None]
        k2 = (Y**2) @ r
        k3 = (Y**3) @ r
        m4 = (Y**4) @ r
        L = X + np.log(r)[None, :]
        L -= L.max(1, keepdims=True)
        logP = L - np.log(np.exp(L).sum(1, keepdims=True))
        out["kl"][s : s + chunk] = (np.exp(logP) * (logP - np.log(r)[None, :])).sum(1)
        out["k2"][s : s + chunk] = k2
        out["k3"][s : s + chunk] = k3
        out["k4"][s : s + chunk] = m4 - 3 * k2**2
        out["sd"][s : s + chunk] = np.sqrt(k2)
    # Eq. 10 check: model conditional at u_bar vs the unigram q
    L0 = ubar @ C.T + np.log(r)
    L0 -= L0.max()
    p0 = np.exp(L0) / np.exp(L0).sum()
    eq10 = float((q * np.log(q / p0)).sum())
    return out, eq10


def main() -> None:
    res = {}
    with threadpool_limits(1):
        for name in RUNS:
            path = ROOT / f"results/norm_tests/oyama/{name}.npz"
            if not path.exists():
                continue
            z = np.load(path)
            st = json.loads(str(z["settings"]))
            counts = z["counts"]
            q = counts / counts.sum()
            r = counts ** st["ns_exponent"]
            r /= r.sum()
            rows = []
            for e, W, C in zip(z["epochs"], z["W"], z["C"]):
                if int(e) not in (5, 50):
                    continue
                o, eq10 = stats(W.astype(np.float64), C.astype(np.float64), q, r)
                quad = o["k2"] / 2
                third = o["k3"] / 3
                rows.append(
                    dict(
                        epoch=int(e),
                        window=st["window"],
                        median_sd_delta=float(np.median(o["sd"])),
                        median_rel_third=float(np.median(np.abs(third) / quad)),
                        median_rel_error_quadratic=float(
                            np.median(np.abs(o["kl"] - quad) / o["kl"])
                        ),
                        pearson_kl_quad=float(pearsonr(o["kl"], quad).statistic),
                        pearson_kl_quad_plus_third=float(
                            pearsonr(o["kl"], quad + third).statistic
                        ),
                        mean_kl=float(o["kl"].mean()),
                        mean_quad=float(quad.mean()),
                        eq10_kl_q_vs_model_at_ubar=eq10,
                    )
                )
                print(name, rows[-1], flush=True)
            res[name] = rows
    (ROOT / "results/norm_theory/t3.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
