"""Causal knockout: cancel SGD's first-order penalty during large-step training.

Smith et al. (2021): per-example SGD with step eps follows gradient flow on
L + (eps/4) sum_k R_k with R_k = ||grad l_k||^2. For Gensim's example (center u,
positive context v_b with r = sigma(-u.v_b), sampled negatives v_l with
p_l = sigma(u.v_l)):
    R_k = ||r v_b - sum_l p_l v_l||^2 + ||u||^2 (r^2 + sum_l p_l^2).
'counter' adds, at every example, the step +eps * (eps/4) * grad R_k (evaluated at the
pre-step point) to the ordinary SGD step, which removes the O(eps) penalty. If the
penalty is the mechanism, counter-SGD at a large step recovers the gradient-flow degree
profile at the same summed learning rate; plain SGD ('plain', same code without the
extra term) shows the large-step shortening. 'counter_context' cancels only the
context-gradient part ||u||^2 (r^2 + sum p^2).
PR #36 graphs, one-hop, exact sigmoid, fixed walk order, Gensim's initial vectors.
Output: results/norm_tests/counter/<variant>_<rate>_s<seed>.npz
Run: .venv/bin/python -I scripts/norm_tests/counter_penalty.py counter 0.0314 0
"""

import argparse
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr36_common  # noqa: E402, F401
from clock_replica import initial  # noqa: E402
from pr36_common import ROOT, T, metrics, setup  # noqa: E402
from stochasticity import TAU, pairs_of  # noqa: E402

MODES = {"plain": 0, "counter": 1, "counter_context": 2}


@numba.njit(cache=True)
def epoch(inp, out, syn0, syn1, cdf, alpha, negative, mode, seed):
    np.random.seed(seed)
    n, d = syn0.shape
    K1 = negative + 1
    tgt = np.empty(K1, np.int64)
    lab = np.empty(K1)
    sig = np.empty(K1)
    gu = np.empty(d)
    dU = np.empty(d)
    dV = np.empty((K1, d))
    coef = alpha * alpha / 4.0
    for t in range(len(inp)):
        a, b = inp[t], out[t]
        m = 0
        tgt[0], lab[0] = b, 1.0
        m = 1
        for dd in range(negative):
            c = np.searchsorted(cdf, np.random.random(), side="right")
            if c >= n:
                c = n - 1
            if c == b:
                continue
            tgt[m], lab[m] = c, 0.0
            m += 1
        uu = 0.0
        for k in range(d):
            uu += syn0[a, k] * syn0[a, k]
        for i in range(m):
            f = 0.0
            for k in range(d):
                f += syn0[a, k] * syn1[tgt[i], k]
            sig[i] = 1.0 / (1.0 + np.exp(-f))
        # residuals: e_0 = r = 1 - sigma(s_b) (toward v_b), e_i = p_i for negatives
        # g_u = r v_b - sum_l p_l v_l  (= -grad_u l)
        for k in range(d):
            gu[k] = 0.0
        for i in range(m):
            e = (1.0 - sig[i]) if i == 0 else -sig[i]
            for k in range(d):
                gu[k] += e * syn1[tgt[i], k]
        # gradient of R (center part C and context part D) if mode > 0
        for k in range(d):
            dU[k] = 0.0
        for i in range(m):
            for k in range(d):
                dV[i, k] = 0.0
        if mode > 0:
            Q = 0.0
            for i in range(m):
                e = (1.0 - sig[i]) if i == 0 else sig[i]
                Q += e * e
            # context part D = ||u||^2 Q
            for k in range(d):
                dU[k] += 2.0 * Q * syn0[a, k]
            for i in range(m):
                s1 = sig[i] * (1.0 - sig[i])
                # d(e^2)/ds: -2 e s1 (positive, e = 1 - sigma); +2 e s1 (negative)
                e = (1.0 - sig[i]) if i == 0 else sig[i]
                de2 = -2.0 * e * s1 if i == 0 else 2.0 * e * s1
                for k in range(d):
                    dU[k] += uu * de2 * syn1[tgt[i], k]
                    dV[i, k] += uu * de2 * syn0[a, k]
            if mode == 1:
                # center part C = ||g_u||^2
                for i in range(m):
                    s1 = sig[i] * (1.0 - sig[i])
                    vg = 0.0
                    for k in range(d):
                        vg += syn1[tgt[i], k] * gu[k]
                    if i == 0:
                        # g_u term r v_b, dr/ds = -s1
                        for k in range(d):
                            dU[k] += 2.0 * (-s1) * vg * syn1[tgt[i], k]
                            dV[i, k] += 2.0 * (
                                (1.0 - sig[i]) * gu[k] + (-s1) * vg * syn0[a, k]
                            )
                    else:
                        # g_u term -p v_l, dp/ds = s1
                        for k in range(d):
                            dU[k] += 2.0 * (-s1) * vg * syn1[tgt[i], k]
                            dV[i, k] += 2.0 * (-sig[i] * gu[k] - s1 * vg * syn0[a, k])
        # SGD step (Gensim order: contexts updated in the loop with the old u)
        for i in range(m):
            g = (lab[i] - sig[i]) * alpha
            for k in range(d):
                syn1[tgt[i], k] += g * syn0[a, k] + coef * dV[i, k]
        for k in range(d):
            syn0[a, k] += alpha * gu[k] + coef * dU[k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=list(MODES))
    p.add_argument("rate", type=float)
    p.add_argument("seed", type=int)
    a = p.parse_args()
    epochs = int(round(TAU / a.rate))
    s = setup(a.seed)
    W, C = initial(s, a.seed)
    inp, out = pairs_of(s["walks_array"])
    cdf = np.cumsum(s["proposal"])
    cdf /= cdf[-1]
    rows = [dict(epoch=0, tau=0.0, **metrics(W, C, s))]
    for e in range(epochs):
        epoch(
            inp,
            out,
            W,
            C,
            cdf,
            a.rate,
            T.negatives,
            MODES[a.variant],
            1000 * a.seed + e,
        )
        rows.append(dict(epoch=e + 1, tau=a.rate * (e + 1), **metrics(W, C, s)))
        print(f"{a.variant} {a.rate} s{a.seed} epoch {e + 1}", flush=True)
    outdir = ROOT / "results/norm_tests/counter"
    outdir.mkdir(parents=True, exist_ok=True)
    keys = rows[0].keys()
    np.savez_compressed(
        outdir / f"{a.variant}_{a.rate}_s{a.seed}.npz",
        degree=s["degree"],
        **{k: np.array([r[k] for r in rows]) for k in keys},
    )


if __name__ == "__main__":
    main()
