"""Is large-step hub shrinkage driven by randomness or by one-pair-at-a-time updates?

PR #36 graphs, one-hop window, exact sigmoid (no skip), Gensim's initial vectors,
constant rate r for 0.0628 / r epochs (summed learning rate 0.0628). Variants:
  sampled    Gensim's update: fixed walk order every epoch, 5 sampled negatives
  expected   same fixed order, negatives replaced by their expectation: every node l
             gets weight 5 q_l (l != target), so the run has no randomness at all
  shuffled   sampled negatives, pair order freshly permuted every epoch
If 'expected' still shortens hubs, randomness is not needed: the effect comes from
applying one pair's gradient at a time with a finite step.
Output: results/norm_tests/stochasticity/<variant>_<rate>_s<seed>.npz
Run: .venv/bin/python -I scripts/norm_tests/stochasticity.py expected 0.0314 0
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

TAU = 0.0628


def pairs_of(walks):
    inp, out = [], []
    for w in walks:
        for i in range(len(w)):
            for j in (i - 1, i + 1):
                if 0 <= j < len(w):
                    inp.append(w[j])  # Gensim: input j predicts i
                    out.append(w[i])
    return np.array(inp, np.int64), np.array(out, np.int64)


@numba.njit(cache=True)
def epoch_sampled(inp, out, order, syn0, syn1, cdf, alpha, negative, seed):
    np.random.seed(seed)
    n, d = syn0.shape
    work = np.zeros(d)
    for t in range(len(order)):
        a, b = inp[order[t]], out[order[t]]
        work[:] = 0.0
        for dd in range(negative + 1):
            if dd == 0:
                target, label = b, 1.0
            else:
                target = np.searchsorted(cdf, np.random.random(), side="right")
                if target >= n:
                    target = n - 1
                if target == b:
                    continue
                label = 0.0
            f = 0.0
            for k in range(d):
                f += syn0[a, k] * syn1[target, k]
            g = (label - 1.0 / (1.0 + np.exp(-f))) * alpha
            for k in range(d):
                work[k] += g * syn1[target, k]
                syn1[target, k] += g * syn0[a, k]
        for k in range(d):
            syn0[a, k] += work[k]


@numba.njit(cache=True)
def epoch_expected(inp, out, syn0, syn1, q, alpha, negative):
    n, d = syn0.shape
    work = np.zeros(d)
    for t in range(len(inp)):
        a, b = inp[t], out[t]
        work[:] = 0.0
        f = 0.0
        for k in range(d):
            f += syn0[a, k] * syn1[b, k]
        g = (1.0 - 1.0 / (1.0 + np.exp(-f))) * alpha
        for k in range(d):
            work[k] += g * syn1[b, k]
            syn1[b, k] += g * syn0[a, k]
        for c in range(n):
            if c == b:
                continue
            f = 0.0
            for k in range(d):
                f += syn0[a, k] * syn1[c, k]
            g = -negative * q[c] / (1.0 + np.exp(-f)) * alpha
            for k in range(d):
                work[k] += g * syn1[c, k]
                syn1[c, k] += g * syn0[a, k]
        for k in range(d):
            syn0[a, k] += work[k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["sampled", "expected", "shuffled"])
    p.add_argument("rate", type=float)
    p.add_argument("seed", type=int)
    a = p.parse_args()
    epochs = int(round(TAU / a.rate))
    s = setup(a.seed)
    W, C = initial(s, a.seed)
    inp, out = pairs_of(s["walks_array"])
    q = s["proposal"].copy()
    cdf = np.cumsum(q)
    cdf /= cdf[-1]
    rng = np.random.default_rng(50_000 + a.seed)
    rows = [dict(epoch=0, tau=0.0, **metrics(W, C, s))]
    for e in range(epochs):
        if a.variant == "expected":
            epoch_expected(inp, out, W, C, q, a.rate, T.negatives)
        else:
            order = (
                rng.permutation(len(inp))
                if a.variant == "shuffled"
                else np.arange(len(inp))
            )
            epoch_sampled(
                inp, out, order, W, C, cdf, a.rate, T.negatives, 1000 * a.seed + e
            )
        rows.append(dict(epoch=e + 1, tau=a.rate * (e + 1), **metrics(W, C, s)))
        print(f"{a.variant} {a.rate} s{a.seed} epoch {e + 1}", flush=True)
    outdir = ROOT / "results/norm_tests/stochasticity"
    outdir.mkdir(parents=True, exist_ok=True)
    keys = rows[0].keys()
    np.savez_compressed(
        outdir / f"{a.variant}_{a.rate}_s{a.seed}.npz",
        degree=s["degree"],
        **{k: np.array([r[k] for r in rows]) for k in keys},
    )


if __name__ == "__main__":
    main()
