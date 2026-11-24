"""Test 3 (intervention): does Gensim's skip of scores outside (-6, 6) cause the
schedule dependence?

Numba SGD replica of Gensim's skip-gram negative-sampling update on the PR #36 corpus
(window 1, 5 negatives, exponent 1, same pair order, same initial vectors as Gensim).
Variants: 'skip' (no update when |score| >= 6, as Gensim) and 'exact' (exact sigmoid,
always update). Same schedules and readouts as clock_gensim.py.
'gf' runs full-batch gradient descent on the per-epoch objective with tiny steps
(gradient flow in tau, the summed learning rate), from the same start.
Output: results/norm_tests/clock/replica_<variant>_<schedule>_s<seed>.npz
Run: .venv/bin/python -I scripts/norm_tests/clock_replica.py exact orig 0
     .venv/bin/python -I scripts/norm_tests/clock_replica.py gf-exact gf 0
"""

import argparse
import sys
from pathlib import Path

import numba
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr36_common  # noqa: E402, F401
from clock_gensim import TAU_END, rate, tau  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from pr36_common import MAX_EXP, ROOT, T, gensim_model, metrics, setup  # noqa: E402


@numba.njit(cache=True)
def train_epoch(walks, syn0, syn1, cdf, a_start, a_end, negative, skip, seed):
    np.random.seed(seed)
    n_walks, length = walks.shape
    d = syn0.shape[1]
    n = syn0.shape[0]
    work = np.zeros(d)
    for w in range(n_walks):
        alpha = a_start + (a_end - a_start) * w / n_walks
        for i in range(length):
            for j in (i - 1, i + 1):
                if j < 0 or j >= length:
                    continue
                inp, out = walks[w, j], walks[w, i]  # Gensim: input j predicts i
                work[:] = 0.0
                for dd in range(negative + 1):
                    if dd == 0:
                        target, label = out, 1.0
                    else:
                        target = np.searchsorted(cdf, np.random.random(), side="right")
                        if target >= n:
                            target = n - 1
                        if target == out:
                            continue
                        label = 0.0
                    f = 0.0
                    for k in range(d):
                        f += syn0[inp, k] * syn1[target, k]
                    if skip and (f <= -6.0 or f >= 6.0):
                        continue
                    g = (label - 1.0 / (1.0 + np.exp(-f))) * alpha
                    for k in range(d):
                        work[k] += g * syn1[target, k]
                        syn1[target, k] += g * syn0[inp, k]
                for k in range(d):
                    syn0[inp, k] += work[k]


def initial(s, seed):
    model = gensim_model(s["walks"], seed, 0.025, 1e-4, 1, Word2Vec)
    slots = np.array([model.wv.key_to_index[n] for n in range(T.nodes)])
    return (
        model.wv.vectors[slots].astype(np.float64).copy(),
        model.syn1neg[slots].astype(np.float64).copy(),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["skip", "exact", "gf-exact", "gf-skip"])
    p.add_argument("schedule")
    p.add_argument("seed", type=int)
    p.add_argument("--step", type=float, default=2e-6, help="gf step in tau")
    p.add_argument("--out", type=Path, default=ROOT / "results/norm_tests/clock")
    a = p.parse_args()
    s = setup(a.seed)
    W, C = initial(s, a.seed)
    rows = []

    def capture(epoch, t):
        rows.append(dict(epoch=epoch, tau=t, **metrics(W, C, s)))

    capture(0, 0.0)
    if a.variant.startswith("gf"):
        P, N = s["positive"], s["negative"]
        marks = np.round(np.arange(1, 41) * TAU_END / 40, 10)
        t, mark = 0.0, 0
        while mark < len(marks):
            S = W @ C.T
            D = -P * expit(-S) + N * expit(S)  # dL/dS, L summed over one epoch
            if a.variant == "gf-skip":
                D[(S <= -MAX_EXP) | (S >= MAX_EXP)] = 0.0
            gW, gC = D @ C, D.T @ W
            W -= a.step * gW
            C -= a.step * gC
            t += a.step
            if t >= marks[mark] - 1e-12:
                capture(mark + 1, float(marks[mark]))
                print(
                    f"gf tau {t:.4f} loss {rows[-1]['expected_loss']:.5f}", flush=True
                )
                mark += 1
    else:
        epochs = 20 if a.schedule == "orig" else int(round(TAU_END / float(a.schedule)))
        cdf = np.cumsum(s["proposal"])
        cdf /= cdf[-1]
        walks = s["walks_array"]
        for e in range(epochs):
            train_epoch(
                walks,
                W,
                C,
                cdf,
                rate(a.schedule, e),
                rate(a.schedule, e + 1),
                T.negatives,
                a.variant == "skip",
                1000 * a.seed + e,
            )
            capture(e + 1, tau(a.schedule, e + 1))
    a.out.mkdir(parents=True, exist_ok=True)
    keys = rows[0].keys()
    np.savez_compressed(
        a.out / f"replica_{a.variant}_{a.schedule}_s{a.seed}.npz",
        degree=s["degree"],
        **{k: np.array([r[k] for r in rows]) for k in keys},
    )
    print(f"replica {a.variant} {a.schedule} s{a.seed} done", flush=True)


if __name__ == "__main__":
    main()
