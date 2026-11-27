# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "numba", "networkx", "matplotlib"]
# ///
"""Minimal Gensim-equivalent skip-gram negative sampling (sg=1, hs=0) in numba.

Mirrors gensim w2v_inner fast_sentence_sg_neg: input = syn0[context word], output =
syn1neg[center word]; negatives drawn from count^exponent, skipped if equal to target;
updates skipped when |score| >= 6; syn0 init uniform(-0.5, 0.5)/d, syn1neg init 0;
linear lr decay alpha -> min_alpha over all sentences; shrinking windows; subsampling.

Extras for diagnosis: fresh=True draws a new iid corpus every epoch (infinite data).
"""

from pathlib import Path

import numba as nb
import numpy as np

# Per-run outputs (*.npz, gitignored): regenerate with run_all.sh.
OUT = Path(__file__).resolve().parents[2] / "results" / "norm_front" / "runs"


@nb.njit(fastmath=True, cache=True)
def _epoch(
    walks,
    fresh,
    tok_cum,
    neg_cum,
    keep,
    W,
    C,
    a_start,
    a_end,
    window,
    shrink,
    negative,
    n_upd,
    eta2,
    seed,
    freeze_c,
):
    np.random.seed(seed)
    S, L = walks.shape
    d = W.shape[1]
    buf = np.empty(L, np.int32)
    work = np.zeros(d, np.float32)
    for s in range(S):
        alpha = np.float32(a_start - (a_start - a_end) * s / S)
        m = 0
        for t in range(L):
            if fresh:
                w = np.searchsorted(
                    tok_cum, np.random.random() * tok_cum[-1], side="right"
                )
            else:
                w = walks[s, t]
            if keep[w] >= 1.0 or keep[w] > np.random.random():
                buf[m] = w
                m += 1
        for i in range(m):
            b = np.random.randint(0, window) if shrink else 0
            lo = max(0, i - window + b)
            hi = min(m, i + window + 1 - b)
            for j in range(lo, hi):
                if j == i:
                    continue
                word = buf[i]
                inp = buf[j]
                for k in range(d):
                    work[k] = 0.0
                for dd in range(negative + 1):
                    if dd == 0:
                        tgt = word
                        label = np.float32(1.0)
                    else:
                        tgt = np.searchsorted(
                            neg_cum, np.random.random() * neg_cum[-1], side="right"
                        )
                        if tgt == word:
                            continue
                        label = np.float32(0.0)
                    f = np.float32(0.0)
                    for k in range(d):
                        f += W[inp, k] * C[tgt, k]
                    if f <= -6.0 or f >= 6.0:
                        continue
                    g = (
                        label - np.float32(1.0) / (np.float32(1.0) + np.exp(-f))
                    ) * alpha
                    for k in range(d):
                        work[k] += g * C[tgt, k]
                    if not freeze_c:
                        for k in range(d):
                            C[tgt, k] += g * W[inp, k]
                for k in range(d):
                    W[inp, k] += work[k]
                n_upd[inp] += 1
                eta2[inp] += alpha * alpha


def keep_probability(counts, sample):
    if sample == 0:
        return np.ones_like(counts, dtype=float)
    thr = sample * counts.sum()
    with np.errstate(divide="ignore"):
        return np.minimum(1.0, (np.sqrt(counts / thr) + 1) * (thr / counts))


def train(
    walks,
    prob,
    *,
    dim,
    window,
    shrink,
    negative,
    exponent,
    sample,
    epochs,
    alpha=0.025,
    min_alpha=1e-4,
    seed=0,
    fresh=False,
    callback=None,
    schedule=None,
    freeze_c=False,
    shuffle=False,
    corpora=None,
):
    """walks: int32 [S, L] corpus (also defines counts); prob: token distribution for fresh."""
    N = len(prob)
    counts = np.bincount(walks.ravel(), minlength=N).astype(float)
    neg_cum = np.cumsum(counts**exponent)
    tok_cum = np.cumsum(prob)
    keep = keep_probability(counts, sample)
    rng = np.random.default_rng(seed)
    W = ((rng.random((N, dim)) - 0.5) / dim).astype(np.float32)
    C = np.zeros((N, dim), np.float32)
    n_upd = np.zeros(N, np.int64)
    eta2 = np.zeros(N, np.float64)
    if schedule is None:  # linear decay over the whole run, like Gensim
        edges = alpha - (alpha - min_alpha) * np.arange(epochs + 1) / epochs
        schedule = list(zip(edges[:-1], edges[1:]))
    for ep, (a0, a1) in enumerate(schedule):
        corpus = walks[rng.permutation(len(walks))] if shuffle and ep > 0 else walks
        if corpora is not None:
            corpus = corpora[ep]
        _epoch(
            corpus,
            fresh and ep > 0,
            tok_cum,
            neg_cum,
            keep,
            W,
            C,
            a0,
            a1,
            window,
            shrink,
            negative,
            n_upd,
            eta2,
            seed * 1000 + ep,
            freeze_c,
        )
        if callback is not None:
            callback(ep, W, C, n_upd, eta2)
    return W, C, counts, n_upd, eta2


def balanced_left(W, C):
    """Left factor of the balanced SVD gauge of M = W C^T (rows), without forming M."""
    W = W.astype(np.float64)
    C = C.astype(np.float64)
    Qw, Rw = np.linalg.qr(W)
    Qc, Rc = np.linalg.qr(C)
    U, s, Vt = np.linalg.svd(Rw @ Rc.T)
    return Qw @ (U * np.sqrt(s)), s


def ba_corpus(seed, N=10_000, m=5, walks_per_node=10, length=80):
    import networkx as nx

    g = nx.barabasi_albert_graph(N, m, seed=seed)
    deg = np.array([d for _, d in sorted(g.degree())], dtype=float)
    p = deg / deg.sum()
    walks = np.random.default_rng(5000 + seed).choice(
        N, size=(walks_per_node * N, length), p=p
    )
    return walks.astype(np.int32), deg, p


def binned(x, y, per_bin=10):
    lo, hi = np.floor(np.log10(x.min()) * 4) / 4, np.ceil(np.log10(x.max()) * 4) / 4
    edges = 10 ** np.arange(lo, hi + 0.25, 0.25)
    idx = np.digitize(x, edges)
    xs, ys = [], []
    for b in np.unique(idx):
        sel = idx == b
        if sel.sum() >= per_bin:
            xs.append(np.exp(np.log(x[sel]).mean()))
            ys.append(y[sel].mean())
    return np.array(xs), np.array(ys)


SETTINGS = {
    "A": dict(
        dim=100, window=5, shrink=True, negative=5, exponent=0.75, sample=1e-3, epochs=5
    ),
    "B": dict(
        dim=300, window=1, shrink=False, negative=5, exponent=1.0, sample=0.0, epochs=5
    ),
    "SW": dict(
        dim=100,
        window=10,
        shrink=True,
        negative=5,
        exponent=0.75,
        sample=1e-4,
        epochs=20,
    ),
}


def graph_walks(graph_seed, walk_seed, N=10_000, m=5, walks_per_node=10, length=80):
    """Uniform random walks on the actual BA graph, started in proportion to degree."""
    import networkx as nx
    g = nx.barabasi_albert_graph(N, m, seed=graph_seed)
    A = nx.to_scipy_sparse_array(g, nodelist=range(N), format="csr")
    indptr, indices = A.indptr, A.indices
    deg = np.diff(indptr).astype(float)
    rng = np.random.default_rng(walk_seed)
    S = walks_per_node * N
    cur = rng.choice(N, size=S, p=deg / deg.sum())
    out = np.empty((S, length), np.int32)
    out[:, 0] = cur
    for t in range(1, length):
        cur = indices[indptr[cur] + (rng.random(S) * deg[cur]).astype(np.int64)]
        out[:, t] = cur
    return out, deg, deg / deg.sum()
