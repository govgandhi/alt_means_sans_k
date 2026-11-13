"""R6 check: does retained-token exposure scale like sqrt(frequency) in the paper's corpora?

Reproduces the calibration corpora (configuration graphs seeds 0-4, 300n stationary
walks of length five) with the worktree's own generators, applies Gensim 4.4.0's
retention rule token by token (keep = sqrt(t/p) + t/p, capped at one, redrawn per
epoch exactly as in train_batch_sg), and measures per node:
  * retained tokens (the m_i of Eq. 38),
  * realized window-one positive pairs after deletion (the quantity Gensim trains on),
  * the fraction of retained pairs whose partner was not adjacent in the raw walk.
Reports the local log-log slope across degree bins next to Eq. (39)'s elasticity.

Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_subsampling_exposure.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
WT = "<scratch>/round/wt"
sys.path.insert(0, WT + "/scripts")

import numpy as np  # noqa: E402
from simulate_norm_frequency import configuration_graph, graph_arrays, sample_walks  # noqa: E402

THRESHOLDS = (3e-4, 1e-3, 3e-3)
EPOCHS = 3  # Monte Carlo epochs per threshold


def retained_statistics(walks: np.ndarray, keep: np.ndarray, rng: np.random.Generator):
    """Vectorised over walks: pair (a, b) survives iff both kept and all between dropped."""
    n = len(keep)
    tokens = np.zeros(n)
    pairs = np.zeros(n)
    skip_pairs = np.zeros(n)
    length = walks.shape[1]
    for _ in range(EPOCHS):
        mask = rng.random(walks.shape) < keep[walks]
        tokens += np.bincount(walks[mask], minlength=n)
        for a in range(length):
            for b in range(a + 1, length):
                alive = mask[:, a] & mask[:, b]
                for c in range(a + 1, b):
                    alive &= ~mask[:, c]
                both = np.concatenate((walks[alive, a], walks[alive, b]))
                add = np.bincount(both, minlength=n)
                pairs += add
                if b - a > 1:
                    skip_pairs += add
    return tokens / EPOCHS, pairs / EPOCHS, skip_pairs / EPOCHS


def bin_slopes(x: np.ndarray, y: np.ndarray, degree: np.ndarray, bins: int = 8):
    log_d = np.log(degree)
    edges = np.unique(np.quantile(log_d, np.linspace(0, 1, bins + 1)))
    labels = np.clip(np.digitize(log_d, edges[1:-1], right=True), 0, len(edges) - 2)
    centers, lx, ly = [], [], []
    for b in range(len(edges) - 1):
        chosen = labels == b
        if chosen.any():
            centers.append(float(np.exp(log_d[chosen].mean())))
            lx.append(float(np.log(x[chosen]).mean()))
            ly.append(float(np.log(y[chosen]).mean()))
    lx, ly = np.asarray(lx), np.asarray(ly)
    return np.asarray(centers), np.diff(ly) / np.diff(lx)


def main() -> None:
    rows = []
    for seed in range(5):
        graph = configuration_graph(600, 3, seed)
        _, neighbors, degree = graph_arrays(graph)
        walks, counts = sample_walks(
            neighbors,
            degree / degree.sum(),
            600 * 300,
            5,
            np.random.default_rng(seed + 10_000),
        )
        walks = np.asarray(walks)
        p = counts / counts.sum()
        rng = np.random.default_rng(2026 + seed)
        base_pairs = np.zeros(600)
        np.add.at(base_pairs, walks[:, :-1].ravel(), 1)
        np.add.at(base_pairs, walks[:, 1:].ravel(), 1)
        for t in THRESHOLDS:
            keep = np.minimum(1.0, np.sqrt(t / p) + t / p)
            tok, pairs, skip = retained_statistics(walks, keep, rng)
            # closed-form expectation of retained tokens (Eq. 38) vs Monte Carlo
            expected = counts * keep
            centers, tok_slope = bin_slopes(counts, tok, degree)
            _, pair_slope = bin_slopes(counts, pairs, degree)
            elasticity = 1 / (2 * (1 + np.sqrt(t / p)))
            elasticity[keep >= 1] = 1.0
            hub = degree >= np.quantile(degree, 0.95)
            leaf = degree <= 3
            rows.append(
                {
                    "seed": seed,
                    "t": t,
                    "subsampled_nodes": int((keep < 1).sum()),
                    "max_p_over_t": float(p.max() / t),
                    "mc_vs_formula_max_rel_err": float(
                        np.max(np.abs(tok - expected) / expected)
                    ),
                    "bin_degree": np.round(centers, 2).tolist(),
                    "token_slope_by_bin": np.round(tok_slope, 3).tolist(),
                    "pair_slope_by_bin": np.round(pair_slope, 3).tolist(),
                    "eq39_elasticity_top5pct": float(elasticity[hub].mean()),
                    "pairs_per_retained_token_hub": float(
                        pairs[hub].sum() / tok[hub].sum()
                    ),
                    "pairs_per_retained_token_leaf": float(
                        pairs[leaf].sum() / tok[leaf].sum()
                    ),
                    "pairs_per_token_no_subsampling": float(
                        base_pairs.sum() / counts.sum()
                    ),
                    "nonadjacent_pair_share_leaf": float(
                        skip[leaf].sum() / pairs[leaf].sum()
                    ),
                    "nonadjacent_pair_share_hub": float(
                        skip[hub].sum() / pairs[hub].sum()
                    ),
                }
            )
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
