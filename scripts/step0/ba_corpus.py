"""Corpus for Fig. 1(d): walks on the expected graph of a Barabasi-Albert graph.

For seed k, draw G = networkx.barabasi_albert_graph(N, M, seed=k) and keep its degrees
k_i. The expected graph at these degrees (configuration-model expectation; the mean-
field BA link probability has the same product form) links every pair (i, j) with weight
k_i k_j / 2E, so a walk started in proportion to degree steps to j with probability k_j
/ 2E wherever it is: every token is an independent draw in proportion to degree. That is
the process whose skip-gram optimum theory.py computes exactly (flat center norms under
every setting). Same shape as the other walk corpora: 10 x N sequences of 80 tokens. N =
10,000 and M = 5 are our choices (mean degree 10, degrees from 5 to a few hundred); they
are not taken from a source.
Output: data/derived/step0/ba_walks_s{k}.npz (walks, strength = degree, ids), seeds 0-4.
Run: python scripts/step0/ba_corpus.py --out data/derived/step0
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np

N, M = 10_000, 5
WALKS_PER_NODE, LENGTH, SEEDS = 10, 80, range(5)

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
for k in SEEDS:
    graph = nx.barabasi_albert_graph(N, M, seed=k)
    degree = np.array([d for _, d in sorted(graph.degree())], dtype=float)
    tokens = np.random.default_rng(5000 + k).choice(
        N, size=(WALKS_PER_NODE * N, LENGTH), p=degree / degree.sum()
    )
    np.savez_compressed(
        a.out / f"ba_walks_s{k}.npz",
        walks=tokens.astype(np.int32),
        strength=degree,
        ids=np.arange(N),
    )
    print(f"seed {k}: degrees {int(degree.min())}-{int(degree.max())}", flush=True)
