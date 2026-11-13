"""Corpus for Fig. 1(c): random walks on the BlogCatalog social network.

Input:  --mat data/external/blogcatalog/downloads/blogcatalog.mat (fetch_data.py;
        `network` is the symmetric 0/1 adjacency, 10,312 nodes)
Output: --out data/derived/step0/blog_walks_s{k}.npz (walks, strength = degree, ids),
        seeds 0-4.
Same walk law as the PNAS walks in corpora.py: 10 x n walks of 80 steps, started in
proportion to degree, each step to a uniformly chosen neighbour (p = q = 1).
Run: python scripts/step0/blogcatalog_corpus.py --mat ... --out data/derived/step0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpora import SEEDS, csr, walks  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mat", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
adjacency = sp.coo_matrix(scipy.io.loadmat(a.mat)["network"])
if (adjacency != adjacency.T).nnz or adjacency.diagonal().any():
    raise ValueError("expected a symmetric adjacency without self-loops")
n = adjacency.shape[0]
graph = csr(
    n, adjacency.row.astype(np.int64), adjacency.col.astype(np.int64), adjacency.data
)
a.out.mkdir(parents=True, exist_ok=True)
for k in SEEDS:
    np.savez_compressed(
        a.out / f"blog_walks_s{k}.npz",
        walks=walks(graph, np.random.default_rng(6000 + k)),
        strength=graph[3],
        ids=np.arange(n),
    )
degree = graph[3]
print(
    n, "nodes,", adjacency.nnz // 2, "edges, degrees", degree.min(), "-", degree.max()
)
