#!/bin/bash
export TMPDIR=/nobackup/gogandhi/tmp
mkdir -p $TMPDIR
cd /nobackup/gogandhi

/nobackup/gogandhi/miniconda3/envs/kmeans_env/bin/python << 'PYTHONEOF'
import os
os.environ['TMPDIR'] = '/nobackup/gogandhi/tmp'

import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

random.seed(42)

data = np.load('/nobackup/gogandhi/alt_means_sans_k/notebooks/laplacian_n2v_theory/matrices.npz')
A = data['A']
n = A.shape[0]

print('Training Node2Vec...')
G = nx.from_numpy_array(A)

def random_walk(G, start, walk_len=40):
    walk = [start]
    for _ in range(walk_len-1):
        cur = walk[-1]
        neigh = list(G.neighbors(cur))
        if not neigh: break
        walk.append(random.choice(neigh))
    return walk

walks = []
nodes = list(G.nodes())
for _ in range(10):
    random.shuffle(nodes)
    for v in nodes:
        walks.append(random_walk(G, v, 40))

print(f'Generated {len(walks)} walks')

model = Word2Vec(
    sentences=walks, vector_size=64, window=10,
    sg=1, negative=5, hs=0, min_count=0, workers=1, epochs=5, seed=42
)

E = np.zeros((n, 64), dtype=float)
Cctx = np.zeros((n, 64), dtype=float)
for i in range(n):
    E[i] = model.wv[i]
    idx = model.wv.key_to_index[i]
    Cctx[i] = model.syn1neg[idx]

print(f'E shape={E.shape}')
np.savez('/nobackup/gogandhi/alt_means_sans_k/notebooks/laplacian_n2v_theory/embeddings.npz', E=E, Cctx=Cctx)
print('Done!')
PYTHONEOF
