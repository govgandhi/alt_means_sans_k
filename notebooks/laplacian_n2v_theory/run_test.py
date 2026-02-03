#!/usr/bin/env python
"""Test script for Laplacian-Node2Vec theory"""
import os
os.environ['TMPDIR'] = '/nobackup/gogandhi/tmp'

import numpy as np
from numpy.linalg import eigh, svd
import networkx as nx
import random
from gensim.models import Word2Vec
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
random.seed(42)

# =============================================================================
# PART 1: Generate DCSBM
# =============================================================================
sizes = [300, 400, 300]
K = len(sizes)
n = sum(sizes)
B = np.array([
    [0.15, 0.02, 0.01],
    [0.02, 0.12, 0.02],
    [0.01, 0.02, 0.10]
])

z = np.zeros(n, dtype=int)
start = 0
for k, s in enumerate(sizes):
    z[start:start+s] = k + 1
    start += s

theta_ranges = [(1.0, 0.4), (1.0, 0.5), (1.0, 0.3)]
theta = np.zeros(n)
start = 0
for k, s in enumerate(sizes):
    theta[start:start+s] = np.linspace(theta_ranges[k][0], theta_ranges[k][1], s)
    start += s
theta = theta / theta.max()

Z_mat = np.zeros((n, K))
for i in range(n):
    Z_mat[i, z[i]-1] = 1.0
Theta = np.diag(theta)
Omega = Theta @ Z_mat @ B @ Z_mat.T @ Theta
np.fill_diagonal(Omega, 0)
Omega = np.clip(Omega, 0, 1 - 1e-12)

R = np.random.rand(n, n)
M = (R < Omega).astype(np.int8)
A = np.triu(M, 1)
A = A + A.T
deg = A.sum(axis=1).astype(float)

print(f'DCSBM: n={n}, edges={int(A.sum()//2)}, avg_deg={deg.mean():.2f}')

# =============================================================================
# PART 2: Compute theoretical matrices
# =============================================================================
D_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
D_inv = 1.0 / np.maximum(deg, 1e-12)
D_sqrt = np.sqrt(np.maximum(deg, 1e-12))

S = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]
T = D_inv[:, None] * A

window = 10
C = np.zeros_like(A, dtype=float)
T_power = T.copy()
for r in range(1, window + 1):
    C += T_power
    T_power = T_power @ T
C = C * D_inv[None, :]

C_tilde = (D_sqrt[:, None] * C) * D_sqrt[None, :]

# Eigenspaces
top_k = K
evals_S, evecs_S = eigh(S)
idx_S = np.argsort(evals_S)[::-1]
U_S = evecs_S[:, idx_S[:top_k]]

evals_C, evecs_C = eigh(C)
idx_C = np.argsort(evals_C)[::-1]
U_C = evecs_C[:, idx_C[:top_k]]

evals_Ctilde, evecs_Ctilde = eigh(C_tilde)
idx_Ctilde = np.argsort(evals_Ctilde)[::-1]
U_Ctilde = evecs_Ctilde[:, idx_Ctilde[:top_k]]

print(f'Top-{K} eigenvalues of S: {evals_S[idx_S[:top_k]].round(4)}')

# Verify relationships
_, svals_SC, _ = svd(U_S.T @ U_C, full_matrices=False)
print(f'\nspan(S) vs span(C): {svals_SC.round(4)}')

_, svals_SCtilde, _ = svd(U_S.T @ U_Ctilde, full_matrices=False)
print(f'span(S) vs span(C_tilde): {svals_SCtilde.round(4)}')

# =============================================================================
# PART 3: Train Node2Vec
# =============================================================================
G = nx.from_numpy_array(A)

def random_walk(G, start, walk_len=40):
    walk = [start]
    for _ in range(walk_len-1):
        cur = walk[-1]
        neigh = list(G.neighbors(cur))
        if not neigh: break
        walk.append(random.choice(neigh))
    return walk

print('\nTraining Node2Vec...')
walks = []
nodes = list(G.nodes())
for _ in range(10):
    random.shuffle(nodes)
    for v in nodes:
        walks.append(random_walk(G, v, 40))

model = Word2Vec(
    sentences=walks, vector_size=64, window=10,
    sg=1, negative=5, hs=0, min_count=0, workers=4, epochs=5, seed=42
)

E = np.zeros((n, 64), dtype=float)
for i in range(n):
    E[i] = model.wv[i]

Cctx = np.zeros((n, 64), dtype=float)
for i in range(n):
    idx = model.wv.key_to_index[i]
    Cctx[i] = model.syn1neg[idx]

print(f'Embedding shape: {E.shape}')

# =============================================================================
# PART 4: Compare embeddings to theory
# =============================================================================
U_E, _, _ = svd(E, full_matrices=False)
U_E_topk = U_E[:, :top_k]

print('\n' + '='*60)
print('EMBEDDING SUBSPACE vs THEORETICAL MATRICES')
print('='*60)

_, s1, _ = svd(U_E_topk.T @ U_S, full_matrices=False)
_, s2, _ = svd(U_E_topk.T @ U_C, full_matrices=False)
_, s3, _ = svd(U_E_topk.T @ U_Ctilde, full_matrices=False)

print(f'E vs S:       {s1.round(4)}  mean={s1.mean():.4f}')
print(f'E vs C:       {s2.round(4)}  mean={s2.mean():.4f}')
print(f'E vs C_tilde: {s3.round(4)}  mean={s3.mean():.4f}')

# Row-normalized
E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)
U_Enorm, _, _ = svd(E_norm, full_matrices=False)
U_Enorm_topk = U_Enorm[:, :top_k]

print('\nRow-normalized E:')
_, s1n, _ = svd(U_Enorm_topk.T @ U_S, full_matrices=False)
_, s2n, _ = svd(U_Enorm_topk.T @ U_C, full_matrices=False)
_, s3n, _ = svd(U_Enorm_topk.T @ U_Ctilde, full_matrices=False)

print(f'E_norm vs S:       {s1n.round(4)}  mean={s1n.mean():.4f}')
print(f'E_norm vs C:       {s2n.round(4)}  mean={s2n.mean():.4f}')
print(f'E_norm vs C_tilde: {s3n.round(4)}  mean={s3n.mean():.4f}')

# =============================================================================
# PART 5: D^alpha transformation
# =============================================================================
print('\n' + '='*60)
print('TRANSFORMATION: D^alpha * U_S alignment with E')
print('='*60)

for alpha in [-0.5, -0.25, 0, 0.25, 0.5]:
    if alpha == 0:
        U_trans = U_S.copy()
    else:
        U_trans = (deg ** alpha)[:, None] * U_S
    U_trans = U_trans / np.linalg.norm(U_trans, axis=0, keepdims=True)
    _, svals, _ = svd(U_E_topk.T @ U_trans, full_matrices=False)
    print(f'  alpha={alpha:+.2f}: {svals.round(4)} mean={svals.mean():.4f}')

# =============================================================================
# PART 6: Norm analysis
# =============================================================================
print('\n' + '='*60)
print('NORM ANALYSIS')
print('='*60)

norms_E = np.linalg.norm(E, axis=1)
print('Spearman correlations with ||E||:')
print(f'  deg:           {spearmanr(norms_E, deg).correlation:.4f}')
print(f'  sqrt(deg):     {spearmanr(norms_E, np.sqrt(deg)).correlation:.4f}')
print(f'  log(deg):      {spearmanr(norms_E, np.log(deg)).correlation:.4f}')
print(f'  1/sqrt(deg):   {spearmanr(norms_E, 1/np.sqrt(deg)).correlation:.4f}')
print(f'  theta:         {spearmanr(norms_E, theta).correlation:.4f}')

print('\nMean ||E|| by community:')
for k in range(1, K+1):
    mask = (z == k)
    print(f'  Comm {k}: ||E||={norms_E[mask].mean():.3f}, deg={deg[mask].mean():.1f}')
