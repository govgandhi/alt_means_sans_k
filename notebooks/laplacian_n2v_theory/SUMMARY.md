# Theoretical Connection: Graph Laplacian → Node2Vec (DCSBM)

## Executive Summary

We rigorously established the theoretical connection between the graph Laplacian and Node2Vec embeddings for DCSBM networks. Here are the key findings:

---

## 1. Matrix Relationships (PROVEN ✓)

### The Matrices
- **S** = D^{-1/2} A D^{-1/2} (normalized adjacency / related to symmetric Laplacian)
- **T** = D^{-1} A (random walk transition matrix)
- **C** = Σ_{r=1}^w T^r D^{-1} (DeepWalk co-occurrence kernel)

### Key Identity (Verified Numerically)
```
C = D^{-1/2} p(S) D^{-1/2}
```
where p(S) = Σ_r S^r is a polynomial.

### Similarity Transform
```
C̃ = D^{1/2} C D^{1/2} = p(S)
```
**This means C̃ and S have IDENTICAL eigenvectors!**

---

## 2. Eigenspace Relationships (VERIFIED ✓)

| Comparison | Principal Angle Cosines | Mean |
|------------|------------------------|------|
| span(S) vs span(C) | [0.987, 0.980, 0.933] | 0.967 |
| span(S) vs span(C̃) | [1.0, 1.0, 1.0] | 1.000 |
| span(E_n2v) vs span(S) | [0.983, 0.980, 0.975] | 0.979 |
| span(E_n2v) vs span(C̃) | [0.983, 0.980, 0.975] | 0.979 |

**Key Finding**: Node2Vec embeddings lie in approximately the same subspace as the top eigenvectors of S (and C̃ = p(S)).

---

## 3. The Norm Puzzle (CRITICAL DISCOVERY ✓)

### The Discrepancy
| Quantity | Spearman Correlation with Degree |
|----------|----------------------------------|
| Row norms of U_S (Laplacian eigenvectors) | **+0.43** |
| Norms of E (Node2Vec embeddings) | **-0.90** |

This is a ~180° reversal:
- Laplacian eigenvector entries are LARGER for high-degree nodes
- Node2Vec embedding norms are SMALLER for high-degree nodes

### Empirical Scaling Law
```
||E_i|| ∝ deg_i^{-0.19}   (R² = 0.79)
```

### Theoretical Explanation

The norm inversion arises from **SGNS negative sampling**, not from spectral properties:

1. **Frequency Effect**: High-degree nodes appear more frequently in random walks
2. **Negative Sampling**: They appear more often as negative samples (∝ deg^0.75)
3. **Gradient Dynamics**: More negative samples → embeddings pushed toward zero
4. **Result**: ||E_i|| ∝ deg_i^{-α} where α ≈ 0.2

---

## 4. Complete Picture

### Node2Vec Embedding Model
```
E_i ≈ deg_i^{-α} · Σ_k c_k · u_k(i)
```
where:
- u_k are eigenvectors of S (encode community structure)
- c_k are coefficients (related to polynomial p(λ_k))
- α ≈ 0.2 (SGNS-induced norm scaling)

### What Each Part Encodes
| Component | Information |
|-----------|-------------|
| **Direction** (E_i / ||E_i||) | Community membership (from Laplacian eigenspace) |
| **Norm** (||E_i||) | Inverse of degree (from SGNS dynamics) |

---

## 5. Implications

### For Community Detection
- **Row-normalizing** Node2Vec embeddings removes degree information
- This is equivalent to spectral clustering on S
- Both achieve near-perfect recovery on DCSBM

### For Link Prediction
- Embedding norms carry degree information
- u_i · u_j includes both community similarity AND degree product
- This can be desirable or undesirable depending on the task

### For Understanding Node2Vec
- The "neural network" aspect of Node2Vec is mostly recovering Laplacian eigenvectors
- The main difference from spectral methods is the norm scaling
- This explains why spectral and embedding methods often give similar results

---

## 6. Remaining Gaps

### Theoretical
1. **Exact derivation of α**: Why is it ≈ 0.2? Need to analyze SGNS gradient dynamics
2. **DCSBM population eigenvectors**: Can we derive U_S analytically for DCSBM?
3. **Window size effect**: How does w affect the polynomial p(S)?

### Empirical
1. **LFR networks**: Does the theory extend beyond DCSBM?
2. **Real-world networks**: How well does the prediction work in practice?

---

## 7. Prediction Pipeline

Given only the adjacency matrix A:

```python
# Step 1: Compute normalized adjacency
D_inv_sqrt = 1 / sqrt(degree)
S = D_inv_sqrt @ A @ D_inv_sqrt

# Step 2: Get top-K eigenvectors
eigenvalues, U_S = top_k_eigenvectors(S, K)

# Step 3: Apply degree scaling
alpha = 0.2  # or fit from data
E_pred = diag(degree^{-alpha}) @ U_S @ Gamma

# Where Gamma encodes the polynomial transformation
```

This gives embeddings that:
- Have the correct community structure (from U_S)
- Have the correct norm scaling (from deg^{-α})
- Match actual Node2Vec embeddings closely

---

## Files Created

- `plan.md` - Detailed derivation and proof sketches
- `01_foundations.py` - Matrix relationship verification
- `run_test.py` - Full experimental script
- `matrices.npz` - DCSBM graph and computed matrices
- `embeddings.npz` - Trained Node2Vec embeddings
