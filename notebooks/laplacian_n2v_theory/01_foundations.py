"""
Phase 1: Rigorous Foundations
============================
Verify the mathematical relationships between:
- S = D^{-1/2}AD^{-1/2} (normalized adjacency / symmetric Laplacian related)
- T = D^{-1}A (random walk transition matrix)
- C = Σ_r T^r D^{-1} (DeepWalk co-occurrence kernel)

Key theoretical question: Do these matrices share eigenspaces?
"""

import numpy as np
from numpy.linalg import eigh, svd, matrix_rank
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# PART 1: Generate a DCSBM graph
# =============================================================================

def generate_dcsbm(sizes, B, theta_ranges, seed=42):
    """
    Generate DCSBM graph.

    Args:
        sizes: list of block sizes
        B: K x K block connection probability matrix
        theta_ranges: list of (min, max) for theta in each block

    Returns:
        A: adjacency matrix
        z: community labels (1-indexed)
        theta: degree parameters
    """
    np.random.seed(seed)
    K = len(sizes)
    n = sum(sizes)

    # Build community labels
    z = np.zeros(n, dtype=int)
    start = 0
    for k, s in enumerate(sizes):
        z[start:start+s] = k + 1  # 1-indexed
        start += s

    # Build theta (degree heterogeneity parameters)
    theta = np.zeros(n)
    start = 0
    for k, s in enumerate(sizes):
        theta[start:start+s] = np.linspace(theta_ranges[k][0], theta_ranges[k][1], s)
        start += s
    theta = theta / theta.max()  # normalize

    # Build membership matrix Z
    Z = np.zeros((n, K))
    for i in range(n):
        Z[i, z[i]-1] = 1.0

    # Expected adjacency: Ω = Θ Z B Z^T Θ
    Theta = np.diag(theta)
    Omega = Theta @ Z @ B @ Z.T @ Theta
    np.fill_diagonal(Omega, 0)  # no self-loops
    Omega = np.clip(Omega, 0, 1 - 1e-12)

    # Sample adjacency
    R = np.random.rand(n, n)
    M = (R < Omega).astype(np.int8)
    A = np.triu(M, 1)
    A = A + A.T  # symmetric

    return A, z, theta, Omega

# Generate graph
sizes = [300, 400, 300]
K = len(sizes)
n = sum(sizes)
B = np.array([
    [0.15, 0.02, 0.01],
    [0.02, 0.12, 0.02],
    [0.01, 0.02, 0.10]
])
theta_ranges = [(1.0, 0.4), (1.0, 0.5), (1.0, 0.3)]  # heterogeneous degrees

A, z, theta, Omega = generate_dcsbm(sizes, B, theta_ranges)
deg = A.sum(axis=1).astype(float)
avg_deg = deg.mean()
print(f"Generated DCSBM: n={n}, edges={int(A.sum()//2)}, avg_deg={avg_deg:.2f}")
print(f"Degree range: [{deg.min():.0f}, {deg.max():.0f}]")

# =============================================================================
# PART 2: Define the key matrices
# =============================================================================

# S = D^{-1/2} A D^{-1/2} (normalized adjacency)
D_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
S = (D_inv_sqrt[:, None] * A) * D_inv_sqrt[None, :]

# T = D^{-1} A (random walk transition matrix)
D_inv = 1.0 / np.maximum(deg, 1e-12)
T = D_inv[:, None] * A

print(f"\nMatrix properties:")
print(f"  S symmetric: {np.allclose(S, S.T)}")
print(f"  T row-stochastic: {np.allclose(T.sum(axis=1), 1)}")

# =============================================================================
# PART 3: Verify similarity relation T = D^{-1/2} S D^{1/2}
# =============================================================================

D_sqrt = np.sqrt(np.maximum(deg, 1e-12))
T_from_S = (D_inv_sqrt[:, None] * S) * D_sqrt[None, :]

print(f"\nVerify T = D^{{-1/2}} S D^{{1/2}}: {np.allclose(T, T_from_S)}")

# =============================================================================
# PART 4: Eigendecomposition
# =============================================================================

# Eigenvalues/vectors of S (symmetric, so use eigh)
evals_S, evecs_S = eigh(S)
idx_S = np.argsort(np.abs(evals_S))[::-1]
evals_S = evals_S[idx_S]
U_S = evecs_S[:, idx_S]

print(f"\nTop-{K+2} eigenvalues of S: {evals_S[:K+2].round(4)}")

# For T (non-symmetric), eigenvalues should be same as S due to similarity
# But eigenvectors differ by D^{1/2}
evals_T = np.linalg.eigvals(T)
evals_T_sorted = np.sort(np.abs(evals_T))[::-1]
print(f"Top-{K+2} eigenvalues of T (magnitude): {evals_T_sorted[:K+2].round(4)}")
print(f"Eigenvalues match (similarity): {np.allclose(sorted(evals_S, reverse=True)[:10], sorted(np.real(evals_T), reverse=True)[:10], atol=1e-6)}")

# =============================================================================
# PART 5: Build DeepWalk kernel C = Σ_r T^r D^{-1}
# =============================================================================

def build_deepwalk_kernel(T, D_inv, window=10):
    """Build C = Σ_{r=1}^w T^r D^{-1}"""
    n = T.shape[0]
    C = np.zeros_like(T, dtype=float)
    T_power = T.copy()
    for r in range(1, window + 1):
        C += T_power
        T_power = T_power @ T
    C = C * D_inv[None, :]  # right-multiply by D^{-1}
    return C

window = 10
C = build_deepwalk_kernel(T, D_inv, window=window)

print(f"\nDeepWalk kernel C (window={window}):")
print(f"  C symmetric: {np.allclose(C, C.T)}")

# =============================================================================
# PART 6: KEY TEST - Verify C = D^{-1/2} p(S) D^{-1/2}
# =============================================================================

# Compute p(S) = Σ_{r=1}^w S^r
def polynomial_of_S(S, window=10):
    """Compute p(S) = Σ_{r=1}^w S^r"""
    n = S.shape[0]
    p_S = np.zeros_like(S, dtype=float)
    S_power = S.copy()
    for r in range(1, window + 1):
        p_S += S_power
        S_power = S_power @ S
    return p_S

p_S = polynomial_of_S(S, window=window)

# C should equal D^{-1/2} p(S) D^{-1/2}
C_from_pS = (D_inv_sqrt[:, None] * p_S) * D_inv_sqrt[None, :]

print(f"\nVerify C = D^{{-1/2}} p(S) D^{{-1/2}}: {np.allclose(C, C_from_pS)}")

# =============================================================================
# PART 7: CRITICAL TEST - Do S and C share eigenspaces?
# =============================================================================

# Get top-K eigenvectors of S and C
top_k = K

# S eigenvectors (already computed)
U_S_topk = U_S[:, :top_k]

# C eigenvectors
evals_C, evecs_C = eigh(C)
idx_C = np.argsort(np.abs(evals_C))[::-1]
evals_C = evals_C[idx_C]
U_C = evecs_C[:, idx_C]
U_C_topk = U_C[:, :top_k]

# Principal angles between subspaces
# cos(θ_i) = singular values of U_S^T @ U_C
_, svals_SC, _ = svd(U_S_topk.T @ U_C_topk, full_matrices=False)

print(f"\n{'='*60}")
print(f"CRITICAL TEST: Principal angle cosines between span(S) and span(C)")
print(f"  Cosines: {svals_SC.round(6)}")
print(f"  If all ≈ 1: eigenspaces are IDENTICAL")
print(f"  If spread out: eigenspaces are DIFFERENT")
print(f"{'='*60}")

# =============================================================================
# PART 8: Test the transformation: Do eigenvectors of C equal D^{-α} U_S?
# =============================================================================

# Theory suggests eigenvectors of C might be D^{-1/2} times eigenvectors of p(S)
# And p(S) has same eigenvectors as S
# So eigenvectors of C should be D^{-1/2} U_S (up to normalization/sign)

# Let's test this
U_S_transformed = D_inv_sqrt[:, None] * U_S_topk
# Normalize columns
U_S_transformed = U_S_transformed / np.linalg.norm(U_S_transformed, axis=0, keepdims=True)

# Compare to actual C eigenvectors
_, svals_transform, _ = svd(U_S_transformed.T @ U_C_topk, full_matrices=False)

print(f"\nTest: Are eigenvectors of C = D^{{-1/2}} × eigenvectors of S?")
print(f"  Cosines: {svals_transform.round(6)}")

# Alternative: maybe it's D^{1/2}?
U_S_transformed2 = D_sqrt[:, None] * U_S_topk
U_S_transformed2 = U_S_transformed2 / np.linalg.norm(U_S_transformed2, axis=0, keepdims=True)
_, svals_transform2, _ = svd(U_S_transformed2.T @ U_C_topk, full_matrices=False)

print(f"\nTest: Are eigenvectors of C = D^{{1/2}} × eigenvectors of S?")
print(f"  Cosines: {svals_transform2.round(6)}")

# =============================================================================
# PART 9: Derive the correct relationship algebraically
# =============================================================================

print(f"\n{'='*60}")
print("ALGEBRAIC DERIVATION")
print("="*60)
print("""
Given: C = D^{-1/2} p(S) D^{-1/2}  where p(S) = Σ_r S^r

p(S) has eigenvectors U_S with eigenvalues p(λ_S)

For C = D^{-1/2} p(S) D^{-1/2}, let's find its eigenvectors:

If Cv = μv, then D^{-1/2} p(S) D^{-1/2} v = μv
Let w = D^{-1/2} v, so v = D^{1/2} w
Then D^{-1/2} p(S) w = μ D^{1/2} w
     p(S) w = μ D w

So (D^{-1} p(S)) w = μ w

The eigenvectors of C are v = D^{1/2} w where w is eigenvector of D^{-1} p(S).

But D^{-1} p(S) = D^{-1/2} (D^{-1/2} p(S) D^{1/2}) D^{-1/2}
               = D^{-1/2} (something) D^{-1/2}

Let M = D^{-1/2} p(S) D^{1/2}. Note that:
D^{-1} p(S) = D^{-1/2} M D^{-1/2}

Hmm, this is getting complicated. Let's just verify numerically.
""")

# Actually compute D^{-1} p(S) and check its eigenvectors
D_inv_pS = D_inv[:, None] * p_S
evals_DinvpS, evecs_DinvpS = np.linalg.eig(D_inv_pS)
idx = np.argsort(np.abs(evals_DinvpS))[::-1]
U_DinvpS = np.real(evecs_DinvpS[:, idx[:top_k]])
U_DinvpS = U_DinvpS / np.linalg.norm(U_DinvpS, axis=0, keepdims=True)

# Transform: v = D^{1/2} w
U_C_predicted = D_sqrt[:, None] * U_DinvpS
U_C_predicted = U_C_predicted / np.linalg.norm(U_C_predicted, axis=0, keepdims=True)

_, svals_predicted, _ = svd(U_C_predicted.T @ U_C_topk, full_matrices=False)

print(f"\nTest: Eigenvectors of C = D^{{1/2}} × eigenvectors of D^{{-1}}p(S)?")
print(f"  Cosines: {svals_predicted.round(6)}")

# =============================================================================
# PART 10: The SYMMETRIC case - C̃ = D^{1/2} C D^{1/2}
# =============================================================================

print(f"\n{'='*60}")
print("SYMMETRIC TRANSFORMATION: C̃ = D^{1/2} C D^{1/2}")
print("="*60)

C_tilde = (D_sqrt[:, None] * C) * D_sqrt[None, :]

print(f"C̃ symmetric: {np.allclose(C_tilde, C_tilde.T)}")

# Verify C̃ = p(S)
print(f"C̃ = p(S): {np.allclose(C_tilde, p_S)}")

# So C̃ and S have the SAME eigenvectors!
evals_Ctilde, evecs_Ctilde = eigh(C_tilde)
idx_Ctilde = np.argsort(evals_Ctilde)[::-1]
U_Ctilde = evecs_Ctilde[:, idx_Ctilde[:top_k]]

_, svals_Ctilde_S, _ = svd(U_Ctilde.T @ U_S_topk, full_matrices=False)

print(f"\nPrincipal angles between span(C̃) and span(S):")
print(f"  Cosines: {svals_Ctilde_S.round(6)}")
print(f"\n>>> C̃ = p(S), so they have IDENTICAL eigenvectors by construction <<<")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print("SUMMARY OF FINDINGS")
print("="*60)
print(f"""
1. T = D^{{-1/2}} S D^{{1/2}}  [VERIFIED: similarity transform]

2. C = Σ_r T^r D^{{-1}} = D^{{-1/2}} p(S) D^{{-1/2}}  [VERIFIED]

3. C̃ = D^{{1/2}} C D^{{1/2}} = p(S)  [VERIFIED: EXACT equality]

4. EIGENSPACE RELATIONSHIPS:
   - C̃ and S have IDENTICAL eigenvectors (since C̃ = p(S))
   - C and S do NOT have identical eigenvectors
   - Principal angles S vs C: {svals_SC.round(4)}

5. KEY INSIGHT:
   The similarity transform C̃ = D^{{1/2}} C D^{{1/2}} = p(S)
   shares eigenspace with S EXACTLY.

   But the raw kernel C = D^{{-1/2}} p(S) D^{{-1/2}} does NOT.
""")

# =============================================================================
# PART 11: What does this mean for Node2Vec embeddings?
# =============================================================================

print(f"\n{'='*60}")
print("IMPLICATIONS FOR NODE2VEC")
print("="*60)
print("""
DeepWalk/Node2Vec SGNS approximately factorizes a matrix related to C.

If SGNS factorizes M ≈ C, then embeddings E satisfy E @ E^T ≈ C
   => Low-rank E lives in eigenspace of C (not S!)

If SGNS factorizes M ≈ log(C) + const, then it's more complex.

NEXT STEP: Train actual Node2Vec and compare its subspace to:
  (a) Eigenspace of S
  (b) Eigenspace of C
  (c) Eigenspace of C̃ = p(S)

To see which one the embeddings actually align with.
""")
