# Theoretical Plan: Graph Laplacian → Node2Vec for DCSBM

## The Core Question
**Can we rigorously derive Node2Vec embeddings from spectral properties of the graph Laplacian for DCSBM networks?**

---

# VERIFIED RESULTS (as of current analysis)

## Result 1: Matrix Relationships (VERIFIED ✓)

**Claim**: The DeepWalk kernel C and normalized adjacency S are related by:
```
C = Σ_{r=1}^w T^r D^{-1} = D^{-1/2} p(S) D^{-1/2}
```
where T = D^{-1}A, S = D^{-1/2}AD^{-1/2}, and p(S) = Σ_r S^r.

**Verification**: Numerically confirmed with machine precision.

**Corollary**: The similarity transform C̃ = D^{1/2}CD^{1/2} = p(S) EXACTLY.

## Result 2: Eigenspace Relationships (VERIFIED ✓)

| Comparison | Principal Angle Cosines | Conclusion |
|------------|------------------------|------------|
| span(S) vs span(C) | [0.987, 0.980, 0.933] | Similar but NOT identical |
| span(S) vs span(C̃) | [1.0, 1.0, 1.0] | IDENTICAL (by construction) |
| span(E) vs span(S) | [0.983, 0.980, 0.975] | Very similar |
| span(E) vs span(C̃) | [0.983, 0.980, 0.975] | Very similar |

**Key Finding**: Node2Vec embeddings E lie approximately in the eigenspace of S (and C̃ = p(S)),
NOT in the eigenspace of C directly.

## Result 3: Norm Behavior (CRITICAL FINDING ✓)

| Quantity | Spearman(·, degree) |
|----------|---------------------|
| ||U_S row norms|| | +0.43 (positive) |
| ||E|| (Node2Vec norms) | -0.90 (negative) |

**The Discrepancy**:
- Laplacian eigenvector row norms are POSITIVELY correlated with degree
- Node2Vec embedding norms are NEGATIVELY correlated with degree
- This is a ~180° reversal!

**Log-linear fit**: ||E|| ~ deg^{-0.186} with R² = 0.79

## Result 4: Theoretical Explanation for Norm Inversion

The norm inversion arises from the SGNS optimization, not from spectral properties.

**SGNS Loss**: L = -Σ_{(i,j)} [log σ(u_i·v_j) + k·E_{neg}[log σ(-u_i·v_k)]]

**Key mechanism**:
1. High-degree nodes appear more frequently in random walks
2. They appear more often as NEGATIVE samples (proportional to deg^{0.75} in standard SGNS)
3. Negative sampling pushes embeddings apart
4. High-frequency nodes get pushed more → smaller norms

**Mathematical form**:
```
E_i ≈ deg_i^{-α} · (projection onto span{U_S})
```
where α ≈ 0.2 empirically.

---

---

## Phase 0: Establish Rigorous Foundations

### 0.1 DCSBM Definition (Standard)
- Nodes i ∈ {1,...,n} with community labels z_i ∈ {1,...,K}
- Degree parameters θ = (θ_1,...,θ_n) with θ_i > 0
- Block connectivity matrix B ∈ R^{K×K}, symmetric, B_{kl} ∈ [0,1]
- Edge probability: **P(A_{ij} = 1) = θ_i θ_j B_{z_i,z_j}** for i ≠ j, A_{ii} = 0

**Population quantities**:
- Expected adjacency: Ω = E[A] where Ω_{ij} = θ_i θ_j B_{z_i,z_j}
- Expected degree: d̄_i = Σ_j Ω_{ij} = θ_i · Σ_k B_{z_i,k} · (Σ_{j:z_j=k} θ_j)

### 0.2 DeepWalk/Node2Vec (p=q=1) - What It Actually Does
1. Generate random walks of length L starting from each node
2. For each walk, create (target, context) pairs within window w
3. Train SGNS: minimize -Σ_{(i,j)} [log σ(u_i·v_j) + k·E_{j'~P_n}[log σ(-u_i·v_{j'})]]

**Key result (Levy & Goldberg 2014)**:
At convergence, SGNS satisfies: u_i · v_j = PMI(i,j) - log(k)
where PMI(i,j) = log(P(i,j) / (P(i)P(j)))

### 0.3 What Matrix Does DeepWalk Factorize? (Qiu et al. 2018, NetMF)

**Theorem (NetMF)**: DeepWalk with window T implicitly factorizes:
```
M = log(vol(G) · (1/T) Σ_{r=1}^T (D^{-1}A)^r · D^{-1}) - log(k)
```
where vol(G) = Σ_i d_i, and k = number of negative samples.

**Critical observation**: This involves powers of T = D^{-1}A (the random walk matrix).

---

## Phase 1: Random Walk Matrix ↔ Laplacian Connection

### Claim 1.1: Eigenspace Equivalence
**Claim**: The matrices T^r (for any r) and the normalized adjacency S = D^{-1/2}AD^{-1/2} share the same eigenvectors (up to a diagonal scaling).

**Proof sketch**:
- Let S = D^{-1/2}AD^{-1/2} with eigenpairs (λ_k, v_k): Sv_k = λ_k v_k
- Then T = D^{-1}A = D^{-1/2}SD^{1/2}
- So T(D^{1/2}v_k) = D^{-1/2}S(D^{1/2}·D^{-1/2})D^{1/2}v_k = D^{-1/2}Sv_k·something...

Wait, let me be more careful:
- T = D^{-1}A
- S = D^{-1/2}AD^{-1/2}
- Relation: D^{1/2}TD^{-1/2} = D^{-1/2}AD^{-1/2} = S

So T and S are **similar matrices**: T = D^{-1/2}SD^{1/2}

**Consequence**: T and S have the SAME eigenvalues. If Sv = λv, then T(D^{1/2}v) = λ(D^{1/2}v).

**Consequence for powers**: T^r = D^{-1/2}S^r D^{1/2}, so T^r and S^r are also similar.

### Claim 1.2: Polynomial Preserves Eigenspaces
**Claim**: For any polynomial p(x), the matrix p(S) has the same eigenvectors as S.

**Proof**: If Sv = λv, then p(S)v = p(λ)v. Trivial but important.

### Claim 1.3: The DeepWalk Matrix Eigenspace
**Claim**: The matrix M_DW = Σ_{r=1}^T T^r · D^{-1} shares its top eigenspace with S (after appropriate transformation).

**To prove rigorously**: Need to work out the exact similarity transform.

Let C = Σ_{r=1}^T T^r · D^{-1}
    = Σ_{r=1}^T D^{-1/2}S^r D^{1/2} · D^{-1}
    = D^{-1/2} (Σ_{r=1}^T S^r) D^{-1/2}
    = D^{-1/2} p(S) D^{-1/2}    where p(x) = Σ_{r=1}^T x^r

Now, p(S) has the same eigenvectors as S.

**Key question**: Does D^{-1/2} p(S) D^{-1/2} have the same eigenvectors as S?
NO! The D^{-1/2} terms change the eigenvectors.

Let's compute: If Sv = λv, then p(S)v = p(λ)v
For C = D^{-1/2}p(S)D^{-1/2}:
C(D^{1/2}v) = D^{-1/2}p(S)D^{-1/2}D^{1/2}v = D^{-1/2}p(S)v = D^{-1/2}p(λ)v = p(λ)·D^{-1/2}v

This is NOT an eigenvector equation for C unless D^{-1/2}v ∝ D^{1/2}v, which is false in general.

**IMPORTANT**: The simple eigenspace equivalence may NOT hold. Need to investigate more carefully.

---

## Phase 2: Rigorous Derivation - What IS the Relationship?

### 2.1 Define Matrices Precisely
- A: adjacency matrix
- D: degree matrix, D_{ii} = d_i
- S: normalized adjacency, S = D^{-1/2}AD^{-1/2}
- T: random walk (transition) matrix, T = D^{-1}A
- L: normalized Laplacian, L = I - S

### 2.2 The DeepWalk Co-occurrence Matrix
From a stationary random walk, the probability of visiting j in context of i:
P(j|i, context within window w) ∝ Σ_{r=1}^w [T^r]_{ij}

The co-occurrence count matrix (in expectation, for many walks):
N_{ij} ∝ d_i · [Σ_{r=1}^w T^r]_{ij}

(The d_i factor comes from starting walks proportional to stationary distribution π_i = d_i/vol(G))

### 2.3 SGNS Target Matrix
SGNS factorizes (approximately):
M_{ij} = log(N_{ij} / (N_i · N_j / N_total)) - log(k)

where N_i = Σ_j N_{ij} (row sum), N_j = Σ_i N_{ij} (col sum).

### 2.4 Symmetrized Analysis
For undirected graphs with p=q=1, the co-occurrence matrix should be symmetric.
Let's define: C = D · (Σ_{r=1}^w T^r)  [co-occurrence, up to constants]

Then the PMI matrix:
PMI_{ij} = log(C_{ij}) - log(C_i·) - log(C_·j) + log(C_total)

This is NOT simply related to S by a similarity transform.

---

## Phase 3: What CAN We Prove?

### 3.1 Spectral Properties of Random Walk Powers
**Fact**: T has eigenvalues in [-1, 1] for connected undirected graphs.
**Fact**: The stationary distribution is π = d/vol(G).
**Fact**: T^r converges to rank-1 matrix π·1^T as r → ∞ (for connected, non-bipartite).

### 3.2 Low-Rank Approximation View
The matrices S and T both have the property that their spectrum is concentrated:
- Top eigenvalue λ_1 = 1 (for connected graphs)
- Next K-1 eigenvalues encode community structure (for SBM/DCSBM)
- Remaining eigenvalues are O(1/√d) for dense graphs

**Hypothesis**: The DeepWalk matrix M, while not having identical eigenvectors to S,
has its top-K eigenspace approximately aligned with S's top-K eigenspace.

This is an APPROXIMATION result, not an exact one.

---

## Phase 4: Experimental Tests to Run

### Test 1: Eigenspace Alignment (Exact vs. Approximate)
Generate DCSBM graph, compute:
1. S = D^{-1/2}AD^{-1/2}, get top-K eigenvectors U_S
2. C = Σ_r T^r · D^{-1}, get top-K eigenvectors U_C
3. Compute principal angles: svd(U_S^T @ U_C)

**If eigenspaces are identical**: all singular values = 1
**If approximately aligned**: singular values close to 1
**If unrelated**: singular values spread out

### Test 2: The D^{-1/2} Transformation Effect
Compare:
1. U_S: eigenvectors of S
2. U_C: eigenvectors of C = D^{-1/2}p(S)D^{-1/2}
3. D^{1/2}U_S (transformed eigenvectors)

Does D^{1/2}U_S align with U_C?

### Test 3: Actual SGNS Embeddings
Train actual Node2Vec/DeepWalk via SGNS, get embeddings E.
Compare subspace of E to:
1. Subspace of U_S
2. Subspace of U_C
3. Subspace of D^{1/2}U_S

Which aligns best?

### Test 4: Population vs. Sample
For DCSBM:
1. Compute population S̄ = D̄^{-1/2}Ω D̄^{-1/2} (using expected A and D)
2. Compare to sample S = D^{-1/2}AD^{-1/2}
3. How well do population eigenvectors predict sample eigenvectors?

### Test 5: Norm Analysis
If theory predicts ||E_i|| = f(d_i), test:
1. Compute actual ||E_i||
2. Compute predicted f(d_i) for various candidate f
3. Correlation analysis

---

## Phase 5: For DCSBM Specifically - Analytical Results

### 5.1 Population Laplacian Eigenvectors
For DCSBM with K blocks:
- Ω = Θ Z B Z^T Θ where Θ = diag(θ), Z = membership matrix
- D̄ = diag(Ω·1) = diag(Θ Z B Z^T Θ · 1)

Can we compute eigenvectors of S̄ = D̄^{-1/2}Ω D̄^{-1/2} analytically?

**Special case (SBM, θ_i = 1 ∀i)**:
- Ω = Z B Z^T
- If blocks have equal size n/K: D̄ = (n/K)·diag(B·1)·I (within-block constant degree)
- S̄ has a specific block structure

**General DCSBM**:
More complex, but the key insight is that S̄ has rank at most K (since Ω has rank K).
So S̄ has exactly K non-zero eigenvalues, and the eigenvectors encode community + degree info.

### 5.2 Eigenvector Structure Prediction
**Conjecture to test**: For DCSBM, the eigenvectors of S have the form:
u_i ≈ f(z_i) · g(θ_i)
where f depends only on community, g depends only on degree parameter.

This would explain why:
- Directions encode community
- Norms (magnitudes) encode degree

---

## Execution Order

1. **First**: Verify Claim 1.1-1.3 mathematically (the similarity transforms)
2. **Second**: Run Test 1-2 to check eigenspace alignment numerically
3. **Third**: Run Test 3 with actual SGNS to see if theory matches practice
4. **Fourth**: Derive DCSBM population eigenvectors (Test 4)
5. **Fifth**: Norm analysis (Test 5)

At each step: if test FAILS, revise theory before proceeding.
