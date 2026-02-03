# Rigorous Theoretical Results: Laplacian ↔ Node2Vec for DCSBM

## Proven Results

### Result 1: DeepWalk Kernel Identity (Algebraic)

**Theorem**: For the DeepWalk co-occurrence kernel C = Σ_{r=1}^w T^r D^{-1}, the similarity transform satisfies:
```
C̃ = D^{1/2} C D^{1/2} = Σ_{r=1}^w S^r = p(S)
```

**Proof**:
1. T = D^{-1}A and S = D^{-1/2}AD^{-1/2}
2. T = D^{-1/2} S D^{1/2} (similarity)
3. T^r = D^{-1/2} S^r D^{1/2} (by induction)
4. C = Σ_r T^r D^{-1} = Σ_r D^{-1/2} S^r D^{1/2} D^{-1} = D^{-1/2} (Σ_r S^r) D^{-1/2}
5. C̃ = D^{1/2} C D^{1/2} = Σ_r S^r = p(S) ∎

**Corollary**: C̃ and S have identical eigenvectors.

---

### Result 2: DCSBM Population Eigenvector Structure (Algebraic)

**Theorem**: For DCSBM with parameters (θ, Z, B), the population normalized adjacency
S̄ = D̄^{-1/2} Ω D̄^{-1/2} has eigenvectors with exact structure:
```
v_k(i) = c_k(z_i) · √θ_i
```
where c_k(·) depends only on community label and eigenvector index.

**Proof**:
1. Define Ψ = Θ Z (so Ψ_{ik} = θ_i · 1[z_i = k])
2. Ω = Ψ B Ψ^T
3. Population degree: d̄_i = θ_i · (Bμ)_{z_i} where μ_k = Σ_{j: z_j=k} θ_j
4. Let g = Bμ (K-vector). Then d̄_i = θ_i · g_{z_i}
5. Define Φ = D̄^{-1/2} Ψ. Then:
   - Φ_{ik} = θ_i / √(θ_i · g_{z_i}) · 1[z_i = k] = √(θ_i / g_{z_i}) · 1[z_i = k]
   - So Φ = diag(√θ) · Z · diag(1/√g)
6. S̄ = Φ B Φ^T
7. S̄ has rank K. Eigenvectors are v = Φw for eigenvectors w of Φ^T Φ B
8. v_i = Σ_k Φ_{ik} w_k = √θ_i / √g_{z_i} · w_{z_i} = √θ_i · (w_{z_i} / √g_{z_i})
9. Define c_k(c) = w_k[c] / √g_c. Then v_k(i) = c_k(z_i) · √θ_i ∎

**Verified numerically**: CV(v/√θ within communities) = 0 to machine precision.

---

### Result 3: Perron Eigenvector of Sample S (Algebraic)

**Theorem**: For any connected undirected graph, the Perron eigenvector of S = D^{-1/2}AD^{-1/2} satisfies:
```
v_1(i) = √(deg_i) / √(2m)
```
where m = number of edges.

**Proof**:
S · D^{1/2} · 1 = D^{-1/2} A · 1 = D^{-1/2} · deg = D^{1/2} · 1

So v_1 ∝ D^{1/2} · 1, normalized gives v_1(i) = √(deg_i) / ||D^{1/2} · 1|| = √(deg_i) / √(2m) ∎

---

## Empirically Observed (Requires More Evidence)

### Observation 1: Subspace Alignment
Node2Vec embeddings E have top-K subspace aligned with top-K eigenvectors of S.

**Evidence**: Principal angle cosines ≈ [0.98, 0.98, 0.98] in our experiment.

**Likely reason**: Node2Vec ≈ factorizes matrix related to p(S), which shares eigenvectors with S.

**Status**: Needs verification across multiple parameter settings.

---

### Observation 2: Norm-Degree Relationship
In our single experiment: Spearman(||E||, deg) = -0.90

**Contrast with theory**:
- Population S̄ eigenvector norms: ∝ √θ (positive correlation with expected degree)
- Sample S eigenvector norms: ∝ √deg (positive correlation)
- Node2Vec E norms: negative correlation with deg

**Status**: Single observation. The sign flip needs explanation and verification.

---

## What This Framework Enables

Given the population results, if we knew Node2Vec ≈ some function of Laplacian eigenvectors, we could predict:

1. **Directions**: E_i / ||E_i|| ≈ U_S[i,:] / ||U_S[i,:]|| (community structure from eigenvectors)

2. **For DCSBM population**:
   - U_S[i,:] = √θ_i · c(z_i)
   - So directions encode z_i, magnitudes encode θ_i

3. **The open question**: What function transforms U_S into E?
   - The subspace is preserved (≈ same)
   - The norms are transformed (from positive to negative deg correlation)

---

## Remaining Gaps

### Gap 1: Sample Concentration
How do sample eigenvectors concentrate around population eigenvectors? Need:
- Davis-Kahan type bounds for DCSBM
- Quantify ||U_sample - U_pop||

### Gap 2: Diagonal Zeroing Effect
Population analysis used Ω without zeroing diagonal. Real graphs have no self-loops.
- How does zeroing diag(Ω) affect eigenvectors?
- Does it change the √θ structure?

### Gap 3: SGNS → Eigenvector Relationship
The DeepWalk kernel C has eigenvectors related to S (via C̃ = p(S)).
But SGNS factorizes log(C) - log(k), not C directly.
- How does the log transform affect eigenvectors?
- Does it explain the norm inversion?

### Gap 4: Norm Mechanism
Why does SGNS produce norms negatively correlated with degree?
- Negative sampling hypothesis (unproven)
- PMI log-transform hypothesis (unproven)
- Need theoretical derivation or systematic experiments
