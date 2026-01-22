# Honest Assessment: What We Actually Know

## Rigorous Facts (Algebraically Provable)

### Fact 1: Matrix Identity
```
C̃ = D^{1/2} C D^{1/2} = p(S) = Σ_{r=1}^w S^r
```
where C = Σ_r T^r D^{-1} is the DeepWalk co-occurrence kernel.

**Proof sketch**:
- T = D^{-1}A and S = D^{-1/2}AD^{-1/2}
- T = D^{-1/2} S D^{1/2} (similarity transform)
- T^r = D^{-1/2} S^r D^{1/2}
- C = Σ_r T^r D^{-1} = D^{-1/2} (Σ_r S^r) D^{-1/2}
- C̃ = D^{1/2} C D^{1/2} = Σ_r S^r = p(S) ✓

**Consequence**: C̃ and S have **identical** eigenvectors.

### Fact 2: Perron Eigenvector Structure
The top eigenvector of S = D^{-1/2}AD^{-1/2} has entries:
```
v_1(i) ∝ sqrt(deg_i)
```

**Proof**: S · (D^{1/2} · 1) = D^{-1/2} A D^{-1/2} · D^{1/2} · 1 = D^{-1/2} A · 1 = D^{-1/2} · deg = D^{1/2} · 1

So v_1 = D^{1/2} · 1 / ||D^{1/2} · 1|| is the Perron eigenvector with eigenvalue 1.

---

## Empirically Robust (Likely to Hold Across Parameters)

### Finding 1: Subspace Alignment
Node2Vec embeddings E lie approximately in span{top eigenvectors of S}.

**Evidence**: Principal angle cosines [0.983, 0.980, 0.975] ≈ 1

**Why this is likely robust**:
- The theoretical connection (NetMF, Qiu et al.) shows DeepWalk factorizes a matrix related to p(S)
- Low-rank factorization of p(S) gives the same top eigenvectors as S
- This is structural, not parameter-dependent

### Finding 2: Row Norms of U_S Correlate Positively with Degree
**Evidence**: Spearman(||U_S||, deg) = +0.43

**Why this makes sense**:
- Top eigenvector v_1 has entries ∝ sqrt(deg)
- So ||U_S[i]||² includes deg_i-dependent terms
- This is a consequence of Fact 2

---

## Observations Requiring More Evidence

### Observation 1: E Norms Negatively Correlated with Degree
In our single experiment: Spearman(||E||, deg) = -0.90

**What we DON'T know**:
- Is this universal or specific to this graph?
- Does it depend on SGNS parameters (k, window, epochs)?
- Does it depend on DCSBM parameters (B, theta distribution)?
- What is the functional form (power law? log? something else)?

### Observation 2: The "Sign Flip"
- Laplacian eigenvector norms: +0.43 correlation with degree
- Node2Vec embedding norms: -0.90 correlation with degree

**Possible explanations (hypotheses, NOT proven)**:
1. SGNS negative sampling pushes frequent nodes toward zero
2. The PMI log-transform inverts the degree scaling
3. Some other optimization artifact

**What we'd need to verify**:
- Test multiple DCSBM configurations
- Test multiple SGNS parameter settings
- Test on non-DCSBM graphs
- Derive theoretically from SGNS gradient dynamics

---

## What We Should NOT Claim

1. ❌ "||E|| ~ deg^{-0.19}" — This is a fit to ONE experiment
2. ❌ Any specific power law — The functional form is unknown
3. ❌ That negative sampling "explains" the norm inversion — This is a hypothesis

---

## What We CAN Reasonably Claim

1. ✓ C̃ = p(S) is algebraically exact
2. ✓ Node2Vec embeddings live approximately in the Laplacian eigenspace
3. ✓ The top eigenvector of S has entries ∝ sqrt(deg)
4. ✓ In our experiment, E norms correlate negatively with degree (needs replication)

---

## Next Steps for Rigorous Work

### To strengthen the eigenspace claim:
- Test on multiple DCSBM configurations
- Test on real-world networks
- Compare to NetMF (which directly factorizes the theoretical matrix)

### To understand the norm behavior:
- Run SGNS with different k (negative samples)
- Run SGNS with different window sizes
- Analyze the SGNS gradient at equilibrium theoretically
- Test if the correlation holds on degree-regular graphs (where deg is constant)

### To derive anything about scaling:
- Need multiple experiments varying one parameter at a time
- Need theoretical derivation from SGNS equilibrium conditions
