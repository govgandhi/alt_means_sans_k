# Theory notes: what bends the trained norm curve

Working notes of 2026-11-23 for the norm tests in `scripts/norm_tests/` and the prior-work audit in `analysis/norm_prior_work.md`.
Code is in `scripts/norm_theory/`, outputs in `results/norm_theory/` (large `.npz` files are gitignored).
Setting names: **one-hop** is window 1, negative exponent 1, no subsampling (the trajectory-embedding setting); **default** is Gensim's defaults (window 5, exponent 0.75, subsampling 1e-3); **wide** is window 10, 20 epochs, subsampling 1e-4.
The PR #36 graphs are the 600-node configuration graphs from BA(600, 3), trained at one-hop with d = 64.
All comparisons at matched summed learning rate τ use τ = 0.0628 unless stated; "gap" is the top degree bin minus the preceding bin of the mean-normalized balanced norm.

## 1. Step size: the first-order SGD penalty (T2)

### Derivation

Gensim applies one example at a time: a positive pair (a, b) and K = 5 negatives c₁, …, c_K drawn from q, all sharing the input vector u_a and applied in one step with rate ε.
Write ℓ_k for that example's loss and L = Σ_k ℓ_k for one epoch, so gradient flow in τ is dθ/dτ = −∇L.
For random reshuffling, one epoch of SGD matches, to first order in ε, gradient flow on the modified loss (Smith, Dherin, Barrett & De, ICLR 2021, Eq. 1)

$$L_\varepsilon = L + \frac{\varepsilon}{4}\sum_k \lVert\nabla \ell_k\rVert^2 .$$

Gensim visits the walks in a fixed order and draws negatives at random; we use the reshuffling form and check it numerically.
With r = σ(−s_ab), p_ac = σ(s_ac), and m_a = Σ_c q_c p_ac v_c, the expectation over the negatives is

$$\mathbb{E}\lVert\nabla_{u}\ell_k\rVert^2 = \lVert r\,v_b - K m_a\rVert^2 + K\Big(\sum_c q_c p_{ac}^2\lVert v_c\rVert^2 - \lVert m_a\rVert^2\Big),$$

$$\mathbb{E}\lVert\nabla_{v}\ell_k\rVert^2 = \lVert u_a\rVert^2\Big(r^2 + K\sum_c q_c p_{ac}^2 + K(K-1)\sum_c q_c^2 p_{ac}^2\Big).$$

A Monte Carlo check of these two lines on random factors agrees within 1% (`t2_check_penalty.py`, three pairs, 20,000 draws each).
`t2_modified_flow.py` integrates gradient flow on L_ε by explicit Euler in τ from Gensim's initial vectors; at ε = 0 it reproduces the replica's gradient flow to 1e-4 relative.

### Check against SGD

Five graphs unless marked; "exact SGD" is the numba replica of Gensim's update with an exact sigmoid and no ±6 skip (`clock_replica.py exact`).
Shift columns compare per-node log balanced-norm ratios to gradient flow, model against exact SGD (against Gensim where no exact run exists).

| ε | Gensim gap | exact SGD gap | modified-flow gap | top-bin norm, flow / exact SGD | per-node shift: Spearman, within bins, slope |
|---|---|---|---|---|---|
| 0 (gradient flow) | — | +0.176 | +0.176 | 7.26 / 7.26 | — |
| 0.000625 | +0.128 | +0.141 | +0.141 | 7.02 / 7.03 | 0.85, 0.82, 0.98 |
| 0.00125 | +0.098 | — | +0.110 | 6.81 / — | 0.84, 0.76, 1.25 (vs Gensim) |
| 0.0025 | +0.051 | +0.062 | +0.062 | 6.48 / 6.50 | 0.93, 0.91, 0.99 |
| 0.00625 (3 graphs) | −0.029 | — | −0.018 | 5.83 / 5.66 (Gensim) | 0.95, 0.92, 1.07 (vs Gensim) |
| 0.0125 (3 graphs) | −0.078 | — | −0.070 | 5.28 / 5.17 (Gensim) | 0.95, 0.92, 1.04 (vs Gensim) |
| 0.025 | −0.101 | −0.102 | −0.097 | 4.77 / 4.86 | 0.95, 0.94, 0.98 |

At τ = 0.1 the agreement is the same (ε = 0.000625: flow +0.117 and top-bin 7.07, exact SGD +0.117 and 7.08).
**Verdict: pass.** The first-order penalty reproduces SGD's degree profile and per-node shifts across the whole tested range, including ε = 0.025, where the gap differs by 0.005 and the top-bin norm by 2%.
The step-size effect on these graphs is therefore the O(ε) correction of SGD relative to gradient flow, and Smith et al.'s modified loss computes it.

**Skip offset.**
Gensim's skip of updates with |score| ≥ 6, together with its sigmoid table, lowers the gap by about 0.013 at small steps: gradient flow +0.176 exact vs +0.163 with the skip; at ε = 0.000625 the replica gives +0.141 exact vs +0.128 with the skip (Gensim +0.128); at 0.0025, +0.062 vs +0.050; at 0.025, −0.102 vs −0.100.
The skip shifts the curve by a roughly constant amount and does not produce the step-size dependence, but at small steps it is about a quarter of the remaining distance to gradient flow, so "plays almost no part" overstates it.

### Drift or diffusion?

`t2_diffusion.py` replaces the penalty by matched noise: gradient flow plus Gaussian noise with SGD's per-epoch gradient covariance (Euler–Maruyama, 3 graphs per ε).
'neg' injects only the negative-sampling noise (Gensim cycles every positive pair once per epoch); 'full' adds positive-pair noise as if pairs were drawn with replacement.

| ε | exact SGD gap | 'neg' gap | 'full' gap | per-node slope (SGD on model), 'neg' / 'full' |
|---|---|---|---|---|
| 0.000625 | +0.141 | +0.155 | — | 1.46 / — |
| 0.0025 | +0.062 | +0.095 | +0.057 | 1.38 / 0.93 |
| 0.025 | −0.102 | −0.075 | −0.103 | 1.13 / 0.95 |

Negative-sampling noise alone produces about two thirds of the shift.
Full matched noise reproduces the bin-level gaps about as well as the drift model, with weaker per-node agreement (Spearman 0.87–0.91 vs 0.93–0.95 for the drift).
**Verdict:** both are O(ε) effects of per-example gradient fluctuations; at the level of degree bins these data do not separate a deterministic drift from diffusion.
The drift (modified-loss) form is deterministic, matches node by node slightly better, and has a closed form, so it is the one to use.

### Which term shrinks hubs

Ablated flows at ε = 0.0025 (`t2_ablation.py`, 5 graphs, Euler step 1e-5 in τ, which changes per-node norms by a median 3e-4 relative to step 5e-6); share = variant's gap shift / full gap shift (top-bin log-norm shift in parentheses).

| Penalty part kept | gap | share of gap shift (top-bin shift) |
|---|---|---|
| all | +0.063 | 1.00 (1.00) |
| context-vector gradients ‖∇_v ℓ_k‖² | +0.063 | 0.99 (0.92) |
| center-vector gradients ‖∇_u ℓ_k‖² | +0.173 | 0.02 (0.09) |
| positive-pair terms | +0.122 | 0.47 (0.40) |
| negative-sample terms | +0.101 | 0.66 (0.69) |
| all but the positive–negative cross term | +0.063 | 1.00 (1.00) |

At ε = 0.000625 the shares are the same within 0.06: context part 0.99 (0.91), center part 0.02 (0.09), positive terms 0.42 (0.38), negative terms 0.61 (0.65), cross term 0.00; the full shift there is −0.035 in gap (−0.113 at 0.0025).

The context-vector part carries the effect.
Each example updates its positive context by −εr u_a and each negative by εp u_a, so the squared step it puts on contexts is ‖u_a‖² times the example's residual mass r² + Σ p².
Summed over an epoch, the penalty on a center vector's length is (ε/4)·‖u_a‖²·Σ_{examples with input a}(r² + Σ_l p²): it grows with the number of updates a node makes and with how badly those pairs are fit.
The center-vector part largely cancels because the positive and negative gradients on u_a are summed into one step and nearly balance near calibration (the ‖r v_b − K m_a‖² term); the cross term contributes nothing measurable.
Inference: hubs shrink because they make the most updates and their pairs sit nearest σ = 1/2 (their per-pair loss is highest), which matches the observed within-bin correlation between shrinkage and gradient-flow pair loss.

### A closed-form per-node predictor

From a trained model one can compute, per node, the Fisher mass F_i = Σ_j (P_ij + N_ij) σ(s_ij)(1 − σ(s_ij)), with P and N the per-epoch positive and expected negative counts (row i for center vectors, column i for context vectors).
Near calibration the penalty is (ε/4)·Σ_ij (P_ij + N_ij)σ(1−σ)(‖w_i‖² + ‖c_j‖²) = (ε/4) tr(Hessian), so node i's length is penalized at rate (ε/2)F_i in log units.
Three ways to turn that into a shift, tested against the flow and exact SGD at ε = 0.0025 from gradient-flow factors at τ = 0.0628 (`t2_predict.py`; Spearman global, within degree bins, slope of actual on predicted):

| Predictor | vs modified flow (balanced) | vs exact SGD (balanced) |
|---|---|---|
| static radial, Δlog‖w_i‖ = −(ε/2)‖w_i‖²/⟨s_ij²⟩_i | 0.75, 0.73, 2.8 | 0.66, 0.64, 2.7 |
| static coupled radial (partners respond) | 0.42, 0.28, 1.4 | 0.45, 0.31, 1.4 |
| free drift, Δlog‖w_i‖ = −(ε/2) τ F_i (same for c_i), balanced norm recomputed | 0.92, 0.91, 0.56 | 0.83, 0.82, 0.55 |

At ε = 0.000625 the ordering is the same: static radial 0.75, 0.73, 3.3 against the flow and 0.63, 0.59, 3.2 against exact SGD; coupled 0.40, 0.27, 1.6 and 0.41, 0.30, 1.6; free drift 0.94, 0.94, 0.68 and 0.79, 0.76, 0.67.

The static radial formula ranks nodes reasonably but predicts a third of the hub shrinkage; letting partners respond captures the slight growth of low-degree nodes but ranks worse.
The free-drift form, which integrates the penalty's radial push over τ without a restoring force, ranks nodes best (within degree bins 0.91).
It overshoots the shrinkage of the two highest-degree bins by a factor of about 1.75 and misses the slight growth (under 1%) of the lowest bins, which comes from partners compensating.
**Simplest working predictor:** Δlog(balanced norm)_i ≈ −κ·(ε/2)·τ·F_i applied to both factors, with F_i computed from the trained model and the corpus pair counts.
The factor κ stands for the partial restoring force; it is fitted (0.68 at ε = 0.000625, 0.56 at 0.0025, so the shift grows somewhat less than linearly in ε) and untested on other graphs or settings.
For ranking nodes by how much a given step size shrinks them, κ does not matter.

## 2. Corpus reuse on the flat-optimum corpus (T1)

### Derivation

On the expected-BA corpus every token is an iid draw from q ∝ degree, so every node's population context law is q and the population optimum has flat norms.
A node with n_i tokens has an empirical context law from m_i draws; for unweighted draws (one-hop, m_i = 2n_i)

$$\mathbb{E}\,\mathrm{KL}(\hat C_i\Vert q) = H(q) - \log m_i + \frac{1}{m_i}\sum_j \mathbb{E}[X_j\log X_j],\qquad X_j\sim\mathrm{Poisson}(m_i q_j),$$

which is about H(q) − log m_i while m_i ≪ e^{H(q)} and (V − 1)/(2m_i) once m_i ≫ V.
For shrinking windows we use m_eff = n_i·2(Σw_t)²/Σw_t² (8.2 n_i at window 5).
The model can fit this noise only on a reused corpus, and node i's progress toward it scales with its updates, n_i τ.
Model: norm²_reused − norm²_fresh = b·KL_i·(1 − e^{−n_i τ/c}).
Since KL_i falls with n_i more slowly than 1/n_i in the sparse regime, the product has an interior maximum that moves to smaller n_i as τ grows.

### Check

- The formula matches the realized noise KL node by node: Pearson 0.999 in all three settings and all five corpora; mean KL 1.949 realized vs 1.938 predicted at one-hop (2.47 nats at 339 tokens down to 0.10 at 27,213).
- The excess of the reused over the fresh 50-epoch one-hop run (seed 0, epochs 2–50) fits the model with R² = 0.69 (b = 9.7, c = 92); dropping the KL factor gives R² = 0.20.
- Predicted vs observed peak (tokens): epoch 5, 1,263 vs 4,061; epoch 10, 714 vs 442; epochs 20–50, 442 vs 442.
  Before epoch 5 the observed curve still rises to the top bin, which the model misses.
- Node-level test of noise fitting (Spearman of norm with KL_i − E[KL | n_i] inside count bins): about 0 in the committed 5-epoch one-hop runs (−0.04), 0.17 at default, 0.08 at wide; in the 50-epoch one-hop run it rises from 0 at epoch 2 to 0.22 at epoch 10 and settles at 0.10 by epoch 50.
  Fresh-corpus controls give −0.01 (one-hop), 0.07 (default), and −0.06 (wide).

**Verdict: partial pass.** The noise-KL law is exact, and KL(n)×progress(nτ) explains most of the reused-minus-fresh excess and the late peak position.
The node-specific part of the noise is encoded only weakly, so the hump reflects the count dependence of the noise target more than each node's particular sample.
Early epochs are dominated by speed, which the model does not describe.

## 3. Why Oyama's relation fails at one-hop on graphs (T3)

Let δ_wc = ⟨u_w − ū, v_c⟩ and K_w(t) the cumulant generating function of δ under the base law r.
The model's conditional is r(c)e^{δ}/Z, so KL = K′(1) − K(1) = κ₂/2 + κ₃/3 + κ₄/8 + …, and Oyama et al.'s Eq. 11 keeps κ₂/2.
`t3_expansion.py` computes the terms from the saved factors (`results/norm_tests/oyama/`).

| Run, epoch | median sd of δ | median abs(κ₃/3)/(κ₂/2) | KL(q ‖ model at ū) (Eq. 10 check) | Pearson(exact KL, κ₂/2) |
|---|---|---|---|---|
| BlogCatalog one-hop, 5 | 2.52 | 0.69 | 0.81 | −0.15 |
| BlogCatalog one-hop, 50 | 3.51 | 0.73 | 2.00 | −0.38 |
| BlogCatalog one-hop at window 5, 50 | 0.80 | 0.16 | 0.008 | 0.79 |
| BlogCatalog default, 50 | 0.90 | 0.12 | 0.010 | 0.90 |
| hep-th one-hop, 5 | 1.36 | 0.25 | 0.11 | 0.57 |
| hep-th one-hop, 50 | 2.33 | 0.69 | 0.24 | 0.20 |
| hep-th default, 50 | 1.36 | 0.49 | 0.19 | 0.76 |

At one-hop on BlogCatalog the score spread is about three times larger, the third-order term is comparable to the quadratic one, and Oyama's Eq. 10 assumption that the unigram equals the model's conditional at the mean vector fails by 0.8–2.0 nats (0.01 at window 5).
In hep-th at one-hop the same quantities grow with training, and the quadratic form's correlation with the exact KL falls with them.
**Verdict: pass (diagnostic).** The relation breaks where the expansion it rests on breaks.
Even where the ranking holds, the quadratic form is a small fraction of the exact KL on graphs (0.37 vs 8.3 nats for BlogCatalog at window 5), so the relation there is a rank correlation; graph contexts carry KL far outside the small-KL regime of the derivation.

## 4. Follow-up tests in the main session

Scripts in `scripts/norm_tests/`; five PR #36 graphs unless marked.

**Knockout (`counter_penalty.py`).** At every SGD example, add +ε·(ε/4)·∇R_k (realized negatives, pre-step point), which cancels the first-order penalty.

| Learning rate | plain SGD gap, top bin | full knockout | context-part knockout | gradient flow |
|---|---|---|---|---|
| 0.00628 | −0.021, 5.87 | +0.231, 7.65 | +0.202, 7.33 | +0.176, 7.26 |
| 0.00314 | +0.042, 6.36 | +0.187, 7.35 | +0.179, 7.23 | +0.176, 7.26 |

The knockout converges to gradient flow as ε halves (context-part overshoot +0.026 → +0.003), consistent with an O(ε²) residual. At ε = 0.0314 the full knockout diverges (loss 1.94, one graph). Losses stay above gradient flow (0.390 vs 0.382 at 0.00314), so the shape is the drift and the remaining noise raises loss without changing shape.

**Randomness (`stochasticity.py`).** Expected negatives with a fixed order (no randomness) keep 91% of the top-bin gap shift at 0.0314 and 58% at 0.00628; reshuffling the pair order changes nothing.

**Persistence (`swap_gensim.py`, Gensim).** 0.0314 for 2 epochs, then 0.000625: the gap recovers from −0.10 to about −0.05 within 20 epochs and stays there; at τ = 0.1 the loss equals constant small steps (0.3837 vs 0.3834) with hubs 21% shorter (5.29 vs 6.72).

**Real corpora (`stepsize_real.py`, one seed, matched τ = 0.06275).** One-hop: BlogCatalog peak 231 → 1,342 and top bin 1.20 → 2.27 × mean as the step shrinks (0.0126 → 0.00063); hep-th top-bin drop −0.10 → −0.02. Default setting (window 5): BlogCatalog peak stays at degree 13–23 and the top bin moves only 0.81 → 0.91 × mean, so the falling side there is the target; hep-th peak 416 → 13,291.

**Corpus-only genericity (`genericity.py`).** g_i = E_{j∼P(·|i)}[r/(1+r)], r = 5e^(−PMI): within-bin Spearman with shrinkage −0.31 to −0.45 (PR #36), −0.30 (hep-th), +0.12 (BlogCatalog, fails).

## What a paper could claim as theory

- **Step size.** A first-order backward-error result: SGD on SGNS follows gradient flow on L + (ε/4)E Σ_k‖∇ℓ_k‖², with an explicit expectation over negatives. On the one-hop PR #36 graphs it reproduces the matched-τ degree profiles and per-node shifts of exact SGD for ε from 0.000625 to 0.025. The hub-shrinking part is the context-gradient term ‖u_a‖²(r² + Σp²), which penalizes a center vector in proportion to its update count and per-pair residual. This is new as far as the prior-work audit found; Smith et al. supply the general expansion.
- **A computable predictor.** Shrinkage ∝ ε τ F_i, with F_i the node's Fisher mass, ranks nodes within degree bins at 0.91–0.94 against the flow and 0.76–0.82 against SGD; its scale constant (0.56–0.68) is fitted.
- **Corpus reuse.** On a flat-optimum corpus the reused-corpus excess follows the exact multinomial noise-KL law times update-limited progress, which predicts the late peak position. The node-level noise signal is weak, so this is a count-level account.
- **Limits of Oyama's relation.** The quadratic norm–KL link holds when score spreads are near the second-order regime and Oyama's reference assumption holds; one-hop graph contexts violate both.
- **Causal check.** Cancelling the penalty during SGD restores gradient-flow lengths, converging as ε shrinks (section 4).
- **Not established.** The predictor's scale constant, the per-node predictors on BlogCatalog, the restoring force behind partial persistence, the early-epoch speed regime, and anything about convergence.
