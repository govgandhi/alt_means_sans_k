# Referee report, Reviewer 5 (mathematical theory)

**Manuscript:** "Degree Information and Norm Turnover in Random-Walk Network Embeddings" (Gandhi, Kojaku, Ahn), submitted to Physical Review E.

**Version reviewed:** commit `c958c528`, `paper/current/main.tex` and `main.pdf` (15 pp.), `supp.tex` with `factor_balance.tex` (`supp.pdf`, 3 pp.), `norm_geometry_supplement.tex`, and `note.tex` (historical).

**Scope:** Secs. II–X of the main text, Supplemental Sec. S3 (native factor balance), and `norm_geometry_supplement.tex`. I discuss experiments only where the theory makes a checkable prediction about them.

**Check scripts:** `<scratch>/round/r5/`, with saved outputs in `r5/outputs/` and extracted primary sources in `r5/sources/`. Run any script with `PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with numpy,scipy,sympy,networkx[,cvxpy] python <script>.py`. I did not retrain any embedding from the paper. Every check is a closed-form target, an SVD or eigendecomposition, a symbolic identity, or a toy optimization on at most four nodes.

Throughout, "main.tex:N" is a source line and "p. N" a page of `main.pdf`. Result numbers follow the PDF: Remark 1, Lemma 2, Theorem 3, Corollary 4, Lemma 5, Theorem 6, Theorem 7, Corollaries 8–9, Remark 10, Propositions 11–12, Lemma 13.

## Report to the editor

- **Correctness.** I re-derived all 13 numbered results in Secs. II–VIII, the displayed identities in Secs. IX–X, and both supplements, 28 items in all (table below). I found no false mathematical statement. Several statements lack a scope condition or are imprecise (comments 8–17).
- **Novelty.** Low for the population part. Theorem 3 is Lemma 3.3 of Qin and Rohe (2013) at τ = 0, which I checked against arXiv:1309.4111. Corollary 8 is Lemma S9 of Davison, Morgan, and Ward (2024), which I checked against the NeurIPS PDF. Corollary 9 is the DCSBM instance of context-distribution smoothing (Levy, Goldberg, and Dagan 2015), which the paper does not cite. Lemma 13 and Proposition 11 are elementary. The new pieces are Remark 10, Proposition 11 as a reporting standard, and Proposition 12, which is narrower than the abstract suggests.
- **Significance.** The theory does not reach the experiments. Every experiment lies outside the population model. The one theoretical object that does apply, the unique free optimum of each training objective at full positive support, is never computed. Computing it changes how one of the paper's two interventions should be read (comments 2–3).
- **Recommendation.** Major revision. Confidence 4/5.

## Summary

The manuscript asks what embedding norms encode about degree in random-walk embeddings. Sections II–VI write three population objects built from the DCSBM proxy Ω = ΘZBZᵀΘ in one notation. The normalized-adjacency eigenspace has within-block radius ∝ θ^{1/2} (Theorem 3). The stationary window-pair matrix J has radius ∝ θ (Theorem 6). The pairwise SGNS optimum is block-constant at unigram negatives (Corollary 8) and acquires a context-side term (1−α) log θ_j otherwise (Corollary 9). Section VII treats the factorization gauge, adopts canonical balanced factors, proves that loss convergence fixes balanced norms only when every pair has positive weight (Proposition 11), and constructs near-optimal balanced embeddings with three distinct normalized norm–degree shapes on a four-node graph (Proposition 12). Section VIII gives a deterministic radius and direction stability lemma, Sec. IX relates the results to prior work, and Sec. X proposes a two-input response model ν(log m, KL) whose opposite-signed partial effects are offered as an account of the inverted-U. The remaining sections test the response model with controlled SGNS interventions, corrected Gensim controls, and a hash-frozen prospective predictor.

## Strengths

1. The PMI normalization is handled correctly. Eq. (43) and main.tex:560–571 state that right multiplication by 𝒟⁻¹ already forms the joint-to-marginal ratio, which settles the double-counting question raised in note.tex:35–37. I confirmed that the old draft's matrix log C − log π_i − log π_j equals a block-constant matrix minus log θ_i minus log θ_j, so the old "additive offset" conclusion was an artifact of the double count (`check_population.py`).
2. The ordered versus two-orientation distinction (Corollary 9 against Eqs. 47–48) and the loopless window-one correction in Remark 10 are explicit and correct. The paper does not claim exactness for the loopless model.
3. Proposition 11 and the canonical balanced convention give a precise reporting standard for norm claims, and the experiments report both raw and balanced norms.
4. The fixed-noise expected-loss fits (Sec. XI D) are the right bridge between the objective and trained factors; they isolate a decline at full positive support.

## Weaknesses, ranked

1. The population theorems describe none of the experiments, and the paper never says so in one place (comment 1).
2. The free optimum of each training objective is never shown. Computing it shows that the intended mixed targets have norms increasing with degree, opposite to the trained curves, and that the realized intervention never reaches full support (comments 2–3).
3. Proposition 11's main consequence for this paper is left implicit. All Gensim, controlled, and prospective experiments use one-step contexts, where the infimum is not attained, so their norm curves are optimization-path observables. Proposition 12 is correct, but it rests on full rank and on unconstrained self-pair scores, and gradient descent selects one reproducible shape (comments 4–5).
4. Most population results are known, three relevant lines of prior theory are missing, and one cited convergence theorem is quoted without the norm constraint that removes the paper's own boundary case (comment 6).
5. The response model is an ansatz. The inverted-U inference has a logical gap, and its two inputs are almost collinear with log degree in the test graphs (comment 7).
6. Section VIII adds no finite-sample content (comment 12).
7. Presentation: the random-graph model is undefined, the θ gauge is unstated, notation collides, a bound cited in the main text is missing from the Supplemental Material, and the negative-sampling exponent of the experiments is never stated (comments 8–11 and 13–17).

## Detailed comments

### 1. The population theorems do not describe any experiment

**Location.** Secs. II–VI and IX (pp. 2–7), Sec. XI (pp. 8–13), Discussion main.tex:1177–1182 (p. 13).

**Problem.** Every experiment uses configuration-model graphs, which are DCSBMs with K = 1. With K = 1 and α = 1, F = 1 and Z₁ = sκ, so Eq. (38) gives H⁽¹⁾ = −log k. The population score is a constant matrix, every minimal or balanced factorization has identical rows, and the population prediction is equal norms for every node. The paper notes the constant score at main.tex:879 but does not draw this consequence for its own experiments. All trained norms in Figs. 1–6 depend strongly on degree, so the population results serve as a null model that every experiment rejects.

**Evidence.** The theorem that would carry the cancellation over to trained factors is Davison et al. (2024), Theorems 2–3. It requires bounded propensities θ_u ∈ [C⁻¹, C] and ρ_n ≫ log n/n (scenario iii, NeurIPS PDF p. 6). It applies to minimizers over the bounded set {‖U‖_∞ ≤ Ã_∞, ‖U‖_{2,∞} ≤ Ã_{2,∞}}, which bounds every row norm (p. 6). Factor convergence additionally needs d = rank M* (Theorem 3, p. 7). The experiments violate all three conditions. Mean degree is 5.9 against log 600 = 6.40 (`check_allocation_support.py`, seed 0), degree tails follow Barabási–Albert, and training is unconstrained at d = 24 or 64 against rank M* = 1. Section IX (main.tex:863) cites Theorems 2–3 without these qualifiers.

**Fix.** Add one paragraph to Sec. IX or the Discussion, and a column to Table I, stating for each population object whether any experiment lies within its scope. Quote the three conditions of Davison et al. and note that the bounded row-norm set in their Theorem 2 excludes the divergence of Proposition 11. If the population theory is meant to be tested, add one DCSBM experiment with K ≥ 2, growing degrees, and d = K, where it predicts block-constant balanced center norms.

### 2. The free optimum of each training objective is computable and absent

**Location.** Sec. X, main.tex:881–892 (p. 7); Sec. XI, main.tex:964–1062 (pp. 8–10); Figs. 1–3.

**Problem.** Proposition 11 guarantees a unique finite optimum whenever every pair has positive weight. Its balanced norms take one SVD, yet the manuscript shows them for none of its conditions. The repository computes the window-10 target only to report a normalized distance of 0.0505–0.0537 to the fits (analysis/norm_identifiability_findings.md). I computed the target profiles on graphs from the paper's own generator (copied from scripts/simulate_norm_frequency.py:77).

**Evidence.**
- **Window 10, uniform centers, q = π, k = 5** (the Sec. XI D setting). The free target's balanced norms decrease with degree, with Spearman −0.39, −0.41, and −0.36 on seeds 0–2. The RMS-normalized highest-degree bin is 0.844–0.848, against 1.012–1.015 in the lowest (`check_realized_target.py`). The rank-24 Eckart–Young truncation has the same shape. On seed 0 the free-target norm and the window-10 context KL have Spearman +0.934 (`check_kl_vs_target_norm.py`). This is target-level support for the specificity branch, stronger than anything in the main text.
- **Context mixing with the intended rows T^{(λ)} of Eq. (70).** For λ = 1/3 and 2/3 the intended rows have full support. The free target's balanced norms then increase with degree, with Spearman +0.66 to +0.80 at full rank and +0.66 to +0.78 for the rank-24 truncation (β ∈ {0, 1}, seeds 0–1; `check_mixing_target.py`; confirmed by eigendecomposition in `probe_mixing_sanity.py`). The norms are weakly anti-correlated with the intended KL (Spearman −0.19 and −0.10). The trained β = 0 curves go the other way, with raw-norm Spearman −0.782 at λ = 1/3 and −0.349 at λ = 2/3 (`results/turnover_surface/summary.json`, `calibration_curves.beta_000_lambda_033` and `_067`). At λ = 1 the target is the constant −log k, whose balanced norm √(log 5) = 1.269 is flat, in line with the flat trained value of 0.9488.

At intermediate λ the intended targets therefore have the opposite norm–degree sign from the trained curves, and the four-epoch trained curves sit far from the optimum of the objective being trained. The statement "context mixing removes the decreasing branch" (main.tex:101, 1009, 1019) is a statement about the training path. The reading "lower specificity at a fixed target removes the branch" does not follow from it.

**Fix.** For every condition in Figs. 1–3, plot the free-target balanced profile at full rank and the exact rank-24 expected-loss optimum (the solver already exists in scripts/window_training/expected_window.py) beside the trained curve. Report the window-10 target profile and its KL correlation in Sec. XI D, and rewrite the interpretation of Fig. 1(d) accordingly.

### 3. The realized context intervention never reaches full support

**Location.** main.tex:974–987 and 1021–1025 (p. 8); scripts/simulate_turnover_surface.py:181–250.

**Problem.** `allocate_pairs` rounds m_i T^{(λ)}_i by largest-remainder allocation (`exact_allocate`, scripts/simulate_exact_center_exposure.py:113–126). With 240 events per node, most mixing entries have expected count below one and round to zero. The leftover units go to the largest fractional parts, which are the high-q (hub) contexts.

**Evidence.** On seed 0 the realized positive support covers 1.0%, 13.3%, 24.9%, and 35.0% of the n² pairs at λ = 0, 1/3, 2/3, and 1 (β = 0; 12.7–31.6% at β = 1). Nonedge contexts that receive pairs have mean degree 16.8, 12.1, and 9.9, against a graph mean of 5.9 (`check_allocation_support.py`). Every λ < 1 condition therefore stays in the non-attained case of Proposition 11, and in the realized data the intervention mostly adds hub contexts to each row. The sentence "Integer allocation can change the marginal and support" (main.tex:986) understates a 13- to 35-fold change in support and a degree-biased mixing mass.

**Fix.** Train the controlled conditions on the exact expected loss without rounding, or use stochastic rounding with a budget large enough that most intended pairs appear. Otherwise report realized support and empirical KL for each condition and restate what the intervention manipulates. Fit the response surface on realized KL; main.tex:1025 already flags the difference.

### 4. Proposition 11 makes every one-step-context norm curve an optimization-path observable

**Location.** Proposition 11, main.tex:691–724 (p. 5); main.tex:1079 and 1162; Figs. 4–6.

**Problem.** With one-step contexts on a simple graph, the diagonal and every nonedge have p⁺ = 0, which is about 99% of the pairs on seed 0 (`check_allocation_support.py`, λ = 0). By Proposition 11, any sequence approaching the infimum has ‖M‖_* → ∞. The rank-24 window-one fits already come within 1.3×10⁻⁶ of the unrestricted infimum (main.tex:1042). The Gensim sweeps, the corrected controls, the controlled interventions, and the prospective test all use window one. Their norm–degree curves are therefore set by initialization, learning-rate schedule, and stopping time, and no objective-level quantity fixes them. The schedule dependence in Fig. 5, where the constant-rate schedule shows no interior maximum (main.tex:1118), is what this predicts. The paper states the ingredients (main.tex:1043, 1180) but never the conclusion.

**Fix.** Add a corollary after Proposition 11: with one-step contexts on a simple graph, the objective determines no norm–degree profile, so any reported curve must be indexed by the optimizer path. Cite the results that characterize the limit direction when the infimum is not attained: Soudry et al. (2018) and Ji and Telgarsky (2019) for separable and partially separable losses, and Gunasekar et al. (2017) and Lyu and Li (2020) for factorized models. Present Fig. 5 as a test of this corollary.

### 5. Proposition 12 is correct, and its content is the self-pair degeneracy at full rank

**Location.** main.tex:726–747 (p. 6); norm_geometry_supplement.tex:28–70; Fig. 3(a); abstract main.tex:66.

**Verified.** For δ = 10⁻², 10⁻⁶, and 10⁻¹⁰ the constructed matrices are negative definite, with r_i² = Dα_i, ρ = √D, and 0 < L − L_inf ≤ δ (D = 62.0, 154.1, 246.2 and s = 5.39, 14.60, 23.81). The factors U = (−M)^{1/2}, V = −U are balanced (`check_props.py`). This reproduces `results/norm_identifiability/exact/exact.json`.

**Problems.**
- (a) The entire profile is carried by the zero-positive self-scores u_i·v_i = M_ii = −Dα_i, for example −111.6 for the two degree-two nodes and −12.4 for the other two in the peaked profile at δ = 10⁻². Every context vector is antiparallel to its center vector, and M has rank 4 = n. This freedom disappears once d < n couples the diagonal to the off-diagonal scores.
- (b) The four-node graph is not special. The same construction works on any simple graph with one-step contexts. On a 600-node configuration graph from the paper's generator, rising, falling, and peaked profiles each come within 10⁻⁶ of the infimum, and their RMS-normalized radii match the prescribed profiles with correlation 1.0000000 (`check_props.py`).
- (c) With a window of two, the same four-node graph has full support, and its unique normalized profile is (0.930, 1.016, 1.016, 1.036) for degrees (3, 2, 2, 1), so the construction fails there.
- (d) Training selects a shape. Factorized gradient descent from a word2vec-style start (small uniform U, V = 0) at d = 3 and d = 4, three seeds each, reaches loss gaps near 3×10⁻⁷. It converges slowly toward a decreasing profile near (0.915, 0.945, 0.945, 1.17); the six runs agree within about 0.02 and drift in the same direction (`gd_selection.py`, `outputs/gd_selection.txt`). None approaches the rising or peaked profile. At d = 2 the loss stalls at a gap of 0.15 while the normalized radius of the degree-three node keeps falling, from 0.92 at 10⁴ steps to 0.76 at 3×10⁶. That is norm drift at constant loss inside a rank-constrained problem, closer to the paper's rank-24 setting and outside what Proposition 12 covers, since it is stated against the unrestricted infimum. The minimum-nuclear-norm margin direction on the zero-positive set, (0.910, 1.000, 1.000, 1.082) from cvxpy, has the same sign as the gradient-descent profile and a different limit.

**Verdict.** The proposition is correct and rigorous. For this paper it is a technicality: it shows that the unrestricted window-one objective admits any shape at full rank, which already follows from non-attainment. The question that bears on the experiments is which direction a given optimizer selects.

**Fix.** State the result for general simple graphs with one-step contexts at full rank, name the self-pair mechanism, and add the rank-constrained stalled-loss example. Add "at full rank" to the abstract sentence at main.tex:66. Treat the implicit-bias limit as the identifiable object.

### 6. Attribution and missing prior theory

**Location.** Sec. IX, main.tex:828–865 (p. 7); Table I (p. 5); main.tex:911; references.

- (a) Theorem 3 is Lemma 3.3 of Qin and Rohe at τ = 0. Their Lemma 3.3(1) gives X_τ = Θ_τ^{1/2} Z (ZᵀΘ_τZ)^{−1/2} U, and their text notes that τ = 0 gives Θ₀ = Θ (arXiv:1309.4111, Sec. 3.1, extracted in `r5/sources/qin/`). Their standing assumption is B ≻ 0, so the paper's only extension is to indefinite B of rank K. main.tex:832 ("gives the unregularized structure") should say exactly this.
- (b) Corollary 9 is the DCSBM instance of context-distribution smoothing, PMI_α(w, c) = log[P(w, c)/(P(w) P_α(c))] (Levy, Goldberg, and Dagan, TACL 2015), which is not cited.
- (c) In the NeurIPS PDF of Davison et al. (2024), whose SHA-256 begins 26274fb2 and matches the audit's, I confirmed Lemma S9 (p. 25, Eqs. S49–S51), the two-orientation negatives of Eq. S9 (p. 21), and Proposition S24 (p. 41). Add the qualifiers from comment 1.
- (d) Missing: Modell and Rubin-Delanchy (arXiv:2105.00987, 2021) show that spectral embedding with the random-walk Laplacian is fully degree-corrected under the DCSBM, with asymptotic variance that depends on degree. In the paper's notation the right eigenvectors of 𝒯 are 𝒟^{−1/2}QY, whose rows y_a/√(s_aκ_a) are block-constant (`check_population.py`). This exponent-zero spectral operator is the spectral counterpart of Corollary 8 and belongs in Table I.
- (e) Missing: Arora, Li, Liang, Ma, and Risteski (TACL 2016) derive log p(w) ≈ ‖v_w‖²/(2d) − log Z in a generative random-walk model of text, a monotone norm–frequency law. It is the closest theoretical comparator for a paper about norm–frequency curves.
- (f) main.tex:911 attributes the retention law to Mikolov et al. (2013), whose Eq. (5) gives discard probability 1 − √(t/f), that is, keep = √(t/f) with elasticity exactly 1/2. The form √(t/p) + t/p used in Eq. (65), and with it the elasticity of 0.31 at the cutoff, come from the word2vec C code and Gensim. Cite the implementation explicitly.
- (g) The Kojaku et al. congruence (main.tex:839–861) is algebraically correct. At the largest window used (L = 10) on the paper's graphs, however, max_ij abs(log(1 + E_ij) − E_ij) = 66–232 (`check_realized_target.py`), so the linearized geometry describes no experiment. Say that this discussion is structural.
- (h) The paper never lists what is new. My reading: Theorem 6 as stated, the ordered versus two-orientation comparison, Remark 10, Proposition 11, and Proposition 12. A two-sentence statement in the introduction would help readers.

### 7. The response model of Sec. X

**Location.** Eq. (68) and main.tex:940–953 (pp. 7–8).

- (a) ν is introduced as an approximation with no derivation, and no result in Secs. II–IX gives the sign of ∂ν/∂KL or ∂ν/∂log m. Call it a phenomenological ansatz in the text.
- (b) The inference at main.tex:952–953 is incomplete, because positive partial derivatives with opposite-signed terms imply neither a crossover nor an inverted-U. Along the path log m = ℓ, KL = −ℓ, the function ν = (log m)²/2 + 1.5 KL gives a U-shape, and ν = 0.1 log m + KL gives a monotone decline (`check_misc.py`). An inverted-U requires the ratio of the two terms to cross one from above exactly once; state that condition.
- (c) The KL reference changes. Eq. (64) uses π, while Fig. 1(b) and the controlled fit use q (main.tex:1007, 1025). For the α = 1 SGNS optimum the natural reference is the negative-sampling distribution. Use one definition throughout.
- (d) On the five calibration-family graphs, KL_i has slope −1.00 ± 0.01 on log d_i, the neighbor term is flat, and the correlation is −0.78 to −0.81 (`check_kl_slope.py`). Along the degree path both inputs are therefore close to affine in log d, and the two partial effects are identified only through the interventions and the neighbor-degree scatter. This explains why the degree-and-threshold baseline passes the same criteria (main.tex:1171).
- (e) As written, ν can be any function, and some ν fits any one-dimensional degree path. The falsifiable content is sufficiency, meaning that equal (log m, KL) predict equal norms across conditions, plus the sign restrictions. State both as hypotheses together with the outcome that would reject them. The global quadratic surface already failed at the strongest held-out threshold (main.tex:1026), and comment 2 shows that even exact optima do not share one sign of the KL effect.

### 8. DCSBM setup

**Location.** Sec. II, main.tex:110–161 (p. 2).

- (a) No random-graph model is defined. Karrer and Newman use Poisson multigraphs with E[A_ii] = Ω_ii (self-loops counted twice), which is exactly the proxy with diagonal, while the constraint θ_iθ_jB ≤ 1 (main.tex:117) suggests Bernoulli edges. State the edge distribution, independence, and diagonal convention.
- (b) (θ, B) is identified only up to θ_i → c_{g_i}θ_i, B_ab → B_ab/(c_a c_b). Every result in Secs. II–VI is invariant under this map (checked in `check_population.py`), but the constraint θ_iθ_jB ≤ 1 and the phrase "propensities remain bounded and every block propensity mass s_a grows" (main.tex:160) are not. Use the invariant condition max_i θ_i/s_{g_i} → 0, or fix the Karrer–Newman gauge Σ_{i∈a} θ_i = 1 that Qin and Rohe adopt.
- (c) "Population" means plug-in, that is, walks on Ω. 𝒯^r differs from E[(D_A⁻¹A)^r], and the log of an expectation differs from the expectation of a log. One sentence would stop readers from taking J (main.tex:373) as an expectation over random graphs.
- (d) No sparsity regime is stated. Any statement linking Ω to realized graphs needs one, for example ρ_n ≫ log n/n in Davison et al. or minimum expected degree plus τ of order log n in Theorem 4.1 of Qin and Rohe.

### 9. Self-loops beyond window one

**Location.** Remark 1 (main.tex:149–161), Remark 10 (main.tex:639–647), Theorem 6.

Remark 1 bounds the change in the normalized operator, and Remark 10 gives the loopless score at window one. For raw pairs and for SGNS at windows L ≥ 2 the loopless walk breaks the multiplicative form, and the paper says nothing. On loopless DCSBMs at window three (`check_population.py`, `probe_loopless.py`), the rank-K balanced factor has within-block radial slope 0.86, 0.95, and 0.99 and direction spread 0.11, 0.08, and 0.07 at n = 37, 148, and 592 with Pareto θ. With heavy-tailed θ, ‖J_loopless − J‖/σ_K(J) still lies between 0.35 and 4.8 at n = 592 across windows one, three, and ten, because max θ_i/s_{g_i} decays slowly; with uniform θ it falls to 0.01–0.02. A telescoping argument gives ‖𝒯₀^r − 𝒯^r‖_∞ ≤ 2r max_i η_i, with η_i from Remark 10. Add a remark with this bound and its consequence for J and for the SGNS score (which needs a relative bound because of the logarithm), or restrict Theorem 6 and Corollaries 8–9 explicitly to the proxy.

### 10. Walk starts and the DeepWalk claim

**Location.** main.tex:106 and 380 (pp. 2–3).

J is the pair law of walks started from ϖ. DeepWalk starts γ walks at every node. With uniform starts and the paper's length-five walks, pairs that involve the first token carry no θ_i factor. The expected pair matrix then has rank 2K (σ₇/σ₃ ≈ 10⁻¹⁵ and σ₄/σ₃ ≈ 9×10⁻³), and its within-block radial slope is 0.85, against 1 for stationary starts (`check_population.py`). Qualify main.tex:106 with "for stationary starts, or as walk length grows".

### 11. Degenerate block structures and the wording around Lemma 2

**Location.** Abstract main.tex:56; main.tex:210–214 (p. 2).

Lemma 2 makes direction a function of community, and main.tex:213 correctly conditions distinct directions on non-proportional community vectors. The abstract and Sec. V read as if direction identifies community. In a concrete counterexample, bipartite B with a uniform window of two gives F = (W + W²)/2 of rank one, so J has rank one and both communities share one direction. The α = 1 SGNS center rows are then identical across all nodes, and community information survives only in the context rows (`check_degenerate.py`). Add to Lemma 2 that c_a and c_b are parallel iff rows a and b of the block matrix are positive multiples, which follows from the argument at main.tex:212, and qualify the abstract.

At main.tex:214, "an equivalent balancing condition" is imprecise because balanced factors can be wider than rank M. What holds is that balance gives UUᵀ = (MMᵀ)^{1/2}, hence col(U) = col(M) at any width, which suffices for the ray structure (`check_population.py`).

### 12. Section VIII adds no finite-sample content

**Location.** main.tex:749–826 (p. 6).

Lemma 13 is correct; it combines the reverse triangle inequality with ‖x/‖x‖ − y/‖y‖‖ ≤ 2‖x − y‖/‖y‖. The constant is loose by about a factor of two: the inner-product Dunkl–Williams inequality gives 2‖x − y‖/(‖x‖ + ‖y‖) ≤ 2ε/(2ρ − ε). In random tests the paper's ratio stays at or below 0.966 and the Dunkl–Williams ratio reaches 1.000 (`check_misc.py`). The section has no estimator, no eigengap, and no sampling rate, so main.tex:822–826 only says that an assumed bound is larger when ρ_i is small, and no experiment uses the lemma. Nearest-direction assignment (Eq. 59) needs the population directions and the alignment O, so it is an oracle rule. Either compress the section to a remark or replace it with an entrywise result for the normalized embedding (Abbe, Fan, Wang, and Zhong 2020; Cape, Tang, and Priebe 2019), including the sparse-regime regularization that Qin and Rohe introduced for exactly the low-degree nodes of main.tex:823. The central limit theorem of Modell and Rubin-Delanchy gives the degree dependence of ε_i explicitly.

### 13. Corollary 9 has a norm consequence worth stating

**Location.** main.tex:573–611 (p. 4); Table I, main.tex:457.

Table I says "vectors affine in log δ_i". In every minimal factorization, ‖v_j‖² = ‖e_b‖² + 2(1−α)(log θ_j) e_b·w + (1−α)²(log θ_j)²‖w‖². So ‖v_j‖ = ‖e_b + (1−α)(log θ_j) w‖ is convex in log θ_j, while center norms are constant within blocks. The ordered population target therefore cannot produce an inverted-U in either factor at any α. Stating this takes one line and supports the paper's separation of population geometry from turnover. In my K = 3 example the vertex of the parabola lies outside the data range for α ∈ {0.5, 0.75, 1.25}, so context norms are monotone there (`check_population.py`).

### 14. The two-orientation score has rank above K

**Location.** Eqs. (47)–(48), main.tex:613–637 (p. 5).

The formula is correct. "Generally not multiplicatively separable" is vague, and the consequence that matters is rank. At α = 0.75, K = 3, n = 150, the two-orientation score has numerical rank 7 with σ₄/σ₁ = 4.5×10⁻³ (`check_population.py`), so tied or symmetrized implementations have no K-dimensional exact factorization when α ≠ 1. State this.

### 15. Factor balance

**Location.** main.tex:684–688 (p. 5); factor_balance.tex:4–29 (Supplemental p. 2).

- (a) factor_balance.tex:27 says the one-step drift formula applies "not verbatim to Gensim's sequential updates". Within one Gensim skip-gram update, with one input vector and one positive plus k negative output vectors, both sides use pre-update values. The update is therefore a simultaneous step on a sampled score-only loss and satisfies D_next − D = η²(VᵀGᵀGV − UᵀGGᵀU) exactly (maximum error 6×10⁻¹⁶, `check_gauge_dynamics.py`). The exception is a negative drawn twice within one update. Successive updates compose as in any SGD. Replace the sentence.
- (b) The supplement omits several cases, which I verified symbolically (the first two and the scalar identity) or in small simulations (`check_gauge_dynamics.py`). Equal weight decay λ gives dD/dt = −2λD, so balance is attracting and exact at any stationary point. Unequal factor learning rates conserve UᵀU/η_u − VᵀV/η_v, which yields relative imbalance η_v/η_u − 1 (3.000 in a run with η_v = 4η_u). Gradient noise grows ‖D‖ (from 2.3×10⁻³ to 7.4×10⁻² in a toy run). In the scalar case a finite step contracts D by the factor 1 − η²g², while in the matrix case the η² term can take either sign (‖D‖ grew 15-fold at η = 4).
- (c) The measured relative Gram imbalance of 0.0786–0.163 can be tied deterministically to the raw–balanced discrepancy. Since (UUᵀ)² − MMᵀ = UDUᵀ and the square root is operator Hölder-1/2 on positive semidefinite matrices, max_i abs(r²_raw,i − r²_bal,i) ≤ ‖UDUᵀ‖_op^{1/2} (checked on random factors in `check_imbalance_bound.py`). Reporting this bound beside Table S3 would turn the observed agreement into a certified one.

### 16. A bound cited in the main text is missing from the Supplemental Material

**Location.** main.tex:1062 (p. 9); supp.tex:61.

main.tex:1062 states that "a conservative full-support loss-to-radius bound is uninformative for all ten window-ten fits". The bound is derived only in norm_geometry_supplement.tex, a separate document that supp.tex does not \input and main.tex does not cite, so a reader of main.pdf and supp.pdf cannot find it. I verified the derivation (`check_supp_bound.py`). The scalar curvature inequality is tight, ‖|A| − |B|‖_F ≤ ‖A − B‖_F holds for Hermitian matrices, and the end-to-end radius bound holds with median tightness 0.026 on random six-node problems. The uniform constant h = min_ij a_ij b_ij/(a_ij + b_ij) is what makes the bound uninformative, and a per-entry curvature weighting would give a much sharper certificate. Fix: \input the file into supp.tex and cite it at main.tex:1062.

### 17. Smaller points

- The negative-sampling exponent used in the experiments is never stated. The code uses α = 1 (scripts/check_subsampling_origin.py:82, scripts/prospective_norm_turnover.py:313, scripts/simulate_norm_frequency.py:239), whereas the Gensim default is 0.75. Since α = 1 versus α ≠ 1 is the paper's central theoretical distinction, state it in Sec. XI.
- Notation collides. δ is the expected degree (main.tex:137) and also the loss tolerance (main.tex:731). α is the negative-sampling exponent and also α_i in the proof of Proposition 12 (main.tex:738). s is the block mass and also the tail level (main.tex:737). D is the diagonal scale (main.tex:739) beside 𝒟 and the imbalance D of the supplement. h is the rank (main.tex:173), a profile (main.tex:738), and a column (main.tex:610). C is the community-vector matrix (main.tex:176) and the constant C_ab (main.tex:627). ρ is a radius (main.tex:766) and an RMS (main.tex:727). Δ is a direction gap (main.tex:816) and a loss gap in the supplement.
- The abstract (main.tex:57) and main.tex:94 should add "for the population proxy with F_ab > 0" to the cancellation claim, since Eq. (63) shows that realized graphs reintroduce degree.
- Eq. (20) requires ℬ ⪰ 0, which by congruence is equivalent to B ⪰ 0; stating it for B is simpler.
- Already tracked issue #44, new evidence: at `c958c528`, scripts/simulate_norm_frequency.py:150 `sample_walks` calls `np.add.at(counts, walk, 1)`, which accumulates repeated indices. A direct check gives [0, 2, 1, 0] for the walk [1, 2, 1], so this function appears to count token occurrences. The issue's premise may be stale; please re-check which code path it refers to.

## Derivation-check table

All scripts are in `r5/` and all outputs in `r5/outputs/`.

| # | Result | Location | Assumptions (stated; unstated) | Check | Verdict |
|---|---|---|---|---|---|
| 1 | δ_i = θ_iκ_{g_i}, vol = sᵀBs (Eqs. 3–4) | main.tex:135–147, p. 2 | none | `check_population.py` II | correct |
| 2 | Remark 1: diagonal perturbation norm max θ_iB_aa/κ_a, relative degree change, bound max θ_i/s_{g_i} | main.tex:149–161, p. 2 | proxy normalization; bound needs B_aa > 0 (trivial otherwise); asymptotic clause is gauge-dependent | `check_population.py` II, including gauge map | correct; restate asymptotics in gauge-invariant form |
| 3 | Lemma 2: rays of minimal factors; zero row of A iff c_a = 0; gauge preserves ξ_i | main.tex:166–214, p. 2 | ξ, ζ > 0, minimal; nonempty communities (Sec. II) | balanced, random invertible gauge, non-minimal counterexample, wider balanced factors, zero and proportional rows | correct; "equivalent balancing condition" imprecise; injectivity condition missing |
| 4 | QᵀQ = I, 𝒮 = QℬQᵀ (Eqs. 11–13) | main.tex:226–246, p. 3 | none | `check_population.py` IV | correct |
| 5 | Theorem 3: u_i = √(θ_i/s_a) y_a, ‖u_i‖² = θ_i/s_a | main.tex:248–279, p. 3 | rank B = K | eigendecomposition; repeated eigenvalue (1, 0.625, 0.625); truncation at a repeated eigenvalue | correct; equals Qin–Rohe Lemma 3.3 at τ = 0 (they assume B ≻ 0); truncation leaves the block calibration basis-dependent |
| 6 | Corollary 4 and main.tex:281–282 | p. 3 | all K eigenvectors | as above | correct |
| 7 | Eqs. (18)–(20): weighted embedding, cosine B_ab/√(B_aaB_bb) | main.tex:293–318, p. 3 | ℬ ⪰ 0, equivalently B ⪰ 0 | positive definite and indefinite B | correct; formula fails for indefinite B, as the text says |
| 8 | Lemma 5: (𝒯^r)_ij = (θ_j/s_b)(W^r)_ab | main.tex:344–361, p. 3 | proxy with diagonal | r = 1–7 numerically | correct |
| 9 | Eq. (26) and the ergodic statements at main.tex:380–382 | p. 3 | irreducible 𝒯 | stationarity check | correct |
| 10 | Theorem 6: J = ΘZGZᵀΘ, symmetric G, exponent 1 | main.tex:384–434, p. 4 | proxy with diagonal; stationary starts | windows 3 and 10, ω_r ∝ (ℓ − r); loopless; uniform starts | correct for the proxy; approximate when loopless (slope 0.86–0.99); fails at finite length with uniform starts (rank 2K, slope 0.85) |
| 11 | Theorem 7 and Eqs. (36)–(38) | main.tex:482–544, p. 4 | positive weights; F_ab > 0 | α ∈ {0.5, 0.75, 1, 1.25} | correct |
| 12 | Corollary 8: block-constant score at α = 1 | main.tex:546–558, p. 4 | F_ab > 0 for all blocks | balanced factors | correct; equals Davison et al. Lemma S9 (verified) |
| 13 | Eq. (43), NetMF identity, no double count (note.tex D4) | main.tex:560–571, p. 4 | α = 1 | direct comparison; old draft matrix | correct; old matrix equals block-constant − log θ_i − log θ_j |
| 14 | Corollary 9: X* = ZHZᵀ + (1−α)1(log θ)ᵀ, rank ≤ K, center rows c_{g_i}, context rows affine in log θ_j | main.tex:573–611, p. 4 | F_ab > 0 | rank, balanced rows, affine fit | correct; norm convexity consequence missing (comment 13) |
| 15 | Eqs. (47)–(48), two-orientation score | main.tex:613–637, p. 5 | tied or two-orientation objective | formula check; rank | correct; rank above K at α ≠ 1 (7 for K = 3) |
| 16 | Remark 10: zeros; loopless window-one score with η_i | main.tex:639–647, p. 5 | positive loopless degrees | off-diagonal formula; bipartite F_ab = 0 case | correct |
| 17 | Eqs. (49)–(51): gauge, balanced factors, uniqueness up to orthogonal maps | main.tex:653–682, p. 5 | rank h | random factors; repeated singular values | correct; matches Davison et al. Prop. S24 (verified) |
| 18 | Flow conservation; D = 0 implies UUᵀ = (MMᵀ)^{1/2} | main.tex:684–686; factor_balance.tex:4–20 | equal rates, score-only loss | SymPy (`check_gauge_dynamics.py`) | correct; supplement omits weight decay and unequal rates |
| 19 | One-step drift η²(VᵀGᵀGV − UᵀGGᵀU) | factor_balance.tex:22–28 | simultaneous step | SymPy; simulated Gensim update | correct; also exact for one Gensim pair update, contrary to factor_balance.tex:27 |
| 20 | Proposition 11 | main.tex:690–724, p. 5 | fixed weights, p⁻ > 0 | elementary verification; `scripts/check_sgns_norm_boundary.py` reviewed | correct (strict convexity plus ‖M‖_* ≥ ‖M‖_F) |
| 21 | Proposition 12 and its proof | main.tex:726–747, p. 6; supplement §1 | one-step contexts, d = n, zero-positive self-pairs | δ = 10⁻², 10⁻⁶, 10⁻¹⁰; 600-node generalization; window two; GD at d = 2, 3, 4; cvxpy margin | correct; narrow (comment 5) |
| 22 | Lemma 13 and Eq. (59) | main.tex:757–826, p. 6 | assumed row error; shared O | 2×10⁵ random tests; Dunkl–Williams comparison | correct; constant loose by about 2; no estimator; oracle assignment |
| 23 | Eqs. (60)–(61), congruence and generalized eigenproblem | main.tex:839–861, p. 7 | Φ ⪰ 0 for the factor comparison | algebra; linearization error at L = 10 | correct; linearization far outside its regime on the paper's graphs |
| 24 | Eq. (62), population context KL | main.tex:870–879, p. 7 | proxy | direct KL | correct |
| 25 | Eqs. (63)–(64) and the −log d claim at main.tex:906 | main.tex:881–907, p. 7 | simple graph, window one | realized-graph KL formula; slopes on five graphs | correct (slope −1.00 ± 0.01) |
| 26 | Eqs. (65)–(67): cutoff (3+√5)/2, retained exposure √(tp) + t, elasticity 1/(2(1+√(t/p))) = 0.309 at the cutoff | main.tex:910–938, p. 7 | Gensim keep form | SymPy (`check_misc.py`) | correct; attribution caveat (comment 6f) |
| 27 | Eq. (68) and the inference at main.tex:952–953 | pp. 7–8 | ν arbitrary and smooth | chain rule; counterexamples | chain rule correct; inverted-U inference incomplete |
| 28 | Full-support loss-to-radius bound | norm_geometry_supplement.tex:72–131 | every a_ij, b_ij > 0 | scalar bound; Hermitian Lipschitz; end-to-end on random problems | correct; loose (median tightness 0.026); absent from the Supplemental Material |

## Questions for the authors

1. What does the exact rank-24 expected-loss optimum look like at β = 0 and λ = 1/3 or 2/3 with the intended, unrounded weights? Does its norm–degree sign match the increasing free target?
2. Why use largest-remainder allocation for the context intervention, given that it leaves most mixing pairs empty and favors hub contexts?
3. Do you regard any population result in Secs. II–VI as a prediction about the experiments of Sec. XI? If so, which one, and in what regime?
4. Which results do you consider new?
5. In the exact construction, does the conclusion survive if self-pairs are excluded from negative sampling, or at any d < n?
6. In "a fixed optimizer and gauge" (main.tex:940), does "optimizer" include the learning-rate schedule, the epoch count, and the initialization scale? Fig. 5 suggests that ν must be indexed by all three.
7. Does the window-10 free-target decline, with its +0.93 correlation between norm and KL on my seed-0 graph, hold on all five of your graphs? It would be the strongest target-level evidence for the specificity branch.

## Recommendation and confidence

**Recommendation: major revision.**

The mathematics is sound; none of the 28 checked items is wrong, and the gauge-aware reporting is useful to the field. As a theory contribution to PRE, however, the population part is largely known and connects to no experiment. The applicable object, the free optimum at full positive support, is missing, and it changes the interpretation of one of the two main interventions. The new propositions are correct but narrower than presented. Most fixes use code the authors already have (free-target SVDs and the expected-loss solver) plus rewriting, so a revision is realistic.

**Confidence: 4/5.** I checked every derivation symbolically or numerically, with scripts provided, and checked the Qin–Rohe and Davison et al. attributions against the primary texts. I am less certain about the rank-24 weighted optima behind comments 2–3, which I did not compute because that would require fitting embeddings. The Jin (2015) and Davison and Austern (2023) attributions rest on the authors' audit (analysis/issue31_prior_theory_audit.md), which I did not repeat. Bibliographic details of the added references in comments 4, 6, and 12 come from memory, except Modell and Rubin-Delanchy, which I located online.

## Action items

| ID | Severity | Location | Problem | Proposed fix | Type | Effort |
|---|---|---|---|---|---|---|
| R5-01 | major | Secs. II–VI and IX; Discussion main.tex:1177–1182 | The population theorems cover no experiment (K = 1 gives constant scores; Davison et al. conditions violated) | Add an applicability paragraph and Table I column; quote Davison et al.'s three conditions, including bounded row norms; optionally add one K ≥ 2 DCSBM run with d = K | ISSUE | ≤1day |
| R5-02 | major | Sec. X; Sec. XI A–D; Figs. 1–3 | Free optima never shown; intended mixed targets have norms increasing with degree (+0.66 to +0.80), while the window-10 target declines and tracks KL (+0.93) | Plot the free-target and exact rank-24 expected-loss profiles beside each trained curve; revise the reading of Fig. 1(d); report the window-10 target in Sec. XI D | ISSUE | ≤1day |
| R5-03 | major | main.tex:974–987; scripts/simulate_turnover_surface.py:181–250 | Largest-remainder allocation leaves 65–87% of pairs zero-positive and biases mixing toward hubs, so realized and intended targets differ qualitatively | Train on the exact expected loss, or use stochastic rounding with a larger budget; otherwise report realized support and KL and restate the intervention | ISSUE | >1day |
| R5-04 | major | Proposition 11; main.tex:1079, 1162; Fig. 5 | All window-one curves are optimization-path observables (non-attained infimum), which the paper never states | Add a corollary, cite the implicit-bias literature, and present Fig. 5 as its test | PR | ≤1day |
| R5-05 | major | Proposition 12, main.tex:726–747; abstract main.tex:66; supplement §1 | The construction relies on d = n and free self-pair scores, holds on any simple graph, and is not what GD selects; rank-constrained drift is not covered | Restate generally, name the mechanism, add the d < n stalled-loss example and the GD-selected shape, add "at full rank" to the abstract | PR | ≤1day |
| R5-06 | major | Sec. IX; Table I; references | Theorem 3 equals Qin–Rohe Lemma 3.3 at τ = 0; Corollary 9 is Levy et al. 2015; missing Modell and Rubin-Delanchy (exponent-zero row), Arora et al. 2016, and implicit-bias work; no list of new results | Revise the attributions, add the exponent-zero row and the citations, and add a two-sentence novelty statement | PR | ≤1day |
| R5-07 | major | main.tex:1062; supp.tex:61 | The cited full-support bound exists only in norm_geometry_supplement.tex, which the Supplemental Material omits | \input it in supp.tex and cite it; optionally sharpen with per-entry curvature | PR | ≤1h |
| R5-08 | minor | Sec. VIII, main.tex:749–826 | Lemma 13 is a triangle inequality with a loose constant and no estimator; nearest-direction rule is an oracle | Compress to a remark with the Dunkl–Williams constant, or replace with an entrywise perturbation result | PR | ≤1h |
| R5-09 | minor | Eq. (68), main.tex:940–953 | Ansatz presented without that label; inverted-U inference incomplete; KL reference inconsistent; falsifiable content unstated | Label the ansatz, state the single-crossing condition, fix one KL reference, and state sufficiency and sign hypotheses with rejection criteria | PR | ≤1h |
| R5-10 | minor | Corollary 9; Table I main.tex:457 | Norm consequence missing | Add that context norms are convex in log θ and center norms constant, so no population inverted-U arises at any α | PR | ≤1h |
| R5-11 | minor | Eqs. (47)–(48) | "Not multiplicatively separable" is vague | State that the two-orientation score has rank above K at α ≠ 1 | PR | ≤1h |
| R5-12 | minor | Sec. II, main.tex:110–161 | Random-graph model, θ gauge, plug-in meaning of "population", and sparsity regime unstated | Define the edge model and diagonal convention, fix a gauge or use max θ_i/s_{g_i} → 0, add one sentence on plug-in quantities | PR | ≤1h |
| R5-13 | minor | Remarks 1 and 10; Theorem 6 | Loopless walks at L ≥ 2 unaddressed; deviations persist with heavy-tailed θ | Add the telescoping bound ‖𝒯₀^r − 𝒯^r‖_∞ ≤ 2r max_i η_i and its consequences, or restrict claims to the proxy | ISSUE | ≤1day |
| R5-14 | minor | main.tex:106, 380 | "Apply to DeepWalk" ignores uniform starts, which give rank 2K and slope 0.85 at length 5 | Qualify with stationary starts or long walks | PR | ≤1h |
| R5-15 | minor | Abstract main.tex:56; main.tex:212–214 | "Direction depends only on community" is not injective (bipartite window-two counterexample); balance is called equivalent to minimality | Add the parallelism condition to Lemma 2; reword main.tex:214 | PR | ≤1h |
| R5-16 | minor | factor_balance.tex:22–29; main.tex:684–688 | Gensim sentence imprecise; weight decay and unequal rates omitted; no deterministic link from imbalance to norm discrepancy | Replace the sentence; add both cases; report the bound max_i abs(r²_raw − r²_bal) ≤ ‖UDUᵀ‖_op^{1/2} beside Table S3 | PR | ≤1day |
| R5-17 | minor | Sec. XI | Negative-sampling exponent of the experiments never stated (code uses α = 1) | State α = 1 | PR | ≤1h |
| R5-18 | minor | main.tex:911 | Retention formula attributed to Mikolov et al. 2013, whose Eq. (5) differs | Cite the word2vec C and Gensim implementation; note that the published Eq. (5) form gives elasticity exactly 1/2 | PR | ≤1h |
| R5-19 | minor | main.tex:137, 173, 176, 610, 627, 727, 731, 737–739, 766, 816 | Symbol collisions (δ, α, s, D, h, C, ρ, Δ) | Rename the proof-local symbols | PR | ≤1h |
| R5-20 | minor | main.tex:839–861 | Kojaku et al. linearization discussed without its regime; error 66–232 at L = 10 on the paper's graphs | Say that the discussion is structural and does not describe the experiments | PR | ≤1h |
| R5-21 | minor | main.tex:57, 94 | Abstract-level cancellation claim lacks its scope | Add "for the population proxy with F_ab > 0" | PR | ≤1h |
