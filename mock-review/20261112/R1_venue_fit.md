# Referee report, Reviewer 1 (venue fit, novelty, framing, claims vs evidence)

Manuscript: "Degree Information and Norm Turnover in Random-Walk Network Embeddings" (Gandhi, Kojaku, Ahn), submitted to Physical Review E.
Reviewed at commit c958c528 (paper/current/main.tex and main.pdf, 15 pp.; supp.tex and supp.pdf, 3 pp.).
Locations give main.tex line numbers plus compiled numbers (PDF page, theorem, equation, figure).
Repository numbers were read from the named files or recomputed with scratch scripts in `<scratch>/round/r1/checks/` (`peak_null.py`, `support_check.py`, `top_bin.py`, `lcc.py`).
Every external source cited below was checked at its arXiv, publisher, proceedings, or PMC page or PDF during this review; the two exceptions (SCORE Lemma 2.1 and Davison & Austern Eqs. 16–17) are marked as relying on the authors' own audit.

## 1. Summary

The manuscript asks what the Euclidean norm of a random-walk skip-gram (SGNS) node embedding encodes about degree.
Part one (Secs. II–IX, pp. 2–7) writes three population objects for the degree-corrected stochastic block model (DCSBM) in one notation: the normalized-adjacency eigenspace (row norm ∝ √θ_i within a block, Thm. 3), the stationary window-averaged pair matrix (row norm ∝ θ_i, Thm. 6), and the unconstrained ordered SGNS score (block-constant at negative-sampling exponent α=1, Cor. 8; a context-side (1−α) log θ_j term otherwise, Cor. 9).
It adds the GL(d) factorization gauge and balanced factors (Sec. VII), a loss/norm boundary (Prop. 11), and a four-node construction in which near-optimal losses allow different RMS-normalized norm profiles (Prop. 12).
Part two (Secs. X–XI, pp. 7–13) trains SGNS on five n=600 configuration-model graphs.
Exact exposure allocations m_i ∝ π_i^β give decreasing (β=0), increasing (β=2/3, 1), and inverted-U (β=1/3) norm–degree curves, and context mixing at equal exposure flattens the decreasing curve (Fig. 1).
Exact expected-loss fits at windows 5 and 10 show decreasing norms (Figs. 2–3).
A Gensim sweep shows turnover that persists without subsampling under the default annealed learning rate and disappears under a constant rate (Figs. 4–5).
An ExtraTrees regressor on (log realized pair count, context KL), frozen before training, passes four declared criteria on direct Barabási–Albert (BA), Holme–Kim, and NetScience targets (Fig. 6).
The Discussion states that the mechanism of the Gensim curves is not established (main.tex:1182, 1189).

## 2. Contribution and fit (focus 1 and 6)

Contribution in two sentences.
The paper restates known DCSBM population identities for where degree enters spectral, pair-probability, and SGNS targets, and adds two elementary identifiability propositions for balanced norms.
Its new content is a set of controlled finite-training observations on small graphs showing that the SGNS norm–degree curve can be decreasing, increasing, or inverted-U depending on exposure allocation, context mixing, and learning-rate history, plus a two-feature regression that ranks node norms within curves better than degree does.

A PRE referee will read this as a collection of controlled numerical observations framed by largely known theory.
There is no central result.
I count seven loosely coupled results (population identities; gauge and identifiability; exposure/context interventions; fixed-noise window fits; the Gensim threshold sweep; learning-rate history; the prospective regressor), produced by three different optimizers (a custom mini-batch trainer at d=24, an exact expected-loss L-BFGS fit, and Gensim at d=64) that disagree where they overlap (comment 14).
The two halves barely touch.
Every experiment uses one-block graphs (configuration model, BA, Holme–Kim, unlabelled NetScience), where the population α=1 score is the constant −log k (main.tex:879); Theorems 3 and 6 and Corollaries 8–9 therefore make no testable prediction for any experiment in the paper.
The trained norms are explained by neither the population theory nor the realized-graph target (main.tex:889–892, 953), and the paper says so.
That candor is a strength, but it leaves the title phenomenon, norm turnover, unexplained.
Given this admission, the contribution is insufficient for PRE as submitted unless the paper either (a) supplies a derived account of the trained curve (Sec. 9 of this report) or (b) is cut down to the result that is both new and robust, the learning-rate-history dependence, with the population material compressed to credited background.

## 3. Strengths

1. Unusually transparent scoping: the paper states what each experiment does not establish, discloses the failed declared response surface (main.tex:1024–1028), reports a post hoc baseline that also passes the declared criteria (main.tex:1171), and ships hash manifests and per-node outputs.
2. The schedule comparison (Sec. XI F, Fig. 5) is new and consequential: with identical graph, corpus, initialization, and negatives, an interior maximum is present in 5/5 graphs under the annealed schedule and absent in 5/5 under a constant rate at epoch 20 (results/training_trajectory/README.md). Anyone reading norms as structural quantities, as Murray et al. (2023) did, needs this control.
3. Eq. (63) with Prop. 11 makes a sharp point: at window one on a sparse graph, nonedge scores are −∞ at the free optimum, so the objective has no finite minimizer and trained norms must depend on optimization. This is the paper's most physically useful theoretical observation, and it is undersold (comment 22).
4. The exposure–KL features carry node-level information beyond degree. On the saved prospective outcomes, the frozen predictor's within-curve Spearman correlation (median 0.842) exceeds an in-sample degree-only oracle (median 0.376, range 0.242–0.620; `peak_null.py`).

## 4. Weaknesses, ranked

1. No central result; the title phenomenon has no mechanism; the theory makes no prediction for the experiments (Sec. 2; comments 1–3).
2. The population results are largely prior work, and the closest precedents are uncited: Qin & Rohe's Lemma 3.3 contains Thm. 3, norm formula included, at τ=0; residual2vec (two of the present authors) contains Lemma 5's DCSBM endpoint formula and the statement that unigram noise cancels the random-walk degree bias; Levy, Goldberg & Dagan (2015) contains the α-smoothed PMI behind Cor. 9 and the balanced factorization (comments 4–9).
3. The controlled interventions repeat the design and qualitative findings of Wilson & Schakel (2015), which the paper cites only, and incorrectly, for the inverted-U (comments 10–12).
4. The prospective evidence is weaker than the abstract and conclusion state: node-free predictors pass the peak criteria, the BA and Holme–Kim targets lie inside the calibration support, the only out-of-distribution target carries the failures, and the reported run used the pre-correction implementation without saying so in the main text (comments 15–18).
5. The inverted-U is weakly resolved: about 1.5 decades of degree, 4–6 occupied quantile bins, a falling branch carried by the top ≈12% of nodes in one bin, and no uncertainty on the interior-maximum classification (comment 19).
6. The manuscript reads as an audit trail of code corrections and superseded outputs (comments 23–25).
7. Scope is narrow relative to the general abstract: α=1 only (never stated), p=q=1, window one in Gensim, n≤600, d=24 or 64, at most 20 epochs (comment 26).
8. Required end matter is missing or placeholder: data availability without URL or DOI; no acknowledgments, funding, author contributions, or AI-use disclosure (comments 27–28).

## 5. Detailed comments

### A. Contribution and coherence

1. **The title and abstract promise an account of norm turnover that the paper does not deliver.**
Location: title; abstract main.tex:65; Discussion main.tex:1182, 1189, 1201.
Problem: the Discussion says the complete explanation "remains a future question" and that the exposure–specificity account "has not been established as a complete mechanism for the Gensim curves".
Evidence: Figs. 1 and 4 disagree on where turnover appears (comment 14), and Fig. 5 shows the curve shape flips with the schedule.
Fix: decide the paper's single claim (Sec. 9). If no mechanism is added, retitle around what is shown (for example, "Learning-rate history shapes the norm–degree relation of skip-gram network embeddings") and demote the population material.

2. **The DCSBM theory makes no prediction for any experiment.**
Location: Secs. IV–VI versus Sec. XI A, G (main.tex:960–961, 1160).
Problem: all experiments are K=1, and at K=1 the population α=1 score is the constant −log k (main.tex:879). No experiment samples a K>1 DCSBM, checks the √δ or δ radii, or checks community directions.
Fix: either add a bridging experiment (sampled DCSBMs across densities and windows; spectral, pair-factorization, and SGNS norms against √δ, δ, and flat; directions against Eqs. (15), (30), (45)) or move Secs. III–VI to a short background section plus an appendix.

3. **The "two-force" response model is the chain rule.**
Location: Eq. (68), main.tex:940–953; Discussion main.tex:1186; Fig. 6 caption ("two-force norm predictor").
Problem: for any smooth ν(log m, KL) with positive partials and two degree-correlated inputs moving in opposite directions, Eq. (68) restates that an interior maximum lies where the two terms cancel. Nothing constrains ν, and the predictor actually tested is a nonparametric ExtraTrees fit.
Fix: present Eq. (68) as a descriptive decomposition and drop the "two-force" labels, or derive ν from a dynamical model and test its parameters.

### B. Novelty and attribution (focus 2)

4. **Theorem 3 is Qin & Rohe (2013), Lemma 3.3, at τ=0.**
Location: main.tex:248–279, 831–832; Table I row 1.
Evidence: Qin & Rohe (NeurIPS 26; arXiv:1309.4111, pp. 3–4) write L_τ = Θ_τ^{1/2} Z B_L Z^T Θ_τ^{1/2} in Lemma 3.2 and state the τ=0 case explicitly; Lemma 3.3 gives X_τ = Θ_τ^{1/2} Z (Z^T Θ_τ Z)^{−1/2} U, same-block rows parallel with squared length θ_i^τ / Σ_{j in block} θ_j^τ, and orthogonal rows across blocks. These are Eqs. (14)–(15) and the paragraph after the proof. The sentence "Theorem 3 gives the unregularized structure" (main.tex:832) implies the τ=0 case was missing.
Fix: state Thm. 3 as Qin & Rohe's result and keep the √θ exponent in Table I with that credit.

5. **Lemma 5, Theorem 6, and Corollary 8 overlap residual2vec, which is uncited.**
Location: main.tex:344–432, 546–558; introduction main.tex:85–95.
Evidence: Kojaku, Yoon, Constantino & Ahn (NeurIPS 2021; arXiv:2110.07654), Sec. 3.1, Eq. (11), give the window-averaged dcSBM random-walk transition P0(j|i) = (d_j/D_{g_j}) [T^{−1} Σ_t P_SBM^t]_{g_i g_j}. In the present notation d_j/D_{g_j} = θ_j/s_b, so this is Lemma 5 with uniform ω_r; multiplying by the stationary distribution gives Thm. 6. Their Secs. 2.2–3.1 state that unigram noise with exponent one negates the random-walk degree bias, and identify node2vec/NetMF noise ∝ d^γ with a soft configuration-model baseline. Two of the present authors wrote that paper.
Fix: cite residual2vec at Lemma 5, Thm. 6, Cor. 8, and in the introduction, and state what is added (rows of minimal exact factors; finite ordered windows).

6. **Corollary 9 is the DCSBM instance of context-distribution smoothing.**
Location: main.tex:573–611; Table I row 5.
Evidence: Levy, Goldberg & Dagan (TACL 3, 211–225, 2015; doi:10.1162/tacl_a_00134), Sec. 3.2, Eq. (3), define PMI_α(w,c) = log P̂(w,c) / (P̂(w) P̂_α(c)) with P̂_α(c) ∝ #(c)^α. Relative to α=1 this adds the context-only term (1−α) log #(c) up to a constant, which is the present (1−α) log θ_j. Davison et al. (2024), App. F, Eqs. (S209)–(S211), treat α≠1 with symmetrized negatives (credited at main.tex:864).
Fix: cite LGD 2015 and present Cor. 9 as an instantiation whose new part is the minimal-factor row form v_j = e_b + (1−α) log θ_j w.

7. **The gauge and balanced-factor material is standard in this literature.**
Location: Sec. VII, main.tex:653–688; SM Sec. S3.
Evidence: LGD 2015, Sec. 3.3, Eq. (5), already use the symmetric split W = U_dΣ_d^{1/2}, C = V_dΣ_d^{1/2} and remark that SGNS training treats W and C symmetrically; Davison et al. (2024) Prop. S24 and Du, Hu & Lee (2018) are credited. The SM's one-step identity D_next − D = η²(V^T G^T G V − U^T G G^T U) is the matrix case of the finite-learning-rate breaking of rescale-symmetry conservation in Kunin et al. (ICLR 2021; arXiv:2012.04728, Sec. 4, Eq. (13); Sec. 6, Eq. (20)).
Fix: cite LGD 2015 and Kunin et al., and cut Sec. VII to the definitions used later.

8. **The raw-pair versus PMI norm contrast has a direct precedent.**
Location: Table I; Sec. V; Discussion main.tex:1178.
Evidence: Huang, Silva & Singh (KDD 2021; doi:10.1145/3447548.3467300), Observations 2–3 and Fig. 7: autocovariance (centered stationary-pair) embeddings have squared norms proportional to degree, PMI-embedding norms are not directly degree-related, and they measure the norm–degree correlation on data.
Fix: cite it and say the present addition is the exact minimal-factor statement under the DCSBM.

9. **The PRE-native framing is missing: a radial coordinate for popularity and an angular coordinate for similarity.**
Location: introduction main.tex:74–95; Table I.
Evidence: in the hyperbolic model of Krioukov et al. (PRE 82, 036106, 2010), Eq. (37), the radial coordinate is logarithmic in expected degree, κ = κ0 e^{ζ(R−r)/2}, and the angle carries similarity; see also Papadopoulos et al. (Nature 489, 537, 2012) and Boguñá et al. (Nat. Rev. Phys. 3, 114, 2021). Spectral DCSBM work describes the same geometry of communities as rays from the origin (Sanna Passino, Heard & Rubin-Delanchy, Technometrics 64, 346, 2022, Sec. 2.3) and clusters on the sphere for that reason (Lei & Rinaldo, Ann. Stat. 43, 215, 2015, spherical k-median). Modell & Rubin-Delanchy (arXiv:2105.00987) show that the random-walk Laplacian embedding removes degree effects, so DCSBM communities concentrate around K points; this is the spectral counterpart of Cor. 8, and Table I omits it. Murray et al. (2023) already tested a hyperbolic-hierarchy explanation of their norm–size curve.
Fix: open with this geometry (is the SGNS norm a popularity coordinate, and when?) and add the random-walk Laplacian row to Table I.

10. **The inverted-U is misattributed to Wilson & Schakel.**
Location: main.tex:80, 1184.
Evidence: Wilson & Schakel (arXiv:1510.02675v2) train CBOW without subsampling on a 482-million-word Wikipedia corpus and report vector length varying approximately linearly on a logarithmic frequency axis at fixed co-occurrence, and decreasing approximately linearly with co-occurrence noise at fixed frequency (Secs. 4–5, Figs. 4 and 7); they report no reversal. The reversal appears in Schakel & Wilson (arXiv:1508.02297) for one corpus of hep-th abstracts, skip-gram with sample=10^−4, window 10, and 20 iterations: the band of lengths rises with term frequency and turns down near a term frequency of 30 (Sec. 3.3, Fig. 3). "Domain-specific corpora" is plural; there is one corpus.
Fix: cite Schakel & Wilson alone for the inverted-U, with its settings (subsampling on).

11. **The interventions repeat Wilson & Schakel's controlled design without credit.**
Location: Sec. XI A–C, Fig. 1, Eq. (70).
Evidence: Wilson & Schakel vary frequency at fixed co-occurrence by splitting a word into pseudowords (Sec. 4) and add co-occurrence noise at fixed frequency by interspersing occurrences uniformly at random, which mixes the context distribution toward the unconditional word distribution (Sec. 5.1), the same operation as T^(λ) = (1−λ)T + λq. Their findings (longer with frequency, shorter with noise) match Fig. 1c at β=1 and Fig. 1d.
Fix: credit the precedent and state what is new: exact per-node exposure allocation on graphs, and an inverted-U at β=1/3 with strictly increasing exposure.

12. **Attribute the two-force explanation and report its prior test.**
Location: main.tex:82.
Evidence: the competing exposure and context-diversity account is Schakel & Wilson's (Sec. 3.3, where both forces are said to set the length). Murray et al. (PNAS 120, e2305414120, 2023; PMC10756268) cite it as "specificity", test it, and report that their experiments do not support specificity as the sole mechanism, pointing to frequency and network connections (SI Figs. S32–S33). Mikolov et al. (2013), Sec. 2.3, motivated subsampling by saturation: frequent-word vectors stop changing appreciably after several million examples.
Fix: attribute the explanation, report Murray et al.'s negative test, and state what the present experiments add beyond it.

13. **Other relevant work the paper should engage.**
Arora et al. (TACL 4, 385, 2016), Thm. 2.2, Eq. (2.4): ‖v_w‖²/(2d) = log p(w) + log Z, a monotone norm–frequency law at their model's optimum.
Kojaku et al. (2024), cited only for the linearization, report that DeepWalk learns degree as a primary embedding dimension while node2vec is degree-agnostic (arXiv:2306.13400v2, Sec. 3), which bears on main.tex:106.
Karkada et al. (arXiv:2502.09863v3, 2025) give closed-form gradient-flow dynamics of word2vec-like models with sequential subspace learning; Saxe, McClelland & Ganguli (arXiv:1312.6120) give exact learning dynamics in linear networks; Soudry et al. (JMLR 19(70), 2018) describe norm growth when logistic targets are unattainable, the setting of Prop. 11. These are the natural tools for the schedule dependence.
Barot, Bhamidi & Dhara (arXiv:2111.05267) analyze non-backtracking node2vec walks, relevant to main.tex:108.
Zhang & Tang's DCSBM extension assumes max θ / min θ = O(1) (arXiv:2101.07354v2, Remark 7); say so at main.tex:862 in a paper about heterogeneous degree.

### C. Claims versus evidence (focus 4)

14. **The controlled and Gensim pipelines disagree, and both are labelled validation.**
Location: Fig. 1f; Fig. 4d and its in-figure title ("Literal Word2Vec validation of the turnover prediction"); Sec. XI E heading; introduction main.tex:102.
Evidence: in the controlled trainer the interior peak appears only under the strongest retention law (t=3×10^−4, peak 7.73), and the curves peak in the top bin at 10^−3 and 3×10^−3 (Fig. 1f). In Gensim the strongest threshold gives a decreasing curve (peak 2.98), and interior peaks appear at weaker thresholds and with subsampling off (Fig. 4b; results/subsampling_origin/README.md). The controlled interpolation misplaces the Gensim peaks by 4, 2, and 1 bins (main.tex:1104).
Fix: rename the section and figure (for example, "Gensim threshold sweep"), state the disagreement where the results are summarized, and match dimension, budgets, and schedules before comparing pipelines.

15. **The peak criteria cannot distinguish the predictor from node-free baselines.**
Location: abstract main.tex:63; main.tex:1165–1166; Fig. 6d; Conclusion main.tex:1213.
Evidence (`peak_null.py` on results/prospective_validation/outcomes.jsonl; it reproduces the reported peak bins): curves have 4 (BA), 5 (Holme–Kim), and 6 (NetScience) occupied bins. A predictor that ignores nodes and places the peak at bin = threshold rank + 1 has median error 1 bin and 9/9 peaks within one bin; a constant middle-bin predictor has median 1 and 8/9. Both pass declared criteria 1–2, and criterion 4 is predictor-independent (SM Sec. S2). The pooled Spearman of 0.881 mixes between-threshold level shifts; degree plus threshold reaches 0.652 (Table S1).
Fix: remove the peak-bin and pooled numbers from the abstract; lead with the within-curve result (0.842 versus 0.293 for the fitted degree-plus-threshold baseline; 0.376 for an in-sample degree-only oracle); add the null predictors to Table S1.

16. **The prospective run used the pre-correction implementation, and the main text does not say so.**
Location: Sec. XI G; Fig. 6 caption; abstract main.tex:62–63.
Evidence: results/prospective_validation/README.md states that these files predate the repeated-visit counter, integer-token negative-table, and BLAS corrections, and that their numbers must not be described as results of the corrected code. Fig. 4's caption flags its historical status; Fig. 6's caption does not. The corrected rerun is retrospective (results/negative_sampling_revalidation/README.md) and gives pooled Spearman 0.875 and 0.873 with 8/9 peaks within one bin.
Fix: state this in Sec. XI G and in the caption, and report historical and corrected values side by side.

17. **"New graph topologies" are mostly interpolation.**
Location: introduction main.tex:104; main.tex:1160; Conclusion main.tex:1213.
Evidence (`support_check.py`): the calibration graphs are configuration models built from BA (m=3, n=600) degree sequences, and the direct BA and Holme–Kim targets share n, m, and the degree law; 99.8–100% of their node feature vectors fall inside the calibration ranges of (log pairs, KL) at each threshold. NetScience is the only out-of-distribution target: 26.6% of its nodes have degree below 3, against 0.4% of calibration node rows, and 6.0% of its rows fall outside the calibration feature ranges at t=3×10^−4. The failures concentrate there: the two-bin peak miss, absolute overestimates of 0.37–0.53 in mean norm (corrected rerun), and within-curve ρ=0.506 at t=3×10^−3.
Fix: describe BA and Holme–Kim as in-distribution transfer, and run a second prospective round on graphs that differ in size, density, degree exponent, and degree correlations, including at least two further empirical networks.

18. **The declared response surface failed, and the replacements were chosen afterward.**
Location: main.tex:1024–1028, 1155; abstract main.tex:65.
Evidence: the declared quadratic surface predicted a curve rising through the top bin (16.04) where the held-out peak was at 7.73; the matching degree-bin interpolation and the ExtraTrees class were chosen after that failure. The abstract's "consistent with a weakening norm response to exposure relative to context specificity" rests on these exploratory fits.
Fix: say in the abstract that the declared model failed and that the supporting fits are exploratory, or drop the sentence.

19. **The inverted-U is weakly resolved.**
Location: Figs. 1, 4–6; peak definition in scripts/prospective_norm_turnover.py:433–472.
Evidence: degrees run mostly from 3 to about 85 (about 1.5 decades). Six or eight requested quantile bins collapse to 4–6 occupied bins, and a curve counts as inverted-U whenever its binned maximum is interior, with no uncertainty or effect-size threshold. In the corrected no-subsampling sweep every graph peaks in bin 5 of 6, so the falling branch is the single top bin (degrees 9–75, 11.5–12.5% of nodes); within that bin the node-level Spearman correlation of norm with degree is −0.71 to −0.82 (`top_bin.py`), so the decline is real but its shape is unresolved. In the controlled β=1/3 curve the rise above the lowest bin is 0.093, about 3% (main.tex:998). Pooling hides graph-level shapes: in the clustered family at t=3×10^−4 the pooled peak is interior while only 1 of 5 graphs has an interior peak (results/negative_sampling_revalidation/README.md). The motivating word2vec observation spans term frequencies from 1 to about 10^6 (Schakel & Wilson, Fig. 3).
Fix: define turnover operationally in Sec. XI; show node-level scatter with a smoother and graph-level bootstrap bands; test unimodality against monotone fits; use graphs whose degrees span at least three decades.

20. **"Persists" under the original schedule is weak evidence.**
Location: abstract main.tex:61; main.tex:1117.
Evidence: the continuation runs at 10^−4, 250 times below the initial rate. Expected loss falls 0.30–0.39% from epoch 5 to 20 under this schedule, against 6.85–24.38% under the constant rate (analysis/training_trajectory_findings.md). With subsampling off, the annealed run has the lower epoch-20 loss (0.3847 against 0.3893), so the inverted-U accompanies the better-optimized solution.
Fix: report these numbers, and add schedules that differ only in the initial rate or only in the floor.

21. **The interventions do not isolate the factors they are named after.**
Location: Eq. (70), main.tex:984–987, 1021–1022; abstract main.tex:60.
Evidence: mixing changes the realized marginal and therefore the negatives (stated in the text). At λ=1 and β=0, every node has identical contexts and identical exposure, so equal norms follow from exchangeability.
Fix: say "progressively flattens", base the claim on intermediate λ, and add a fixed-noise version of the mixing intervention like the one in Sec. XI D.

22. **Connect Prop. 11 to the schedule result.**
Location: Prop. 11, main.tex:690–724; Sec. XI F.
Evidence: at window one on a sparse graph, nonedges have p+=0, so no finite free optimum exists (Eq. (63)). Gensim skips updates with |score| ≥ 6, and up to 89% of expected negative mass lies outside (−6, 6) (results/training_trajectory/README.md). Norms are then set by finite-time dynamics plus an implementation clip.
Fix: make this the organizing idea of Sec. XI F and test it with sweeps of dimension (24, 64, 128), clip value, initial rate, and floor rate.

### D. Framing and presentation (focus 5)

23. **Audit-trail language.**
Location: abstract main.tex:61 ("Corrected word2vec controls"); Sec. XI E ("The original, uncorrected Gensim sweep"); Fig. 4 caption ("the historical plotting routine computed node-based standard errors"); Sec. XI F heading; main.tex:1103 ("have not yet been repeated with the corrected implementation"); Conclusion main.tex:1213 ("Hash-locked").
Evidence: in about 6,700 words of text, "corrected" or "uncorrected" appears 13 times, "original" 9, "exploratory" 6, and "historical" 3.
Fix: describe the final implementation once in a Methods section, and move the bug history, hashes, and freeze mechanics to an SM reproducibility section.

24. **Figures from superseded runs.**
Fig. 4 shows the uncorrected sweep with node-based standard errors although a corrected sweep with graph-level errors exists (results/subsampling_origin/); its panel (c) has no corrected counterpart (main.tex:1103).
Fig. 1 is a historical PDF with three text labels patched in place (results/realized_graph_audit/README.md:33).
Fix: regenerate every figure from corrected outputs with one plotting script, and drop panel 4(c) or rerun it.

25. **What a referee would ask to cut or move to the SM.**
Shrink Sec. III to a lemma statement and Sec. IV to a credited sentence.
Move Eqs. (18)–(20) and (47)–(48), Remark 10, Sec. VIII (Lemma 13 is a generic deterministic bound that no experiment uses), the congruence discussion around Eqs. (60)–(61), Sec. XI C's response-surface bookkeeping with Fig. 1e–f, Fig. 3c, and the criteria accounting to the SM.
Keep Table I (with credit), Eqs. (63)–(64), Props. 11–12, Fig. 1a–d, Fig. 5, and a prospective section that leads with within-curve ranking and null baselines.
The main text also cites a "conservative full-support loss-to-radius bound" (main.tex:1062) and the √2 cap (Fig. 3 caption) that are derived only in norm_geometry_supplement.tex, which is not part of the SM; include the derivation in supp.tex or drop the statements.

### E. Scope (focus 7)

26. **Generality is too narrow for the abstract's framing.**
Evidence: every Gensim run sets ns_exponent=1.0 (scripts/check_subsampling_origin.py:82; scripts/prospective_norm_turnover.py:313), yet the manuscript never states α for the experiments; the standard word2vec choice is 0.75 (Mikolov et al. 2013, Sec. 2.2; LGD 2015, Sec. 3.2), where Cor. 9 predicts a different context geometry. The Gensim runs use window 1, walks of length 5, and d=64. The controlled trainer uses d=24, batch size 256, and a rate decaying from 0.025 to 0.0025 over four epochs (scripts/simulate_turnover_surface.py:80–88); none of this appears in Sec. XI A, although the paper shows that the schedule changes the curve. Graphs have n≤600, and there is no text or trajectory corpus, although the motivating PNAS study used d=300 (PMC10756268).
Fix: state α, schedule, batch size, dimension, and initialization for every pipeline; add α=0.75, windows 5–10, a dimension sweep, and n≥10^4; or restrict the abstract to small graphs, one-step contexts, and unigram noise.

### F. Required statements (focus 8)

27. **Data availability is a placeholder.**
Location: main.tex:1216–1220.
Evidence: no repository URL or DOI, and "A permanent public archive will be supplied before submission." APS requires a data availability statement for all Physical Review journals (APS news release, August 2024); authors choose a pre-scripted template at submission, the statement prints after the Acknowledgments, and public datasets are cited in the references (journals.aps.org/authors/data-availability-statements-pilot; PRE author guidelines, journals.aps.org/pre/authors).
Fix: deposit code, frozen predictions, and outputs in an archive with a DOI, cite it, and cite the NetScience source (the repository copy reproduces the 379-node, 914-edge largest component; `lcc.py`).

28. **No acknowledgments or funding, no author contributions, and no AI-use disclosure.**
Location: end matter.
Evidence: PRE's author guidelines list Author Contribution and Acknowledgment statements and require disclosure in the paper of substantive AI use; the APS ethics policy strongly encourages CRediT contributor roles. The repository history includes merged branches named codex/* (for example PRs #41 and #42).
Fix: add all three statements.

### G. Minor

29. **DeepWalk naming.** main.tex:106–107 equates DeepWalk with node2vec at p=q=1, then notes that the SGNS results assume negative sampling in place of DeepWalk's hierarchical softmax. Kojaku et al. (2024) and residual2vec (Sec. 3.1, where hierarchical softmax corresponds to a uniform-noise baseline) show that the two differ exactly in their treatment of degree. Call the analyzed model SGNS on simple random walks (node2vec with p=q=1).
30. **Source of the retention formula.** Eq. (65) cites Mikolov et al. (2013), whose Eq. (5) gives the discard probability 1−√(t/f); √(t/p)+t/p is the implementation form, so cite the C code or Gensim (Řehůřek & Sojka, LREC 2010 workshop). LGD 2015, Sec. 3.1, already describe how "dirty" subsampling changes context windows (main.tex:929).
31. **Raw norms include a common-mode component.** The KL theory the paper invokes concerns centered, whitened norms (Oyama et al. 2023), and SGNS vectors share a nonzero mean direction (Schakel & Wilson, Sec. 3.1; Mimno & Thompson, EMNLP 2017, pp. 2873–2878). Check whether turnover survives centering. Oyama et al. (Sec. 6) also show that empirical KL is biased upward for low-count items, which matters when strong external subsampling leaves hubs with few retained pairs.
32. **Small items.** The supplement's Kojaku affiliation omits the city (supp.tex:12). Cite the SM with the APS "See Supplemental Material at [URL]" reference. Fig. 6d tick labels overlap. Fig. 1 panel titles ("Mixing removes the falling limb", "Exploratory peak shift") read as lab notes. references.bib carries unused duplicates (schakel2015, murray2023, kojaku2023, oyama2023).

## 6. Novelty table (focus 2)

| Result in manuscript | Closest prior work (checked at source) | What is new here |
|---|---|---|
| Thm. 3: normalized-adjacency rows √(θ_i/s_a) y_a, orthogonal block directions | Qin & Rohe, NeurIPS 26 (2013), arXiv:1309.4111, Lemmas 3.2–3.3, τ=0 stated; Sanna Passino et al., Technometrics 64, 346 (2022), Sec. 2.3 (rays); Lei & Rinaldo, Ann. Stat. 43, 215 (2015) (spherical clustering under the DCSBM) | Notation only |
| Table I raw-adjacency row u_i = θ_i c_a | Jin, Ann. Stat. 43, 57 (2015), Lemma 2.1 (credited; page check from the authors' audit) | None |
| Lemma 5 and Thm. 6: (T^r)_ij = (θ_j/s_b)(W^r)_ab; J = ΘZGZ^TΘ with radius ∝ θ_i | Kojaku et al., NeurIPS 2021, arXiv:2110.07654, Sec. 3.1, Eq. (11) (uncited); Huang et al., KDD 2021, doi:10.1145/3447548.3467300, Obs. 2 (uncited) | Minimal exact factorization rows and exponent 1 for the uncentered pair matrix; small |
| Thm. 7, first part: x* = log(p+/p−) | Levy & Goldberg, NeurIPS 27 (2014) (credited) | None |
| Cor. 8: block-constant score at α=1 | Davison, Morgan & Ward, NeurIPS 37 (2024), Lemma S9, p. 25 (credited); Kojaku et al. 2021, Secs. 2.2–3.1 (uncited); Qiu et al., WSDM 2018 (substitution into NetMF takes two lines; credited) | Finite ordered-window proxy and the loopless correction (Remark 10) |
| Cor. 9: context-side (1−α) log θ_j, block-constant center rows | Levy, Goldberg & Dagan, TACL 3, 211 (2015), Sec. 3.2, Eq. (3) (uncited); Davison et al. 2024, App. F, Eqs. (S209)–(S211) (credited) | Minimal-factor row form for ordered SGNS under the DCSBM; small |
| Sec. VII: gauge, balanced factors, conservation of U^TU−V^TV | LGD 2015, Sec. 3.3, Eq. (5) (uncited); Davison et al. 2024, Prop. S24 (credited); Du, Hu & Lee, NeurIPS 31 (2018) (credited); Kunin et al., ICLR 2021, arXiv:2012.04728, Sec. 6, Eq. (20) (uncited; matches the SM one-step formula) | The 20-run native-factor audit (empirical) |
| Prop. 11: no finite optimum when p+=0; nuclear-norm divergence | Soudry et al., JMLR 19(70) (2018), unattainable logistic targets (uncited) | Matrix/SGNS statement; elementary |
| Prop. 12: four-node construction with distinct RMS-normalized shapes | None found | New; relies on free zero-weight diagonal scores at full rank n=4 |
| Exposure and context interventions (Fig. 1) | Wilson & Schakel, arXiv:1510.02675 (2015), Secs. 4–5, Figs. 4, 7 (cited only for the inverted-U) | Exact per-node exposure on graphs; inverted-U at β=1/3 with increasing exposure |
| Inverted-U and the two-force account | Schakel & Wilson, arXiv:1508.02297 (2015), Sec. 3.3, Fig. 3; Murray et al., PNAS 120, e2305414120 (2023), Fig. 5 and SI (specificity tested, judged not the sole mechanism) | Turnover in Gensim without subsampling |
| Learning-rate-history dependence (Fig. 5) | No direct precedent found; frameworks in Saxe et al., arXiv:1312.6120, and Karkada et al., arXiv:2502.09863v3 | New; the paper's strongest result |
| Degree in radius, community in angle | Krioukov et al., PRE 82, 036106 (2010), Eq. (37); Papadopoulos et al., Nature 489, 537 (2012); Boguñá et al., Nat. Rev. Phys. 3, 114 (2021); Modell & Rubin-Delanchy, arXiv:2105.00987 (2021) | Uncited; the framing PRE readers expect |

## 7. Abstract and conclusion, sentence by sentence (focus 4)

| # | Claim (main.tex line) | Support | Verdict |
|---|---|---|---|
| A1 | Each node has a direction and a norm whose meaning depends on operator, graph, and training (54) | Whole paper | Framing; acceptable |
| A2 | Compares three population geometries in a common notation (55) | Secs. IV–VI, Table I | Supported; known results, credit absent from the abstract |
| A3 | The first two have exact radial-angular structure (56) | Thms. 3, 6 | Supported for the diagonal-inclusive proxy with rank-K B; Thm. 3 is Qin & Rohe's |
| A4 | Unigram negatives cancel degree from the optimal score (57) | Cor. 8 | Supported when every F_ab>0; prior (Davison et al. Lemma S9; residual2vec) |
| A5 | Controlled finite-sample experiments probe exposure and context sampling (58) | Sec. XI A–C | Acceptable |
| A6 | β=0 decreasing, β=1 increasing, β=1/3 inverted-U (59) | Fig. 1c; main.tex:996–1000 | Supported for one trainer (d=24, 4 epochs, n=600); the β=1/3 rise is 3% and the fall is one bin |
| A7 | Mixing removes the decreasing branch (60) | Fig. 1d | The λ=1 endpoint follows from exchangeability and mixing also changes the negatives; "progressively flattens" is accurate |
| A8 | Turnover persists without subsampling under the original schedule and is absent at epoch 20 under a constant rate (61) | Fig. 5 | Supported; "persists" is weak at a 10^−4 floor; "Corrected" is undefined for the reader |
| A9 | Predictions frozen before training on two new synthetic families and NetScience (62) | Sec. XI G | Procedurally supported; pre-correction code undisclosed; BA and Holme–Kim are in-distribution |
| A10 | Median peak error 0 bins, 8/9 within one bin, pooled ρ=0.881 (63) | results/prospective_validation/summary.json | Numbers correct; low evidential value (node-free predictors pass; pooled ρ mixes thresholds) |
| A11 | Degree-and-threshold baseline passes the criteria but ranks nodes worse within curves (64) | Table S1 | Supported; the informative result, and it should replace A10 |
| A12 | Curves consistent with a weakening exposure response relative to specificity; exposure need not decrease (65) | Fig. 1; Eq. (68) | First clause weak (declared model failed; supporting fits post hoc; they misplace the Gensim peaks); second clause supported by β=1/3 |
| A13 | Arbitrarily small loss gaps permit distinct balanced shapes after removing scale (66) | Prop. 12; Fig. 3a | Supported for n=4, full rank, free diagonal; relevance to low-rank fits unshown |
| A14 | Results distinguish three levels and provide "testable controls" (67) | Whole paper | Vague; the controls are not specified |
| C1 | No single norm–degree law (1210) | Table I; Figs. 1, 5 | Supported |
| C2 | The four objects encode degree differently (1211) | Table I; Sec. XI | Supported; the population part is prior work |
| C3 | The exposure and context-sampling interventions produce an inverted-U with increasing exposure (1212) | Fig. 1c | Partly; only the exposure intervention at β=1/3 produced it, and context mixing flattens |
| C4 | Hash-locked predictions recover approximate peak movement across new topologies (1213) | Fig. 6 | Overclaimed: peak movement is criterion 4 (predictor-independent), node-free predictors match the peak accuracy, the synthetic targets are in-distribution, and "its" conflates the controlled trainer with Gensim |
| C5 | Interpretation requires naming operator, contexts, noise, gauge, training, and checking stability (1214) | Whole paper | Reasonable advice; "therefore" overstates what was derived |

## 8. Citation checks (focus 3)

| Location | Cited for | Checked at | Finding |
|---|---|---|---|
| main.tex:80 | Inverted-U in "domain-specific corpora" (Schakel & Wilson) | arXiv:1508.02297, Sec. 3.3, Fig. 3, footnote 8 | Correct for one hep-th corpus, skip-gram, sample=10^−4; "corpora" should be singular |
| main.tex:80, 1184 | Same claim (Wilson & Schakel) | arXiv:1510.02675v2, Secs. 3–5, Figs. 4, 7 | Misattribution: monotone, roughly linear dependences; CBOW; general corpus; no subsampling |
| main.tex:81 | Norms rise to about 10^3 researchers, then fall (Murray et al.) | PMC10756268, text at Fig. 5A | Correct for U.S. organizations; concave curves in most countries shown; SGNS with w=1, d=300, noise exponent 1.0 |
| main.tex:82 | Exposure versus context-diversity explanation (no citation) | Schakel & Wilson, Sec. 3.3; Murray et al. | Attribute to Schakel & Wilson; Murray et al. found specificity insufficient as the sole mechanism |
| main.tex:87, 908 | Centered, transformed norm related to KL (Oyama et al.) | EMNLP 2023, pp. 2108–2130; arXiv:2212.09663v3 | Correct; add their finite-sample KL bias (Sec. 6) |
| main.tex:444, 831–832 | Normalized-eigenspace rays (Qin & Rohe) | arXiv:1309.4111, Lemmas 3.2–3.3 | Correct, and it covers τ=0 completely; the "unregularized structure" sentence overstates novelty |
| main.tex:558 | Lemma S9 (Davison et al. 2024) | NeurIPS PDF, p. 25; SHA-256 matches the authors' audit | Correct |
| main.tex:681 | Prop. S24 | Same PDF | Correct |
| main.tex:684 | Gradient-flow balance (Du et al.) | arXiv:1806.00900 | Correct |
| main.tex:839 | Large-window linearization (Kojaku et al. 2024) | arXiv:2306.13400v2, Sec. 4.1 | Correct; the same paper's Sec. 3 contradicts main.tex:106 on DeepWalk |
| main.tex:862 | Exact recovery under DCSBM variants (Zhang & Tang) | arXiv:2101.07354v2, Remark 7 | Correct; requires max θ / min θ = O(1) |
| main.tex:864 | Symmetrized negatives, App. F (Davison et al. 2024) | Eqs. (S209)–(S211) | Correct |
| main.tex:911 | Retention formula (Mikolov et al.) | NeurIPS 2013 PDF, Sec. 2.3, Eq. (5) | The paper gives discard 1−√(t/f); the square-root-plus-linear form is the implementation's |
| main.tex:1160 | NetScience largest component, 379 nodes and 914 edges (Newman 2006) | Repository GML file | Correct |
| main.tex:833, 836–837 | SCORE Lemma 2.1; Davison & Austern Eqs. (16)–(17), Remark 28 | Not rechecked; analysis/issue31_prior_theory_audit.md records page-level checks | Accepted on the audit |

## 9. The single change that would most raise the paper's chance at PRE (focus 10)

Replace the regression with a derived, testable account of the trained norm–degree curve, and make that account the paper.
The ingredients are already in the manuscript: at window one on sparse graphs the objective has no finite minimizer (Prop. 11, Eq. (63)); norms then depend on exposure-weighted learning-rate integrals, on rank, and on Gensim's |score|<6 update window; and the schedule flips the curve (Fig. 5).
A per-node growth model under annealed SGD, or the closed-form gradient-flow dynamics of Karkada et al., should predict where the peak sits and how it moves with initial rate, decay, epochs, dimension, and β, ideally as a scaling relation for the peak degree with a data collapse, tested prospectively on larger graphs.
With such a result, the population identities shrink to one credited background page and the paper offers a physics contribution.
Without it, the manuscript remains a careful audit of the conditions under which embedding norms resist structural interpretation.

## 10. Jargon for a PRE readership (focus 9)

| Term | First use | Problem | Suggestion |
|---|---|---|---|
| Norm turnover | Title; abstract main.tex:61 | Never defined; idiosyncratic | "nonmonotonic (inverted-U) norm–degree relation", with an operational definition |
| Exposure | Abstract main.tex:58 | Defined only at main.tex:98 | "number of updates in which a node is the center" |
| Context KL | Abstract main.tex:62 | Defined at Eq. (64) | "KL divergence of a node's context distribution from the noise distribution" |
| Balanced norm | Abstract main.tex:66 | Defined at Eq. (51) | Define in the abstract or omit |
| Unigram negative sampling; marginal normalization | Abstract main.tex:57 | α=1 defined only at Eq. (33) | "negatives drawn in proportion to stationary frequency" |
| Corrected word2vec controls; original schedule; constant-rate alternative | Abstract main.tex:61 | Refers to internal code history | Name the schedules (linear decay from 0.025 to 10^−4; constant 0.0025) |
| Graph-family–threshold curves; degree bins | Abstract main.tex:63 | "Threshold" undefined | "subsampling threshold t" |
| Gauge | Sec. VII | Physicists expect a local symmetry; here it is a global GL(d) reparametrization | One explanatory sentence |
| Hash-locked | Conclusion main.tex:1213 | Procedural detail | "frozen before training" |
| Two-force; falling limb; held-out retention law | Fig. 6 caption; Fig. 1 | Lab shorthand | Plain descriptions |

## 11. Questions for the authors

1. What single result should a PRE reader take away, and which section establishes it?
2. Does any experiment test Thms. 3 and 6 or Cors. 8–9 on sampled DCSBM graphs with K>1? If none does, why do they occupy the main text?
3. How do the prospective metrics compare with node-free predictors (threshold rank, constant middle bin) and with an in-sample degree oracle, and why were the declared criteria set at levels these predictors pass?
4. Does the peak location move with embedding dimension at a fixed schedule, with the initial rate at a fixed decay, and with Gensim's score clip?
5. Under the annealed schedule the expected loss changes by 0.3–0.4% between epochs 5 and 20. What happens if the floor rate after annealing is raised to 0.0025?
6. Which negative-sampling exponent did each experiment use, and does the curve change at the common default of 0.75?
7. Does turnover survive centering the embeddings, as assumed by the KL theory you cite?
8. Why does Fig. 4 show the uncorrected sweep when results/subsampling_origin/ contains a corrected one?
9. What does your account predict for the location of the peak near 10^3 researchers in Murray et al., and can you test it on trajectory-like data?

## 12. Recommendation and confidence

Recommendation: reconsider after major revision.
Confidence: 4 of 5. I checked the novelty and citation claims at the primary sources listed and recomputed the prospective-criteria nulls, support ranges, and top-bin structure from saved outputs; I did not re-derive every proof or rerun training.

Justification: the work is careful, reproducible, and candid, and the learning-rate-history result is new and useful.
As submitted, the population theory is largely prior work with key precedents uncited or misattributed, the title phenomenon has no mechanism by the authors' own account, the prospective evidence is weaker than the abstract states, and the scope is narrow.
PRE asks for a clear advance of interest to its readership.
If the revision supplies neither a derived account of the trained norms (Sec. 9) nor a narrowed paper built around the schedule result with correct credit, I would recommend rejection.

## 13. Action items

| ID | Severity | Location | Problem | Proposed fix | Type | Effort |
|---|---|---|---|---|---|---|
| R1-01 | blocker | Title; abstract; Secs. X–XII (main.tex:65, 1182, 1189) | No central result; turnover mechanism unexplained; DCSBM theory predicts nothing for the K=1 experiments | Choose one claim: derive and test a dynamical account of trained norms (Sec. 9), or retitle and narrow around the schedule result with population material as credited background | ISSUE | >1day |
| R1-02 | major | Thm. 3 (main.tex:248–279); main.tex:831–832; Table I | Thm. 3 is Qin & Rohe's Lemma 3.3 at τ=0 | Attribute at the theorem; reword "gives the unregularized structure" | PR | ≤1h |
| R1-03 | major | Lemma 5, Thm. 6, Cor. 8; intro main.tex:85–95 | residual2vec (Eq. 11; Secs. 2.2–3.1) uncited | Cite and state what is added | PR | ≤1h |
| R1-04 | major | Cor. 9 (main.tex:573–611); Table I | Context-side term is LGD 2015's PMI_α | Cite; present as an instantiation | PR | ≤1h |
| R1-05 | major | Sec. VII; SM Sec. S3 | Balanced split and finite-step imbalance formula are prior (LGD 2015 Eq. 5; Kunin et al. 2021 Eq. 20) | Cite; trim Sec. VII | PR | ≤1h |
| R1-06 | major | Table I; Sec. V | Huang et al. 2021 raw-pair versus PMI norm result uncited | Cite | PR | ≤1h |
| R1-07 | major | Intro main.tex:74–95; Table I | No network-geometry framing; random-walk Laplacian row missing | Add a framing paragraph (Krioukov 2010; Papadopoulos 2012; Boguñá 2021; Sanna Passino 2022; Modell & Rubin-Delanchy 2021) and a random-walk Laplacian row | PR | ≤1day |
| R1-08 | major | main.tex:80, 82, 1184 | Inverted-U misattributed to Wilson & Schakel; "corpora"; explanation unattributed; Murray et al.'s negative test omitted | Correct citations and wording | PR | ≤1h |
| R1-09 | major | Sec. XI A–C; Fig. 1 | Interventions repeat Wilson & Schakel's design without credit | Credit; state what is new | PR | ≤1h |
| R1-10 | major | Fig. 1f versus Fig. 4d; Sec. XI E; intro main.tex:102 | Pipelines disagree on where turnover appears; both called validation | Rename section and figure; state the disagreement | PR | ≤1h |
| R1-11 | major | Abstract main.tex:63; main.tex:1165–1166; Conclusion main.tex:1213; Table S1 | Peak criteria pass for node-free predictors; pooled ρ mixes thresholds | Drop from abstract; lead with within-curve ranking; add null predictors and degree oracle to Table S1 | ISSUE | ≤1day |
| R1-12 | major | Sec. XI G; Fig. 6 caption | Pre-correction prospective run not disclosed in main text | Disclose; report historical and corrected values side by side | PR | ≤1h |
| R1-13 | major | Intro main.tex:104; main.tex:1160; Conclusion main.tex:1213 | BA and Holme–Kim targets lie inside calibration support; failures concentrate on the only out-of-distribution target | Reword as in-distribution transfer; run a new prospective round on different graphs and empirical networks | ISSUE | >1day |
| R1-14 | major | Abstract main.tex:65; main.tex:1024–1028 | Declared model failed; supporting fits post hoc | Rewrite or drop the abstract sentence | PR | ≤1h |
| R1-15 | major | Figs. 1, 4–6; Sec. XI | Inverted-U resolved by 4–6 bins; falling branch is one top bin; no uncertainty; pooling hides per-graph shapes | Operational definition; node-level plots with bootstrap bands; unimodality test; graphs spanning ≥3 decades | ISSUE | >1day |
| R1-16 | major | Abstract main.tex:61; main.tex:1117 | "Persists" at a 10^−4 floor where loss barely moves | Report loss changes; add initial-rate-only and floor-only schedule variants | ISSUE | ≤1day |
| R1-17 | major | Prop. 11; Sec. XI F | No link drawn between the missing finite optimum, Gensim's update clip, and schedule dependence | Make it the organizing idea; sweep dimension, clip, initial rate, floor | ISSUE | >1day |
| R1-18 | major | Abstract main.tex:61; Secs. XI E–G; Conclusion | Audit-trail language | One final pipeline in Methods; history, hashes, freeze mechanics to SM | PR | ≤1day |
| R1-19 | major | Figs. 1, 4 | Superseded and label-patched figures; node-based standard errors | Regenerate from corrected outputs; drop or rerun Fig. 4c | ISSUE | ≤1day |
| R1-20 | major | Sec. XI A, E, G | α, optimizer, schedule, batch, dimension, and initialization missing from Methods | State them for every pipeline | PR | ≤1h |
| R1-21 | major | Abstract; Sec. XI | Scope limited to α=1 (word2vec standard is 0.75), p=q=1, window 1, n≤600, d=24/64 | Add α=0.75, windows 5–10, dimension sweep, n≥10^4, one text or trajectory corpus; or narrow the abstract | ISSUE | >1day |
| R1-22 | major | Data Availability (main.tex:1216–1220) | Placeholder without URL or DOI | Archive with DOI; APS template; cite NetScience source | PR | ≤1day |
| R1-23 | major | End matter | Acknowledgments and funding, author contributions, AI-use disclosure missing | Add all three (CRediT roles encouraged) | PR | ≤1h |
| R1-24 | minor | Eq. (68); main.tex:1186; Fig. 6 caption | "Two-force" model is the chain rule | Present as a descriptive decomposition; drop the label | PR | ≤1h |
| R1-25 | minor | Abstract main.tex:60; Conclusion main.tex:1212 | Context mixing flattens and does not produce the inverted-U; "removes" | Correct the wording | PR | ≤1h |
| R1-26 | minor | Secs. III, IV, VIII, IX; Eqs. (18)–(20), (47)–(48); Remark 10; Figs. 1e–f, 3c | Material to compress or move | Move to SM | PR | ≤1day |
| R1-27 | minor | main.tex:1062; Fig. 3 caption | Bound and √2 cap derived only in norm_geometry_supplement.tex | Fold into supp.tex or drop | PR | ≤1h |
| R1-28 | minor | main.tex:106–107 | DeepWalk naming conflicts with Kojaku et al. 2024 and residual2vec | Call the model SGNS on simple random walks (node2vec, p=q=1) | PR | ≤1h |
| R1-29 | minor | Eq. (65); main.tex:911–929 | Retention formula cites Mikolov et al., whose Eq. (5) differs; the cited form is the implementation's; dirty subsampling uncredited | Cite the implementation and LGD 2015 Sec. 3.1 | PR | ≤1h |
| R1-30 | minor | Sec. XI; link to Oyama et al. | Raw norms include a common-mode component; empirical KL bias unaddressed | Report centered norms; bias-correct KL | ISSUE | ≤1day |
| R1-31 | minor | Title; abstract; intro | Undefined jargon | Define or replace per Sec. 10 | PR | ≤1h |
| R1-32 | minor | Prop. 12 | Construction uses free diagonal scores at full rank n=4 | Add an off-diagonal, low-rank version or move to SM | ISSUE | ≤1day |
| R1-33 | minor | main.tex:862; references.bib | Missing dynamics references (Saxe; Karkada; Soudry), Arora 2016, Barot 2021; Zhang–Tang homogeneity caveat | Add citations and caveat | PR | ≤1h |
| R1-34 | minor | supp.tex:12; Fig. 6d; Fig. 1 panel titles; references.bib; SM citation | Small presentation items | Fix | PR | ≤1h |
