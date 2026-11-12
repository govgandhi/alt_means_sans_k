# Physical Review E: Report of Referee 4 (presentation, figures, notation, internal consistency)

Manuscript: "Degree Information and Norm Turnover in Random-Walk Network Embeddings" (Gandhi, Kojaku, Ahn).
Reviewed version: commit `c958c528`, `paper/current/main.pdf` (15 pp.) and `supp.pdf` (3 pp.), with figure data and plotting scripts under `results/` and `scripts/`.
Every page and every figure was rendered and inspected at print size (150 dpi page renders; figure PDFs rendered at their printed 510 pt width at 300 dpi).
Measurements below come from small scratch scripts listed at the end; nothing in the worktree was modified.

## Summary

The manuscript compares the degree dependence of three population objects under the DCSBM (normalized-adjacency eigenspace, stationary random-walk pair matrix, SGNS optimal score), proves exact radial-angular factorizations for the first two, and shows that unigram negative sampling removes node degree from the optimal SGNS score.
It then argues that the empirical inverted-U relation between degree and embedding norm arises in finite training: in a controlled SGNS trainer, the exponent of center exposure sets whether norms fall, peak, or rise with degree, and mixing contexts toward their marginal flattens the curve.
Gensim word2vec controls show that the unfiltered turnover depends on the learning-rate schedule, and a frozen ExtraTrees predictor based on realized pair counts and context KL is evaluated on two new synthetic graph families and NetScience (median peak error 0 bins, 8/9 peaks within one bin, pooled Spearman 0.881).
Two propositions establish that near-optimal loss does not pin down balanced norms, even after removing overall scale.

From the presentation side, the numbers are unusually well traced: of roughly 35 reported quantities I checked against the committed summaries, all match except two rounding slips.
The problems are in how the evidence is shown.
All six figures have text below 7 pt at print size (Fig. 1 legends are 3.4 pt), two captions misstate what the error bars are, the colors in Figs. 1 and 4 collapse under color-vision deficiency, the main Gensim figure (Fig. 4) comes from a runtime the authors themselves classify as defective, and the readouts the abstract relies on (degree bin, peak, turnover, the four criteria) are never defined in the main text.
The notation is heavily overloaded, and the prose narrates the project's history ("original", "historical", "corrected") in section titles and captions.
These are all fixable, and most of the fixes can use data already in the repository.

## Strengths

1. Traceability. Nearly every number in the text can be found in a committed JSON summary, and the prospective freeze is backed by a hash manifest (`results/prospective_validation/freeze_manifest.json`) with `target_outcomes_present_at_freeze = false`.
2. Exploratory and confirmatory analyses are labeled, and failures are reported (global surface failure at main.tex:1026; peak misses at main.tex:1167; baseline that also passes, main.tex:1171).
3. Table I is a compact summary of the population results, and Figs. 2, 3, and 5 already use graph-level standard errors with faint per-graph curves; Figs. 3 and 6 already use the Okabe–Ito palette.
4. The SM baseline tables (Tables S1–S2) report pooled and within-curve correlations with ranges and say explicitly that the ranges are not confidence intervals (supp.tex:30).

## Weaknesses, ranked

1. Figures fail print legibility and accessibility: text 2.4–6.8 pt in every figure; color-only encodings that collapse under deuteranopia/protanopia in Figs. 1 and 4; inconsistent panel labeling (comments 3–6).
2. Uncertainty is misdescribed: the Fig. 1 and Fig. 6 captions say "standard errors across five graph seeds", while the code computes standard errors over pooled nodes; three figures use node-pooled SE and three use graph-level SE (comment 1).
3. Superseded runs carry main-text evidence: Fig. 4 comes from the pre-correction runtime and its in-figure title claims a "validation" that the text refutes; the Fig. 6 outcomes are also pre-correction and the text does not say so; the Introduction claims a corrected-Gensim result that the Results section never reports (comments 2, 8, 9).
4. The readouts are undefined: degree bins (8 or 6 requested quantile bins, 4–6 occupied depending on family), peak, turnover, inverted-U, and the four declared criteria appear only in code or in an uncited SM section (comments 10, 11).
5. Notation and terminology are overloaded: q, c/C, D, e, ρ, r, h, s, α, δ, Z, G, L, d each carry two to four meanings; "exposure" and "context KL" each have three or four operational definitions (comments 12, 13).
6. Structure and prose: a 301-word, 14-sentence abstract whose first three sentences do not state the main result; about 6.5 pages of theory before any data; section titles and captions that narrate the project's history; long hedging chains (comments 14–18).
7. Figure reproducibility: Fig. 1 is a byte-patched PDF containing a non-embedded Helvetica font, and the node-level data behind Figs. 1 and 4 are not committed, which contradicts the Data Availability statement (comment 6).

## Detailed comments

### 1. Error bars in Figs. 1 and 6 are not what the captions say

- Location: Fig. 1 caption (main.tex:1012; p. 9); Fig. 6 caption (main.tex:1146; p. 13).
- Problem: Fig. 1 says "Points and error bars show degree-bin means and standard errors across five graph seeds." Fig. 6 says "standard errors over five graph seeds for the two synthetic families and five walk and optimizer seeds for NetScience."
- Evidence: Fig. 1 is drawn by `plot_binned` → `binned_curve` (scripts/simulate_exact_center_exposure.py:285–328). Its inputs come from `selected(rows, name, field)` with `seed=None` (scripts/simulate_turnover_surface.py:275–288), which pools every node from all five graphs, and the bar is `std(y)/sqrt(n_nodes)` over that pool. Panels 1(e) and 1(f) are drawn with plain `plot` calls (simulate_turnover_surface.py:1017–1056) and have no error bars at all. Fig. 6 uses its own `binned_curve` (scripts/prospective_norm_turnover.py:433–455) with the same node-pooled formula. Recomputing the Fig. 6 bins from `results/prospective_validation/outcomes.jsonl`, the replicate-level SE divided by the plotted SE ranges from 0.13 to 1.90 across the 45 family–threshold–bin cells (median 0.62). For NetScience the plotted bars are about 2–8 times wider than the replicate SE (five walk seeds on one graph); for the clustered family they can be about half the replicate SE (t = 3×10⁻³, degree-4 bin: plotted 0.019, replicate 0.034).
- Fix: now, correct both captions to describe node-pooled standard errors and say that Fig. 1(e,f) has none. Then replot Figs. 1 and 6 with graph- or replicate-level SE, as Figs. 2, 3, and 5 already do, so that all six figures use one definition.

### 2. Fig. 4 shows the pre-correction Gensim runtime, and its in-figure title contradicts the text

- Location: Sec. XI.E (main.tex:1076–1104); Fig. 4 (main.tex:1086–1098; p. 11); title strings in scripts/simulate_gensim_threshold_sweep.py:221, 255, 261.
- Problem: The caption calls the figure "Historical Gensim word2vec sweep, before the implementation corrections described below", yet the figure itself is titled "Literal Word2Vec validation of the turnover prediction" and panel (d) "External peak check". The text says the exploratory controlled fit misplaces the peak at all three thresholds (main.tex:1104), so the figure title asserts a validation that the paper refutes. The runtime used here assigned negative-table weights to the wrong vocabulary slots when node names are integers (analysis/negative_sampling_control_protocol.md:44–52), so the noise distribution in this sweep differs from the unigram sampler assumed by Corollary 8. Error bars are node-based (the caption says so).
- Evidence: corrected epoch-5 runs exist for the same graphs, corpora, and initializations. `results/training_trajectory/README.md` (baseline schedule, epoch 5) gives interior maxima 0/5, 5/5, 5/5 and pooled peak degrees 2.98, 6.00, 7.73 at t = 3×10⁻⁴, 10⁻³, 3×10⁻³, plus 5/5 and 7.73 with filtering off; per-node norms are in `results/training_trajectory/runs/` and graph-level SE in `summary.json`.
- Fix: replace Fig. 4(b,d) with the corrected epoch-5 sweep at all four filtering settings (graph-level SE, interior-maximum counts in the legend or caption); move the historical sweep and the unreplicated budget-matched panel (c) to the SM; delete the in-figure title. This also supplies the missing Results support for main.tex:102 (comment 8).

### 3. Text in every figure is below 7 pt at print size

- Location: all six figures; worst in Fig. 1 (p. 9).
- Evidence: each figure is included at `width=\textwidth` (510 pt). Scaling the native PDF font sizes by 510/native width gives:

| Figure | Native width (pt) | Scale | Tick and axis labels | Legends | Smallest text |
|---|---|---|---|---|---|
| 1 | 1044 | 0.489 | 4.9 pt | 3.4 pt | 2.4 pt (exponent in "3×10⁻⁴") |
| 2 | 720 | 0.708 | 7.1 pt | 5.7 pt | 5.7 pt |
| 3 | 864 | 0.590 | 7.1–7.7 pt | 6.5–6.8 pt | 5.0 pt (tick exponents) |
| 4 | 828 | 0.616 | 6.2 pt | 4.9 pt | 4.3 pt |
| 5 | 576 | 0.885 | 8.9 pt | 6.2 pt | 6.2 pt |
| 6 | 756 | 0.675 | 6.8 pt | 5.4 pt | 4.7 pt |

  Tick labels collide in Fig. 1(f) and Fig. 4(d) ("3×10⁻⁴" overlaps "4×10⁻⁴") and in Fig. 6(d) ("3×10⁰4×10⁰"). Log-degree axes in Figs. 1, 4, and 6 are labeled "3×10⁰, 4×10⁰, 6×10⁰, 10¹", while Figs. 2, 3, and 5 use plain integers.
- Fix: set `figsize` to the printed size (about 7.0 in for `figure*`, 3.4 in for one column), use at least 7 pt (8 pt preferred) for all text including legends and exponents, and use integer ticks on degree axes (3, 4, 6, 8, 12, 16) as in Fig. 5.

### 4. Figs. 1 and 4 are not colorblind-safe, and color is the only cue

- Location: Fig. 1 (all panels), Fig. 4 (all panels).
- Evidence: CIE76 ΔE between line colors, before and after the Machado et al. (2009) simulation: matplotlib tab10 green vs red 119.8 → 7.3 under deuteranopia (Fig. 1: β = 2/3 vs β = 1, λ = 2/3 vs λ = 1); blue vs purple 38.2 → 8.0 (deuteranopia) and 5.7 (protanopia) (Fig. 1(a,c): β = 0 vs the held-out curve); orange vs green 100.6 → 4.6 under protanopia (Fig. 4: t = 10⁻³ vs 3×10⁻³). Every line uses the same circle marker. The PRE author page lists "accessible color palettes" and "Use of words or symbols (instead of color) to provide meaning" among its accessibility requirements. The Okabe–Ito colors in Fig. 6 stay at ΔE ≥ 51 under both simulations. In Fig. 3(b,c) the window-5/window-10 pair (green #009E73 vs pink #CC79A7) drops from 87.5 to 17.9 under deuteranopia, which is weak for overlapping curves drawn with the same marker.
- Fix: β, λ, and t are ordered parameters, so use one sequential ramp per parameter plus distinct markers, or direct labels at line ends; reuse the same color for the same t in Figs. 1, 4, and 6.

### 5. Panel labels, in-figure titles, and encodings are inconsistent

- Location: Figs. 1–6.
- Evidence: Figs. 1, 3, and 4 print "A"–"F" in the figure while captions use "(a)"–"(f)". Figs. 2 and 5 have no panel labels (Fig. 5 caption uses "Left/Right"; Fig. 2 caption names no panels). Fig. 6 has no labels in the figure, but its caption uses "(a–c)" and "(d)". Figs. 1, 2, 4, and 5 carry suptitles that repeat or contradict the caption (Fig. 4; comment 2). The degree axis is "degree" (Figs. 1, 4, 6), "Degree (geometric bin mean)" (Figs. 2, 5), and "Degree-bin center" (Fig. 3b). Windows 5 and 10 are dark green and purple in Fig. 2 but teal and pink in Fig. 3(b). Fig. 3 uses blue/orange/green for profile type in (a) and for window in (c). Thresholds appear as "3×10⁻⁴" (text, Fig. 1), "3e−04" (Fig. 4), "0.0003" (Fig. 6, Table S3), and "t = 0.003" (main.tex:1119). Fig. 1 legends round β and λ to 0.33 and 0.67 while the text uses 1/3 and 2/3. Fig. 5 calls center vectors "raw input norms" (main.tex:1126), and its y-axis labels are field names ("raw norm", "balanced norm").
- Fix: one shared style module for all plotting scripts: (a), (b), … labels matching the captions; no suptitles; fixed color and marker per condition across figures; one degree-axis label; one threshold notation (10⁻³ style); "center-vector norm" everywhere.

### 6. Fig. 1 is a byte-patched PDF with a non-embedded font, and Figs. 1 and 4 cannot be regenerated from the repository

- Location: Fig. 1 (p. 9); scripts/audit_realized_graph.py:590–660; scripts/simulate_turnover_surface.py:979, 997, 1068; .gitignore:7; Data Availability (main.tex:1218).
- Evidence: `qualify_historical_figure` redacts three labels in `results/turnover_surface/turnover_surface.pdf` and writes replacements in Helvetica with PyMuPDF (audit_realized_graph.py:634–640). `pdffonts main.pdf` lists Helvetica (Type 1, WinAnsi) with the embedded flag set to "no"; the remaining figure text is DejaVu Sans, so Fig. 1 mixes two typefaces. The plotting script still writes the old labels ("Causal two-force test for the norm-frequency turnover", "C  Exposure creates the rising limb", "B  Specificity interventions"), so rerunning it reverts the edits. The node-level inputs (`node_results.csv`) are excluded by `*.csv` in .gitignore, and `results/turnover_surface/` and `results/gensim_threshold_sweep/` contain only summaries and images. The Data Availability statement says that "generated data needed to reproduce the figures accompany this manuscript".
- Fix: change the labels in the plotting script, commit the node-level CSVs (or regenerate them), rebuild Fig. 1 from the script, and check that `pdffonts` reports every font embedded. Comments 1, 3, 4, and 26 all depend on this.

### 7. Fig. 6(d) hides one of nine points and cannot show the metric it reports

- Location: Fig. 6(d) (p. 13); scripts/prospective_norm_turnover.py:838–870.
- Evidence: from `results/prospective_validation/summary.json`, Direct BA at t = 10⁻³ and at t = 3×10⁻³ both sit at (6.77, 6.77), so the orange circle is completely covered by the green one; the clustered t = 3×10⁻⁴ square and the NetScience t = 10⁻³ triangle both sit at (4.00, 4.00). Eight markers are visible for nine curves. The criterion is in bins, but the axes are log degree. The clustered t = 3×10⁻³ point (predicted 6.34, observed 13.41) is visually the largest miss, yet it counts as "within one bin" because 13.41 is the adjacent, last bin; the reported two-bin miss (NetScience t = 3×10⁻⁴, 1.66 vs 4.00) looks smaller. With six requested quantile bins, Direct BA has 4 occupied bins (59% of its nodes in the lowest), the clustered family 5, and NetScience 6, so "within one bin" covers different fractions of the degree range per family. None of this is stated.
- Fix: plot predicted against observed peak-bin index on an integer grid with small offsets per threshold (or give a 3×3 table of peak bins); state the number of occupied bins per family in the caption; consider reporting the error as a fraction of each family's bin count.

### 8. The Introduction claims a corrected-Gensim result that the Results section never reports

- Location: main.tex:102 ("Corrected Gensim controls reproduce the peak movement with weaker subsampling under the original learning-rate schedule") vs Sec. XI.F (main.tex:1106–1120).
- Evidence: Sec. XI.F reports only the filtering-off condition at epoch 5 and the trajectory contrasts. The corrected epoch-5 peak locations (2.98, 6.00, 7.73) appear only in `results/training_trajectory/README.md`.
- Fix: report the corrected epoch-5 peak locations and interior-maximum counts in Sec. XI.F and show them in the replacement Fig. 4 (comment 2).

### 9. The prospective outcomes come from the pre-correction runtime, and the main text does not say so

- Location: Sec. XI.G (main.tex:1153–1173), Fig. 6 caption, abstract (main.tex:62–64).
- Evidence: `results/prospective_validation/README.md` states that these files "predate the repeated-visit counter, integer-token negative-table and BLAS corrections" and that "Their original numbers must not be described as results of the corrected code." The abstract and Fig. 6 report these numbers (0.881, 8/9). The corrected arms (pooled Spearman 0.875 and 0.873, 8/9 each; Table S1) are mentioned only as "corrected sampling comparisons" (main.tex:1172), and the arms "retained noise" and "raw noise" are not defined in the table captions (related to #17). Fig. 4 flags its historical status; Fig. 6 does not.
- Fix: one sentence in Sec. XI.G and in the Fig. 6 caption naming the runtime that produced the frozen outcomes, followed by the corrected values from Table S1; define both corrected arms in the Table S1 and S2 captions.

### 10. Degree bins, peak, turnover, and inverted-U are never defined in the main text

- Location: first uses at main.tex:61 ("turnover"), 63 ("degree bins"), 1110 and 1117 ("interior maxima"), 1167 ("weakly unimodal", "boundary classifications").
- Evidence: bins are quantiles of log degree. The controlled and trajectory pipelines request 8 bins (6 occupied; scripts/simulate_exact_center_exposure.py:285–309; Fig. 5 caption), Gensim uses 6 occupied bins, and the prospective test requests 6 (scripts/prospective_norm_turnover.py:433–455; 4–6 occupied). The bin x-coordinate is the geometric-mean degree in Figs. 1(a–d), 2, 4, 5, and 6, but the arithmetic mean in Figs. 1(e,f) (16.08 vs 16.04 for the same top bin; scripts/simulate_turnover_surface.py:912–934). The code defines an inverted-U as an interior maximum that exceeds both end bins (simulate_turnover_surface.py:577–598); "turnover", "inverted-U", "interior maximum", "peak", "nonmonotonic", "falling limb", "decreasing branch", and "weakly unimodal" are used interchangeably in the text.
- Fix: a short "Readouts" paragraph in Sec. XI.A defining the bins for each pipeline, the bin coordinate, "peak bin", and "inverted-U" (with one name for it), and stating that peak errors are measured in occupied bins.

### 11. The four declared criteria are not stated in the main text

- Location: main.tex:1165–1166 ("The four declared criteria all passed"); abstract main.tex:64.
- Evidence: thresholds appear only in SM Sec. S2 (supp.tex:52–53), which the main text never cites, and in `results/prospective_validation/protocol.json`.
- Fix: list them where they are used (median absolute peak-bin error ≤ 1; at least 2/3 of peaks within one bin; pooled Spearman ≥ 0.5; observed peak nondecreasing with weaker subsampling in at least two of three families) and cite SM Sec. S2.

### 12. Notation is overloaded

- Location: throughout; see the notation table below.
- Evidence: q denotes the negative distribution (main.tex:469), the negative proposal (882), and the realized context marginal (980; the Fig. 1(b) axis). c and C denote community vectors and their matrix (176–180), a block constant C_ab (627), and the center share c_i (882, 983). D denotes the degree matrix (844, 858), a scalar in the proof of Proposition 12 (738–740), and the Gram imbalance (factor_balance.tex:11; Table S3 header). e_i is a standard basis vector in Eq. (18) (main.tex:296) and e_a a community context vector in Eq. (7) and Table I. ρ is an RMS radius (727), a row norm (766–822), and the Spearman coefficient (Tables S1–S2). r_i is the balanced radius in the main text (727) and the raw norm in the SM (factor_balance.tex:35). The population stationary distribution ϖ and the window weights ω appear in the same product in Eq. (27), and the two glyphs are hard to tell apart in print. σ is never defined (main.tex:488). d_i, G, vol(G), and γ are first used without definition (main.tex:651, 883). d is both degree and the number of retained spectral components (main.tex:860).
- Fix: add a notation table (in the main text or an appendix) and make the renames listed below.

### 13. Key terms are undefined or carry several meanings

- Location: abstract and Secs. X–XI; SM tables.
- Evidence:
  - "Exposure" means positive-center updates (main.tex:98), expected retained tokens m_i ∝ p_i keep_t(p_i) (Eq. 66, main.tex:926), exact center counts m_i(β) ∝ π_i^β (Eq. 69, main.tex:967), and the realized positive-pair count (main.tex:1156); the same symbol m_i is used for the second and third.
  - "Context KL" is measured against π in Eq. (64), against the realized center-weighted marginal q in Fig. 1(b), and against the retained-pair context marginal in the prospective test (scripts/prospective_norm_turnover.py:252–265). Units (nats) are stated only in Fig. 2.
  - The λ intervention is called the "context intervention" (main.tex:99), "specificity intervention" (974), and "context mixing" (1009); the Fig. 2 caption then uses "Context mixing" for window widening (main.tex:1048).
  - "Population operator" (54, 1214), "realized-graph targets" (67), "two-force norm predictor" (1144), "Literal word2vec" (1076), "full support" (1052), "score-preserving balanced restarts" (1034), "collision-allowing negatives" (730), and "Hash-locked" (1213) are not defined where used.
  - The predictor is "exposure–KL" in the main text and "Pairs + KL" in Tables S1–S2.
  - "Balanced norm", "balanced radius", and "balanced center norm" are used interchangeably across Figs. 2, 3, and 5.
- Fix: define each term once (Sec. XI.A or a short glossary), use one name per object, and state the reference distribution every time a KL is plotted.

### 14. The abstract is long, front-loads setup, and uses undefined terms

- Location: main.tex:53–68.
- Evidence: 301 words in 14 sentences. Sentences 1–3 introduce the setting and notation; the finite-training result begins at sentence 6, and the prospective test occupies sentences 9–11. The abstract uses "population operator", "context KL", "realized-graph targets", and "Corrected word2vec controls" without saying what was corrected. The exact construction (sentence 13) comes after the prospective test, out of the paper's order. Sentence 4's cancellation result is attributed in the text to prior work (main.tex:558), so it is unclear which population results the abstract claims as new.
- Fix: aim for about 200 words in this order: (1) the question and the empirical inverted-U; (2) the main finding about finite training (exposure exponent sets the shape; full context mixing flattens it; the learning-rate schedule matters in Gensim); (3) the population contrast in one sentence (radius ∝ δ^{1/2} and ∝ δ; degree cancels from the unigram SGNS score); (4) the prospective test with its numbers and the baseline; (5) what readers should do when interpreting norms. Drop "Corrected", or say once what was corrected.

### 15. Section titles and captions narrate the project's history

- Location: Sec. XI.E title "Literal word2vec validation" (main.tex:1076); Sec. XI.F title "Corrected no-subsampling and training-trajectory controls" (1106); "The original, uncorrected Gensim sweep" (1078); "Historical Gensim word2vec sweep" (1089); "the historical plotting routine" (1095); "have not yet been repeated with the corrected implementation" (1103); "added after inspecting this failure" (1027); "The initial 500-iteration endpoints suggest …" (1041); "We chose a more flexible prospective predictor after the exploratory analyses above" (1155).
- Evidence: "corrected" occurs 13 times in main.tex, and the corrections themselves are listed in a single clause (main.tex:1108: "corrected repeated-visit counts, integer-token negative-table indexing, and compiled inner-product arithmetic"). A reader cannot tell from the paper whether these are Gensim defects or defects in the authors' wrapper, or which figures they affect. The analysis notes explain both (analysis/negative_sampling_control_protocol.md:44–52 and 70).
- Fix: present the corrected analyses as the results and title sections by content (e.g., "Gensim word2vec: subsampling threshold" and "Gensim word2vec: learning-rate schedule"); describe each correction once in an SM paragraph (what was wrong, how it was fixed, which figures and tables changed); give the exploratory or confirmatory status in one sentence per analysis.

### 16. Hedging chains bury the findings

- Location: Sec. XI and the Discussion, e.g., main.tex:1043, 1060–1062, 1138, 1173, 1181–1182, 1189, 1201.
- Evidence: Sec. XI contains at least 15 disclaimers of the form "does not", "do not", "need not", "rather than", "not establish", or "without proving" in about 120 sentences, and several paragraphs end with two or more qualifications in a row (main.tex:1060–1062, 1138–1139, 1181–1182).
- Fix: state each result first, keep one qualifier per claim where it matters, and move the rest into the existing Limitations paragraph (main.tex:1192–1201).

### 17. The theory sections are long, and their link to the experiments is thin

- Location: Secs. II–X (pp. 2–8).
- Evidence: Secs. II–X take about 6.5 pages (pp. 2–8) and contain 61 labeled equations, 36 of which are never referenced; the first figure appears on p. 9. Several results are not used by the experiments: Remarks 1 and 10 (diagonal and zero-count conventions), the symmetric score (Eqs. 47–48), the NetMF form (Eq. 43), Lemma 13 with Eq. (59), and the Kojaku congruence (Eqs. 60–61). Sec. IX (prior theory) sits between the theory and the experiments, and the Discussion calls these "established population results" (main.tex:1177), but the paper never says which statements are new. The Introduction has no roadmap paragraph. Eq. (68), the response model behind both predictors, is never cited in Sec. XI. Proposition 12's objective (p⁺ = A_ij/(4d_i), p⁻ = 5d_j/32) appears only inside the proof (main.tex:736). Remarks are set in italics because they use the plain theorem style (main.tex:39). Statements do precede proofs throughout, and assumptions (rank K, F_ab > 0) are visible.
- Fix: a 1–1.5-page "Population geometry" section built around Table I, with proofs in appendices; mark new results explicitly; add a roadmap at the end of Sec. I; cite Eq. (68) in Secs. XI.C and XI.G; move the objective into the statement of Proposition 12; use `\theoremstyle{remark}`.

### 18. A schematic and a pipeline table are missing

- Location: new Fig. 0 and Table II.
- Evidence: the paper uses five pipelines that differ in trainer, dimension (24 or 64), epochs (4, 5, or 1–20), window (1, 2, 5, 10), noise distribution (realized context marginal, π, raw-token unigram, legacy table), binning, and replicate structure, described in scattered sentences (main.tex:960–991, 1033–1034, 1078–1080, 1108–1115, 1160–1162). The reader cannot see which experiment varies what, and main.tex:1139 has to warn that dimension and pair budgets are unmatched.
- Fix: a schematic running population operator → realized-graph or corpus target (Eqs. 63–64) → finite SGNS training → raw or balanced readout, with arrows marking where β, λ, L, t, and the schedule act; plus a table listing each pipeline's settings and the figures it feeds.

### 19. Several panels duplicate data or carry a second message

- Location: Figs. 1(e,f), 2(b), 3, 4(d).
- Evidence: Fig. 1 carries two messages (interventions in a–d; exploratory surface fit in e–f). Fig. 1(f) and Fig. 4(d) plot the same exploratory-fit series (7.73, 16.08, 16.08; `controlled_predicted_peak_degrees` in results/gensim_threshold_sweep/summary.json) under two names ("exploratory fit", "exploratory controlled fit"), and the "actual" series in 1(f) is identical to the fit, so it is hidden. The fit is also called "exploratory exposure-matched fit" (legend), "degree-bin interpolation" (caption), and "degree-bin-specific interpolation across β" (text). Fig. 2(b) and Fig. 3(b) show the same window-5/10 balanced norms before and after RMS normalization. Fig. 3(a) illustrates Proposition 12 (p. 6) but is first cited on p. 9. Read together, Figs. 1(f) and 4(d) obscure a contrast that matters for #21: in the controlled pipeline the held-out inverted-U appears only at t = 3×10⁻⁴ (monotone increase at 10⁻³ and 3×10⁻³; results/turnover_surface/summary.json, `held_out_targets`), while in Gensim the inverted-U appears only at t ≥ 10⁻³ (monotone decrease at 3×10⁻⁴).
- Fix: keep Fig. 1(a–d) and move (e,f) to the SM; merge Fig. 3(b) into Fig. 2 as a third panel; place Fig. 3(a) next to Proposition 12 and cite it there; move Fig. 3(c) to the SM; state the cross-pipeline threshold contrast in the text.

### 20. Table I

- Location: main.tex:436–461 (p. 5).
- Evidence: it is never cited (no `\ref{tab:radial-exponents}` in main.tex). Row 1 cites Qin and Rohe (regularized spectral clustering) for the unregularized result proved in Theorem 3. The third column mixes norm proportionalities, a vector statement ("vectors affine in log δ_i"), and "no degree signal for any α"; the norm of v_i = e_a + (1−α)(log θ_i)w is not affine in log δ_i. The caption defines none of u_i, v_i, c_a, e_a, w, or α. The exponents 1/2 and 1 that the Introduction and Discussion quote (main.tex:93, 1178) are not a column. The realized-graph score (Eq. 63), where degree re-enters and which drives every experiment, is missing.
- Fix: cite the table in Secs. I and XII; cite Theorem 3 in row 1; columns "object | row form | norm within block | exponent"; add a realized-graph SGNS row; write a self-contained caption.

### 21. Supplement tables

- Location: Tables S1–S3 (supp.pdf pp. 1–3).
- Evidence: the arms "Original", "Corrected, retained noise", and "Corrected, raw noise" are not defined in the captions. "Pooled ρ" is not defined in the caption, and ρ also denotes the RMS radius and a row norm in the main text. Median and range run together ("0.842[0.506, 0.935]") because the cell is in math mode. Table S2 drops the pooled column without saying why. The Table S3 header uses D (the degree matrix in the main text) and r (the balanced radius in the main text, the raw norm here). Precision differs between Table S3 (0.1630) and the SM text (0.163). "Requested relative Gram imbalance" (factor_balance.tex:35) is unclear. The main text never cites Table S3 or SM Sec. S2.
- Fix: define the arms and ρ in the captions, reuse the main-text predictor names, add a space or parentheses around the ranges, rename the conflicting symbols, and cite Table S3 at main.tex:687.

### 22. Precision and rounding

- Location: main.tex:996–1000, 1020, 1035–1042, 1058–1059.
- Evidence: several quantities are reported to 4–5 significant figures while their standard errors sit in the second digit, e.g., "−0.3258 ± 0.0155", "−0.2173 ± 0.0123", "−0.2980 ± 0.0142" (1035), "−0.1129 ± 0.0054" (1058), "12.31% to 10.03%" (1037). Two values are misrounded: the largest window-5 normalized RMS difference is 0.0045122 (results/norm_identifiability/empirical/summary.json), reported as 0.00452 (1059); the full-mixing end bins are 0.948800 and 0.948878 (results/turnover_surface/summary.json), reported as "0.9488 in both" (1020). The Spearman values at 996–1000 are node-level and pooled over five graphs (`curve_metrics` applies `safe_spearman` to all nodes of a condition; simulate_turnover_surface.py:643–647 and 772–786), which the text does not say.
- Fix: two significant figures on standard errors with matching decimals on means; correct the two values (0.00451; "0.949 in both"); state the unit of analysis for each correlation.

### 23. Smaller factual and wording slips

- Location and evidence:
  - main.tex:1041 "windows one through ten": only L ∈ {1, 2, 5, 10} were fit (results/window_training_continuation/plan.json).
  - Fig. 2 averages all ten fits per window, including the three window-5 fits that failed the stability checks (scripts/window_training/report_continuation.py:63–75). The caption counts them but does not say they are included.
  - main.tex:1167 counts three "boundary classifications", including the NetScience two-bin miss, while main.tex:1200 lists "missed one peak by two bins, misclassified two boundary peaks". The count agrees; the categories do not.
  - main.tex:991 says raw and balanced norms "give the same qualitative results", but no balanced curve for the controlled experiment is shown, and at λ = 1 the summary classifier labels the balanced curve an inverted-U and the raw curve not (results/turnover_surface/summary.json).
  - Abstract main.tex:60 "Mixing node contexts toward their marginal distribution removes the decreasing branch": at λ = 2/3 the curve still decreases (Spearman −0.35); only full mixing removes it.
  - main.tex:1185 "generate increasing, decreasing, and inverted-U curves even while exposure grows": the decreasing curve is the equal-exposure condition.
  - The PDF subject field reads "finite-training mechanisms" (main.tex:13), while the text repeatedly says that no mechanism is identified.
  - Fig. 1(a) shows that the t = 3×10⁻⁴ retention exposure nearly coincides with β = 1/3 (the purple and orange curves overlap). This is consistent with Eq. (67): with p_i = d_i/vol(G) and vol(G) = 3504–3532 in the five graphs (results/realized_graph_audit/graphs.json), p_i/t runs from about 2.8 to 15 over the plotted degrees, so the log elasticity runs from about 0.31 to 0.40. Saying so in the caption would explain why this threshold is the reference, and the overlap needs distinct markers.
- Fix: correct each as indicated.

### 24. LaTeX and production

- Location: main.tex:1222; main.log:508, 547; main.tex:837, 863; supp.tex:12; main.tex:1220.
- Evidence:
  - Article titles are missing from the main-text bibliography even though `longbibliography` is set. `mainNotes.bib` is 0 bytes and `main.bbl` reports "production of article title (-1) disabled", while the SM bibliography prints titles. In a scratch copy, deleting the explicit `\bibliographystyle{apsrev4-2}` (main.tex:1222) restored the REVTeX control entry and the titles (e.g., "[6] J. Jin, Fast community detection by SCORE, The Annals of Statistics 43, 57 (2015)"), with the page count unchanged.
  - The non-embedded Helvetica font from Fig. 1 (comment 6).
  - Two underfull lines show as loose spacing on p. 1 (right column, "SCORE [6] and regularized spectral clustering [7].") and p. 9 (right column, "The original, uncorrected Gensim sweep trained").
  - "(Remark 28)" (main.tex:837) and "(Theorems 2 and 3)" (main.tex:863) read as this paper's own numbered results; this paper has a Theorem 3. Write "[11, Remark 28]" and "[12, Theorems 2 and 3]".
  - The text spells out "Equation (n)" 20 times; Physical Review copyediting will abbreviate to "Eq. (n)" mid-sentence.
  - The SM affiliation for Kojaku omits the city (supp.tex:12 vs main.tex:47).
  - "A permanent public archive will be supplied before submission" (main.tex:1220) reads as an internal to-do; give the archive identifier or remove the sentence.
  - There are no overfull boxes or undefined references or citations, and the xr labels into the SM resolve.
- Fix: as listed.

### 25. Two cited results are derived only outside the Supplemental Material

- Location: main.tex:1062 ("A conservative full-support loss-to-radius bound is uninformative for all ten window-ten fits"); Fig. 3 caption, main.tex:1070 ("the dotted line is the geometric bound √2").
- Evidence: both are derived only in paper/current/norm_geometry_supplement.tex (section "A conservative full-support bound", lines 72–128; the √2 cap for nonnegative unit-RMS profiles at line 121), which supp.tex does not include and main.tex does not cite.
- Fix: move the derivation into the SM and cite it, or drop the sentence and explain √2 in the caption.

### 26. The central controlled inverted-U lacks a per-graph robustness count

- Location: main.tex:998; Fig. 1(c).
- Evidence: the β = 1/3 peak exceeds the lowest bin by 0.093. It is shown only with node-pooled bars (comment 1), and no per-graph count of interior maxima is given, although the Gensim trajectory analysis reports such counts (5/5, 0/5; results/training_trajectory/README.md). The node data needed to compute the count are not committed (comment 6).
- Fix: report, for raw and balanced norms, how many of the five graphs have an interior maximum at β = 1/3, and the per-graph peak-minus-end gaps.

## Per-figure assessment

| Figure | Page | Message (one line) | Problems | Fix |
|---|---|---|---|---|
| Fig. 1, `fig:controlled-turnover` | 9 | The exposure exponent β sets whether norms fall, peak, or rise with degree; full context mixing flattens the curve | Text 2.4–6.4 pt; tab10 green/red and blue/purple collapse under CVD; caption's error-bar claim is false and (e,f) has no bars; A–F vs (a)–(f); suptitle; byte-patched PDF with non-embedded Helvetica; two messages; (f) duplicates Fig. 4(d) and hides its "actual" series; held-out curve in (c) not described in caption (c); three names for the interpolation fit; KL units missing in (b); colliding tick labels in (f) | Regenerate from the script at print size with a sequential palette and markers; keep (a–d), move (e,f) to SM; correct the caption; describe the held-out curve; state units |
| Fig. 2, `fig:fixed-noise-windows` | 10 | Wider windows weaken the upper-degree declines in both context KL and balanced norm | Caption title "Context mixing" reuses the λ-intervention name; no panel labels; suptitle; legends 5.7 pt; unstable window-5 fits included without saying so; "full support" undefined; overlaps Fig. 3(b) | Retitle "Wider context windows …"; (a), (b) labels; merge Fig. 3(b) as (c); say that all fits are averaged |
| Fig. 3, `fig:norm-identifiability` | 10 | Near-optimal loss does not fix normalized shape; saved fits nonetheless agree at L = 5 and 10 | Three messages; (a) belongs with Prop. 12 (p. 6) and is not cited there; same colors mean profile type in (a) and window in (c); window colors differ from Fig. 2; window-5/10 green and pink fall to ΔE 17.9 under deuteranopia with identical markers; √2 bound unexplained and derived outside the SM; editorial text inside panels; "Degree-bin center" label | Move (a) to Sec. VII; (b) into Fig. 2; (c) to SM; explain √2 in the caption; remove in-panel commentary |
| Fig. 4, `fig:gensim-turnover` | 11 | In Gensim the norm peak moves to higher degree as subsampling weakens | Pre-correction runtime; in-figure title "validation of the turnover prediction" contradicts text; node-based SE; orange/green collapse under protanopia; "fixed epochs" hidden under "budget matched" in (d); "3e−04" notation; colliding ticks; budget-matched panel never replicated | Replace with the corrected epoch-5 sweep at four filtering settings (graph-level SE, interior-maximum counts); move historical panels to SM |
| Fig. 5, `fig:training-schedule` | 12 | With filtering off, a constant learning rate gives increasing curves at epoch 20 where the original schedule gives a peak | No panel labels ("Left/Right"); suptitle; legend 6.2 pt; "raw input norms"; field-name y-labels; panels could share a y-axis | (a), (b) labels; "center-vector norm"; shared y-axis; drop the suptitle |
| Fig. 6, `fig:prospective-validation` | 13 | Frozen exposure–KL predictions track observed curves and peak movement in unseen graphs | No panel labels in the figure; caption's error-bar claim is false; (d) hides one of nine points; log-degree axes cannot show the bin metric; occupied bins per family (4, 5, 6) unstated; legend order reads poorly; colliding ticks; "Clustered power law" vs "clustered scale-free"; "two-force" undefined; pre-correction runtime undisclosed | Peak-bin index plot or 3×3 table; state bins; correct the caption; name families consistently; disclose the runtime |
| Missing | – | Where each intervention acts in the population → realized → trained pipeline | No schematic; pipeline settings scattered over five sections | Add Fig. 0 and a pipeline table (comment 18) |

## Consistency of reported numbers

| Quantity | Locations | Values | Consistent? |
|---|---|---|---|
| Prospective pooled node-level Spearman | abstract (63); main.tex:1166; Table S1; prospective summary.json | 0.881; 0.881; 0.881; 0.8806 | Yes |
| Peaks within one bin | abstract (63); 1166; Fig. 6 caption (1148); Table S1; summary.json | 8; 8/9; 8/9; 8/9; 0.889 | Yes (only 8 markers visible in Fig. 6(d)) |
| Median peak error | abstract; 1166; Table S1 | 0; 0; 0 | Yes |
| Within-curve Spearman, exposure–KL vs degree + threshold | 1171; Table S1 | 0.842 vs 0.293 (both places) | Yes |
| Calibration leave-one-seed-out | 1158; calibration_summary.json | R² 0.828, ρ 0.932; 0.8284, 0.9325 | Yes |
| Boundary failures | 1167; 1200; Fig. 6 caption | "three boundary classifications" (includes NetScience); "one peak by two bins, two boundary peaks"; "NetScience misses by two bins" | Count yes; categories differ |
| Controlled β = 0, 1, 2/3 | 996–1000; turnover_surface/summary.json | 3.507→2.297 (−0.758); 2.496→4.802 (0.895); 2.861→3.960 (0.821) | Yes |
| Controlled β = 1/3 peak | 998; summary.json | 3.315 at degree 6.00, +0.093, −0.361 | Yes |
| Full mixing end bins | 1020; summary.json | "0.9488 in both"; 0.948800, 0.948878 | No (fourth digit) |
| Held-out peak vs surface prediction | 1026; summary.json; Fig. 1(f) | 7.73 vs 16.04; 7.731 vs 16.044; 7.73 vs 16.08 | Yes (Fig. 1(f) uses the arithmetic bin mean) |
| Interpolation RMSE | 1027; summary.json | 0.013; 0.0132 | Yes |
| Gensim historical peaks | 1082, 1102; Fig. 4(d); gensim summary.json | lowest bin, 6, 7.73; 2.98, 6.00, 7.73 | Yes |
| Gensim rise and drop | 1084; summary.json | 0.372→0.569; 0.537→0.238 | Yes |
| Interpolation miss on Gensim | 1104 | 4, 2, 1 bins | Yes |
| Corrected Gensim peak movement | Intro 102; Results 1106–1120; training_trajectory README | "reproduce the peak movement"; not reported; 2.98, 6.00, 7.73 | No (claim absent from Results) |
| Corrected filtering-off peak | 1110; trajectory summary.json | 7.73, drop 0.244 | Yes |
| Schedule contrast | abstract 61; 1117–1118; README | 5/5 interior from epoch 2 (original); 0/5 at all checkpoints and monotone increase at epoch 20 (constant) | Yes |
| Window gaps, norm and KL | 1035; report/summary.json | −0.3258±0.0155→−0.2173±0.0123; −0.2980±0.0142→−0.1303±0.0063 | Yes |
| Stability counts | 1038; Fig. 2 legend | 7/10, 10/10 | Yes |
| Window set | 1041; plan.json | "one through ten"; [1, 2, 5, 10] | No (wording) |
| Window-1 loss gaps and norm changes | 1042; report/summary.json | 1.303e−6, 7.327e−7; 5.01%, 1.13% | Yes |
| Normalized upper-bin gap | 1058; empirical/summary.json | −0.1129±0.0054→−0.0934±0.0053 | Yes |
| Start differences | 1059; Fig. 3(c); empirical/summary.json | 0.984–1.187; 0.487–0.886; ≤0.00452; ≤5.54e−5 vs 0.0045122, 5.538e−5 | Partly (0.00452 should read 0.00451) |
| Factor balance | 687; factor_balance.tex:40–42; Table S3 | 0.55–1.15%, 12.45%; Gram imbalance 0.0786–0.163 vs 0.0786–0.1630 | Yes (precision differs) |
| Learning rates | 1114; controlled config | 0.025→0.0001; constant 0.0025 = controlled final rate | Yes |
| Epochs and dimensions | 989, 1079, 1162; configs | 4 epochs, 24 d (controlled); 5 epochs, 64 d (Gensim) | Yes |
| Degree bins | Fig. 5 caption; code | 8 requested → 6 occupied (controlled, trajectories); 6 requested → 4, 5, 6 occupied (prospective) | No (undefined, and they differ) |
| Error-bar definition | captions vs code | Figs. 1, 6: caption says seeds, code uses nodes; Fig. 4: nodes (stated); Figs. 2, 3, 5: graphs | No |
| Threshold notation | text; Figs. 1, 4, 6; Table S3; 1119 | 3×10⁻⁴; 3e−04; 0.0003; t = 0.003 | No (format) |
| Exposure | 98; 919–929; 965–972; 1156 | updates; expected retained tokens; exact center counts; realized positive pairs | No (one term, four quantities) |
| Context KL reference | 897–904; Fig. 1(b); prospective code | π; realized center-weighted marginal q; retained-pair marginal | No (one term, three references) |
| Kojaku affiliation | main.tex:47; supp.tex:12 | "Binghamton, New York"; "New York" | No |
| Node-level target predictions | 1163; outcomes.jsonl | 23,685; 3 × (3000 + 3000 + 1895) | Yes |
| NetScience size | 1160; outcomes.jsonl | 379 nodes; 1895/5 | Yes (edge count not checked) |

## Notation audit

| Symbol | Meanings and locations | Suggested change |
|---|---|---|
| q | negative distribution δ^α/Z_α (469); negative proposal (882); realized context marginal Σ c_i T_i (980; Fig. 1(b)) | Keep q for negatives; use q̄ for the context marginal |
| c, C | community vectors c_a and matrix C (176–180); block constant C_ab (627); center share c_i (882, 983) | Center share μ_i; C_ab → H^sym_ab |
| D, 𝒟 | population degree matrix 𝒟 (140); degree matrix D (844, 858); scalar in Prop. 12 proof (738–740); Gram imbalance (SM Eq. S1, Table S3) | Scalar τ; imbalance Ξ |
| e | basis vector e_i (296); community context vector e_a (180, 457, 589); Euler's number | Write row i of U\|Λ\|^{1/2} directly in Eq. (18) |
| ρ | RMS radius (727, 740); row norm ρ_i (766–822); Spearman (Tables S1–S2) | RMS radius r̄; write "Spearman" in tables |
| r | lag index (363); rank bound (724); balanced radius r_i(M) (727); raw norm r_i (SM) | Separate letters for lag, rank, radius; one meaning of r_i across main text and SM |
| h | rank(M) (173); column h_b of H^(α) (609–610); profile h_i (738) | Profile f_i |
| s | block propensity mass s_a (130); nonedge score magnitude s (737) | Rename the latter |
| α | negative-sampling exponent (469); α_i in Prop. 12 proof (738) | φ_i |
| δ | expected degree δ_i (137); loss tolerance δ (731) | Tolerance ε_L |
| a, b | block indices; scalar weights in proof of Prop. 11 (705) | Use p⁺, p⁻ as in the statement |
| Z | membership matrix (114); normalizer Z_α (471) | 𝒵_α or N_α |
| G | block matrix G_ab (392); realized graph G (883, undefined); loss gradient (SM) | Define G; gradient ∇𝓛 |
| L | window length (363, 1033); loss L(M) (695); 𝓛 in SM | 𝓛 for loss throughout |
| d | realized degree d_i (first used 651, never defined); number of spectral components (860) | Define d_i; another letter for components |
| ϖ, ω | stationary distribution ϖ_i (370) and window weights ω_r (363) in the same product (377) | π^pop for the population stationary distribution |
| m_i | expected retained tokens (Eq. 66); exact center counts (Eq. 69) | m_i^ret and m_i(β) |
| η | ratio η_i (646); step size (SM Eq. S3) | Rename one |
| σ, γ | logistic function (488) and exponent in ‖u_i‖ ∝ d_i^{−γ} (651), both undefined | Define |

## Questions for the authors

1. How many of the five graphs show an interior maximum at β = 1/3 for raw and for balanced norms, and what are the per-graph peak-minus-end gaps?
2. Can `node_results.csv` for Figs. 1 and 4 be committed? Without them neither figure can be restyled or given graph-level error bars without rerunning training.
3. Why does the main text show the pre-correction Gensim sweep when corrected epoch-5 runs exist for the same corpora? Are corrected budget-matched runs planned?
4. In the controlled pipeline the held-out inverted-U appears only at t = 3×10⁻⁴, while in Gensim it appears only at t ≥ 10⁻³ (see #21). Do you read this as one mechanism acting differently in the two trainers, and should the figures make the contrast explicit?
5. Which reference distribution defines "context KL" in each experiment, and should the prospective predictor's KL be compared with Eq. (64)?
6. Were the Fig. 6 target embeddings trained with the legacy negative table? If so, would Fig. 6 be better drawn from the corrected arms, with the frozen arm reported as the confirmatory test?
7. Which population results do you claim as new (Theorem 3, Theorem 6, Corollaries 8–9, Propositions 11–12)? The answer determines what the first sentences of the abstract should say.
8. What do "weakly unimodal" and "boundary classification" mean operationally (main.tex:1167)?
9. Is the √2 in Fig. 3(c) the bound for nonnegative unit-RMS profiles, and where should readers find the loss-to-radius bound cited at main.tex:1062?

## Recommendation and confidence

Recommendation: major revision.
The scientific content may well support publication in PRE, but the paper cannot be evaluated as printed. The primary figure is unreadable at print size, two captions misstate the uncertainty shown, the main Gensim figure and the prospective outcomes come from a runtime the authors themselves classify as superseded (only the former is labeled), and the readouts behind the abstract's headline numbers are undefined in the main text.
None of these require new science; most fixes use data already committed (Figs. 2, 3, 5, 6 and the corrected Gensim trajectories), while Figs. 1 and 4 need their node-level data restored.

Confidence: 4 of 5.
I verified figure fonts, colors, error-bar code paths, bin counts, and about 35 reported numbers directly against the committed scripts and summaries, and I confirmed the bibliography fix in a scratch build. I did not rerun any training, so I cannot say how much the graph-level error bars for Figs. 1 and 4 will differ from the plotted ones, and I have not assessed the proofs beyond notation and organization.

## Action items

| ID | Severity | Location | Problem | Proposed fix | Type | Effort |
|---|---|---|---|---|---|---|
| R4-01 | blocker | main.tex:1012 (Fig. 1), main.tex:1146 (Fig. 6) | Captions say SE across graph seeds or replicates; code plots node-pooled SE; Fig. 1(e,f) has no bars | Correct both captions now; state that 1(e,f) has no error bars | PR | ≤1h |
| R4-02 | major | scripts/simulate_exact_center_exposure.py:285–328; scripts/prospective_norm_turnover.py:433–455 | Node-pooled SE differs from replicate SE by factors of 0.13–1.90 (Fig. 6); three figures use one SE definition and three another | Replot Figs. 1, 4, 6 with graph- or replicate-level SE as in Figs. 2, 3, 5 | ISSUE | ≤1day (Fig. 6); >1day for Fig. 1 if data must be regenerated |
| R4-03 | blocker | Fig. 4; main.tex:1076–1104; scripts/simulate_gensim_threshold_sweep.py:221–261 | Main figure from the pre-correction runtime; in-figure title claims a validation the text refutes; node SE | Replace (b,d) with the corrected epoch-5 sweep from results/training_trajectory (four settings, graph SE, interior counts); move historical and budget-matched panels to SM; drop the suptitle | ISSUE | ≤1day |
| R4-04 | blocker | All figures, worst Fig. 1 | Text 2.4–6.8 pt at print size; colliding tick labels | Draw at printed size; ≥7 pt (8 pt preferred) for all text; integer degree ticks | ISSUE | ≤1day (Fig. 1 needs R4-07) |
| R4-05 | major | Figs. 1, 4 (and Fig. 3(b,c)) | Color pairs collapse under CVD (ΔE 7.3, 5.7, 4.6; 17.9 in Fig. 3); color is the only cue | Sequential ramp per ordered parameter plus markers or direct labels; same color per t across Figs. 1, 4, 6 | PR | ≤1day (Fig. 1 needs R4-07) |
| R4-06 | major | Figs. 1–6 | A–F vs (a)–(f); missing panel labels (Figs. 2, 5, 6); suptitles; inconsistent axis labels, window colors, threshold notation, "input norms" | One shared plotting style; labels matching captions; no suptitles; consistent names | ISSUE | ≤1day |
| R4-07 | major | Fig. 1; scripts/audit_realized_graph.py:590–660; scripts/simulate_turnover_surface.py:979–1068; .gitignore:7; main.tex:1218 | Byte-patched PDF with non-embedded Helvetica and mixed fonts; script still writes old labels; node CSVs for Figs. 1 and 4 not committed, contradicting Data Availability | Fix labels in the script; commit or regenerate node CSVs; rebuild; verify all fonts embedded | ISSUE | >1day |
| R4-08 | major | Fig. 6(d); scripts/prospective_norm_turnover.py:838–870 | Two coincident pairs hide one of nine markers; log-degree axes cannot show the bin metric; occupied bins per family (4, 5, 6) unstated | Plot peak-bin indices with offsets (or a 3×3 table); state bins per family | PR | ≤1day |
| R4-09 | major | main.tex:102 vs 1106–1120 | Intro claims corrected Gensim peak movement; Results never report it | Report 2.98, 6.00, 7.73 and 0/5, 5/5, 5/5 interior maxima at epoch 5 | PR | ≤1h |
| R4-10 | major | main.tex:1153–1173; Fig. 6 caption; abstract 62–64; Tables S1–S2 | Prospective outcomes are pre-correction (results/prospective_validation/README.md), undisclosed in the main text; corrected arms undefined (#17) | Name the runtime in text and caption; give corrected values; define "retained noise" and "raw noise" | PR | ≤1h |
| R4-11 | major | Sec. XI.A (main.tex:958–991) | Degree bins (8 or 6 requested; 4–6 occupied), bin coordinate, peak, turnover, inverted-U, weakly unimodal, boundary classification undefined | Add a "Readouts" paragraph; one name for the inverted-U | PR | ≤1h |
| R4-12 | major | main.tex:1165–1166; supp.tex:52–53 | The four criteria thresholds appear only in an uncited SM section | State them in the main text and cite SM Sec. S2 | PR | ≤1h |
| R4-13 | major | Throughout (notation audit table) | q, c/C, D, e, ρ, r, h, s, α, δ, Z, G, L, d, m_i, η overloaded; ϖ/ω adjacent; σ, γ, d_i, G undefined | Notation table and renames | PR | ≤1day |
| R4-14 | major | Abstract; Secs. X–XI; Tables S1–S2 | "Exposure" (four quantities), "context KL" (three references), λ intervention (three names, and windows called "context mixing"), plus undefined terms (population operator, realized-graph target, two-force, literal, full support, hash-locked) | Define each once; one name per object; state KL reference and units in every figure | PR | ≤1day |
| R4-15 | major | main.tex:53–68 | 301 words and 14 sentences; main result not in the first three sentences; undefined terms; unexplained "Corrected" | Rewrite to about 200 words following the five-slot order in comment 14 | PR | ≤1h |
| R4-16 | major | main.tex:1076, 1078, 1089, 1095, 1103, 1106, 1108, 1027, 1041, 1155 | Section titles and captions narrate the project's history; corrections never explained | Content-based titles; present corrected results; one SM paragraph describing each correction and the figures it affects | PR | ≤1day |
| R4-17 | minor | Sec. XI; Discussion (e.g., main.tex:1043, 1060–1062, 1138, 1173, 1181–1182, 1189, 1201) | At least 15 stacked disclaimers in Sec. XI bury the findings | State results first; consolidate qualifiers in the Limitations paragraph | PR | ≤1h |
| R4-18 | major | Secs. II–X | About 6.5 pages of theory before data; 36 of 61 labeled equations unreferenced; tangential results; no roadmap; Eq. (68) not linked to Sec. XI; Prop. 12 objective only in its proof; italic remarks | Condense around Table I with proofs in appendices; mark new results; add a roadmap; cite Eq. (68); restate Prop. 12; `\theoremstyle{remark}` | ISSUE | >1day |
| R4-19 | major | New Fig. 0 and Table II | No schematic or pipeline summary for the five experimental pipelines | Add the schematic and a pipeline-settings table | ISSUE | ≤1day |
| R4-20 | minor | Figs. 1(e,f), 2(b), 3, 4(d) | Duplicated series and multi-message figures; Fig. 3(a) far from Prop. 12; cross-pipeline threshold contrast hidden (#21) | Reorganize as in comment 19; state the contrast | ISSUE | ≤1day |
| R4-21 | minor | Table I (main.tex:436–461) | Never cited; row 1 cites the regularized result; mixed column semantics; no exponent column; missing realized-graph row; caption not self-contained | Revise as in comment 20 | PR | ≤1h |
| R4-22 | minor | Tables S1–S3; factor_balance.tex:35–40 | Arms and ρ undefined; predictor names differ from main text; D and r clash with main text; spacing; precision; S2 and S3 uncited | Revise captions and headers; cite Table S3 at main.tex:687 | PR | ≤1h |
| R4-23 | minor | main.tex:996–1000, 1020, 1035–1042, 1058–1059 | Excess precision; 0.00452 should be 0.00451; "0.9488 in both" should be 0.949; Spearman unit of analysis unstated | Two significant figures on SEs; correct values; state units of analysis | PR | ≤1h |
| R4-24 | minor | main.tex:13, 60, 991, 1041, 1167 vs 1200, 1185; Fig. 2 caption | Wording slips: windows "one through ten"; unstable fits included silently; boundary-miss categories; balanced curves claimed but not shown; abstract overstates partial mixing; "even while exposure grows" covers the equal-exposure case; PDF subject says "mechanisms" | Bundle of wording fixes listed in comment 23 | PR | ≤1h |
| R4-25 | minor | main.tex:1222, 837, 863, 1220; supp.tex:12; main.log:508, 547 | Explicit `\bibliographystyle` suppresses article titles; ambiguous external theorem references; affiliation; archive wording; loose lines; "Equation" spelled out | Delete main.tex:1222 (verified in a scratch build); write "[11, Remark 28]" and "[12, Theorems 2 and 3]"; fix the rest | PR | ≤1h |
| R4-26 | major | main.tex:1062, 1070; norm_geometry_supplement.tex:72–128 | Loss-to-radius bound and √2 cap cited but derived only in a document outside the SM | Move the derivation into the SM and cite it, or drop the claim and explain √2 in the caption | PR | ≤1h |
| R4-27 | major | main.tex:998; Fig. 1(c) | The β = 1/3 inverted-U (0.093 rise) has no per-graph robustness count | Report interior-maximum counts and per-graph gaps for raw and balanced norms | ISSUE | ≤1day (needs the node data from R4-07) |

## Reviewer checks (scratch, reproducible)

- Font sizes at print scale: <scratch>/round/r4/fontsizes.py (PyMuPDF; scales native span sizes by 510 pt / page width).
- Color-vision simulation and ΔE: <scratch>/round/r4/cvd.py (Machado et al. 2009, severity 1); simulated images in <scratch>/round/r4/figs/cvd/.
- Node-pooled vs replicate SE for Fig. 6: <scratch>/round/r4/se_compare.py.
- Occupied bins per prospective family: <scratch>/round/r4/bins.py.
- Bibliography test build (explicit `\bibliographystyle` removed): <scratch>/round/r4/build/paper/current/main.pdf.
- Page renders: <scratch>/round/r4/p-01.png through p-15.png and s-1.png through s-3.png.
