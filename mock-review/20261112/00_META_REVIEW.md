# Meta-review, round 2026-11-12

Manuscript: `paper/current/main.tex` at `c958c52` (main, 15 pp.; supplement 3 pp.), reviewed as a full draft for Physical Review E.
Panel: six independent reviewers, one lens each, on Claude Opus at maximum effort.
None saw the others' reviews; each saw only the titles of the open issues (#17, #21, #22, #44).
The repo's HQ card lists the project at the drafting stage, but the draft has a full introduction, discussion, and conclusion, so the panel got the full-draft prompt.

## Scores

| Reviewer | Lens | Recommendation | Confidence |
|---|---|---|---|
| R1 | Venue fit, novelty, framing | Major revision ("reconsider after major revision") | 4/5 |
| R2 | Statistics and methods | Major revision | 4/5 |
| R3 | Data processing and reproducibility | Major revision | 4/5 |
| R4 | Presentation | Major revision | 4/5 |
| R5 | Theory (DCSBM, spectral, SGNS optima) | Major revision | 4/5 |
| R6 | SGNS training as implemented (Gensim) | Major revision | 4/5 |

## Consensus

All six recommend major revision, and none recommends rejection.
The record is unusually auditable, and the mathematics checks out.
The problems are in what the experiments can show and how the paper reports them:

- The population theory never reaches the experiments.
- The controlled curve shapes depend on the training budget.
- The prospective test's criteria are weak, and its headline numbers predate the corrections, which the main text does not say.
- Two captions misstate the error bars.
- Several published results are restated without credit.

Themes ranked by how many reviewers raised them independently (tracker IDs in brackets):

1. Prospective headline numbers come from the pre-correction pipeline without a main-text label: R1, R2, R3, R4, R6 [T-003].
2. Fig. 1 and Fig. 4 cannot be regenerated or come from superseded outputs: R1, R2, R3, R4 [T-007].
3. Training settings and the three implementation corrections are not described: R1, R3, R4, R6 [T-010].
4. Error bars are node-pooled standard errors captioned as seed-level: R2, R3, R4 [T-005].
5. The declared prospective criteria pass for predictors without node information: R1, R2, R3 [T-004].
6. Context-mixing allocation does not deliver the intended targets: R2, R5, R6 [T-011].
7. The inverted-U rests on 4–6 occupied bins with a one-bin falling branch: R1, R2, R4 [T-013].
8. Results depend on training budget and progress: R6, with R1 and R2 on the schedule and trainer disagreement [T-002].
9. No experiment tests the population theorems: R1, R5 [T-001].
10. Published theorems are restated without credit: R1, R5 [T-008].

## Previous round status

None; this is the first round.

## What holds up

- R5 re-derived 28 theory items, including all 13 numbered results and both supplements, and found no mathematical error.
- R3 traced 80 numbers to committed outputs with one mis-rounding, and R2 and R4 reproduced about 28 and 35 numbers respectively.
- All 492 recorded file hashes verify (R3). Tables S1–S2 regenerate byte-identically, and Figs. 2, 3, and 5 regenerate pixel-identically.
- The README revalidation recipe reruns in about four minutes, and the paper builds from a fresh clone (R3).
- The corrected retrospective revalidation gives numbers close to the frozen ones: pooled Spearman 0.875/0.873, 8/9 peaks within one bin, median peak error 0 (R3).
- The learning-rate-history result is new and useful (R1), though R6 questions its interpretation (T-002).
- The Discussion is candid about exploratory choices: post hoc interpolation, the post hoc model class, and the boundary misses.

## Cross-cutting problems

### Theory and experiments don't meet (T-001, T-002)

Every experiment uses configuration-model graphs, which are one-block DCSBMs.
For K = 1 the paper's own result gives a constant unigram score, `−log k` (main.tex:879), so the population theorems predict equal norms for every node and no experiment tests them (R1-01, R5-01).
R5 computed the free optima the paper never shows (R5-02).
For λ = 1/3 and 2/3 the intended mixed targets have norms rising with degree (Spearman +0.66 to +0.80), while the trained β = 0 curves fall.
"Mixing removes the decreasing branch" therefore describes the training path; the target itself has norms rising with degree.
At window one the infimum isn't attained (Prop. 11), so every one-step-context curve is an optimization-path observable (R5-04); R1-17 links this to the schedule dependence in Fig. 5.

R6-01 reran the authors' controlled trainer with only the epoch count changed.

| Budget (relative to the paper) | β = 0 | β = 1/3 | β = 1 |
|---|---|---|---|
| 1× (4 epochs, as in the paper) | decreasing | inverted-U | increasing |
| 4× (16 epochs) | inverted-U | increasing | increasing |
| 12× (48 epochs) | increasing | increasing | inverted-U |

I reran R6's script on graph seed 3, which R6 had not used, and got the same pattern.
The abstract's exposure sentence (main.tex:59) holds only at the chosen budget, and the response model ν(log m, KL) in Eq. (68) has no time argument.
R6-02 adds that the constant-rate arm at epoch 20 is behind the annealed arm in loss in 5/5 graphs, so the schedule contrast compares unequal training progress.

### Prospective test (T-003, T-004)

`results/prospective_validation/README.md` says the frozen outcomes predate the repeated-visit counter, integer-token negative-table, and BLAS corrections, and "must not be described as results of the corrected code."
Sec. XI.G and the abstract report 0 bins, 8/9, and 0.881 without saying so.
The frozen run also used the permuted negative sampler, which the main text does not mention (R6-06).
PR #47 adds a caveat about walk-visit counting to this paragraph, but not about the other corrections.

The declared criteria (`protocol.json`: median peak error ≤ 1 bin, ≥ 2/3 of peaks within one bin, pooled ρ ≥ 0.5, ≥ 2 families nondecreasing) don't discriminate.
I reran R2's null predictor: the calibration mean per threshold, plus a 0.01 bump at a fixed bin, passes the first three criteria (median error 1, 6–7/9 within one bin, ρ 0.58–0.59), and the fourth is predictor-independent.
R1-11 finds that placing the peak at threshold rank + 1 gives 9/9 within one bin.
R2-02 and R2-03 show that 46% of pooled rank variance lies between curves, and about 80% of the within-curve variance lies inside exact-degree classes.
A graph-only predictor (degree, threshold, structural KL) ties the frozen predictor's within-curve 0.842.
R1-13 and R2-13 note that the two synthetic target families lie 99.8–100% inside the calibration feature ranges, while all the failures fall on NetScience, the one out-of-distribution target.
R2-09 adds that the declared response surface failed (R² −2.19, ρ −0.67), which the paper doesn't report.
R2-04 and R3-06 add that the freeze is self-attested: protocol, manifest, predictions, and outcomes first appear together in commit `5902c3c`, and the binning and peak definitions are not hashed.

### Reporting of uncertainty and figures (T-005, T-006, T-007)

The Fig. 1 caption (main.tex:1012) and the Fig. 6 caption (main.tex:1146) say the error bars are standard errors across seeds.
`binned_curve` in `scripts/simulate_exact_center_exposure.py:308` and `scripts/prospective_norm_turnover.py:455` computes `std/sqrt(n_nodes)` over nodes pooled across seeds, and Fig. 1(e,f) has no error bars.
Seed-level standard errors differ from the plotted bars by factors of 0.13–1.90 in Fig. 6 (R4-02) and up to 13.8 across both figures (R3-04).
At print width (≈510 pt), the smallest and median figure text are 2.4 and 4.9 pt in Fig. 1, 4.3 and 4.9 pt in Fig. 4, and 4.7 and 6.7 pt in Fig. 6 (my measurement of the committed PDFs; R4-04).
Fig. 1 is a PDF label edit of an older plot with a non-embedded Helvetica font, its node-level CSVs are gitignored, and the plotting script still writes the old labels (R3-05, R4-07).

### Credit (T-008, T-009)

- Theorem 3 is Qin & Rohe 2013, Lemma 3.3, at τ = 0 (R1-02 and R5-06, each checked against arXiv:1309.4111).
- Corollary 9 is the α-smoothed PMI of Levy, Goldberg & Dagan 2015, which is not cited (R1-04, R5-06).
- residual2vec (Kojaku et al. 2021, two of the present authors) is not cited, although R1-03 finds that it overlaps Lemma 5, Theorem 6, and Corollary 8.
- Wilson & Schakel 2015 is cited for the inverted-U (main.tex:80), but its abstract (arXiv:1510.02675) reports length "more or less linearly" dependent on frequency.
- Schakel & Wilson 2015 (arXiv:1508.02297, Sec. 3.3 and Fig. 3, skip-gram on hep-th abstracts) does report the inverted-U, and it also proposes the consistent-use versus many-contexts explanation that main.tex:82 states without a citation.

## Compliance and veracity

- The main text presents pre-correction outcomes without a label, although the repo's own README forbids this (T-003). Before submission, this is the most serious veracity item.
- Two captions misdescribe their error bars (T-005).
- One citation is misattributed (T-009).
- The data-availability statement is a placeholder with no URL or DOI, and generated data for Figs. 1 and 4 are not in the repo (R1-22, R3-07).
- There are no acknowledgments or funding statement and no author contributions (R1-23).
- R1-23 also lists an AI-use disclosure; check APS's current policy before treating it as required.
- PRE review is single-blind, so anonymity doesn't apply.

## Reviewer errors and unverified claims

- R5 says the premise of #44 "may be stale" because `sample_walks` now uses `np.add.at`. The code at `c958c52` is fixed (since PR #35), but #44 concerns outputs produced before that fix, which PR #47 reruns. The issue is not stale.
- R4 says deleting `\bibliographystyle{apsrev4-2}` (main.tex:1222) restores article titles in the references. I didn't verify this.
- R1-03's residual2vec overlap (specific equation correspondences) comes from R1 alone; I confirmed only that residual2vec is uncited.
- R1-23's AI-use disclosure: APS requirements not checked.
- Of the blockers I verified, none was a false positive: T-001 (main.tex:879, 960), T-002 (rerun on seed 3), T-003 (README and Sec. XI.G), T-004 (null predictor rerun, `protocol.json`), T-005 (code and captions), and T-006 (font measurement).

## Disagreements

- R3 says the corrections leave the qualitative conclusions intact, while R6 says the controlled conclusions reverse with budget. Both hold: the corrections barely move the numbers at the paper's budget, but the budget itself changes the shapes. Running the grid at three budgets would settle it.
- R2 says the main text labels the Gensim sweep "uncorrected" (main.tex:1078), and R3 and R4 call Fig. 4 a blocker. The text does label it. The in-figure title "validation of the turnover prediction" and the use of superseded outputs remain, so I rate it major (T-007).

## Already in flight

As of 2026-11-13, PRs #46 and #47 have merged and #44 is closed; the gaps noted below remain.

- PR #47 reruns the historical sweeps with token counts, adds a walk-visit caveat to Sec. XI.G, and updates the Fig. 4 text: peak bins unchanged. It covers part of T-003 and T-007, but doesn't disclose the negative-table and BLAS corrections or swap the Fig. 4 image.
- PR #46 adds import guards so `make test` collects on a fresh clone; this is part of T-014.
- Issue #22 (exposure adds little norm once mixed): R2 adds new evidence that the β = 1, λ = 1 cell is not fully mixed (T-011).
- Issue #17 (external subsampling flattens the negative distribution) relates to R6-06's permuted sampler (T-003).
- Issue #44 is covered by PR #47.

## Triage

**Must fix before submission**

- T-003: say in Sec. XI.G, the abstract, and Fig. 6 that the frozen outcomes predate the corrections, and report the corrected revalidation alongside them.
- T-005: fix the two captions, or plot seed-level standard errors.
- T-009: fix the Wilson & Schakel citation.
- T-004: report the null predictors and the failed declared surface, and present the prospective result as what it is.
- T-001 and T-002: decide the paper's central claim, given that the theory doesn't reach the experiments and the shapes depend on budget.
- T-008: credit Qin & Rohe, Levy–Goldberg–Dagan, and residual2vec where the paper restates their results.
- T-006: make figure text at least 7 pt at print size.

**Should fix**

- T-007 (figure provenance), T-010 (methods and corrections), T-011 and T-012 (interventions), T-013 (shape resolution and definitions).
- T-014 (the reproduction gate on the extension hash), T-015 (required statements), T-016 (notation), T-017 (structure).

**Defer**

- T-018 (Prop. 12 scope), T-019 (scope beyond Gensim and α = 1), T-020, T-021.
