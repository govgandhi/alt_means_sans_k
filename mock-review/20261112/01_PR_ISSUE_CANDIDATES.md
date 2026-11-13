# PR and issue candidates, round 2026-11-12

Priorities: P0 desk-reject or veracity, P1 wrong as printed, P2 presentation, P3 later.
Tracker IDs refer to `../TRACKER.md`; source IDs refer to the action tables in `R<n>_*.md`.

## Filed as

| Item | Filed |
|---|---|
| PR-A | [#56](https://github.com/govgandhi/alt_means_sans_k/pull/56) |
| PR-B | [#57](https://github.com/govgandhi/alt_means_sans_k/pull/57) |
| I-1 | [#50](https://github.com/govgandhi/alt_means_sans_k/issues/50) |
| I-2 | [#51](https://github.com/govgandhi/alt_means_sans_k/issues/51) |
| I-3 | [#52](https://github.com/govgandhi/alt_means_sans_k/issues/52) |
| I-4 | [#53](https://github.com/govgandhi/alt_means_sans_k/issues/53) |
| I-5 | [#54](https://github.com/govgandhi/alt_means_sans_k/issues/54) |
| I-6 | [#55](https://github.com/govgandhi/alt_means_sans_k/issues/55) |
| R2-20 on #22 | [#22 comment](https://github.com/govgandhi/alt_means_sans_k/issues/22#issuecomment-6857138529) |

## PR candidates

### PR-A (P0, T-005): say what the Fig. 1 and Fig. 6 error bars are

Source: R2-07, R3-04, R4-01.

- [ ] Fig. 1 caption (main.tex:1012): bars in (a)–(d) are standard errors over nodes pooled across the five graph seeds, and (e,f) have none.
- [ ] Fig. 6 caption (main.tex:1146): bars are standard errors over nodes pooled across seeds or replicates, which don't measure variation between seeds.
- Plotting seed-level standard errors instead is part of I-3.

### PR-B (P0, T-009): cite Schakel & Wilson 2015 alone for the inverted-U and for its explanation

Source: R1-08.

- [ ] main.tex:80: drop `wilson2015controlled`, which reports linear dependence (arXiv:1510.02675), and say "a domain-specific corpus" (hep-th abstracts, arXiv:1508.02297 Sec. 3.3).
- [ ] main.tex:82: cite `schakel2015measuring` for the competition explanation, which it proposes (Sec. 3.3).
- Crediting Wilson & Schakel's intervention design (R1-09) is part of I-4.

### Not opened this round

- T-003 disclosure in Sec. XI.G: #47 (merged) added the walk-visit caveat to that paragraph; the negative-table and BLAS caveat still needs adding, and I-2 decides how.
- T-014: compare `blas_check` results without the extension hash (R3-03). This is the authors' provenance design, so it is raised in I-5.
- T-017: include `norm_geometry_supplement.tex` in the Supplemental Material or move the cited bound into it (R4-26, R5-07).
- T-020: drop or support the Intro's corrected-Gensim claim at main.tex:102 (R4-09).
- T-021: R4's `\bibliographystyle` removal (unverified) and the minor wording items.

## Issue candidates

### I-1 (P1; T-001, T-002, T-018): experiments test no population theorem, and their shapes depend on training budget

Source: R1-01, R1-10, R1-16, R1-17, R2-10, R5-01, R5-02, R5-04, R5-05, R6-01, R6-02, R6-13, R6-14.
Decision: what is the paper's central claim? Options the panel raised: (a) derive and test a dynamical account of trained norms; (b) narrow the paper around the learning-rate and budget dependence, with the population material as credited background; (c) add multi-block experiments that the theorems do cover.
Analyses: the β × λ grid at three or more budgets, including one matched to the Gensim runs; norms against per-node training progress; free optima of the realized targets.

### I-2 (P0; T-003, T-004): the prospective test's headline numbers predate the corrections, and its criteria pass without node information

Source: R1-11, R1-12, R1-13, R1-14, R2-01–R2-06, R2-09, R2-13, R2-21, R3-01, R3-06, R4-10, R6-06.
Decision: how to present the prospective result. Report the frozen and corrected numbers side by side; report the null predictors, the failed declared surface, and a shape metric; call it exploratory transfer, or run a new freeze with discriminating criteria and an out-of-distribution target set.

### I-3 (P2; T-005, T-006, T-007): figures are unreadable at print size, and Figs. 1 and 4 can't be regenerated as printed

Source: R1-19, R2-08, R3-02, R3-05, R4-02–R4-08.
Work: seed-level standard errors; text at least 7 pt at print width; colorblind-safe palettes with marker or line-style cues; panel labels that match captions; regenerate Fig. 1 from its script and commit its node data; swap Fig. 4 for the corrected sweep that #47 produced.

### I-4 (P1; T-008, T-009): published results are restated without credit

Source: R1-02–R1-09, R5-06.
Decision: for each of Theorem 3, Lemma 5, Theorem 6, Corollaries 8–9, and the balanced split, cite the precedent and say what is new, or move the result to background.
Each citation the reviewers propose is listed with its source; verify them before adding.

### I-5 (P1; T-010, T-014, T-015): methods, corrections, and release statements are missing

Source: R1-18, R1-20, R1-22, R1-23, R3-03, R3-07–R3-10, R4-16, R6-04, R6-05.
Work: a methods table (ns_exponent, Gensim version and patch, controlled optimizer, init, batch, seeds, workers); one paragraph that names each correction, what it changed, and which results it affects; retitle sections that narrate the project's history; a reproduction map from figure to script; a decision on the extension-hash gate; and the archive DOI, NetScience source and license, acknowledgments and funding, and author contributions.

### I-6 (P1; T-011, T-012, T-013): the interventions don't isolate exposure from context specificity, and the inverted-U is thinly resolved

Source: R1-15, R2-06, R2-11, R2-12, R4-08, R4-11, R4-12, R4-27, R5-03, R6-03, R6-07.
Analyses: report realized versus intended mixed targets (zero-positive share, mass by degree, marginal TV); control or report output-side updates under the exposure intervention; per-graph shape counts and a resolution check with finer or equal-count bins; define bin, peak, turnover, and the criteria in the main text.
Issue #22 gets a comment instead of a duplicate: R2-20 finds the β = 1, λ = 1 cell not fully mixed (realized KL 0.359, slope −0.117).

### Not filed this round

T-016 (notation and abstract), T-017 (structure), T-019 (scope), T-020, T-021: see the reviews; they can wait until I-1 settles the paper's shape.
