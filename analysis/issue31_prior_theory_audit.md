# Issue 31: attribution of population geometry and degree cancellation

Checked 2026-11-02 against the published primary PDFs, including visual checks of
the relevant formulas. Scope: the two attribution questions in
[issue #31](https://github.com/govgandhi/alt_means_sans_k/issues/31), not a complete
novelty audit or a revalidation of the manuscript's experiments.

## Decisions

1. Credit Davison, Morgan and Ward directly beside `cor:alpha-one`: their
   Lemma S9 already states block constancy at unigram exponent one. Keep our
   corollary as the identity for the specified finite population-walk proxy,
   ordered objective and minimal exact factors, without claiming cancellation
   itself is new.
2. Add Davison and Austern as earlier normalization theory. Their original
   nonedge-only sampler has a finite-density correction, so do not call its
   target exactly identical to ours.
3. Credit Jin's SCORE for the raw-adjacency population eigenvector rays. Split
   that table row from stationary pairs so the citation does not imply SCORE
   derived our walk formula.
4. Describe the theory as a comparison using established foundations, including
   in the abstract, introduction and contribution paragraph. No equations,
   theorem assumptions, empirical results or personal drafts are changed.
   Also credit Davison, Morgan and Ward's Proposition S24 for the existing
   statement that minimal balanced factors are unique up to orthogonal rotation.
5. Use the established theory as a benchmark for the unresolved turnover
   mechanism. The README and manuscript outlook prioritize separating
   realized-graph sparsity from training dynamics, with learning-rate controls
   before transfer to natural sequences. Mark the older strategy note as
   historical. These are research priorities, not new results or claims of
   novelty; the learning-rate evidence remains in PR #36 for YY's review.

## Primary evidence and scope

### Davison, Morgan and Ward (NeurIPS 2024)

[Published PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/015a8c69bedcb0a7b2ed2e1678f34399-Paper-Conference.pdf),
[official record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/015a8c69bedcb0a7b2ed2e1678f34399-Abstract-Conference.html),
DOI: 10.52202/079017-0022.

- **p.25, Lemma S9, Eqs. S49–S51:** the unconstrained population target is
  block-constant when the negative-sampling exponent is one. This is direct
  precedent, not merely a related recovery result.
- **pp.6–7, scenario (iii), Theorems 2–3:** score and balanced minimal-dimensional
  embedding convergence require their asymptotic assumptions, including bounded
  degree propensities and sufficient density. Our algebraic proxy identity is
  not an additional convergence theorem.
- **p.21, Eq. S9; p.47, Appendix F:** negative weights combine both orientations.
  At other exponents their dependence involves both propensities, matching the
  structure of our symmetric-score discussion rather than our ordered
  context-only log-degree term. Sampling constants and walk definitions should
  not be equated across the papers.
- **p.41, Proposition S24:** directly supports the orthogonal freedom of minimal
  balanced factors already used in our gauge section.

### Davison and Austern (JMLR 2023)

[Published PDF](https://www.jmlr.org/papers/volume24/21-0841/21-0841.pdf),
[official record and BibTeX](https://www.jmlr.org/papers/v24/21-0841.html).
Verified metadata: volume 24, article 138, pp.1–120, 2023.

- **p.17, Eqs.16–17:** the unconstrained logistic score is a log weight ratio;
  their unigram example normalizes by both graphon degree functions.
- **pp.27–28, Algorithm 4, Lemma 27, Remark 28:** positives are adjacent walk
  pairs and negatives are restricted to nonedges. The resulting
  `1 - rho_n W(x,y)` correction disappears when that restriction is omitted.
- **Our inference:** substituting a degree-corrected block kernel cancels the
  product of propensities, but the nonedge correction retains propensity
  dependence at finite density. Thus this is related normalization precedent,
  not an exact finite-density restatement of our ordered-window corollary.
- **pp.14, 18–19:** symmetric definite/indefinite representations and their
  convergence assumptions differ from unrestricted ordered factors.

Source caution: Eq.17's walk-length constant differs from Lemma 27 and the
general expression on p.29. Remark 28 also labels the affected weight differently
from Lemma 27. The comparison uses the displayed sampling weights and the
degree-dependent terms; no disputed global constant is imported into our text.

### Jin (Annals of Statistics 2015)

[Published PDF hosted by the author](https://www.stat.cmu.edu/~jiashun/Research/Year/SCORE.pdf),
DOI: [10.1214/14-AOS1265](https://doi.org/10.1214/14-AOS1265).
Verified metadata: volume 43(1), pp.57–89, 2015.

- **Published p.72, Lemma 2.1:** each population eigenvector is proportional
  within a block to the node's propensity. Collecting the eigenvectors gives
  `u_i = theta_i c_{g_i}` in our notation.
- **pp.70 and 74, Eq.2.2 and Sec.2.5:** ratios to the leading eigenvector remove
  the node multiplier.
- **p.71, Eqs.2.4–2.5:** the exact population operator includes its diagonal;
  the loopless mean is `Omega - diag(Omega)`, consistent with our proxy
  distinction. The lemma's assumptions include simple nonzero eigenvalues;
  sample-graph consistency has additional conditions. Do not attribute exact
  sample-adjacency rays or the stationary-pair calculation to SCORE.

## Source reproducibility

Download the PDFs at the URLs above and extract with `pdftotext -layout`.
The relevant NeurIPS and JMLR printed page numbers equal PDF page numbers;
SCORE's printed p.72 is PDF page 16. Render equations with
`pdftoppm -f PAGE -l PAGE -singlefile -png INPUT.pdf OUTPUT` to check extraction.
SHA-256 hashes of the PDFs inspected:

| Source | SHA-256 |
|---|---|
| NeurIPS 2024 | `26274fb2810beb3024ccb25e3a787125b174684aa79f4d71e34a6147b75f32e1` |
| JMLR 2023 | `cbe214e65770cac7795f4cd3288df3e072e81dd460d9f10ffe72c09cf7eac323` |
| SCORE 2015 | `9d328a3631b18e8705a9549682a5158ca93d88c2f067eb8cd671f51da7618f6e` |

The audit supports the attribution corrections. It does not establish that our
remaining formulation details are globally novel or identify the mechanism of
norm turnover. PRs #35 and #36 remain separate and await YY's review.

Build the revised manuscript from `paper/current` with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
