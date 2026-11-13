# Referee report: Reviewer 2 (statistics and experimental methods)

**Manuscript:** "Degree Information and Norm Turnover in Random-Walk Network Embeddings," G. Gandhi, S. Kojaku, and Y.-Y. Ahn. Submitted to Physical Review E.

**Version reviewed:** commit `c958c528` (`paper/current/main.tex` and `main.pdf`, 15 pp.; `supp.tex` and `supp.pdf`, 3 pp.).

**Scope:** Sec. XI (numerical experiments, `main.tex` l.955–1173), the parts of the abstract, introduction, Discussion, and Conclusion that rest on it, Supplemental Sec. S1–S2, and the stored outputs, protocols, code, and git history. I did not evaluate the population theory (Secs. II–X) and retrained nothing. Every number in this report was recomputed from stored outputs with the scripts listed in Sec. 10. Line numbers refer to `main.tex`; page numbers to `main.pdf`.

**Recommendation:** major revision. **Confidence:** 4/5.

## 1. Summary

The numerical part makes four claims.
(i) In a custom 24-dimensional mini-batch SGNS trainer on five configuration-model graphs, fixing center exposure at m_i ∝ π_i^β produces decreasing (β=0), inverted-U (β=1/3), and increasing (β ≥ 2/3) norm–degree curves, and mixing each transition row toward its context marginal flattens the equal-exposure curve (l.995–1028, Fig. 1).
(ii) In Gensim word2vec, turnover persists without subsampling under the original learning-rate decay and is absent at 20 epochs under a constant rate (l.1106–1139, Fig. 5).
(iii) In exact expected-loss fits at fixed exposure and fixed stationary noise, the upper-degree norm decline is weaker at window 10 than at window 5 (l.1030–1062, Figs. 2–3).
(iv) An extremely randomized tree regressor on log realized pair count and empirical context KL, frozen before training on direct Barabási–Albert, Holme–Kim, and NetScience targets, passes four declared criteria (median peak error 0 bins, 8/9 peaks within one bin, pooled node Spearman 0.881, nondecreasing observed peaks) and ranks nodes within curves better than a post hoc degree-and-threshold baseline (median 0.842 versus 0.293) (l.1153–1173, Fig. 6, Table S1).

I reproduced every headline number I checked (Sec. 5). The problems lie in what these numbers can show. The declared criteria cannot separate the frozen predictor from trivial predictors. The one metric that does separate it measures mostly the ranking of nodes that share a degree, and a predictor built from graph structure alone matches it. The "prospective" freeze has no evidence of temporal precedence beyond self-report. The headline numbers come from an implementation that the repository itself marks as bug-affected, and the main text does not say so. Two figures carry mislabeled error bars. The qualitative shape labels that carry the mechanistic narrative are never tested.

## 2. Strengths

1. Provenance is unusually complete and internally consistent. All four hashes in `results/prospective_validation/freeze_manifest.json` match the files; the 23,685 frozen predictions equal those stored with the outcomes (0 mismatches); refitting the frozen ExtraTrees with scikit-learn 1.8.0 reproduces the predictions exactly (maximum absolute difference 0); the plan files of the later controls hash their protocols, and every one of those hashes matches the current files.
2. Much of the exploratory history is disclosed: the degree-bin interpolation was chosen after the declared surface failed (l.1027–1028), the tree model after initial experiments (l.1199), the baselines are post hoc (l.1171; Supp. S1), and criterion 4 is predictor-independent (Supp. S2).
3. The corrected Gensim and training-trajectory controls are well designed (paired corpora, initializations, and negative tables; an exact five-epoch bridge), and their main claims survive a selection-corrected, per-graph check (comment 12).
4. The decreasing equal-exposure branch replicates in a second optimizer: the exact expected-loss L-BFGS fits at uniform center share and q = π (Fig. 2) decline at windows 5 and 10, and the 5→10 weakening holds in 10/10 graph-starts and after RMS normalization.

## 3. Weaknesses, ranked

**W1. The declared criteria have almost no power against trivial alternatives** (comments 1–2). A predictor that outputs the calibration mean norm for each threshold plus a 0.01 bump at the third degree bin of every curve, with no exposure, KL, model, or family information, passes all four criteria (median error 1, 7/9 within one bin, pooled ρ = 0.590). Uniform random peak guesses pass both peak criteria with probability 0.40. The abstract (l.63–64) and Conclusion (l.1213) still present these criteria as the evidence.

**W2. The discriminating metric does not measure the turnover shape** (comment 3). About 80% of the within-curve rank variance of observed norms lies within exact-degree classes, the best possible function of degree reaches a within-curve median Spearman of only about 0.37, and a post hoc predictor built from the adjacency matrix alone (log degree, log threshold, structural one-step KL) matches the frozen predictor's within-curve median of 0.842. The 0.842 versus 0.293 contrast therefore shows that neighbor-degree composition predicts the norms of equal-degree nodes. That finding is interesting, but it does not show that realized exposure and empirical KL predict the turnover curve.

**W3. "Prospective" and "hash-locked" rest on self-attestation** (comment 4). Protocol, manifest, predictions, outcomes, summary, and the manuscript text that already reports the outcomes all first appear in one commit made about 21 h after the recorded freeze. The evaluation code that defines bins and peaks is not hashed and is absent from both protocol files.

**W4. The headline prospective numbers come from the pre-correction pipeline without disclosure** (comment 5). The repository marks those outcomes as predating the repeated-visit, negative-table, and BLAS corrections; Sec. XI.G and the abstract do not, while Sec. XI.E labels the Gensim sweep "uncorrected."

**W5. Mechanistic language outruns the tests** (comments 9–12). The declared response surface failed badly on held-out data (R² = −2.19, node ρ = −0.67; unreported), its own exposure derivative is negative on 28% of calibration rows, the controlled trainer and Gensim respond to subsampling in opposite directions, the interventions are coupled through q, the λ = 1 endpoint is flat by construction, and no inverted-U, decreasing, or increasing label is tested with seed-level inference.

**W6. Uncertainty is misreported or unrecoverable** (comments 6–8). Figs. 1 and 6 show node-level standard errors under captions that say "standard errors across five graph seeds"; node-level data for Figs. 1 and 4 are not archived; the prospective curves have only 4–6 occupied bins, and "median error 0" turns into 1 under 5 of 15 single-replicate deletions.

**W7. Transfer is mostly near-distribution** (comment 13). Two of three target families share the calibration graphs' degree-sequence model (n = 600, m = 3). NetScience is the only out-of-support target; it is a single graph, and it hosts the large misses.

## 4. Detailed comments

### Comment 1. The declared criteria cannot discriminate (R2-01)

- **Location.** Abstract l.63–64; Sec. XI.G l.1165–1169; Conclusion l.1213; `analysis/prospective_validation_protocol.md` l.43–53; Supp. S2.
- **Problem.** Criteria 1–3 (median absolute peak error ≤1 bin, ≥2/3 of peaks within one bin, pooled ρ ≥ 0.5) are passed by predictors that contain none of the proposed mechanism, and criterion 4 is the same for every predictor. The protocol states that passing "supports qualitative transport of the two-force mechanism" (l.52). That inference does not follow.
- **Evidence** (`r2/peak_null.py`, `r2/null_predictor.py`).
  - The curves have 4 (BA), 5 (Holme–Kim), and 6 (NetScience) occupied bins. With independent uniform guesses over occupied bins, P(median ≤1 and ≥6/9 within one) = 0.397 by exact enumeration over 1.7×10⁶ combinations.
  - Predicting the same bin index in every curve passes criteria 1–2 for bins 1, 2, and 3 (zero-based, as in the code; 6/9, 7/9, and 7/9 within one bin; median 1).
  - Predicting the target peak at the bin nearest the calibration curve's peak degree for the same threshold (no model and no node features) gives 8/9 within one bin and median error 1, the same within-one count as the frozen predictor.
  - A threshold-only predictor (the calibration mean raw norm for each threshold, constant within threshold) has pooled ρ = 0.586.
  - The calibration mean per threshold plus a 0.01 bump at bin 2 of each curve passes all four criteria (median 1, 7/9, ρ = 0.590).
  - The frozen predictor does beat uniform guessing on exact hits: P(median error = 0) = 0.022 and P(≥8/9 within one) = 0.043 under uniform guesses. That comparison, against the trivial predictors above and with uncertainty, is the one worth reporting.
- **Fix.** Stop citing "passes the declared criteria" as evidence in the abstract and Conclusion. In the main text, report the frozen predictor beside the constant-bin, calibration-peak-transfer, threshold-only, degree-and-threshold, and graph-only predictors on discriminating metrics (exact hits, peak error in log-degree units, level-centered binned-shape error, within-degree ranking), each with replicate-bootstrap intervals. State in Sec. XI.G that the declared criteria turned out to be weak.

### Comment 2. The pooled Spearman correlation mixes curve levels with node ranking (R2-02)

- **Location.** Abstract l.63; l.1166; `supp.tex` l.31; `scripts/prospective_norm_turnover.py:747`.
- **Problem.** The pooled correlation runs over all 23,685 node-replicate-threshold rows. Thresholds shift the overall norm level, so between-curve differences inflate it. The supplement says so; the abstract still headlines 0.881.
- **Evidence** (`r2/prospective_repro.py`, `r2/peak_bootstrap.py`). Between-curve variation accounts for 46% of the rank variance of observed norms. Centering predictions and outcomes within each curve lowers ρ from 0.881 to 0.843. The threshold-only predictor reaches 0.586, and degree-and-threshold reaches 0.652 with a within-curve median of 0.293. A replicate-cluster bootstrap (replicates resampled within family with thresholds kept together, B = 2000) gives 0.881 [0.878, 0.883] for the pooled value and 0.842 [0.837, 0.848] for the within-curve median. These intervals are conditional on the three families; across families the within-curve values range from 0.506 to 0.935.
- **Fix.** Report the within-curve median and range and the centered pooled ρ in the abstract and Sec. XI.G, with replicate-bootstrap intervals, and state that the nine curves are three families crossed with three within-graph conditions.

### Comment 3. Within-curve ranking is mostly ranking within degree classes (R2-03)

- **Location.** Abstract l.64; l.1171–1173; Supp. S1 and Table S1; Fig. 6 caption ("two-force norm predictor").
- **Problem.** The within-curve Spearman correlation is dominated by variation among nodes of equal degree, which carries no information about the norm–degree curve, and the degree baseline is capped by construction. The comparison is matched in tuning budget (none) but unequal in information, and its estimand differs from turnover.
- **Evidence** (`r2/baseline_refit.py`, `r2/noise_tracking.py`, `r2/graph_only_baseline.py`, `r2/shape_compare.py`).
  - Exact-degree classes account for a median 20% (range 12–39%) of within-curve rank variance of observed norms. After exact-degree class means are removed, the frozen predictor's within-curve rank correlation rises to a median 0.877.
  - Within single degree classes (for example the 1,199 degree-3 BA node-replicates), Spearman(prediction, outcome), averaged over the classes with at least 50 node-replicates, is 0.75–0.94 in eight of nine curves and 0.38 for NetScience at t = 3×10⁻³.
  - The best possible degree-only function (the observed mean norm at each exact degree, cross-fitted across replicates) reaches a within-curve median of 0.366 (range 0.189–0.574); fitted in-sample it reaches 0.376.
  - I regenerated all target graphs with the repository's generators and seeds (networkx 3.6.1; degrees match the stored ones exactly). Within degree classes, the structural one-step KL D(T_i‖π) explains a median 86–99% of the variance of empirical KL (range across curves) and correlates with norms at a median ρ = 0.78–0.92 (0.45 for NetScience at t = 3×10⁻³). The corpus-sampling residual of empirical KL correlates at −0.12 to +0.31, and for NetScience only 5–9% of norm variance is within-node across replicates. Graph structure drives the ranking; realized-corpus noise contributes little.
  - A post hoc graph-only ExtraTrees model (log degree, log threshold, structural KL; frozen settings with all features considered at each split; fit on calibration graphs only) gives a within-curve median of 0.842 (range 0.334–0.922), pooled ρ = 0.848, median peak error 0, and 7/9 peaks within one bin.
  - At the level that does concern turnover, the frozen predictor beats degree-and-threshold: after level centering, its binned-curve RMSE is lower in 7/9 curves (median 0.054 versus 0.142). The paper reports no shape-level metric.
- **Fix.** Report within-degree-class association as its own analysis; it is the cleanest observational evidence for the specificity hypothesis in the paper, with the caveat that neighbor-degree composition is not randomized. Report a level-centered binned-shape metric with intervals, add the graph-only comparator, and drop "two-force" from the Fig. 6 caption unless a comparison isolates information the graph-only model lacks.

### Comment 4. Temporal precedence of the freeze is self-attested (R2-04)

- **Location.** l.1155, l.1163; Conclusion l.1213 ("Hash-locked"); Data Availability l.1218; `results/prospective_validation/{protocol.json,freeze_manifest.json}`; `scripts/prospective_norm_turnover.py:436, 458–505, 586–588, 677, 747`.
- **Problem.** Hashes prove the files are unchanged since the manifest was written. They cannot show when it was written, and the git history gives no ordering.
- **Evidence** (`git log`, `shasum -a 256`).
  - `protocol.json` has `created_utc` 2026-08-13T20:30:54Z and the manifest `frozen_utc` 20:31:44Z. All four manifest hashes match, `outcomes_sha256` in `summary.json` matches `outcomes.jsonl`, and the NetScience GML matches `netscience_sha256`.
  - The protocol markdown, protocol JSON, manifest, calibration, predictions, outcomes, summary, figure, and a manuscript abstract already reporting 0.881 all first appear in commit `5902c3c` (2026-08-14 13:33:52 −0400), about 21 h after the recorded freeze. `outcomes.jsonl` and `summary.json` carry no timestamps.
  - `"target_outcomes_present_at_freeze": False` is a literal (l.677), guarded only by a check that `outcomes.jsonl` does not exist in the same output directory (l.586–588). It cannot exclude target runs elsewhere.
  - The manifest hashes neither the evaluation code nor `frozen_prediction_summary.json`. Neither protocol file defines the degree binning (six requested log-degree quantile bins, l.436, which collapse to 4–6 occupied bins), the peak (argmax of bin means pooled over five replicates, l.458–505), or the pooling unit of the Spearman criterion (all rows, l.747). These definitions exist only in code committed together with the outcomes.
  - Mitigating facts: the peak results are insensitive to binning (comment 6), and I found no trace in the repository of target-family runs before the freeze.
  - The same single-commit pattern holds for the subsampling-origin and trajectory protocols (`b3bd902`) and the window amendment (`baf7de5`). Their plan files hash the protocols, which shows the protocols were unchanged after the plans were written; it cannot show that the plans preceded the outcomes.
- **Fix.** Replace "hash-locked" with wording that matches this evidence, add one sentence on the commit history to Sec. XI.G, and state the evaluation definitions in the Methods. For any future test, hash the evaluation code with the protocol and push or timestamp both externally (public remote, OSF, or an RFC 3161 service) before target training.

### Comment 5. The headline numbers come from the pre-correction pipeline (R2-05)

- **Location.** Abstract l.62–64; Sec. XI.G l.1153–1173; compare l.1078 and the Fig. 4 caption; `results/prospective_validation/README.md` l.3–7; `results/README.md`.
- **Problem.** The repository states that the original prospective outcomes "predate the repeated-visit counter, integer-token negative-table and BLAS corrections" and that "their original numbers must not be described as results of the corrected code." The main text reports them without that qualification; only l.1172 points to "corrected sampling comparisons" in the supplement. The Gensim sweep, affected by the same corrections, is labeled "original, uncorrected" (l.1078). The corrected reruns (pooled ρ 0.875 and 0.873) are retrospective because the outcomes were already known.
- **Fix.** State in the abstract and Sec. XI.G that the frozen evaluation used the pre-correction implementation, and give the corrected retrospective values beside the original ones.

### Comment 6. Peak location is coarse and fragile (R2-06)

- **Location.** l.63, l.1148, l.1166–1167; Fig. 6 (p. 13); `scripts/prospective_norm_turnover.py:433–470`.
- **Problem.** Each peak is the argmax of 4–6 pooled bin means. With so few bins, "within one bin" covers half to three quarters of the four BA bins, and several argmaxes are decided by differences smaller than their standard error.
- **Evidence** (`r2/degree_bins.py`, `r2/prospective_repro.py`, `r2/peak_bootstrap.py`, `r2/calib_and_se.py`, `r2/split_peak_test.py`).
  - BA curves have four occupied bins with centers 3.29, 5.00, 6.77, and 14.31; bin 0 merges degrees 3 and 4 and holds 59% of node-replicates. Holme–Kim curves have five bins and NetScience six. The manuscript never states this.
  - The gap between the two highest observed bins is below one node-level SE in 5/9 curves; for NetScience at t = 3×10⁻³ it is 0.001 (0.04 SE).
  - Under the replicate bootstrap, the median error is 0 in 63% of resamples and 1 otherwise; the within-one count is 7–8 (95% interval) and exact hits 3–7. Deleting any one of replicates 201, 202, 203, 303, or 305 changes the median error to 1.
  - Per replicate (45 curves on the pooled-curve bin edges), 39/45 peaks are within one bin and 24/45 exact.
  - Choosing the peak on two replicates and estimating it on the other three, the Holme–Kim t = 3×10⁻⁴ interior peak, counted as an exact hit, has a positive held-out contrast in 5/10 splits (median +0.009).
  - The within-one count is robust: 8/9 for 5–10 requested quantile bins and for 6 or 8 equal-width log-degree bins, 9/9 with 3–4 bins, and 7/9 with 12.
  - Fig. 6(d) overplots coincident points (for example the BA peaks at t = 10⁻³ and 3×10⁻³), so fewer than nine markers are visible.
- **Fix.** Report occupied bins and bin counts per curve, peak error in log-degree units with replicate-bootstrap intervals, the per-replicate error distribution, and the binning sensitivity table; jitter or tabulate Fig. 6(d).

### Comment 7. Error bars are mislabeled in Figs. 1 and 6 (R2-07)

- **Location.** Fig. 1 caption l.1012 (p. 9); Fig. 6 caption l.1146 (p. 13); `scripts/simulate_exact_center_exposure.py:308` through `plot_binned`, called at `scripts/simulate_turnover_surface.py:946–1008`; `scripts/prospective_norm_turnover.py:454, 806–815`; identical code at `5902c3c`.
- **Problem.** Both captions say "standard errors across five graph seeds." Both figures plot SD/√N over all pooled node-replicate rows in a bin. Fig. 4 discloses exactly this for the historical sweep (l.1095); Figs. 1 and 6 do not.
- **Evidence** (`r2/calib_and_se.py`). For Fig. 6, seed-level SEs (SD of the five replicate bin means divided by √5) are 0.58–1.90 times the plotted node-level SEs in the synthetic families and 0.13–0.60 times in NetScience, where the replicates share one graph and measure only walk and optimizer variability.
- **Fix.** Relabel Fig. 1 now. Recompute Fig. 6 with seed-level SEs from `outcomes.jsonl` and mark the NetScience bars as algorithmic variability on one graph.

### Comment 8. Node-level data behind Figs. 1 and 4 are not archived (R2-08)

- **Location.** Data Availability l.1218; `.gitignore:7` (`*.csv`); `results/turnover_surface/` and `results/gensim_threshold_sweep/` (summaries and figures only).
- **Problem.** The `node_results.csv` files of both historical runs are ignored by git. The β = 1/3 inverted-U (a rise of 0.093, 2.9% of the lowest-bin norm, with node-level ρ = 0.023 against degree), its consistency across the five graphs, and both figures' error bars therefore cannot be recomputed. The package versions of these runs are not recorded, so a rerun is not guaranteed to reproduce them.
- **Fix.** Archive the node-level files (allowlist them as was done for `results/subsampling_origin/outcomes.csv`) or regenerate them in the pinned environment; otherwise qualify the Data Availability statement.

### Comment 9. The declared response surface failed more severely than reported (R2-09)

- **Location.** Abstract l.65; l.940–953 (response model; sign premise at l.952); l.1024–1028; `results/turnover_surface/summary.json` (`held_out_targets`, `response_surface`); `scripts/simulate_turnover_surface.py:28–31, 685–711`.
- **Problem.** The text reports the failure only as a misplaced peak. The fitted surface also contradicts the sign premise of the response model behind the abstract's interpretive sentence.
- **Evidence** (`r2/controlled_check.py`).
  - Held out at t = 3×10⁻⁴: R² = −2.19 and Spearman(observed, predicted) = −0.673. At t = 10⁻³ and 3×10⁻³, R² = 0.495 and 0.708, but both observed curves are monotone, and the surface placed the peak in the highest bin under all three laws.
  - The fitted ∂ν/∂log m is negative on 27.9% of calibration rows, while ∂ν/∂KL is positive on 99.93%. A falling limb can then come from a negative exposure response, which the two-force reading (l.952) excludes by assumption.
  - The interpolation that succeeded is specific to degree bins, so it conditions on degree directly and is not a response in the two scalars.
  - The only records of the declaration are the script docstring and the run's own `analysis_plan` field, and the docstring already says the surface "fails at the strongest subsampling threshold" (l.30–31). The declaration cannot be checked.
- **Fix.** Report R² and ρ for all three held-out laws and the derivative-sign fractions. Either delete the abstract sentence at l.65 or restate it as a prediction that an observation could contradict, for example the sign of ∂ν/∂log m above a stated exposure, estimated on calibration conditions and checked on a new one.

### Comment 10. The controlled trainer does not reproduce Gensim's threshold dependence (R2-10)

- **Location.** Abstract l.59–61 and l.65; Intro l.100–102; l.1104; l.1113–1120; Discussion l.1185–1189.
- **Problem.** In the controlled trainer, the three held-out retention laws give an inverted-U only at the strongest subsampling (peaks at degree 7.73, 16.04, and 16.04 for t = 3×10⁻⁴, 10⁻³, 3×10⁻³). In Gensim the strongest subsampling gives a monotone decrease and the weaker settings give inverted-Us (peaks 2.98, 6.00, 7.73), and a constant learning rate removes the unfiltered turnover at 20 epochs. The controlled curves show that the exposure law can produce each shape in one trainer. They do not account for the Gensim curves, and the sequence of abstract sentences (l.59–61, l.65) invites that reading.
- **Fix.** State in the abstract and Discussion that the two pipelines respond to subsampling in opposite directions, and confine "consistent with" statements to the trainer in which the pattern was observed.

### Comment 11. The interventions are coupled, and one endpoint is flat by construction (R2-11)

- **Location.** Abstract l.59–60; l.964–987; l.1017–1022; Fig. 1(d) panel title "Mixing removes the falling limb"; `results/realized_graph_audit/allocations.json.gz`.
- **Evidence** (`r2/confound_kl.py`, `r2/controlled_check.py`).
  - The exposure intervention changes specificity. The realized slope dKL/d log d is −0.825 at β = 0 and −0.993 at β = 1 with λ = 0 (graph ranges −0.841 to −0.795 and −1.003 to −0.975). The text says that q moves (l.985–987) but never quantifies the consequence.
  - At β = 0 and λ = 1, every center receives the same integer allocation of contexts drawn from the same q (realized KL is exactly 0), so center norms are exchangeable and the flat curve follows from the design. The empirical content of the mixing series lies at λ = 1/3 and 2/3.
  - At λ = 1/3 the absolute endpoint decline shrinks from 1.210 to 0.824, but the relative decline is unchanged (−34.5% to −35.2%) and node ρ against degree goes from −0.758 to −0.782. Flattening in relative terms appears only at λ = 2/3 (−5.8%). From λ = 0 to λ = 1 the lowest-bin norm falls by 73% and the highest-bin norm by 59%. The paper separates scale from shape for the window comparison (l.1037, l.1058) but not here.
  - "Equal exposure" fixes center updates only; context-side and negative-sampling exposure still follow q.
- **Fix.** Report the KL slope for every (β, λ) cell and relative or RMS-normalized contrasts for the λ series. Rephrase abstract l.59–60 to say "equal center exposure," that partial mixing weakens the absolute decline, and that full mixing equalizes all centers by construction.

### Comment 12. Shape labels are untested, and the classifier has a high false-positive rate (R2-12)

- **Location.** l.998, l.1083, l.1110, l.1117–1118, l.1165–1167; `curve_metrics` in `scripts/simulate_turnover_surface.py:643–661` and `peak_metrics` in `scripts/prospective_norm_turnover.py:458–470`.
- **Problem.** A curve is called an inverted-U when the argmax of 6–8 bin means is interior and exceeds both endpoints. For a flat curve with independent noise and six bins this happens about two thirds of the time. No margin, test, or seed-level inference is applied to any shape label in the paper.
- **Evidence** (`r2/trajectory_check.py`, `r2/readout_agreement.py`, `r2/split_peak_test.py`).
  - At epoch 0, before any training, raw norms are flagged as interior maxima in 2/5 graphs under every schedule and threshold (margins 0.0002 and 0.0007).
  - At β = 0 and λ = 1, flat by construction, the balanced readout is flagged as an inverted-U while the raw readout is not, which contradicts l.991 ("both give the same qualitative results").
  - Several claims do survive a selection-corrected test. In the corrected Gensim runs (`results/subsampling_origin/outcomes.csv`), choosing the peak on a random half of each bin's nodes and testing on the other half, the interior maximum holds in at least 193 of 200 splits for every graph at t = 0, 10⁻³, and 3×10⁻³, and in none at 3×10⁻⁴. The trajectory contrast is equally clear: at epoch 20 the constant-rate curves rise by at least 0.15 between every pair of adjacent bins, and the original-schedule curves drop by 0.18–0.26 from peak to highest bin, in all five graphs.
- **Fix.** Apply one prespecified test to every shape claim: choose the breakpoint on held-out data and estimate both slopes on the rest (a two-lines test in the sense of Simonsohn, AMPPS 2018), with graphs as the unit of inference, and report how many of the five graphs show each shape. The β = 1/3 claim needs its per-seed data (comment 8) before it can be tested.

### Comment 13. Transfer is near-distribution for two of three families (R2-13)

- **Location.** Abstract l.62; Intro l.103–104; l.1160–1161; Discussion l.1190 and l.1194.
- **Evidence** (`r2/support_check.py`).
  - The calibration graphs are configuration models with Barabási–Albert degree sequences (n = 600, m = 3), and the direct BA and Holme–Kim targets use the same n and m. At most 0.2% of their rows fall outside the calibration feature box, and 6.7–8.4% lie in sparse regions (fewer than 40 calibration rows, one leaf, within a box spanning 5% of each feature's range).
  - In NetScience, 26.6% of nodes have degree ≤2 (calibration: 0.4%). At t = 3×10⁻⁴, 6.0% of its rows lie outside the calibration box and 73.5% in sparse regions. The two-bin miss and the +0.37 to +0.51 bias in mean norm occur here.
  - The five NetScience replicates share one graph, so they cannot estimate between-graph uncertainty for empirical networks.
- **Fix.** Describe the synthetic targets as families sharing the calibration degree model, add covariate-shift diagnostics, and report NetScience as a single out-of-support case.

### Comment 14. The corrected Gensim peak movement is partly a pooling effect (R2-14)

- **Location.** Intro l.102; l.1082 and l.1102. This is related to tracked issue #21 but distinct from it: it concerns how the peak moves, not where turnover appears.
- **Evidence** (`r2/gensim_corrected_check.py`). In the corrected sweep the pooled peak moves from 2.98 to 6.00 to 7.73, as in the historical sweep. Per graph, the move from the lowest bin to an interior bin between t = 3×10⁻⁴ and 10⁻³ happens in 5/5 graphs; the move between 10⁻³ and 3×10⁻³ happens in 1/5 (graph 2, whose top-two margin at 3×10⁻³ is 0.009). The corrected revalidation report documents the same pooling problem for the Holme–Kim targets (`results/negative_sampling_revalidation/README.md` l.75).
- **Fix.** Report per-graph peak bins beside the pooled curves, and limit "peak movement" to the transition from 3×10⁻⁴ to 10⁻³.

### Comment 15. Window comparison: figure subset and the timing of "predefined" (R2-15)

- **Location.** l.1035–1043; Fig. 2 (p. 10); `analysis/window_optimization_amendment.md` l.3–7 and l.59–60; `results/window_training_continuation/report/kl_and_trained_norms.png`.
- **Problem.** The amendment was written after the initial 500-iteration endpoints had been seen and declares that "the primary figure places context KL and trained balanced norms against degree side by side for every window." The manuscript shows windows 5 and 10 only. The stability gates described as "predefined" (l.1038, Fig. 2 caption) were defined in that amendment.
- **Evidence** (`r2/window_check.py`). Recomputed from the stored per-fit curves, the upper gap is −0.3258 ± 0.0155 at window 5 and −0.2173 ± 0.0123 at window 10, weaker in 10/10 graph-starts, and −0.318 versus −0.211 when restricted to the seven stable window-5 fits. The declared all-window figure shows relative norm declines of +1.0% (window 1, a rise), −37.9% (window 2), −12.3% (window 5), and −10.0% (window 10), while relative KL declines increase monotonically.
- **Fix.** Put the amendment's declared figure in the SM, state when the gates were fixed, and keep the window 5 versus 10 conclusion, which holds.

### Comment 16. Calibration adequacy pools thresholds (R2-16)

- **Location.** l.1158.
- **Evidence** (`r2/calib_and_se.py`). The leave-one-seed-out R² = 0.828 and ρ = 0.932 reproduce exactly. Within thresholds, R² = 0.699, 0.755, and 0.694 and ρ = 0.866, 0.923, and 0.841.
- **Fix.** Report the within-threshold values.

### Comment 17. Agreement between norm readouts is overstated (R2-17)

- **Location.** l.991 and l.1196.
- **Evidence** (`r2/readout_agreement.py`, `r2/trajectory_check.py`). Besides the β = 0, λ = 1 disagreement (comment 12), the centered-score readout computed in the same run goes unreported; for the held-out t = 3×10⁻⁴ law it peaks at degree 6.00 instead of 7.73. After training, raw and balanced classifications agree throughout the trajectory runs; all 16 disagreements occur at epoch 0.
- **Fix.** Qualify both sentences, and report the third readout or say why it is omitted.

### Comment 18. Multiplicity and selective emphasis (R2-18)

- **Location.** Abstract l.59–65; Sec. XI throughout.
- **Problem.** Sec. XI rests on several hundred descriptive shape calls (16 grid cells × 3 readouts, 3 held-out laws, 6 historical Gensim curves, 40 corrected curves, 480 trajectory classifications, 40 window fits, and 9 prospective curves × 5 predictors × 3 arms × 2 readouts) without tests or any accounting for multiplicity. The abstract's inverted-U at β = 1/3 is the only interior-peaked raw curve among the 16 grid cells, with a 2.9% rise.
- **Fix.** Add an SM table listing every condition with its shape call, margin, and number of graphs, and state that all calls are descriptive unless tested as in comment 12.

### Comment 19. Reproducibility conventions (R2-19)

- **Location.** `results/prospective_validation/protocol.json`; Methods.
- **Problem.** The protocol and manifest record no package versions for the original prospective run (the exact refit suggests scikit-learn 1.8.0). The prospective evaluation requests six quantile bins while the controlled, Gensim, and trajectory analyses request eight. "Spearman with degree" is node-level in `turnover_surface/summary.json` and bin-level in `gensim_threshold_sweep/summary.json`.
- **Fix.** Document the environment and all three conventions in the Methods.

### Comment 20. Note for tracked issue #22 (R2-20)

New evidence only: the β = 1, λ = 1 cell is not fully mixed after integer allocation. Its realized KL has mean 0.359 and slope dKL/d log d = −0.117 across the five graphs (`r2/confound_kl.py`), and its node-level ρ against degree is −0.344 while its endpoint contrast is +6.4%. Any statement about exposure under full mixing needs to account for this residual specificity gradient.

## 5. Number reproduction

| Claim | Location | Recomputed | Source | Match |
|---|---|---|---|---|
| Pooled node-level Spearman 0.881 | Abstract l.63; l.1166 | 0.8806 (centered within curve: 0.843) | `results/prospective_validation/outcomes.jsonl`, 23,685 rows | Yes |
| Median absolute peak error 0 bins | l.63; l.1166 | 0 (errors 1, 0, 0, 0, 0, 1, 2, 0, 1); bootstrap P(=0) = 0.63 | `outcomes.jsonl` with `binned_curve` (6 requested bins) | Yes, fragile |
| 8/9 peaks within one bin | l.63; l.1148; l.1166 | 8/9; 8/9 for 5–10 requested bins | same | Yes |
| Observed peaks nondecreasing in 3/3 families | l.1166 | 3/3 with ties (BA 6.77 = 6.77; NetScience 4.0 = 4.0) | `summary.json` `peak_order` | Yes |
| Three boundary misclassifications; NetScience t = 3×10⁻⁴ two bins off | l.1167 | BA 3×10⁻⁴, Holme–Kim 3×10⁻³, NetScience 3×10⁻⁴ (observed bin 2, predicted 0) | `summary.json` `curves` | Yes |
| NetScience norms overestimated | l.1168 | mean bias +0.376, +0.506, +0.374 | `outcomes.jsonl` | Yes |
| Calibration LOSO R² 0.828, ρ 0.932 | l.1158 | 0.8284, 0.9325 (within threshold: R² 0.69–0.76) | `calibration.jsonl`, refit with scikit-learn 1.8.0 | Yes |
| 23,685 frozen node predictions, hashed | l.1163 | 23,685 rows; 0 mismatches with outcomes; 4/4 manifest hashes match; exact refit | `frozen_predictions.jsonl`, `freeze_manifest.json` | Yes |
| Within-curve median ρ 0.842 versus 0.293 | l.1171; Table S1 | 0.842 [0.506, 0.935]; 0.293 [0.163, 0.610] | refit; `results/prospective_baselines/summary.json` | Yes |
| Degree-and-threshold passes all four; pooled 0.652 | l.1171; Table S1 | passes; 0.652 | refit | Yes |
| Degree only: pooled 0.178, fails | Table S1 | 0.178 | refit | Yes |
| Exposure–KL beats degree-and-threshold in 45/45 replicate groups (raw); 44/45 balanced, original; 45/45 corrected | Supp. S1 | 45/45; 44/45; 45/45 in both corrected arms | `prospective_baselines/summary.json` | Yes |
| Corrected arms: pooled 0.875 and 0.873; degree-and-threshold 9/9 with retained noise | Table S1 | 0.875, 0.873; 9/9 | same | Yes |
| β = 0: 3.507 → 2.297, ρ = −0.758 | l.996 | 3.5068 → 2.2972, −0.7581 | `results/turnover_surface/summary.json` | Yes |
| β = 1: 2.496 → 4.802, ρ = 0.895 | l.997 | 2.4960 → 4.8016, 0.8950 | same | Yes |
| β = 1/3: peak 3.315 near degree 6; +0.093 over lowest, +0.361 over highest | l.998 | 3.3147 at 6.00; 0.0933; 0.3609 | same | Yes |
| β = 2/3: 2.861 → 3.960, ρ = 0.821 | l.1000 | 2.8610 → 3.9596, 0.821 | same | Yes |
| Full mixing: 0.9488 in lowest and highest bins | l.1020 | 0.9488 and 0.9489 | same | Yes |
| Declared surface: observed peak 7.73, predicted monotone to 16.04 | l.1026 | 7.731 versus 16.044; also R² = −2.19, ρ = −0.673 (unreported) | same, `held_out_targets` | Yes |
| Exploratory interpolation: pooled RMSE 0.013, peak in correct bin | l.1027 | 0.0132; 7.73 = 7.73 | same | Yes |
| Historical Gensim peaks 2.98, 6.00, 7.73 (fixed epochs and budget-matched) | l.1082; l.1102 | 2.985, 6.000, 7.731 in both | `results/gensim_threshold_sweep/summary.json` | Yes |
| Rise 0.372 → 0.569; drop 0.537 → 0.238 | l.1084 | 0.3724 → 0.5688; 0.5372 → 0.2382 | same | Yes |
| Exploratory fit misses by 4, 2, 1 bins | l.1104 | 4, 2, 1 | same | Yes |
| No subsampling: 5/5 graphs interior; pooled peak 7.73; drop 0.244 | l.1110 | 5/5; 7.731; 0.2440 | `results/subsampling_origin/outcomes.csv` | Yes |
| Original schedule interior 5/5 from epoch 2 to 20; constant rate 0/5 at every checkpoint, increasing at epoch 20 | l.1117–1118 | 5/5 (epochs 2–20); 0/5 (epochs 1–20); smallest bin step 0.151 | `results/training_trajectory/summary.json` | Yes |
| Window gaps −0.3258 ± 0.0155 → −0.2173 ± 0.0123; 7/10 and 10/10 stable | l.1035; l.1038 | −0.3258 ± 0.0155; −0.2173 ± 0.0123; 7/10, 10/10 | `results/window_training_continuation/report/summary.json` | Yes |
| Relative decline 12.31% → 10.03%; weaker in every graph and start | l.1036–1037 | 12.31% → 10.03%; 10/10 | same | Yes |
| KL gap −0.2980 ± 0.0142 → −0.1303 ± 0.0063 | l.1035 | −0.298046 ± 0.014201; −0.130276 ± 0.006333 | `report/tables.md` | Yes (read) |
| Normalized gap −0.1129 ± 0.0054 → −0.0934 ± 0.0053 | l.1058 | −0.112947 ± 0.005360; −0.093384 ± 0.005299 | `results/norm_identifiability/` and `analysis/norm_identifiability_findings.md` | Yes (read) |

## 6. Questions for the authors

1. Which model classes and feature sets were fit to calibration data before the tree model was chosen, and were any of them evaluated on graphs other than the five calibration graphs?
2. Is there any record outside this repository, such as a pushed branch, CI log, or message, that predates the generation of target outcomes on 2026-08-13 or 2026-08-14?
3. Where was the evaluation binning (six requested quantile bins, pooled across replicates) written down before the freeze? Why six bins here and eight in every other analysis?
4. Can the per-seed node data behind Figs. 1 and 4 be recovered? In how many of the five graphs is the β = 1/3 curve interior-peaked, and what is the seed-level SE of its 0.093 rise?
5. What observation would contradict "a weakening norm response to exposure relative to context specificity," given that the fitted surface has a negative exposure derivative on 28% of calibration rows?
6. How do you reconcile the opposite threshold dependence of the controlled trainer and Gensim?
7. What does realized-corpus information add beyond graph structure, given that a graph-only comparator matches the frozen predictor's within-curve ranking?
8. Why do Sec. XI.G and the abstract not state that the frozen evaluation used the pre-correction implementation?
9. Would you replace the five NetScience replicates with several empirical networks, so that between-graph uncertainty can be estimated?
10. What do the λ-series curves look like after RMS normalization, as done for the windows?

## 7. What can be fixed before submission, and what cannot

**Fixable now from stored outputs and text:** R2-01 to R2-07, R2-09 to R2-11, R2-13 to R2-17, and R2-19. Most of R2-12 can also be done now, because the corrected Gensim, trajectory, window, and prospective runs store node-level outcomes.

**Fixable with reruns:** R2-08 and the β = 1/3 part of R2-12 need the historical controlled-grid and Gensim node data, recovered or regenerated in a pinned environment. Separating exposure from specificity (R2-11) needs runs with a fixed negative distribution, as in the window experiments.

**Not repairable by analysis:** the temporal precedence of the original freeze; the post hoc choice of the tree model and of the interpolation; the lack of between-graph replication for empirical networks; and the near-distribution design of two target families. Only a new prospective test (R2-21) addresses these, and the current test should be described as an internal, self-attested check.

## 8. Recommendation and confidence

**Recommendation: major revision.** The stored outputs reproduce the reported numbers, the code is auditable, and the paper discloses much of its exploratory history. The numerical claims, however, are framed around criteria that trivial predictors pass, a ranking metric whose advantage comes from neighbor composition among equal-degree nodes, and shape labels that were never tested, and the headline prospective numbers are presented without saying they come from the pre-correction pipeline. Most of this can be repaired by reanalysis of stored data and rewriting, without new training. The confirmatory status of the prospective test cannot be repaired retroactively.

**Confidence: 4/5.** I reproduced the numbers and traced the code paths behind them, but I did not evaluate the theory sections or retrain models, and some judgments, such as what counts as near-distribution transfer, are open to argument.

## 9. Action items

| ID | Severity | Location | Problem | Proposed fix | Type | Effort |
|---|---|---|---|---|---|---|
| R2-01 | blocker | Abstract l.63–64; l.1165–1169; Concl. l.1213; protocol l.43–53 | Trivial predictors pass the declared criteria: constant bins 1–3, calibration-peak transfer (8/9), threshold mean plus bump (all four); random guesses pass the peak criteria with p = 0.40 | Add trivial predictors to `scripts/prospective_baselines.py` and Table S1; report discriminating metrics with replicate-bootstrap intervals; reframe abstract and Conclusion | ISSUE | ≤1day |
| R2-02 | major | Abstract l.63; l.1166 | Pooled ρ inflated by curve levels: 46% of rank variance between curves; centered 0.843; threshold-only predictor 0.586 | Report within-curve median and range and centered ρ, with bootstrap intervals | PR | ≤1h |
| R2-03 | major | Abstract l.64; l.1171–1173; Table S1; Fig. 6 caption | About 80% of within-curve rank variance lies within degree classes; degree-only ceiling about 0.37; graph-only predictor ties at 0.842 | Add graph-only comparator, within-degree and level-centered binned-shape metrics; drop "two-force" label | ISSUE | ≤1day |
| R2-04 | major | l.1163; l.1213; l.1218; `prospective_norm_turnover.py:436, 458–505, 586–588, 677, 747` | Freeze precedence self-attested (one commit `5902c3c` about 21 h after freeze; literal flag); binning, peak, and pooling undocumented and unhashed | Disclose history; replace "hash-locked"; document evaluation definitions; external timestamp for future tests | PR | ≤1h |
| R2-05 | blocker | Abstract l.62–64; Sec. XI.G | Headline numbers come from the pre-correction pipeline without a label | State it; give corrected retrospective values beside the originals | PR | ≤1h |
| R2-06 | major | l.63; l.1148; l.1166–1167; Fig. 6 | 4–6 occupied bins; top-two margin below 1 SE in 5/9 curves; median 0 flips under 5/15 deletions; Holme–Kim 3×10⁻⁴ hit not robust | Report bins, margins, log-degree peak error with intervals, per-replicate errors, binning sensitivity; fix Fig. 6(d) overplotting | PR | ≤1day |
| R2-07 | major | Fig. 1 caption l.1012; Fig. 6 caption l.1146 | Node-level SEs labeled as seed-level | Relabel Fig. 1; recompute Fig. 6 with seed-level SEs; flag NetScience bars | PR | ≤1day |
| R2-08 | major | l.1218; `.gitignore:7` | Node-level data for Figs. 1 and 4 not archived; β = 1/3 claim unverifiable; versions unrecorded | Recover or regenerate and allowlist; otherwise qualify the statement | ISSUE | ≤1day |
| R2-09 | major | Abstract l.65; l.952; l.1024–1028 | Declared surface R² = −2.19, ρ = −0.67 unreported; exposure derivative negative on 28% of rows; declaration unverifiable | Report magnitudes and sign fractions; delete l.65 or restate it as a falsifiable prediction | PR | ≤1h |
| R2-10 | major | Abstract l.59–61, l.65; Intro l.100–102; Disc. l.1185–1189 | Controlled trainer and Gensim respond to subsampling in opposite directions; schedule dependence | State the discrepancy; confine mechanistic statements to the trainer that shows them | PR | ≤1h |
| R2-11 | major | Abstract l.59–60; l.964–987; l.1017–1022; Fig. 1(d) | β changes the KL slope (−0.83 versus −0.99); λ = 1 flat by exchangeability; λ = 1/3 relative decline unchanged | Report KL slopes per cell and relative or RMS-normalized λ contrasts; reword | PR | ≤1day |
| R2-12 | major | l.998; l.1083; l.1110; l.1117–1118; l.1165–1167 | Shape labels untested; classifier flags random initialization (2/5) and the flat λ = 1 balanced curve | Prespecified held-out-breakpoint two-lines test with graphs as units; per-graph counts for every shape claim | ISSUE | ≤1day |
| R2-13 | major | Abstract l.62; Intro l.103–104; l.1160–1161; Disc. l.1190 | Two target families share the calibration degree model; NetScience is out of support (73.5% of rows sparse at 3×10⁻⁴) and a single graph | Qualify "unseen"; add covariate-shift diagnostics; treat NetScience as one case | PR | ≤1day |
| R2-14 | minor | Intro l.102; l.1082; l.1102 | Corrected pooled shift 6.00 → 7.73 occurs in 1/5 graphs | Add per-graph peak table; limit the movement claim to 3×10⁻⁴ → 10⁻³ | PR | ≤1h |
| R2-15 | minor | l.1035–1043; Fig. 2; amendment l.3–7, l.59–60 | Declared all-window figure replaced by windows 5 and 10; gates called "predefined" were set after initial endpoints | Add the all-window figure to the SM; state when the gates were fixed | PR | ≤1h |
| R2-16 | minor | l.1158 | LOSO metrics pool thresholds | Report within-threshold R² (0.69–0.76) and ρ (0.84–0.92) | PR | ≤1h |
| R2-17 | minor | l.991; l.1196 | Readout agreement overstated; centered-score readout unreported | Qualify both sentences; report the third readout or explain its omission | PR | ≤1h |
| R2-18 | minor | Abstract; Sec. XI | Several hundred untested shape calls; the abstract features a single 2.9% bump | SM table of every condition with call, margin, and graph count | ISSUE | ≤1day |
| R2-19 | minor | `protocol.json`; Methods | No package versions; inconsistent bin counts and Spearman conventions across pipelines | Document environment and conventions | PR | ≤1h |
| R2-20 | minor | Tracked #22; `allocations.json.gz` | β = 1, λ = 1 not fully mixed (KL 0.359, slope −0.117) | Add this evidence to #22 | ISSUE | ≤1h |
| R2-21 | major | Discussion l.1203–1205 | Confirmatory status cannot be restored for the current targets | New prospective test with externally timestamped protocol and code, prespecified discriminating baselines, and independent graphs or datasets | ISSUE | >1day |

## 10. Check scripts

All scripts read the worktree without writing to it and save their output in `r2/logs/`. Run each from `r2/` with `PYTHONDONTWRITEBYTECODE=1 uv run --with numpy,scipy python <script>`; the scripts that fit models or regenerate graphs need `--python 3.13 --with numpy==2.4.1,scipy==1.17.1,scikit-learn==1.8.0,networkx==3.6.1`.

- `common.py`: paths and an exact copy of the prospective `binned_curve`.
- `prospective_repro.py`: pooled ρ, peak errors, within-curve ρ, top-two margins, centered ρ, between-curve share, threshold-only predictor.
- `peak_null.py`: constant-bin, uniform-guess, and calibration-peak-transfer predictors against the peak criteria.
- `null_predictor.py`: threshold mean plus bump against all four criteria.
- `peak_bootstrap.py`: replicate-cluster bootstrap, leave-one-replicate-out, and binning sensitivity.
- `baseline_refit.py`: refit of all baselines, oracle degree ceiling, within-degree-class ranking.
- `baseline_summary_check.py`: replicate-group win counts from `prospective_baselines/summary.json`.
- `shape_compare.py`: level-centered binned-shape error, frozen predictor versus degree-and-threshold.
- `noise_tracking.py`: NetScience within-node tracking; structural versus residual KL within degree classes.
- `graph_only_baseline.py`: graph-only comparator.
- `calib_and_se.py`: calibration LOSO, node-level versus seed-level SEs, per-replicate peak errors.
- `support_check.py`: covariate shift between calibration and targets.
- `split_peak_test.py`: selection-corrected interior-maximum checks.
- `controlled_check.py`, `readout_agreement.py`, `confound_kl.py`: controlled grid, readouts, KL slopes by condition.
- `trajectory_check.py`, `gensim_corrected_check.py`, `window_check.py`: trajectory classifications, corrected Gensim per-graph peaks, window gaps.
- `degree_bins.py`: occupied bins and counts per family.
