# Physical Review E: referee report, Reviewer 3 (data processing and computational reproducibility)

**Manuscript:** "Degree Information and Norm Turnover in Random-Walk Network Embeddings" (Gandhi, Kojaku, Ahn).
**Material reviewed:** project repository at commit `c958c528` (worktree), `paper/current/main.tex` and `main.pdf` (15 pp.), `supp.tex` and `supp.pdf` (3 pp.), `factor_balance.tex`, `norm_geometry_supplement.tex`, all `results/*` folders cited by `\graphicspath` (`main.tex:16–22`) or by the text, the scripts that produce them, the protocols in `analysis/`, `tests/`, and git history.
**Scope of this report:** traceability of every number, regenerability of every figure and table, provenance and currency of the stored outputs, the patched Gensim build, the prospective hash manifest, the fresh-clone experience, data availability, and repository hygiene. Theory is outside my brief except where a number depends on it.

All checks ran on a scratch clone and scratch virtual environments; nothing was written to the worktree. The commands are in `r3/run_checks.sh`, with helper scripts `r3/check_manifests.py`, `r3/compare_images.py`, `r3/compare_json.py`, `r3/error_bars.py`, and `r3/replay_factor_balance.py`. Platform: macOS 27.2 arm64, Apple clang 21, Python 3.13.12 from uv, with the pinned requirements in `analysis/negative_sampling_control_requirements.txt`. The authors' recorded build is also darwin, so cross-platform (Linux/x86) identity remains untested.

## Summary

The paper compares population geometry of three random-walk embedding operators under the DCSBM and then reports six numerical figures produced by four separate pipelines.
These are a custom NumPy mini-batch SGNS trainer (Fig. 1), an exact expected-loss L-BFGS optimizer (Figs. 2–3), Gensim word2vec (historical Fig. 4 and corrected Fig. 5), and an ExtraTrees predictor frozen before Gensim target training (Fig. 6).
The repository is unusually well instrumented: hash manifests, plan files, completion files, protocols written before runs, and archived pre-correction runs.

I traced 80 printed numbers and numeric claims. Every one with a repository source matches its stored field at printed precision, except one mis-rounding (T24); one bound is rounded up (T42), and one value is an external literature figure I did not check (T8).
All 492 file digests recorded in 15 manifests verify.
Tables S1–S2 regenerate byte-identically, Figs. 2, 3, and 5 regenerate pixel-identically from committed data, the Fig. 1 experiment reruns in 75 s and matches its stored summary to 1e-11, the README's corrected revalidation reruns end to end in about four minutes, and the 20 corrected Gensim controls replay bit-identically on my machine.

The problems are concentrated in what the paper says about where its numbers come from, and in gates that stop a reader from running the documented scripts.
Two main-text figures (4 and 6) and the abstract's headline prospective statistics come from the pipeline that the repository itself labels "historical" and "bug-affected"; Fig. 6 and the abstract do not say so.
The corrected counterparts exist (or take under a minute to produce) and change the numbers slightly while leaving every qualitative conclusion I checked intact.
Captions of Figs. 1 and 6 describe seed-level standard errors where the code draws node-pooled standard errors.
The scripts that produce Fig. 5 and Table S3 refuse to run on any machine other than the authors' because they compare the SHA-256 of a locally compiled binary.
Fig. 1 is a PDF-edited copy of an older plot that the repository cannot regenerate, and the Data Availability statement overstates what accompanies the manuscript.

## Strengths

1. Integrity bookkeeping is exceptional. Every one of the 492 file digests in `results/*/completion.json`, `plan.json`, `manifest.json`, `provenance.json`, and the prospective `freeze_manifest.json` matches its file at `c958c528` (`r3/check_manifests.py`).
2. Most artefacts regenerate exactly. `scripts/prospective_baselines.py` reproduces `table.tex`, `balanced_table.tex`, `README.md`, and `summary.json` byte for byte in 13 s; Figs. 2, 3, and 5 are pixel-identical (0 differing pixels at 100 dpi); `simulate_turnover_surface.py` reproduces all 941 numeric fields of `results/turnover_surface/summary.json` to a relative difference of at most 1e-11.
3. The corrected Gensim pipeline is reproducible from the README. `scripts/build_control_gensim.py` builds the patched extension from the checksummed sdist without error and `check_control_gensim.py` passes. The README's revalidation recipe (120 Gensim fits) ran end to end in 4 min 9 s: all 9,000 calibration, 23,685 prediction, and 23,685 outcome rows agree with the committed files to ≤2.2e-15 relative, `check_revalidation_repeat.py` reports `all_fields_exact`, and the report step regenerates `README.md` byte-identically and `predictor_revalidation.png` pixel-identically. With one over-strict gate bypassed, the 20 corrected controls behind Table S3 also reproduce every recorded digest (R3-03).
4. The historical runs are preserved. The defective-runtime sweep, the original calibration source, and the original frozen predictions remain in place, so the effect of each correction can be measured.

## Weaknesses (ranked)

1. **Blocker.** The abstract, Sec. XI H, and Fig. 6 report the prospective test from the pipeline with all three implementation defects (walk-level visit counts, permuted integer-token negative table, BLAS score of −1 returned as 0) without saying so (R3-01).
2. **Blocker.** Fig. 4 and its text numbers are built from outputs that `results/README.md` marks as pre-correction; the corrected fixed-epoch sweep already exists, and I reran the full corrected sweep, budget-matched arm included, in 45 s (R3-02).
3. **Major.** The integrity gate in `blas_check()` includes the compiled extension's SHA-256, so the documented scripts for Fig. 5 and Table S3 fail for every reader, although the underlying computation is bit-reproducible (R3-03).
4. **Major.** Fig. 1 and Fig. 6 captions claim standard errors across seeds; the code computes node-pooled standard errors, which differ from seed-level ones by factors of 0.13–13.8 (R3-04).
5. **Major.** Fig. 1 cannot be regenerated from the repository: it is a PyMuPDF label edit of a Matplotlib 3.11.1 figure whose node-level data are gitignored (R3-05).
6. **Major.** The "prospective" freeze is self-attested. Predictions, manifest, and outcomes enter git in the same commit, the evaluation code and bin definition were not hashed, and the protocol's own statement that this is not an externally registered report is missing from the paper (R3-06).
7. **Major.** The Data Availability statement is inaccurate and the NetScience data lack provenance, terms, and a description of preprocessing (R3-07).
8. **Major.** Eight environment specifications coexist; the README environment cannot collect the test suite, `make test` errors out, and the environment that rendered Figs. 1, 4, and 6 is recorded nowhere (R3-08).
9. **Major.** There is no per-figure reproduction map or single entry point, and the documented root `make all` runs an empty template workflow (R3-09).
10. **Major.** The manuscript names the three implementation corrections in one clause (`main.tex:1108`) and never explains them or maps which results they affect (R3-10).
11. **Minor.** Provenance gaps in folder READMEs, rounding, figure-text mismatches, hygiene, fonts, missing method details, and byte-level manifests (R3-11 to R3-20).

## Detailed comments

### R3-01 (blocker). Headline prospective numbers come from the bug-affected pipeline and are not labeled

- **Location:** abstract (`main.tex:62–64`, p. 1); Sec. XI H (`main.tex:1155–1173`, p. 13); Fig. 6 and caption (`main.tex:1141–1151`, p. 13); Conclusion (`main.tex:1213`).
- **Problem:** The abstract sentence reporting "median peak error is zero degree bins, eight peaks lie within one bin ..., pooled node-level Spearman correlation is 0.881" directly follows "Corrected word2vec controls show ...". A reader is likely to assume the same corrected implementation. The repository says otherwise: `results/prospective_validation/README.md` states that these files "predate the repeated-visit counter, integer-token negative-table and BLAS corrections" and that "their original numbers must not be described as results of the corrected code"; `results/negative_sampling_revalidation/README.md` calls the historical column "the original bug-affected pipeline". Neither the caption nor Sec. XI H mentions this; the only hint is "corrected sampling comparisons are reported in Supplemental Material" (`main.tex:1172`).
- **Evidence:** I confirmed the historical runtime defect with the stock Gensim 4.4.0 wheel under the pinned requirements: `check_control_gensim.py` raises "Gensim our_dot_float returned 0.0 instead of -1.0". The archived script (`archive/negative_sampling_control/calibration_source/scripts/prospective_norm_turnover.py`, identical to `5902c3c:scripts/prospective_norm_turnover.py`) trains with Gensim's default cumulative table and the walk-level counter. Corrected values from the retrospective revalidation (`results/negative_sampling_revalidation/README.md`; `results/prospective_baselines/table.tex`): pooled Spearman 0.875 (retained noise) and 0.873 (raw noise) versus 0.881; peaks within one bin 8/9 in all three; median error 0 in all three; calibration LOSO R² 0.821/0.822 and Spearman 0.933/0.935 versus 0.828 and 0.932 (`calibration_cv.json`). The qualitative conclusions survive the correction. The paper's statement of where the numbers come from has to change.
- **Fix:** State in the abstract, caption, and Sec. XI H that the frozen prospective run used the uncorrected sampler, give the corrected retrospective values alongside (0.875/0.873, 8/9, 0), and show the corrected curves; `results/negative_sampling_revalidation/predictor_revalidation.png` already exists. Keep the original as the prospectively frozen evidence and the corrected run as the numerically valid one, and say which number a reader should cite.

### R3-02 (blocker). Fig. 4 is built from superseded outputs although the corrected rerun is trivial

- **Location:** Fig. 4 (`main.tex:1086–1098`, p. 11); text `main.tex:1078–1084`, `1100–1104`.
- **Problem:** `results/README.md` says `gensim_threshold_sweep/` contains "outputs produced before these corrections" that "must not be presented as corrected results". The caption labels the figure historical, which is honest, but a main-text figure and four text numbers (0.372, 0.569, 0.537, 0.238) still come from a computation the authors know to be defective. `main.tex:1103` says the budget-matched comparison "has not yet been repeated". The committed script no longer produces the committed figure either: its dependency `scripts/simulate_norm_frequency.py` changed in `87e2785` (counting and negative table fixed) after the outputs were committed in `5902c3c`.
- **Evidence:** (a) Historical path: with the `5902c3c` versions of `simulate_gensim_threshold_sweep.py`, `simulate_norm_frequency.py`, `simulate_exact_center_exposure.py`, and `simulate_turnover_surface.py` plus the stock Gensim 4.4.0 wheel, the sweep reruns in 46 s and reproduces all 156 numeric fields of `summary.json` to ≤5e-15 relative; stderr shows five "Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'" events, so the −1 defect fired in a run that reproduces the published numbers. (b) Corrected path: the current script under the patched runtime runs in 45 s. Peak locations are unchanged (2.98, 6.00, 7.73 for both arms), but the amplitudes move:

  | Quantity (raw norm) | Paper (historical) | Corrected (my rerun; identical to `subsampling_origin` pooled curves) |
  |---|---:|---:|
  | Rise to peak, t = 1e-3 / 3e-3, fixed epochs | 0.372 / 0.569 | 0.375 / 0.560 |
  | Drop from peak, t = 1e-3 / 3e-3, fixed epochs | 0.537 / 0.238 | 0.527 / 0.257 |
  | Drop, t = 3e-4, fixed epochs | 0.748 | 0.759 |
  | Drop, t = 3e-4 / 1e-3, budget-matched (8 and 6 epochs) | 0.854 / 0.603 | 0.845 / 0.593 |
  | Mean expected retained centers, t = 3e-4 | 2.32 M | 2.60 M |

  Panel (a) and the 8/6/5 epoch choice were computed from walk-level counts (the defect tracked as #44); the corrected counts raise expected retained centers by about 12%.
- **Fix:** Rerun the sweep with the corrected code (both arms), regenerate Fig. 4 with graph-level standard errors, update `main.tex:1084` and `1102–1104`, and delete the "not yet been repeated" sentence. Remove the embedded title "Literal Word2Vec validation of the turnover prediction" (`simulate_gensim_threshold_sweep.py:261`), which contradicts the caption and the text's finding that the controlled fit misses by four, two, and one bins.

### R3-03 (major). The BLAS gate compares a machine-specific binary hash; Fig. 5 and Table S3 are not reader-runnable

- **Location:** `scripts/check_control_gensim.py:47–51` (adds `extension_sha256` to the check result); comparisons in `scripts/factor_balance.py:129–130`, `scripts/training_trajectory/run.py:206–208`, `scripts/window_training/schedule_fork.py:290–295`, `scripts/window_training/verify.py:471–474`; claims in `factor_balance.tex:47–50` and `paper/current/README.md`.
- **Problem:** The gate is meant to certify that the compiled dot product handles −1 correctly. It also requires the SHA-256 of the compiled `.so` to equal the authors' build. The generated C embeds absolute include paths of the virtual environment (6 occurrences in my build), and compiler and Python builds differ, so no reader can match it.
- **Evidence:** Following the README exactly (with the build cache redirected to scratch), my extension hash is `4a642bfe...`; the recorded one is `f966cc17...` (`results/subsampling_origin/plan.json`). The numeric checks are identical (`our_dot_float`; −2, −1, −0.5, 0, 1 all exact). `python scripts/factor_balance.py --output <scratch>` aborts with "Patched BLAS validation differs from source plan"; `python scripts/training_trajectory/run.py --output <scratch>` aborts with "Numerical extension differs from the baseline." With only that key ignored (`r3/replay_factor_balance.py`), all 20 corrected controls reproduce every graph, walk, count, initialization-signature, token-count, center-hash, and context-hash check, and Table S3 values are identical (max |Δ norm| = 0), in 41 s. The README's revalidation recipe escapes the problem only because it writes a fresh `plan.json` with the local hash; every script that checks against a committed plan fails.
- **Fix:** Compare `wrapper`, `dot_products`, and `passed`; record `extension_sha256` as information only. Add a test that a rebuilt extension with a different hash passes the gate. This one change makes Table S3, Fig. 5, and the schedule-fork verification reader-runnable.

### R3-04 (major). Error bars in Figs. 1 and 6 are node-pooled standard errors, but the captions say seed-level

- **Location:** Fig. 1 caption (`main.tex:1012`): "standard errors across five graph seeds"; Fig. 6 caption (`main.tex:1146`): "standard errors over five graph seeds ... and five walk and optimizer seeds for NetScience". Code: `scripts/simulate_exact_center_exposure.py:308` and `scripts/prospective_norm_turnover.py:454`, both `np.std(y[bin]) / sqrt(n_nodes_in_bin)` over nodes pooled across all seeds.
- **Problem:** This is the same flaw that the Fig. 4 caption acknowledges ("the historical plotting routine computed node-based standard errors", `main.tex:1095`), but Figs. 1 and 6 use the same routine and claim the opposite. For NetScience the five "replicates" are the same 379 nodes, so a node-pooled SE does not measure replicate variation at all.
- **Evidence:** `r3/error_bars.py` recomputes both from node data (Fig. 6 from committed `outcomes.jsonl`; Fig. 1 from my verified rerun). Ratio of seed-level SE to the plotted node SE: Fig. 6 median 0.62, range 0.13–1.90 (NetScience 0.13–0.59, because the five runs on one graph barely differ); Fig. 1 median 1.15, range 0.42–13.8 (the λ = 1 curve: plotted ≈1e-4, seed-level ≈6e-4).
- **Fix:** Redraw both figures with SEs of seed-level bin means in common bins (the newer report code for Figs. 2, 3, and 5 already does this), or change the captions to state node-pooled SEs and their meaning. Panels 1(e) and 1(f) show no error bars although the caption implies them.

### R3-05 (major). Fig. 1 is a PDF-edited historical plot that the repository cannot regenerate

- **Location:** Fig. 1 (`main.tex:1002–1015`, p. 9) uses `results/realized_graph_audit/turnover_surface_qualified.pdf`, produced by `qualify_historical_figure()` in `scripts/audit_realized_graph.py:590–660`.
- **Problem:** The function opens `results/turnover_surface/turnover_surface.pdf` (rendered on 2026-08-13 with Matplotlib 3.11.1, per its PDF metadata), redacts three labels, and stamps replacements in Helvetica. The plotting code still emits "Causal two-force test for the norm-frequency turnover", "B Specificity interventions", and "C Exposure creates the rising limb" (`scripts/simulate_turnover_surface.py:979, 997, 1068`). The node-level data needed for `--reuse` (`node_results.csv`, `exposure_matched_predictions.csv`) are excluded by `.gitignore:7` (`*.csv`), so the figure cannot be regenerated from committed files.
- **Evidence:** `pdffonts` on `main.pdf` shows one font not embedded: Helvetica Type 1 (object 347), coming from this figure. Rerunning the experiment (75 s) reproduces the summary to ≤1e-11, so the data are fine. The regenerated plot under the pinned Matplotlib 3.10.8, however, has the old titles and a different log-axis autoscale (x axes extend to about 0.4 in panels A–D), so pinned code plus pinned environment does not reproduce the published figure.
- **Fix:** Change the titles in `make_figure`, commit the node CSV (gzip; about 11 MB raw), regenerate under the pinned environment with explicit axis limits, and delete the PDF surgery step. This also fixes the non-embedded font.

### R3-06 (major). The prospective freeze is not independently verifiable

- **Location:** `main.tex:1155`, `1163` ("no outcome file existed at the freeze"), `1213` ("Hash-locked target predictions"); `results/prospective_validation/freeze_manifest.json`; `analysis/prospective_validation_protocol.md:4`.
- **Problem:** The manifest's four hashes match `protocol.json`, `analysis/prospective_validation_protocol.md`, `calibration.jsonl`, and `frozen_predictions.jsonl`. Three things limit what that proves. (1) Git cannot order the freeze before the outcomes: `freeze_manifest.json`, `frozen_predictions.jsonl`, `protocol.json`, the human protocol, and `outcomes.jsonl` were all first committed together in `5902c3c` (2026-08-14T13:33−04:00), after the self-recorded `frozen_utc` (2026-08-13T20:31Z). (2) The evaluation code was not hashed: `protocol.json` contains no script hash, and the bin definition (six pooled log-degree quantile bins, `prospective_norm_turnover.py:432–455`) and the peak rule live only in code, while the hashed protocol says "one degree bin" without defining bins. (3) The protocol itself says "This is a code-frozen prospective validation within the project, not an externally registered report"; the paper omits this.
- **Fix:** Say in Sec. XI H that the freeze is internal and self-timestamped, that the first commit containing predictions also contains outcomes, and that the evaluation code was not hash-locked. For the next test, push a signed commit or tag (or an OSF/Zenodo deposit) containing predictions plus the evaluation script hash before training. Add a sidecar file mapping manifest keys to file paths (R3-17).

### R3-07 (major). Data Availability statement is inaccurate; NetScience provenance, terms, and preprocessing undocumented

- **Location:** `main.tex:1216–1220` (p. 14); `README.md:94`; `are_angles_important/netscience/`.
- **Problem:** (1) The statement gives no URL, commit, or DOI. The GitHub repository is public, but the paper does not cite it. (2) "Generated data needed to reproduce the figures accompany this manuscript" is false for Fig. 1 and Fig. 4, whose node-level CSVs are gitignored. (3) `README.md:94` says "Data files are not included in the repository." (4) The NetScience file sits in an unrelated subproject folder. Its companion text gives the source and citation request (Newman 2006) but no URL, license, or redistribution terms. The file is weighted (61 distinct edge weights, 0.125–4.75) while the pipeline reads it with `nx.read_gml` and walks uniformly over neighbors, so it is used unweighted; the paper does not say so. The LCC extraction (379 nodes, 914 edges, which I confirmed) happens in code and is not recorded in any result file.
- **Fix:** Cite the repository URL and the exact commit; create a tagged release with a Zenodo DOI; move NetScience to `data/netscience/` with source URL, SHA-256 (`ede685c5...`, already in `protocol.json`), citation, terms, and a note that weights are ignored and only the LCC is used. State a data license (for example CC BY 4.0) for results and figures separately from the MIT code license, and correct `README.md:94`.

### R3-08 (major). Environment specification is fragmented, partly wrong, and incomplete for the historical figures

- **Location:** `README.md:8–40` and `68–91`; `environment.yml`; `environment_gensim.yml`; `pyproject.toml`; `uv.lock`; `analysis/*requirements*.txt`; PEP 723 headers in `scripts/simulate_*.py` and `scripts/prospective_norm_turnover.py`; `.python-version`.
- **Problem and evidence:**
  - Eight specifications coexist, and nothing maps them to figures. The README's "Setup" still sends readers to conda environments with Python 3.9.18, gensim 4.3.3, `gensim-mod`, Linux-only packages, and hard-coded prefixes (`environment.yml:607`, `environment_gensim.yml:382`: `/nobackup/gogandhi/...`); none of these can reproduce anything in the paper.
  - The README environment cannot run the suite. `python -m pytest` stops at collection (`tests/test_realized_graph_audit.py` imports `pymupdf`, which is only in `analysis/realized_graph_audit_requirements.txt`). With both files installed, 136 tests pass in 9 s; the README says "all 26 tests".
  - `make test` (`uv run --frozen`) reports "7 skipped, 10 errors" (`ModuleNotFoundError: numpy`), contradicting the README's claim that missing optional dependencies are skipped.
  - `.python-version` says `3.13`, so uv selected 3.13.17, while `schedule_fork.py:290–295` rejects any Python other than the recorded 3.13.12. `uv.lock` pins pytest 9.0.3 and ruff 0.14.14; the requirements pin 9.2.0 and 0.17.3; the audit requirements pin pytest 8.4.2.
  - Figs. 1, 4, and 6 were rendered with Matplotlib 3.11.1 (PDF metadata), resolved from PEP 723 ranges at run time. The pinned environment has 3.10.8 and renders visibly different axes. The historical runtime (stock Gensim 4.4.0 with the defect) is documented only as an archive note.
- **Fix:** One lock file covering every paper script (`pyproject.toml` dependencies plus `uv.lock`, Python pinned to 3.13.12); a separate, documented historical lock for the pre-correction runs; move the conda files to `archive/`; make tests skip with `pytest.importorskip`; correct the test count.

### R3-09 (major). No per-figure reproduction map or entry point

- **Location:** `Makefile` (root), `workflow/Snakefile`, `README.md`.
- **Problem:** The root `make all` runs `snakemake` on `workflow/Snakefile`, an unmodified project template whose rules are all commented out (`PROJ_NAME=PROJ_NAME` is still a placeholder). The six figures and four tables come from at least nine scripts in three pinned environments plus the unpinned historical one, with different `--output`/`--root`/`--output-dir` conventions. Fig. 4 can only be reproduced by checking out `5902c3c` versions of four scripts and installing the defective Gensim wheel, which I had to reconstruct from git history.
- **Fix:** Add a table to `README.md` and the SM: figure or table → script@commit → inputs → environment → command → expected output hash or tolerance, with status (historical/corrected). Back it with a `make figures` target or `scripts/reproduce_paper.py` that writes to a fresh directory and compares with committed outputs within tolerances.

### R3-10 (major). The three implementation corrections are not documented in the paper

- **Location:** `main.tex:1108` ("corrected repeated-visit counts, integer-token negative-table indexing, and compiled inner-product arithmetic"); SM has no corresponding section.
- **Problem:** A reader cannot learn what was wrong, how large the effects were, or which results each defect touched. The repository explains all three well (`analysis/negative_sampling_control_protocol.md`): `counts[walk] += 1` counts walks containing a node; `make_cum_table` passes integer positions to `get_vecattr`, which resolves them as integer keys, so for counts {2:8, 0:4, 1:2} the slots get [4, 2, 8] instead of [8, 4, 2]; and the Cython `except -1` declaration on `sdot_ptr`/`dsdot_ptr` turns a valid score of −1 into 0 (upstream issue #3665, PR #3668).
- **Evidence of effect size:** runtime fix alone changes calibration-control norms by up to 0.065 with no peak-bin change (`results/negative_sampling_control/README.md`); all three together change the Fig. 4 amplitudes as tabulated in R3-02 and the Fig. 6 pooled Spearman from 0.881 to 0.875/0.873.
- **Fix:** Add an SM section with the three defects, the toy examples, upstream references, and a table of affected results (Figs. 4 and 6 affected; Fig. 1 custom trainer and Figs. 2–3 expected-loss optimizer unaffected; Fig. 5, Table S3, and the corrected arms of Tables S1–S2 computed with corrected code).

### R3-11 (minor). Folder-level provenance gaps

- **Location:** `results/turnover_surface/`, `results/gensim_threshold_sweep/`, `results/norm_identifiability/`, `results/window_training_continuation/`, `results/factor_balance/` (no README); `results/prospective_validation/README.md` (names no script, commit, or environment); `results/training_trajectory/README.md` (no reproduce command).
- **Problem:** The README for the Fig. 6 data does not point to the archived generating script (`archive/negative_sampling_control/calibration_source/scripts/prospective_norm_turnover.py`), and `scripts/prospective_norm_turnover.py` has changed since (`87e2785`). For Fig. 4, as noted, running the committed script now yields a different (corrected) result. Seeds for the controlled trainer (graphs 0–4, optimizer `100000 + seed`) appear only in code.
- **Fix:** One README per used folder: producing script with commit, exact command, environment file, seeds, status (historical, corrected, superseded), and known defects.

### R3-12 (minor). Rounding and wording

- `main.tex:1020`: "0.9488 in both the lowest and highest degree bins"; the highest bin is 0.948878 (`calibration_curves.beta_000_lambda_100.raw_norm.highest_degree_bin`), so 0.9489.
- `main.tex:1059`: "at most 0.00452"; the maximum is 0.0045122, which rounds to 0.00451.
- `main.tex:991`: "both give the same qualitative results"; at λ = 1 the stored balanced curve is flagged `inverted_u: true` with amplitude about 1e-4 while the raw curve is not. Harmless, but say "up to differences of order 1e-4".
- Sec. XI E is titled "Literal word2vec validation" for a sweep the caption calls historical and pre-correction.

### R3-13 (minor). Figure and caption mismatches

- Fig. 1 panels are lettered A–F; the caption uses (a)–(f). Panel F's "peak shift" goes from 7.73 to the boundary bin (16.08) at t = 1e-3 and 3e-3, so two of the three "peaks" are monotone curves; "tracks the peak shift" (`main.tex:1011`) overstates this.
- Overlapping log tick labels: Fig. 1 panel F, Fig. 4 panel D, and Fig. 6 panel (d) ("3×10^0" collides with "4×10^0").
- Figs. 1, 2, 4, and 5 carry in-figure suptitles; APS figures normally rely on the caption.

### R3-14 (minor; release decision). Repository hygiene

- Tracked build and cache artefacts: `scripts/__pycache__/*.cpython-39.pyc` (4), `notebooks/*/__pycache__/` (4, including `.py` files), `snakemake/.snakemake/log/*` (12), `notebooks/20231209_debugging_parallelizaton/progress.log` (1.6 MB). The LaTeX auxiliaries in `paper/current` are correctly ignored.
- Unrelated material: `notebooks/` (154 files, about 117 MB, 80 of the repository's 87 notebooks) from the k-means/LFR project; `notebooks/data/LFR_benchmark/*.npz` (27 MB, 12.6 MB, and 4 × 2.7 MB, force-added despite `.gitignore`); three 4.6 MB HTML plots; `libs/pyclustering-Warning-Fix.zip` (third-party code, no license notice); `results/Changing Embedding Dimension`, `Changing Embedding Method`, `Changing Mixing Parameter`, `Changing clustering metric`; `Figs/` (NetSci abstract art); `workflow/` (a template with its own LICENSE); `archive/kmeans`.
- The README title is still "Alternative to K-Means clustering methods for community detection of embedding vectors." Legacy scripts with hard-coded `/nobackup/gogandhi/...` paths (`scripts/similarity_scores.py:24`, `clustering_method_metric_change.py:24`, `nets_and_embeddings.py:10`) sit beside the paper's scripts; `scripts/clustering_methods.py` is empty.
- Total tracked size is about 292 MB, of which about 155 MB is paper-relevant; history adds 188 MB.
- **Fix:** Publish a curated release (paper sources, the paper's scripts and tests, protocols, used results, NetScience) as the archived artefact, and move the rest to `archive/` or a separate repository.

### R3-15 (minor). Paper build

- `make main.pdf` from a fresh clone works: supplement first, then main; 15 + 3 pages in 5 s; no undefined references; xr labels resolve (Sec. S1, Tables S1–S2, Sec. S3).
- `main.pdf` does not depend on figure or table files in the Makefile, so make can report a stale PDF as up to date.
- `norm_geometry_supplement.tex` is referenced by neither main nor SM, although `exact_norm_profiles.py` names it as the proof location.
- `make all` also builds the historical `note.tex`.
- APS submission compiles `main.tex` alone; with `xr-hyper`, upload `supp.aux` or hard-code the SM references.

### R3-16 (minor). `plot_norm_identifiability.py --root` crashes for any root outside the repository

`scripts/window_training/plot_norm_identifiability.py:217` calls `Path(__file__).relative_to(config.root)`. With a scratch root it writes all four figure files and then raises `ValueError` before the manifest, so the only working invocation overwrites committed outputs. The regenerated figure is pixel-identical; the CLI needs a fix.

### R3-17 (minor). `freeze_manifest.json` keys name roles and omit file paths

Keys are `protocol_sha256`, `human_protocol_sha256`, `calibration_sha256`, and `predictions_sha256`; I matched them to files by trial. Add a sidecar that maps each key to a path, plus the evaluation-script and environment hashes, without editing the frozen file.

### R3-18 (minor). Fonts

Figs. 1, 2, 4, 5, and 6 embed Type 3 fonts (11 Type 3 fonts in `main.pdf`); Fig. 1 adds the non-embedded Helvetica. Set `pdf.fonttype: 42` and regenerate.

### R3-19 (minor). Method details that exist only in code

The controlled trainer's mini-batch size 256, linear learning-rate decay 0.025 → 0.0025, uniform initialization ±0.5/d with zero contexts, float64, and with-replacement negatives (`simulate_turnover_surface.py:79–89`, `simulate_exact_center_exposure.py:210–275`). Gensim `min_count = 50` in the historical sweep, `shrink_windows=False`, and unweighted use of NetScience. Add them to Sec. XI A, XI E, and XI H.

### R3-20 (minor). Byte-level manifests flag numerically identical reruns

My regenerated `window_training_continuation/report/summary.json` differs from the committed one only at ≤2.5e-15 relative (27 of about 78,000 lines), which changes its hash and the report manifest. The fresh revalidation is the same story at scale: `calibration.jsonl`, `frozen_predictions.jsonl`, and `outcomes.jsonl` differ from the committed files in balanced-norm fields at ≤2.2e-15, so every digest in its `completion.json` and `freeze_manifest.json` differs even though the science is identical. Verification scripts that compare digests of numeric outputs will report such reruns as failures. Compare numeric outputs with explicit tolerances and keep digests for inputs and frozen artefacts.

## Number-tracing table

"Historical" means the bug-affected pre-correction pipeline. All paths are relative to the repository root.

| # | Claim as printed | Location | Source file : field | Stored value | Match |
|---|---|---|---|---|---|
| T1 | median peak error zero bins | abstract, `main.tex:63` | `results/prospective_validation/summary.json` : `criteria.median_absolute_peak_bin_error` | 0.0 | Yes (historical; corrected arms also 0) |
| T2 | eight peaks within one bin | `main.tex:63`, `1166` | same : `criteria.fraction_peaks_within_one_bin` | 0.8889 (8/9) | Yes (corrected 8/9) |
| T3 | pooled Spearman 0.881 | `main.tex:63`, `1166` | same : `criteria.pooled_spearman` | 0.880555 | Yes, historical; corrected 0.874989 / 0.872642 (`negative_sampling_revalidation/README.md`) |
| T4 | nine family–threshold curves | `main.tex:63` | same : `curves` | 9 entries | Yes |
| T5 | equal exposure decreasing, proportional increasing, cube root inverted-U | `main.tex:59`, `100` | `results/turnover_surface/summary.json` : `calibration_curves.beta_{000,100,033}_lambda_000.raw_norm` | ρ −0.758; ρ 0.895; `inverted_u` true | Yes |
| T6 | β = 2/3 still increases | `main.tex:100` | `beta_067_lambda_000.raw_norm` | ρ 0.821, peak at last bin | Yes |
| T7 | mixing removes decreasing branch | `main.tex:60`, `101` | `beta_000_lambda_100.raw_norm` | ρ 0.0013; 0.948800 → 0.948878 | Yes |
| T8 | norms rise to about 10^3 researchers | `main.tex:81` | external (Murray et al. 2023) | — | Not traceable to the repository; not checked |
| T9 | radial exponents 1/2 and 1; Table I row forms | `main.tex:93`, `436–461` | `results/realized_graph_audit/checks.json` : `population[*].checks` | identity residuals ≤5.69e-14 (README) | Numerically checked in repository; the paper does not cite the check |
| T10 | raw vs balanced 0.55–1.15% | `main.tex:687`; `factor_balance.tex:43` | `results/factor_balance/summary.json` : `conditions[*].final.relative_norm_l2_difference` | 0.005479–0.011499 | Yes; bit-identical in my replay |
| T11 | individual discrepancies up to 12.45% | `main.tex:687` | same : `maximum_node_relative_difference` (max) | 0.124533 | Yes |
| T12 | final relative Gram imbalance 0.0786–0.163 | `factor_balance.tex:41` | same : `final.relative_imbalance` | 0.078561–0.163026 | Yes |
| T13 | absolute imbalance 0.407–0.411 → 225–910 | `factor_balance.tex:42` | `initial/final.imbalance_frobenius` | 0.4072–0.4111 → 225.19–910.26 | Yes |
| T14 | cutoff (3+√5)/2 t ≈ 2.62t | `main.tex:911` | analytic | 2.618 | Yes |
| T15 | elasticity about 0.31 at cutoff | `main.tex:937` | analytic | 0.309 | Yes |
| T16 | five graphs, n = 600, attachment 3 | `main.tex:960–961` | `turnover_surface/summary.json` : `config` | 5, 600, 3 | Yes |
| T17 | 144,000 pairs per condition and epoch | `main.tex:965` | `checks.events_per_condition_per_epoch` | 144000 | Yes |
| T18 | β, λ ∈ {0, 1/3, 2/3, 1} | `main.tex:971`, `983` | `analysis_plan.calibration_betas`, `calibration_context_mix` | as stated | Yes |
| T19 | 24 dimensions, 4 epochs, 5 negatives | `main.tex:989` | `config` | 24, 4, 5 (batch 256, LR 0.025→0.0025 unstated) | Yes |
| T20 | 3.507 → 2.297, ρ = −0.758 | `main.tex:996` | `beta_000_lambda_000.raw_norm.{lowest_degree_bin, highest_degree_bin, spearman_degree}` | 3.50680, 2.29722, −0.75812 | Yes |
| T21 | 2.496 → 4.802, ρ = 0.895 | `main.tex:997` | `beta_100_lambda_000.raw_norm` | 2.49603, 4.80164, 0.89536 | Yes |
| T22 | peak 3.315 near degree 6; +0.093 over lowest; 0.361 over highest | `main.tex:998` | `beta_033_lambda_000.raw_norm.{peak, peak_degree, peak_minus_lowest, highest_minus_peak}` | 3.31471, 6.000, 0.09333, −0.36093 | Yes |
| T23 | 2.861 → 3.960, ρ = 0.821 | `main.tex:1000` | `beta_067_lambda_000.raw_norm` | 2.86101, 3.95962, 0.82080 | Yes |
| T24 | 0.9488 in both lowest and highest bins | `main.tex:1020` | `beta_000_lambda_100.raw_norm.{lowest,highest}_degree_bin` | 0.948800, 0.948878 | No: highest rounds to 0.9489 (R3-12) |
| T25 | observed peak near 7.73; model monotone to 16.04 | `main.tex:1026` | `held_out_targets.holdout_t3e-4.raw_norm_prediction.{actual,predicted}_curve.peak_degree` | 7.731; 16.044, `inverted_u` false | Yes |
| T26 | pooled RMSE 0.013; peak in correct bin | `main.tex:1027` | `holdout_t3e-4.exposure_matched_prediction.raw_norm.rmse_of_pooled_curve`; peak bins | 0.013192; 4 = 4 | Yes |
| T27 | c_i = 1/600, q = π, 5 negatives, rank 24 | `main.tex:1033` | `results/window_training_continuation/plan.json` : `config` | 600 nodes, 5, 24 | Yes |
| T28 | up to 3000 additional iterations | `main.tex:1034` | `plan.json` : `block_iterations × max_blocks` | 250 × 12 | Yes |
| T29 | norm gap −0.3258 ± 0.0155 → −0.2173 ± 0.0123 | `main.tex:1035` | `.../report/summary.json` : `windows[2,3].norm_drop.absolute_{mean,se}` | −0.325805 ± 0.015527; −0.217347 ± 0.012346 | Yes |
| T30 | KL gap −0.2980 ± 0.0142 → −0.1303 ± 0.0063 | `main.tex:1035` | `windows[2,3].kl_drop.absolute_{mean,se}` | −0.298046 ± 0.014201; −0.130276 ± 0.006333 | Yes |
| T31 | weakens in every graph and start | `main.tex:1036` | `endpoints[*].curve.high_minus_penultimate` | 10/10 weaken (my check) | Yes |
| T32 | relative norm decline 12.31% → 10.03% | `main.tex:1037` | `windows[2,3].norm_drop.fractional_mean` | 0.123056, 0.100279 | Yes |
| T33 | relative KL declines increase | `main.tex:1037` | `windows[2,3].kl_drop.fractional_mean` | 0.2856 → 0.3193 | Yes |
| T34 | 7/10 and 10/10 stable fits | `main.tex:1038`; Fig. 2 labels | `windows[2,3].stability_reached` | 7, 10 | Yes |
| T35 | 500-iteration endpoints attenuate monotonically | `main.tex:1041` | `results/window_training_exploration/report/tables.md` | −1.256, −0.596, −0.376, −0.217 | Yes |
| T36 | gaps 1.303e-6 and 7.327e-7 | `main.tex:1042` | `report/summary.json` : `endpoints` (seed 3/200003, seed 4/200004) `.objective_suboptimality_upper_bound` | 1.30289e-6, 7.32691e-7 | Yes |
| T37 | highest-bin change 5.01% and 1.13% | `main.tex:1042` | same : `max_relative_bin_change` | 0.050119, 0.011330 (maximum attained at the highest bin; my check) | Yes |
| T38 | forty final fits | `main.tex:1057` | `results/norm_identifiability/empirical/tables.md` | 40 rows | Yes |
| T39 | normalized gap −0.1129 ± 0.0054 → −0.0934 ± 0.0053 | `main.tex:1058` | `empirical/summary.json` : `windows[2,3].normalized_curve.upper_gap_{mean,se}` | −0.112947 ± 0.005360; −0.093384 ± 0.005299 | Yes |
| T40 | weakens in every paired graph and start | `main.tex:1058` | `tables.md` "Absolute decline weakens" | 10/10 | Yes |
| T41 | start differences 0.984–1.187 (w1), 0.487–0.886 (w2) | `main.tex:1059` | `tables.md` "Paired starts" | 0.98445–1.18684; 0.48660–0.88620 | Yes |
| T42 | at most 0.00452 (w5) and 5.54e-5 (w10) | `main.tex:1059` | same | 0.0045122; 5.5382e-5 | Yes as bounds; 0.00451 by rounding (R3-12) |
| T43 | full-support bound uninformative for all ten w10 fits | `main.tex:1062` | `tables.md` "Full-support" | normalized bound 1.4142 (√2) for all 10 | Yes |
| T44 | within 10^-6 of the loss infimum | Fig. 3 caption, `main.tex:1068` | `exact/exact.json` : `runs[1].examples[*].excess_loss` | 4.2857e-7 | Yes |
| T45 | Gensim settings (300n walks, length 5, 64 d, window 1, 5 negatives, 1 worker, 5 epochs) | `main.tex:1079` | `results/gensim_threshold_sweep/summary.json` : `config`, `workers`, `window` | as stated; `min_count` 50 unstated | Yes |
| T46 | peak at lowest bin / 6 / 7.73 | `main.tex:1082` | `peak_comparison.fixed_epochs_actual_peak_degrees` | 2.985, 6.000, 7.731 | Yes (corrected rerun identical) |
| T47 | rise 0.372 → 0.569 | `main.tex:1084` | `conditions.fixed_epochs.{t1e-3,t3e-3}.raw_norm.peak_minus_lowest` | 0.37240, 0.56880 | Yes but superseded: corrected 0.3746, 0.5600 |
| T48 | drop 0.537 → 0.238 | `main.tex:1084` | `...highest_minus_peak` | −0.53716, −0.23816 | Yes but superseded: corrected 0.5272, 0.2570 |
| T49 | eight, six, five epochs | `main.tex:1101` | `budget_matched.*.epochs_min` | 8, 6, 5 | Yes |
| T50 | budget-matched peaks 2.98, 6.00, 7.73 | `main.tex:1102` | `peak_comparison.budget_matched_actual_peak_degrees` | 2.985, 6.000, 7.731 | Yes (corrected rerun: same peaks) |
| T51 | controlled fit too high by 4, 2, 1 bins | `main.tex:1104` | `controlled_predicted_peak_degrees` vs actual | [7.73, 16.08, 16.08] vs [2.98, 6.00, 7.73] | Yes |
| T52 | unfiltered: all five graphs interior maxima, both readouts | `main.tex:1110` | `results/subsampling_origin/summary.json` : `per_graph[threshold=0].interior_peak` | 10/10 true | Yes |
| T53 | pooled maximum near 7.73; raw drop 0.244 | `main.tex:1110` | `pooled[threshold=0, raw_norm].{peak_bin, peak_minus_highest}` | bin 5 (7.731); 0.243978 | Yes |
| T54 | forty trajectories | `main.tex:1113` | `results/training_trajectory/completion.json` | 40 runs | Yes |
| T55 | 0.025 → 0.0001, then floor; constant 0.0025 | `main.tex:1114` | `training_trajectory/plan.json` : `config.constant_rate`; source plan alpha/min_alpha | 0.0025; 0.025/0.0001 | Yes |
| T56 | checkpoints 1, 2, 5, 10, 20 | `main.tex:1115` | `plan.json` : `checkpoints` | [0, 1, 2, 5, 10, 20] | Yes |
| T57 | 20 five-epoch checkpoints reproduce baseline exactly | `main.tex:1116` | `training_trajectory/README.md`; `completion.json` | stated; my independent replay of the 20 baselines is bit-identical | Yes |
| T58 | interior maxima by epoch 2, through epoch 20 | `main.tex:1117` | `summary.json` : `per_graph` (baseline, t = 0) | 0/5 at epoch 1; 5/5 at 2, 5, 10, 20 | Yes |
| T59 | constant rate: no interior maximum; epoch-20 curves increasing for both readouts | `main.tex:1118` | `per_graph` (constant, t = 0) | 0/5 at all checkpoints; 10/10 strictly increasing | Yes |
| T60 | t = 0.003 same contrast | `main.tex:1119` | `per_graph` (t = 0.003) | baseline 5/5 from epoch 2; constant 0/5 | Yes |
| T61 | expected loss decreases at every interval in all 40 | `main.tex:1135` | `summary.json` : `progress[*].expected_loss` | 0 violations | Yes |
| T62 | six occupied bins | Fig. 5 caption, `main.tex:1129` | `pooled[*].degree` | length 6 | Yes |
| T63 | 400 trees, minimum leaf 40 | `main.tex:1157` | `prospective_validation/protocol.json` : `model.parameters` | 400, 40 (`max_features` 2, seed 20260813) | Yes |
| T64 | LOSO R² 0.828, Spearman 0.932 | `main.tex:1158` | `protocol.json` : `calibration_leave_one_seed_out.raw_norm` | 0.828418, 0.932465 | Yes, historical; corrected 0.821/0.933 and 0.822/0.935 |
| T65 | n = 600, attachment 3, triangle probability 0.35 | `main.tex:1160` | `protocol.json` : `config` | as stated | Yes |
| T66 | NetScience LCC 379 nodes, 914 edges | `main.tex:1160` | not stored; recomputed from `are_angles_important/netscience/netscience.gml` | 379, 914 | Yes (recomputed) |
| T67 | 300n stationary-start walks, length 5 | `main.tex:1162` | `protocol.json` : `config`; `prospective_norm_turnover.py:198–216` | stationary starts confirmed in code | Yes |
| T68 | 23,685 node-level predictions | `main.tex:1163` | `frozen_predictions.jsonl` line count | 23685 | Yes |
| T69 | hashes recorded before training; no outcome file at freeze | `main.tex:1163` | `freeze_manifest.json` | 4/4 hashes match; `target_outcomes_present_at_freeze` false (self-reported) | Hashes match; timing not verifiable from git (R3-06) |
| T70 | criteria: 0, 8/9, 0.881, nondecreasing in 3 families | `main.tex:1166` | `summary.json` : `criteria`, `peak_order` | 0, 8/9, 0.8806, 3 | Yes (historical) |
| T71 | three boundary misclassifications; NetScience misses by two bins | `main.tex:1167`; Fig. 6 caption | `summary.json` : `curves[*].{predicted,actual}.peak_bin` | BA 3e-4: 1 vs 0; clustered 3e-3: last bin; NetScience 3e-4: error 2 | Yes |
| T72 | baseline within-curve ρ 0.293 vs 0.842 | `main.tex:1171` | `results/prospective_baselines/table.tex` (Original rows) | 0.293, 0.842 | Yes (byte-identical regeneration) |
| T73 | ExtraTrees seed 20260813, `max_features` 2 | `supp.tex:22` | `protocol.json` | as stated | Yes |
| T74 | 9/9 vs 8/9 (corrected retained) | `supp.tex:36` | `table.tex` | 9/9, 8/9 | Yes |
| T75 | raw-norm ρ beats baseline in all 45 groups, every dataset | `supp.tex:37` | `prospective_baselines/summary.json` : `datasets.*.{pairs_kl,degree_threshold}.raw_norm_replicates` | 45/45 in each of 3 datasets | Yes |
| T76 | balanced: loses 1 of 45 (original), wins 45/45 corrected | `supp.tex:44` | same, balanced | 44/45; 45/45; 45/45 | Yes |
| T77 | degree alone fails pooled criterion; pairs alone pass all four (corrected raw) | `supp.tex:38` | `table.tex` | 0.178/0.169/0.128 < 0.5; "Yes" | Yes |
| T78 | BA peaks at 1e-3 and 3e-3 both about 6.77 | `supp.tex:52` | `prospective_validation/summary.json` : `peak_order.barabasi_albert.actual_peak_degrees` | 6.7696, 6.7696 | Yes |
| T79 | Table S3 entries | SM Sec. S3 | `results/factor_balance/summary.json` | identical to my replay | Yes |
| T80 | NetScience distributed with the repository | `main.tex:1219` | `are_angles_important/netscience/netscience.gml` | SHA-256 `ede685c5...` = `protocol.json` `netscience_sha256` | Yes (location, terms, and weights undocumented; R3-07) |

## Figure and table provenance

| Item (paper) | Producing script (last code commit) | Inputs | Environment (evidence) | Regenerable from committed files? | My regeneration | Status and flags |
|---|---|---|---|---|---|---|
| Fig. 1 `turnover_surface_qualified` (p. 9) | `scripts/simulate_turnover_surface.py` (`5902c3c`) → `turnover_surface.pdf`; then `scripts/audit_realized_graph.py:590–660` (`6e2b79f`) replaces three labels by PDF redaction | `node_results.csv`, `exposure_matched_predictions.csv` (gitignored) | Matplotlib 3.11.1 per PDF metadata, unpinned (PEP 723 ranges); PyMuPDF 1.26.7 for the edit | No | Experiment rerun (75 s): 941 summary fields match to ≤1e-11; pinned-env plot has old causal titles and different axes | Not flagged historical; SE caption wrong (R3-04); non-embedded Helvetica (R3-05) |
| Fig. 2 `wide_window_comparison` (p. 10) | `scripts/window_training/report_continuation.py` (`31108fa`) | `results/window_training_continuation/*.json.gz`, `*.npy.gz`; exploration graphs | pinned (`window_training_exploration_requirements.txt`); Matplotlib 3.10.8 | Yes | Pixel-identical; `tables.md` identical; `summary.json` differs at ≤2.5e-15 | Current |
| Fig. 3 `norm_identifiability` (p. 10) | `scripts/window_training/plot_norm_identifiability.py` after `normalized_profiles.py`, `exact_norm_profiles.py` (`31108fa`) | `results/norm_identifiability/{empirical,exact}` | pinned; Matplotlib 3.10.8 | Yes | Pixel-identical; `--root` crash before manifest (R3-16) | Current |
| Fig. 4 `gensim_threshold_sweep` (p. 11) | `scripts/simulate_gensim_threshold_sweep.py` (`5902c3c`) importing `simulate_norm_frequency.py` as of `5902c3c` (since changed in `87e2785`) | `node_results.csv` (gitignored); `turnover_surface/summary.json` | Matplotlib 3.11.1; stock Gensim 4.4.0 with the −1 defect; walk-level counts; permuted table | No; the current script now produces corrected numbers | Historical reconstruction (46 s): summary to ≤5e-15, 5 defect events; corrected run (45 s): numbers in R3-02 | Superseded per `results/README.md`; caption says historical; embedded title says "validation" (R3-02) |
| Fig. 5 `unfiltered_schedule_comparison` (p. 12) | `scripts/training_trajectory/report.py` (`b3bd902`); training by `run.py` | `results/training_trajectory/runs/*.json` | pinned + patched Gensim; Matplotlib 3.10.8 | Report yes; training refused on other machines (R3-03) | Pixel-identical; `summary.json`, `README.md`, `report_manifest.json` byte-identical | Current, corrected |
| Fig. 6 `prospective_validation` (p. 13) | `archive/negative_sampling_control/calibration_source/scripts/prospective_norm_turnover.py` (= `5902c3c`); plot stage in current script | `results/prospective_validation/outcomes.jsonl` | Matplotlib 3.11.1; stock Gensim 4.4.0 (defect), integer-token table, walk-level counts | Plot yes | Data curves identical; 10% of pixels differ (tick labels and Matplotlib version) | Historical, bug-affected (`results/prospective_validation/README.md`), not labeled in the paper (R3-01); SE caption wrong (R3-04) |
| Table I `tab:radial-exponents` (p. 5) | hand-written, `main.tex:436–461` | — | — | — | Identities numerically checked by `scripts/audit_realized_graph.py` (`results/realized_graph_audit/checks.json`) | Check not cited |
| Tables S1–S2 | `scripts/prospective_baselines.py` (`e97b7ed`) | `prospective_validation/`, `negative_sampling_revalidation/` | pinned | Yes | `table.tex`, `balanced_table.tex`, `README.md`, `summary.json` byte-identical (13 s); refit delta 0 | Current |
| Table S3 | `scripts/factor_balance.py` (`475f3e5`) | replays `subsampling_origin` | pinned + patched Gensim | Script refuses (R3-03) | With the hash key ignored: 20/20 runs bit-identical, all table values identical (41 s) | Current, corrected |

## Questions to the authors

1. Was any commit containing `frozen_predictions.jsonl` pushed, or otherwise timestamped outside your machines, before target training began? If so, which?
2. Was the evaluation code (binning, peak rule, criteria) unchanged between the freeze and evaluation, and how can a reader verify that?
3. Which numbers should a reader cite for the prospective test: the frozen historical ones or the corrected retrospective ones? The abstract currently cites the former without qualification.
4. Why does Fig. 4 remain on pre-correction outputs when the corrected sweep, budget-matched arm included, runs in under a minute?
5. Which Python, NumPy, Gensim, and Matplotlib versions produced Figs. 1, 4, and 6? The PDFs say Matplotlib 3.11.1, which no file pins.
6. Under what terms do you redistribute NetScience, and was ignoring its edge weights a deliberate choice?
7. Will the archived release include the node-level data behind Figs. 1 and 4?
8. Has any corrected run been reproduced on Linux/x86, where bitwise identity is less likely?

## Recommendation and confidence

**Recommendation: major revision.**
The computational record is far better than typical. Hash bookkeeping is complete, most artefacts regenerate exactly, and the corrections I could rerun leave the qualitative conclusions intact.
The revision is still substantive. Two main-text figures and the abstract's headline statistics rest on a pipeline the repository itself declares defective, and the paper discloses this only for Fig. 4. Error bars in two figures are misdescribed. Fig. 1 cannot be regenerated from the repository. A reader following the documentation cannot rerun Fig. 5 or Table S3, and the Data Availability statement is inaccurate.
Each item has a concrete fix, most within a day, and my reruns indicate that the corrected numbers differ only slightly.

**Confidence: 4/5.** I traced and reran the numerical side directly, rebuilt the patched runtime, reran the README revalidation end to end, and checked every recorded hash. I did not re-derive the theory, did not rerun the window-continuation optimizations or the 40 trajectory trainings, and tested only macOS arm64 (the authors' recorded platform is macOS 15.7.9 arm64).

## Action items

| ID | Severity | Location | Problem | Proposed fix | Type | Effort |
|---|---|---|---|---|---|---|
| R3-01 | blocker | abstract `main.tex:62–64`; Sec. XI H `1155–1173`; Fig. 6 caption `1143–1149` | Headline prospective numbers (0, 8/9, 0.881) and Fig. 6 come from the bug-affected pipeline; not disclosed | Label as uncorrected; add corrected retrospective values (0.875/0.873, 8/9, 0) and the corrected curves (`negative_sampling_revalidation/predictor_revalidation.png`) | PR | ≤1day |
| R3-02 | blocker | Fig. 4; `main.tex:1078–1104`; `simulate_gensim_threshold_sweep.py:261` | Main-text figure and four numbers from superseded outputs; "not yet repeated" although the corrected sweep takes 45 s | Rerun corrected sweep (both arms), regenerate Fig. 4 with graph-level SEs, update numbers, remove "validation" suptitle | PR | ≤1day |
| R3-03 | major | `check_control_gensim.py:47–51`; `factor_balance.py:129`; `training_trajectory/run.py:206–208`; `schedule_fork.py:290–295`; `window_training/verify.py:474` | Gate requires the authors' binary hash; Table S3 and Fig. 5 scripts fail for every reader although results are bit-reproducible | Compare `wrapper`, `dot_products`, `passed` only; log the hash; add a test | PR | ≤1h |
| R3-04 | major | Fig. 1 caption `main.tex:1012`; Fig. 6 caption `1146`; `simulate_exact_center_exposure.py:308`; `prospective_norm_turnover.py:454` | Captions claim seed-level SEs; code draws node-pooled SEs (ratio 0.13–13.8) | Recompute seed-level SEs in common bins and redraw, or correct the captions | PR | ≤1day |
| R3-05 | major | Fig. 1; `audit_realized_graph.py:590–660`; `simulate_turnover_surface.py:979, 997, 1068`; `.gitignore:7` | Figure is a PDF label edit; node data gitignored; pinned env renders differently; non-embedded Helvetica | Fix titles in code, commit gzipped node CSVs, regenerate in pinned env with explicit limits, drop PDF surgery | PR | ≤1day |
| R3-06 | major | `main.tex:1155`, `1163`, `1210`; `freeze_manifest.json`; protocol line 4 | Freeze self-attested; predictions and outcomes first committed together (`5902c3c`); evaluation code and bins not hashed | Disclose internal freeze and its limits in the paper; for future tests, externally timestamp predictions plus evaluation-code hash | PR | ≤1h |
| R3-07 | major | Data Availability `main.tex:1216–1220`; `README.md:94`; `are_angles_important/netscience/` | No URL/commit/DOI; "generated data accompany" false for Figs. 1 and 4; NetScience terms, location, weights, and LCC undocumented | Cite URL and commit, mint a Zenodo DOI, move NetScience to `data/` with provenance and terms, state data license, fix README | PR | ≤1day |
| R3-08 | major | `README.md:8–40`, `68–91`; `environment*.yml`; `pyproject.toml`; `uv.lock`; `analysis/*requirements*.txt`; `.python-version` | Eight environment specs; README env fails test collection; `make test` errors; "26 tests" vs 136; Python 3.13 vs 3.13.12; Matplotlib 3.11.1 unrecorded | Single lock for paper scripts, separate historical lock, archive conda files, `importorskip`, correct counts | PR | ≤1day |
| R3-09 | major | `Makefile`; `workflow/Snakefile`; `README.md` | No per-figure map or entry point; `make all` runs an empty template; Fig. 4 reproducible only via undocumented git archaeology | Figure → script@commit → inputs → env → command → tolerance table; `make figures` or `scripts/reproduce_paper.py` | PR | ≤1day |
| R3-10 | major | `main.tex:1108`; SM | Three implementation defects named but not explained; affected results unmapped | SM section: defects, toy examples, upstream issue #3665/PR #3668, effect sizes, affected-results table | PR | ≤1day |
| R3-11 | minor | `results/{turnover_surface, gensim_threshold_sweep, norm_identifiability, window_training_continuation, factor_balance}`; `results/prospective_validation/README.md` | Missing READMEs; no script/commit/env for Fig. 6 data; current scripts diverge from stored outputs | README per folder with script@commit, command, env, seeds, status | PR | ≤1day |
| R3-12 | minor | `main.tex:991`, `1020`, `1059`; heading of Sec. XI E | 0.9488 should be 0.9489 for the highest bin; 0.00452 vs 0.00451; balanced λ = 1 flag; "validation" heading | Correct rounding and wording | PR | ≤1h |
| R3-13 | minor | Figs. 1, 4, 6 | Panel letters vs caption; missing error bars in 1(e,f); boundary "peak shift"; overlapping tick labels; in-figure suptitles | Regenerate with caption-consistent labels and clean ticks | PR | ≤1day |
| R3-14 | minor | repository root, `notebooks/`, `libs/`, `results/Changing *`, `scripts/` | Tracked caches and logs, 117 MB of unrelated notebooks, LFR `.npz`, third-party zip, template folders, K-means README title, hard-coded paths | Decide release scope; publish a curated archive; move legacy material out | ISSUE | ≤1day |
| R3-15 | minor | `paper/current/Makefile`; `norm_geometry_supplement.tex`; `xr-hyper` | No figure dependencies; orphan supplement; xr needs `supp.aux` at APS | Add figure and table prerequisites; reference or drop the orphan; plan xr for submission | PR | ≤1h |
| R3-16 | minor | `scripts/window_training/plot_norm_identifiability.py:217` | `--root` outside the repo crashes after writing figures | Hash the script path relative to the repository root, independent of `--root` | PR | ≤1h |
| R3-17 | minor | `results/prospective_validation/freeze_manifest.json` | Keys name roles; files must be guessed; no code or environment hash | Sidecar mapping keys to paths plus code and environment hashes | PR | ≤1h |
| R3-18 | minor | Figs. 1, 2, 4, 5, 6 | Type 3 fonts; non-embedded Helvetica in Fig. 1 | `pdf.fonttype: 42`; regenerate | PR | ≤1h |
| R3-19 | minor | Sec. XI A, E, H | Batch size, LR schedule, initialization, `min_count = 50`, `shrink_windows=False`, unweighted NetScience appear only in code | Add to methods text | PR | ≤1h |
| R3-20 | minor | verification scripts; `report/summary.json` manifests | Byte-level digests flag reruns that differ at 1e-15 | Tolerance-based comparison for numeric outputs; digests for inputs and frozen files | ISSUE | ≤1day |
