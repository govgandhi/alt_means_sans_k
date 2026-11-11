# Does turnover require Gensim subsampling? (issue 21)

This design is recorded before the corrected runs. It is retrospective: earlier,
buggy runs already suggested a peak without filtering. Those outcomes are not
used to tune this test. This is a bounded first test of issue 21, not a test of
convergence or a complete explanation of the norm curve.

## Fixed design and provenance

Use the configuration graphs and fixed-epoch settings from
`scripts/simulate_gensim_threshold_sweep.py`: 600 nodes, attachment 3, graph seeds
0–4, 180,000 stationary-start walks per graph, five tokens per walk, dimension 64,
window 1, five negative proposals, noise exponent 1, five epochs, minimum count 50,
initial learning rate 0.025 and Gensim's default final learning rate 0.0001.
Walk seeds are graph seed + 10,000 and optimizer seeds graph seed + 100,000, as in
the original sweep. Compare all three original thresholds (0.0003, 0.001, 0.003)
with `sample=0` (filtering disabled), without changing any other setting: 20 fits.
The original no-subsampling condition in `simulate_norm_frequency.py` supplies
that fourth setting. No parameter or seed is chosen using the new outcomes.

Use the corrected repeated-visit counts, vocabulary-slot negative table and
patched Gensim numerical extension from PR 35. Run the BLAS check before training.
Within each graph, verify identical raw walks, counts, vocabulary, initial vectors,
optimizer state and negative CDF across all four settings. Check the actual
`sample_int` array against the retention formula in Gensim 4.4.0's `prepare_vocab`.
Hash graphs, corpora, initial arrays, dependencies, source and numerical extension.
Save the plan before training and retain all node outcomes and all 20 checks.

Subsampling is internal to Gensim. Discarded tokens are removed before contexts
are formed. Fixed epochs and the same learning-rate schedule therefore do not
match accepted updates or positive pairs. Negative proposals have the same
distribution, but their realized draws need not match because filtering changes
RNG consumption. Report Gensim's effective-token accounting, explicitly not as a
positive-pair or gradient-update count. This is a literal filtering intervention,
not an exposure-only intervention.

## Readouts fixed before training

Raw input norm is primary; balanced SVD norm is secondary. Reuse the original
requested eight log-degree quantile bins, dropping duplicate boundaries. Report
each graph's curve, peak bin, peak degree and differences from both endpoint bins.
An interior maximum is descriptive, not a significance test of unimodality.
The peak is selected from the same curve; its gap is not a prespecified contrast.

Also report pooled curves for comparison with the historical sweep. For a separate
mean curve with standard errors, define common bin boundaries from all five
graphs' degrees, independently of norms. First average nodes within each graph
and bin, then average those graph means equally. Compute sample SD / sqrt(number
of observed graphs), not node-based standard errors. Show all individual curves,
bin node counts and graph counts; missing bins remain missing. Five seeds quantify
variation across these graph/walk/optimizer combinations, not universal uncertainty.
Do not silently change binning, select favorable seeds or omit failed settings.

If the unfiltered models show turnover, subsampling is not necessary for that
pattern under these settings. The strength and consistency of this observation
must be reported. A peak shift can still be a subsampling effect. Neither outcome
establishes why turnover appears, nor whether it persists at convergence, different
dimension, graph family or training budget. A matched-budget/epoch trajectory
experiment remains separate future work; no such results are claimed here.

## Reproduction and preservation

Use Python 3.13.12 and the pinned runtime/build requirements already supplied for
PR 35. In a virtual environment:

```sh
python3 -m pip install -r analysis/negative_sampling_control_requirements.txt
python3 scripts/build_control_gensim.py --install
python3 scripts/check_control_gensim.py
python3 -m pytest
python3 scripts/check_subsampling_origin.py
```

The default output is `results/subsampling_origin`. Use `--output` with a fresh
directory for an independent run. A repeated completed invocation verifies
hashes without changing files. An interrupted run can be rerun only with an
identical recorded plan. No historical result, frozen prediction, or manuscript
is modified. PR 35 must be present because this test depends on its fixes.
