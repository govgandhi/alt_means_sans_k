Put results (e.g., figures) here. 

## SGNS implementation corrections

The current corrected revalidation is in
[`negative_sampling_revalidation/`](negative_sampling_revalidation/README.md).
It repeats both calibration and all original target cases with corrected visit
counts, negative-table assignment, and checked BLAS arithmetic. Its design is
retrospective; it does not replace the historical prediction freeze.

`prospective_validation/`, `norm_frequency_simulation/`, and
`gensim_threshold_sweep/` contain outputs produced before these corrections.
The first now has a separate corrected revalidation. The other two have **not**
been revalidated and should not be presented as outputs of the corrected code.
The manuscript's existing figures/numbers still reference those historical files.
Preserve them for comparison; use a new `--output-dir` for any future rerun.

`negative_sampling_control/` is the earlier calibration-only diagnostic with
legacy thinning; its source snapshot and reproducer preserve that distinction.
