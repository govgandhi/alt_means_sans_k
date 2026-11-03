Put results (e.g., figures) here. 

## SGNS implementation corrections

The current corrected revalidation is in
[`negative_sampling_revalidation/`](negative_sampling_revalidation/README.md).
It repeats both calibration and all original target cases with corrected visit
counts, negative-table assignment, and checked BLAS arithmetic. Its design is
retrospective; it does not replace the historical prediction freeze.

`prospective_validation/`, `norm_frequency_simulation/`, and
`gensim_threshold_sweep/` contain outputs produced before these corrections.
The first now has a separate corrected revalidation. The corrected internal
subsampling comparison and no-filter control are in
[`subsampling_origin/`](subsampling_origin/README.md). Forty continuous runs in
[`training_trajectory/`](training_trajectory/README.md) test training duration and
learning-rate history. They show that unfiltered turnover depends on the schedule.
The historical retained-token budget-matched sweep and the other original
`norm_frequency_simulation` conditions have **not** been rerun. Their saved outputs
must not be presented as corrected results. The manuscript labels the historical
Gensim figure separately from the new control and trajectory evidence.
Preserve them for comparison; use a new `--output-dir` for any future rerun.

`negative_sampling_control/` is the earlier calibration-only diagnostic with
legacy thinning; its source snapshot and reproducer preserve that distinction.
