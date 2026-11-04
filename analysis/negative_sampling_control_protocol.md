# Negative-sampling control for issue #17

Status: design recorded before this diagnostic's training. This is a retrospective
calibration diagnostic, not a new prospective validation. The original frozen
protocol and results remain unchanged.

Runtime amendment, before the corrected-runtime sweep: the first complete sweep
reproduced every original calibration raw norm exactly, but emitted Gensim's
`our_dot_float` warning. Source inspection and a scalar check identified an upstream
defect: a valid BLAS dot product of −1 returns 0. The first sweep, its exact design,
script, environment and log are preserved under
`archive/negative_sampling_control/gensim_4_4_0_unpatched/`. Those outputs are
superseded for scientific interpretation by the corrected-runtime sweep.

The corrected runtime rebuilds only the Word2Vec extension from the official
Gensim 4.4.0 source, verifies its archive checksum, and changes `sdot_ptr` and
`dsdot_ptr` from `except -1 nogil` to `noexcept nogil`. This is the declaration fix
proposed in [upstream PR #3668](https://github.com/piskvorky/gensim/pull/3668),
addressing [issue #3665](https://github.com/piskvorky/gensim/issues/3665). No sampling
logic changes in that extension. The experiment independently records the compiled
extension hash and verifies dot products −2, −1, −0.5, 0 and 1 before training.
All three arms use this same corrected runtime. Compare both complete sweeps to
measure sensitivity to the numerical correction rather than assuming cancellation.

## Question and scope

Does the norm-versus-degree pattern persist when negative proposals use the
original token distribution, held fixed across filtering thresholds? Separately,
does correcting integer-token table assignment change the original results?

The reference world is the existing five configuration-model calibration graphs,
seeds 0–4. Use all three original thresholds (0.0003, 0.001, 0.003), original walk
and optimizer seeds, 600 nodes, 300n walks of length five, window one, dimension
64, five nominal negatives, five epochs, alpha 0.025 ending at 0.0001, and one
Gensim worker. These values come from the frozen protocol and original constructor
defaults; none are selected using the new outcomes. No regression model is fit.

## Three arms and the controlled departure

1. **legacy:** original Gensim cumulative table with integer node names.
2. **retained:** correctly map retained-token counts to vocabulary positions.
3. **raw:** correctly map original-token counts to the same vocabulary positions.

The retained-versus-raw contrast answers issue #17. The legacy-versus-retained
contrast isolates a separate implementation defect. A toy check of the declared
Gensim 4.4.0 reproduces it: for node counts {2:8, 0:4, 1:2}, vocabulary order is
[2,0,1], but the original table assigns slot weights [4,2,8], not [8,4,2].
`make_cum_table` passes integer positions to `get_vecattr`, which first interprets
them as token keys. A test with string names supplies an independent reference.

Source: Gensim 4.4.0 `word2vec.py`, `make_cum_table`; `keyedvectors.py`,
`get_index` and `get_vecattr`. The exact installed package version is pinned and
recorded. Correct tables are cumulative counts in vocabulary-slot order, rounded
to Gensim's domain 2^31−1. Small quantization differences are checked explicitly.

Every arm is a deep copy of one untrained template per graph/threshold, preserving
vocabulary order, initial center/context vectors, RNG state, corpus, internal
subsampling disabled, and learning-rate schedule. Only the cumulative negative
table changes. Raw probabilities are fixed across thresholds in node coordinates,
up to table rounding. Gensim skips negative proposals equal to the positive output
token without resampling, so *nominal* draws are fixed, not the number of accepted
negative updates. This behavior belongs to the sampler intervention.

## Preserve the positive examples

Regenerate the exact existing external thinning, then verify degree, positive-pair
counts, retained-token counts and pair-marginal KL against all saved calibration
rows before training. Use that identical retained corpus for all three arms.

The original walk counter undercounts repeated visits (`counts[walk] += 1`). Keep
that legacy behavior only for reconstructing the existing filtering. Independently
count **every occurrence**, including repeats, for the raw negative distribution.
Changing thinning as well would prevent isolation of the sampler effect. This
diagnostic does not repair or validate the original retention procedure.

The stored KL is measured against the positive-pair context marginal; it is not
KL against any arm's negative distribution. It is only checked as a fixed corpus
feature here. No target outcomes or predictions are used for model selection.

## Readouts and interpretation

Save every node's raw and balanced norms, original/retained/legacy counts,
positive-pair counts and normalized cumulative-table masses. These masses approximate
proposal probabilities; integer boundary and modulo effects produce negligible
differences, and skipped collisions are not counted as negative updates.
Report every graph separately
at every threshold, including failures. Primary response: raw center norm;
secondary: balanced norm. Use the existing six log-degree quantile bins, formed
within each graph independently of outcomes. Report each curve, peak bin, peak
height above each endpoint, paired mean changes, RMSE and rank correlation.
An interior maximum is descriptive; it is not a significance test of unimodality.

Plot individual graph curves and their equally weighted seed means where the
numbers of occupied bins agree. Graph seeds, not nodes or thresholds, are the
replication units. Do not use node-based error bars. No equivalence margin or
causal mediation claim is declared; close curves alone do not prove equivalence.

- Persistence under raw noise shows that changing the proposal distribution is
  unnecessary for that pattern in these calibration conditions.
- A shifted peak or amplitude identifies sensitivity to the negative sampler.
- Disappearance or reversal limits the original mechanism explanation.
- Mapping-repair effects must be reported separately from the raw/retained contrast.

Neither contrast proves the complete exposure/context mechanism, extends results
to held-out graph families, nor counts as a new prospective freeze. If new target
validation is needed, it requires a separately specified protocol. Preserve the
original files and distinguish this diagnostic from them in all claims.

## Reproduction and integrity gates

Use Python 3.13.12 and install the complete pinned environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r analysis/negative_sampling_control_requirements.txt
.venv/bin/python -m pip install -r analysis/negative_sampling_control_build_requirements.txt
.venv/bin/python scripts/build_control_gensim.py --install
.venv/bin/python scripts/check_control_gensim.py
.venv/bin/python -m pytest tests/test_negative_sampling_control.py
.venv/bin/python scripts/negative_sampling_control.py
```

For a fresh rerun, supply `--output /path/to/new/output`. The default is
`results/negative_sampling_control`. Repeating a completed invocation verifies
the plan and output hashes. The program writes its input hashes, environment and
design into `plan.json` before training; checks original inputs again afterward;
and writes a completion manifest only after all 45 models and all outputs finish.
The environment is pinned, but cross-platform floating-point identity is not
guaranteed. The legacy-versus-saved differences explicitly measure reproduction.

Tests cover duplicate-visit counts, integer-token assignment against a string-token
reference, preserved initialization/RNG/vocabulary, clone independence, unchanged
corpus training, and an identical-table null giving identical final vectors.
Original frozen features and file hashes must pass before interpreting results.
The build requires an existing C compiler. Its products and backups stay under
`.cache/gensim-control-build`; installation is restricted to a virtual environment.
The official sdist omits an unrelated `fastss.pyx`, so the build script compiles
only the Word2Vec extension used here, not the whole Gensim package. It preserves
the original installed extension and declarations before replacement. Reinstalling
the ordinary Gensim wheel undoes the fix; the runtime regression check detects it.
For historical reproduction in an unpatched environment only, use
`--allow-unpatched-blas --output /path/to/new/historical-output`.
