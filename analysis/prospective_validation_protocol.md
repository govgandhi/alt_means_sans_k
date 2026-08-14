# Prospective norm-turnover validation protocol

Frozen on 2026-08-13 before training any target embedding.
This is a code-frozen prospective validation within the project, not an externally registered report.

## Calibration data

- Graph family: configuration-model graphs derived from Barabasi--Albert degree sequences.
- Graph seeds: 0--4.
- Conditions: external word2vec subsampling at \(t\in\{3\times10^{-4},10^{-3},3\times10^{-3}\}\).
- Calibration outcomes: raw center-vector norm (primary) and balanced norm (secondary).

## Locked predictor

- Inputs available before embedding training:
  1. log realized positive center-context pair count;
  2. empirical \(D_{\mathrm{KL}}(T_i\Vert q)\) in the externally subsampled corpus.
- Model: `ExtraTreesRegressor` with 400 trees, `min_samples_leaf=40`, both features considered at each split, and random seed 20260813.
- No degree, graph-family label, or target outcome enters the predictor.
- Model adequacy on calibration data is reported with leave-one-graph-seed-out predictions; its settings remain fixed regardless of that result.

## Prospective targets

- Direct Barabasi--Albert graphs: \(n=600\), attachment parameter 3, seeds 101--105.
- Holme--Kim power-law cluster graphs: \(n=600\), attachment parameter 3, triangle probability 0.35, seeds 201--205.
- NetScience coauthorship network: largest connected component, with walk and optimizer seeds 301--305.

Each target uses 300 stationary-start walks per node, walk length 5, window size 1, 64 dimensions, five negatives, unigram negative sampling, and five epochs.
Subsampling is applied outside Gensim with the standard retention law, after which Gensim receives the retained walks with internal subsampling disabled.
This makes the realized center-context pair counts and empirical context distributions observable before target training.

## Frozen sequence

1. Fit the predictor using calibration embeddings.
2. Generate target walks and externally subsample them.
3. Record realized pair counts and context KL.
4. Write and hash every target norm and peak prediction.
5. Verify the hashes, then train target embeddings.
6. Report every target condition, including failures.

The evaluation stage may not refit the predictor or replace a frozen prediction.

## Primary tests

For the nine graph-family-by-threshold curves:

- median absolute predicted-peak error no greater than one degree bin;
- at least two thirds of predicted peaks within one degree bin of the observed peak;
- pooled Spearman correlation between predicted and observed node norms at least 0.5;
- observed peak degree nondecreasing with weaker subsampling in at least two of the three graph families.

Passing all four criteria supports qualitative transport of the two-force mechanism.
Failure of any criterion limits the claim to the controlled configuration-model regime and will be reported in the manuscript.

