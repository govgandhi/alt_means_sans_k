# Step 0: the norm–frequency problem (Fig. 1)

Fig. 1 of the manuscript shows that skip-gram vector norms rise and then fall with frequency in
published vectors, in text and in networks, and do so even on a synthetic graph (the expected
Barabási–Albert graph) where the exact optimum of the objective has no frequency dependence.

## Rebuild

From the repository root, in the pinned patched environment (see the root README; `.venv`):

```sh
make step0                       # or: scripts/step0/run_all.sh .venv/bin/python
```

This fetches and checksums the data, builds the corpora, trains 85 runs (about 60–70 minutes
with 8 parallel single-threaded runs on an Apple M4), computes the theory line and checks, and
writes `results/step0/fig1_problem.pdf`. To redraw the figure from the committed runs only:

```sh
.venv/bin/python -I scripts/step0/figure_paper.py --runs results/step0/runs \
  --anchor-factors data/external/pnas_murray2023/extracted/derived/Derived/Embeddings/precedence/factors/d300_ws1_precedence_w2vfactors.csv \
  --anchor-sizes data/external/pnas_murray2023/extracted/additional/Additional/2008-2019_inst_sizes.txt \
  --out results/step0/fig1_problem.pdf
```

Panel (a) needs the fetched PNAS files; the other panels need only `results/step0/runs`.

## Data (git-ignored, fetched by `fetch_data.py`)

| Dataset | Source | SHA-256 |
|---|---|---|
| PNAS `Derived.zip` (3.81 GB) | Figshare 13072790 v1, https://ndownloader.figshare.com/files/25117865 (CC BY 4.0) | `6224ecf0…ba39cc9` |
| PNAS `Additional.zip` (4.2 MB) | https://ndownloader.figshare.com/files/25117640 | `01e4d2bf…ca2425` |
| hep-th abstracts (12 MB) | KDD Cup 2003, https://www.cs.cornell.edu/projects/kddcup/download/hep-th-abs.tar.gz | `d4f33c10…e610026` |
| BlogCatalog `blogcatalog.mat` (1.3 MB) | Tang & Liu 2009, copy in the DeepWalk repository at commit `abe220b6` (the original ASU host did not respond on 2026-11-12) | `d4f4fb89…c259da8f` |

Full hashes are in `fetch_data.py`. Raw Web of Science trajectories behind the PNAS paper are not
public; the deposited network, vectors and organization sizes are used instead.

## Pipeline

| Script | Output |
|---|---|
| `fetch_data.py` | downloads, verifies, extracts the used members into `data/external/` |
| `corpora.py` | weighted walks on the PNAS precedence network (10 per node, 80 steps, started in proportion to strength, steps in proportion to weight), its strength-preserving null, tokenized hep-th text → `data/derived/step0/` |
| `population_corpus.py` | the population process: independent draws in proportion to strength |
| `blogcatalog_corpus.py` | Fig. 1(c): random walks on BlogCatalog, same walk law as the PNAS walks |
| `ba_corpus.py` | Fig. 1(d): walks on the expected graph of `networkx.barabasi_albert_graph(10000, 5, seed=k)`, i.e. independent draws in proportion to degree (N and m are our choices) |
| `train.py` | one Gensim run (corrected build, one worker, seeded) → `results/step0/runs/*.npz` |
| `theory.py` | exact optimum of the expected loss for the population process → `results/step0/theory.json`; `--corpus ba` → `theory_ba.json` |
| `checks.py` | corpus, transition, update-range, cutoff and replay checks → `results/step0/checks.json` |
| `figure_paper.py` | Fig. 1 (PRE double-column, vector PDF + 600-dpi PNG) |
| `figure.py`, `anchor_pnas.py` | diagnostic grid (raw and balanced norms, null) and the PNAS redraw |
| `diagnose_population.py`, `diagnose_unseen_pairs.py` | 5- vs 50-epoch diagnosis (for the next figure) |

## Settings

Gensim defaults (skip-gram): 100 dimensions, window 5 (shrinking), 5 negatives, exponent 0.75,
subsampling 1e-3, 5 epochs, minimum count 5. PNAS settings: 300 dimensions, window 1, exponent 1,
5 epochs, minimum count 50; subsampling off (not reported by the paper; our assumption).
Schakel & Wilson settings: 100 dimensions, window 10, subsampling 1e-4, 20 epochs, minimum count 1.

## Known limitations

- The PNAS deposit has 7,090 organizations in its factors table, against the paper's 6,580; it was
  posted before the final paper and may reflect an earlier analysis version.
- Our hep-th tokenizer approximates Schakel & Wilson's preprocessing (34.5k vocabulary vs their ~44k).
- Minimum counts remove 12% of BlogCatalog nodes under PNAS settings (degree 3 or less, about half
  of degree 4) and almost none under the other settings; none on the BA graph. On the PNAS network
  (diagnostic figure only) they remove 16% and 40%.
- The BA graph (N = 10,000, m = 5) has bins with at least ten nodes only up to degree ~125, so
  Fig. 1(d) spans 1.5 decades; within it the PNAS-settings curve only rises and the Schakel &
  Wilson curve only falls.
- Schakel & Wilson settings are text settings; on networks they are used as a third training
  setting, not as anyone's published network setup.
- At 50 epochs on the population process the curve inverts and the loss moves away from the
  optimum (see `results/step0/diagnose_*.json`); that belongs to the next figure, not Fig. 1.
