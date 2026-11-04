# alt_means_sans_k

Alternative to K-Means clustering methods for community detection of embedding vectors.

## Corrected SGNS experiments

The corrected sampler and target revalidation is documented in
[its protocol](analysis/negative_sampling_revalidation_protocol.md), with
[results and limitations](results/negative_sampling_revalidation/README.md).
Use the pinned experiment environment; the older notebook environments below do
not include the numerical correction:

```sh
uv venv --python 3.13.12 .venv
uv pip install --python .venv/bin/python \
  -r analysis/negative_sampling_control_requirements.txt \
  -r analysis/negative_sampling_control_build_requirements.txt
.venv/bin/python scripts/build_control_gensim.py --install
.venv/bin/python scripts/check_control_gensim.py
.venv/bin/python -m pytest
.venv/bin/python scripts/revalidate_negative_sampling.py \
  --output /tmp/fresh-revalidation > /tmp/revalidation-training.txt 2>&1
.venv/bin/python scripts/check_revalidation_repeat.py --output /tmp/fresh-revalidation
.venv/bin/python scripts/report_negative_sampling_revalidation.py \
  --output /tmp/fresh-revalidation --training-log /tmp/revalidation-training.txt
```

The build requires a C compiler and modifies only the active virtual environment.
Production batch trainers check the numerical wrapper before training. They assign
negative weights by vocabulary position and count every repeated walk visit.
The staged prospective trainer explicitly uses original-token negative counts;
the new revalidation compares both original- and retained-token samplers.

The minimal `make test` environment skips the scientific tests when their optional
dependencies are absent. The pinned environment above runs all 26 tests, including
the mandatory numerical checks; an installed but defective BLAS build fails them.

Historical output status is recorded in [results/README.md](results/README.md).
To reproduce the original calibration-only control with its exact source files,
run `scripts/reproduce_calibration_control.py --output /tmp/fresh-control` in the
same pinned environment. No historical files or manuscript drafts are replaced.

Reproducing the initial results produced by Dr Sadamori Kojaku. Read more [here](./scratchbook/Kmeans_Sadamori.pdf)

## Current research focus

The [PRE manuscript](paper/current/main.tex) studies how graph structure,
sampling, and training affect embedding norms and their inverted-U relation
with degree. Population rays and unigram degree cancellation are established
foundations; the [source audit](analysis/issue31_prior_theory_audit.md) records
their attribution and assumptions.

The research priority is to explain and predict when turnover appears, moves,
or disappears. [PR #36](https://github.com/govgandhi/alt_means_sans_k/pull/36)
provides evidence of learning-rate dependence, with the mechanism and connection
to the PNAS trajectory results still open. Finish the existing issue work first;
PRs #35 and #36 remain with YY for review. Then isolate initial rate, decay, and
duration on fixed graphs and corpora, and test the original PNAS settings.
Keep population predictions, realized-graph targets, and training effects distinct.
The older [strategy note](paper/current/note.tex) is historical.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/govgandhi/alt_means_sans_k.git
cd alt_means_sans_k
```

### 2. Create conda environment

**Main environment (kmeans_env):**
```bash
conda env create -f environment.yml
conda activate kmeans_env
```

**For gensim-related notebooks:**
```bash
conda env create -f environment_gensim.yml
conda activate gensim_mod_env
```

### 3. Install local libraries (if needed)
Some notebooks may require local packages from `libs/`:
Reproducing the initial results produced by Dr Sadamori Kojaku. Read more [here](https://github.com/govgandhi/alt_means_sans_k/blob/ff494bf976c0ce7c4300eb1dda092ff2329d82cf/paper/20220803_testing_the_proposed_method%20-%20Sadamori%20Kojaku.pdf)  
## Setup
Setup virtual environment with these packages to reproduce results. 
Note: If you are using Apple silicon macs, CUDA was made to run on nvidia cards and the gpu functionalities might not work. [Fixing a workable version of the code for it]

```bash
pip install -e libs/node2vec  # if using node2vec modifications
```

### 4. Data
Data files are not included in the repository. See individual notebooks for data generation instructions.

## Project Structure

```
alt_means_sans_k/
├── notebooks/           # Experiment notebooks (YYYYMMDD_description/)
├── libs/                # Local/modified libraries
├── scripts/             # Python scripts for batch processing
├── results/             # Output figures and analysis
├── Figs/                # Publication figures
├── paper/               # Manuscript files
├── workflow/            # Snakemake workflows
├── archive/             # Historical work (kmeans early experiments)
├── are_angles_important/  # Subproject: angle importance in embeddings
├── environment.yml      # Main conda environment (kmeans_env)
└── environment_gensim.yml # Gensim environment for specific notebooks
```

## Key Notebooks

- `notebooks/20240809_changing_clustering_metric/` - Metric comparison experiments
- `notebooks/20250114_inhomogeneity_of_clusters/` - Cluster inhomogeneity analysis
- `notebooks/20250210_modding_node2vec/` - Modified node2vec experiments

## Subprojects

### are_angles_important/
Investigates the role of angular information in network embeddings.
- Angular synchronization experiments
- LFR and SBM network testing

## Note for Apple Silicon Macs
CUDA was made to run on NVIDIA cards. GPU functionalities may not work on Apple Silicon. See PyTorch documentation for Metal/MPS support.
Install in-house package(s):  
`cd libs/LFR-benchmark && python3 setup.py build && pip install -e .`
