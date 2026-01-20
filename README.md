# alt_means_sans_k

Alternative to K-Means clustering methods for community detection of embedding vectors.

Reproducing the initial results produced by Dr Sadamori Kojaku. Read more [here](./scratchbook/Kmeans_Sadamori.pdf)

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
