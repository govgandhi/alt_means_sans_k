# Local Libraries

This directory contains local/modified libraries used by the project.

## Installation

Install packages using editable mode:
```bash
pip install -e <package_name>
```

## Libraries

### node2vec
Modified node2vec implementation for network embedding experiments.
- Source: Modified from [aditya-grover/node2vec](https://github.com/aditya-grover/node2vec)
- Install: `pip install -e node2vec`

### BeliefPropagation
Belief propagation for community detection.
- Source: [skojaku/BeliefPropagation](https://github.com/skojaku/BeliefPropagation)
- Install: `pip install -e BeliefPropagation`

### LFR-benchmark
LFR benchmark network generator for testing community detection.
- Source: [skojaku/LFR-benchmark](https://github.com/skojaku/LFR-benchmark)
- Install: `pip install -e LFR-benchmark`

### libs_for_gensim
Gensim-compatible libraries for specific notebooks.
- Used with `gensim_mod_env` environment

### project_package_name
Template package structure for project-specific utilities.
