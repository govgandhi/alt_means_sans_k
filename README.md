# alt_means_sans_k
Alternative to K Means clustering methods for community detection of embedding vectors.


Reproducing the initial results produced by Dr Sadamori Kojaku. Read more [here](https://github.com/govgandhi/alt_means_sans_k/blob/ff494bf976c0ce7c4300eb1dda092ff2329d82cf/paper/20220803_testing_the_proposed_method%20-%20Sadamori%20Kojaku.pdf)  
## Setup
Setup virtual environment with these packages to reproduce results. 
Note: If you are using Apple silicon macs, CUDA was made to run on nvidia cards and the gpu functionalities might not work. [Fixing a workable version of the code for it]

```bash
conda create -n project_env_name python=3.9  
conda activate project_env_name    
conda install -c conda-forge mamba -y  
mamba install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y  
mamba install -y -c bioconda -c conda-forge snakemake -y  
mamba install -c conda-forge graph-tool scikit-learn numpy numba scipy pandas polars networkx seaborn matplotlib gensim ipykernel tqdm black faiss-gpu==1.7.3 -y 
```

Install in-house package(s):  
`cd libs/LFR-benchmark && python3 setup.py build && pip install -e .`
