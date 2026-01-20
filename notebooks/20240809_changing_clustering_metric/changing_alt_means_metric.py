'''
Writing this script such that if I run it with input parameters, 
it should give me element centric similarity for the methods, we query
each method we query can run withinn this or out. Will decide.
Use chanage_mu_test.py as reference.
'''
import numpy as np
from scipy import sparse
import pandas as pd
#import os
#import networkx as nx
#import gensim
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.cluster import OPTICS, DBSCAN
from sklearn.linear_model import LogisticRegression 
import faiss
import fast_hdbscan
#import lfr
#import embcom
#import csv
import sys
sys.path.append("/nobackup/gogandhi/alt_means_sans_k/")

from scripts.nets_and_embeddings import create_and_save_network_and_embedding
#from scripts.clustering_methods import clustering_method_values
from scripts.nets_and_embeddings import load_net_and_embedding

from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
import numpy as np
import belief_propagation
import infomap
from graph_tool.all import Graph,minimize_blockmodel_dl

# Need net, node_table and emb files
# net is G, emb files are wv, node_table probably
# has some info about ground truth community labels or smth like that.

# Define a function that calculates element-centric similarity:
def calc_esim(y, ypred):

    ylab, y = np.unique(y, return_inverse=True)
    ypredlab, ypred = np.unique(ypred, return_inverse=True)
    
    Ka, Kb = len(ylab), len(ypredlab)
    K = np.maximum(Ka, Kb)
    N = len(y)
    
    UA = sparse.csr_matrix((np.ones_like(y), (np.arange(y.size), y)), shape=(N,K))
    UB = sparse.csr_matrix((np.ones_like(ypred), (np.arange(ypred.size), ypred)), shape=(N, K))    
    
    nA = np.array(UA.sum(axis=0)).reshape(-1)
    nB = np.array(UB.sum(axis=0)).reshape(-1)

# nAB[i][j] is read as the number of elements that belong to ith ground truth label and jth predicrted label.
# nAB[1][0] = 1 For ground truth label with index 1 and predicted label 0 we have 1 element. i.e. 0000|1| vs 1110|0|

    nAB = (UA.T @ UB).toarray()
    nAB_rand = np.outer(nA, nB) / N
    
# assuming that each element has an equal probability of being assigned to any label,
# and the expected counts are calculated based on label frequencies.


    # Calc element-centric similarity
    Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :]) 
    Q = 1 / np.maximum(Q, 1)
    S = np.sum(np.multiply(Q, (nAB**2))) / N
    
    # Calc the expected element-centric similarity for random partitions
    #Q = np.maximum(nA[:, None] @ np.ones((1, K)), np.ones((K, 1)) @ nB[None, :]) 
    #Q = 1 / np.maximum(Q, 1)
    Srand = np.sum(np.multiply(Q, (nAB_rand**2))) / N
    Scorrected = (S - Srand) / (1 - Srand)
    return Scorrected



def louvain(Z, w1, b0, num_neighbors=100, iteration=50, device="cuda:0", return_member_matrix=False, metric="dotsim"):
    num_nodes = Z.shape[0]
    node_size = np.ones(num_nodes)
    U = sparse.identity(num_nodes, format="csr")
    Vt = Z.copy()

    while True:
        cids_t = label_switching(
            Z=Vt,
            num_neighbors=num_neighbors,
            rho=b0 / w1,
            node_size=node_size,
            epochs=iteration,
            device=device,
            metric=metric
        )
        _, cids_t = np.unique(cids_t, return_inverse=True)

        if int(max(cids_t) + 1) == Vt.shape[0]:
            break

        num_nodes_t = len(cids_t)
        k = int(np.max(cids_t) + 1)
        Ut = sparse.csr_matrix((np.ones(num_nodes_t), (np.arange(num_nodes_t), cids_t)), shape=(num_nodes_t, k))
        U = U @ Ut
        Vt = Ut.T @ Vt
        node_size = np.array(Ut.T @ node_size).reshape(-1)

    if return_member_matrix:
        return U
    cids = np.array((U @ sparse.diags(np.arange(U.shape[1]))).sum(axis=1)).reshape(-1)
    return cids
import faiss
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

def find_knn_edges(emb, num_neighbors, target=None, metric="dotsim", device=None):
    k = int(np.minimum(num_neighbors + 1, emb.shape[0]).astype(int))
    # Normalize embeddings if metric is cosine
    if metric == "cosine":
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        if target is not None:
            target = target / np.linalg.norm(target, axis=1, keepdims=True)

    indices, distances = find_knn(
        emb if target is None else target, emb, num_neighbors=k, metric=metric, device=device
    )
    r = np.outer(np.arange(indices.shape[0]), np.ones((1, indices.shape[1]))).astype(int)
    r, c, distances = r.reshape(-1), indices.astype(int).reshape(-1), distances.reshape(-1)
    if len(r) == 0:
        return r, c, distances
    return r, c, distances


def find_knn(target, emb, num_neighbors, metric="dotsim", device=None):
    if metric == "dotsim" or metric == "cosine":
        index = faiss.IndexFlatIP(emb.shape[1])
        if metric == "cosine":
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            target = target / np.linalg.norm(target, axis=1, keepdims=True)
    elif metric == "euclidean":
        index = faiss.IndexFlatL2(emb.shape[1])
    elif metric == "manhattan":
        index = faiss.IndexFlatL1(emb.shape[1]) # THIS DOES NOT EXIST, CAN'T DO INDEXING FOR MANHATTAN I GUESS
    else:
        raise ValueError("Invalid metric specified.")

    if device is None:
        index.add(emb.astype(np.float32))
        distances, indices = index.search(target.astype(np.float32), k=num_neighbors)
    else:
        gpu_id = int(device[-1])
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        index.add(emb.astype(np.float32))
        distances, indices = index.search(target.astype(np.float32), k=num_neighbors)
        index.reset()

    return indices, distances


def label_switching(Z, rho, num_neighbors=50, node_size=None, device=None, epochs=50, metric="dotsim"):
    num_nodes, dim = Z.shape
    if node_size is None:
        node_size = np.ones(num_nodes)
    Z = Z.copy(order="C").astype(np.float32)

    # Normalize Z for cosine similarity without adding extra dimensions
    if metric == "cosine":
        Z1 = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        Zrho = Z1  # Use the normalized Z1 directly without adding extra dimensions
    else:
        Z1 = Z
        Zrho = Z

    # Perform nearest neighbor search with consistent dimensions
    r, c, v = find_knn_edges(Zrho, target=Z1, num_neighbors=num_neighbors, metric=metric, device=device)
    A = sparse.csr_matrix((v, (r, c)), shape=(num_nodes, num_nodes))

    return _label_switching_(A.indptr, A.indices, Z, num_nodes, rho, node_size, epochs, metric=metric)

    
def _label_switching_(A_indptr, A_indices, Z, num_nodes, rho, node_size, epochs=100, metric="dotsim"):
    Nc = np.zeros(num_nodes)
    cids = np.arange(num_nodes)
    Vc = Z.copy()

    if metric in ["dotsim", "cosine"]:
        Vnorm = np.sum(np.multiply(Z, Z), axis=1).reshape(-1)

    for nid in range(num_nodes):
        Nc[nid] += node_size[nid]

    for _ in range(epochs):
        order = np.random.choice(num_nodes, size=num_nodes, replace=False)
        updated_node_num = 0

        for node_id in order:
            neighbors = A_indices[A_indptr[node_id]:A_indptr[node_id + 1]]
            c = cids[node_id]
            clist = np.unique(cids[neighbors])
            next_cid = -1

            if metric == "euclidean":
                dqmin = float("inf")
                qself = np.sum((Z[node_id, :] - Vc[c, :]) ** 2) + rho * node_size[node_id] * (Nc[c] - node_size[node_id])
            elif metric == "manhattan":
                dqmin = float("inf")
                qself = np.sum(np.abs(Z[node_id, :] - Vc[c, :])) + rho * node_size[node_id] * (Nc[c] - node_size[node_id])
            else:  # dotsim or cosine
                dqmax = 0
                qself = np.sum(Z[node_id, :] * Vc[c, :]) - Vnorm[node_id] - rho * node_size[node_id] * (Nc[c] - node_size[node_id])

            for cprime in clist:
                if c == cprime:
                    continue

                if metric == "euclidean":
                    dq = np.sum((Z[node_id, :] - Vc[cprime, :]) ** 2) + rho * node_size[node_id] * Nc[cprime] - qself
                    if dq < dqmin:
                        next_cid = cprime
                        dqmin = dq
                elif metric == "manhattan":
                    dq = np.sum(np.abs(Z[node_id, :] - Vc[cprime, :])) + rho * node_size[node_id] * Nc[cprime] - qself
                    if dq < dqmin:
                        next_cid = cprime
                        dqmin = dq
                else:  # dotsim or cosine
                    dq = (np.sum(Z[node_id, :] * Vc[cprime, :]) - rho * node_size[node_id] * Nc[cprime]) - qself
                    if dq > dqmax:
                        next_cid = cprime
                        dqmax = dq

            if (metric in ["euclidean", "manhattan"] and dqmin >= 0) or (metric in ["dotsim", "cosine"] and dqmax <= 1e-16):
                continue

            Nc[c] -= node_size[node_id]
            Nc[next_cid] += node_size[node_id]

            Vc[c, :] -= Z[node_id, :]
            Vc[next_cid, :] += Z[node_id, :]

            cids[node_id] = next_cid
            updated_node_num += 1

        if (updated_node_num / max(1, num_nodes)) < 1e-3:
            break

    return cids


def proposed_method_labels(emb, device_name, metric="dotsim"):
    if metric == "cosine":
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    rpos, cpos, vpos = find_knn_edges(emb, num_neighbors=500, device=device_name, metric=metric)
    cneg = np.random.choice(emb.shape[0], len(cpos))
    vneg = np.array(np.sum(emb[rpos, :] * emb[cneg, :], axis=1)).reshape(-1)

    model = LogisticRegression()
    model.fit(
        np.concatenate([vpos, vneg]).reshape((-1, 1)),
        np.concatenate([np.ones_like(vpos), np.zeros_like(vneg)]),
    )
    w1, b0 = model.coef_[0, 0], -model.intercept_[0]
    return louvain(emb, w1, b0, device=device_name, metric=metric)


def clustering_method_values(net, community_table, emb, score_keys, device_name):
    X = np.einsum("ij,i->ij", emb, 1 / np.maximum(np.linalg.norm(emb, axis=1), 1e-24))
    X = emb.copy()

    def method_score(key):
        if key == "proposed_euclidean":
            return calc_esim(community_table["community_id"], proposed_method_labels(emb, device_name, metric="euclidean"))
        elif key == "proposed_cosine":
            return calc_esim(community_table["community_id"], proposed_method_labels(emb, device_name, metric="cosine"))
        elif key == "proposed_dot":
            return calc_esim(community_table["community_id"], proposed_method_labels(emb, device_name, metric="dotsim"))
        elif key == "proposed_manhattan":
            return calc_esim(community_table["community_id"], proposed_method_labels(emb, device_name, metric="manhattan"))

    score_dictionary = {key: method_score(key) for key in score_keys}
    return score_dictionary





from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time


N=10000
mu_values = np.round(np.arange(0.05, 1.05, 0.05),decimals=2)
score_keys=['proposed_cosine','proposed_dot'] 
k=10
device_name="cuda:1"

params_template = {
    "N": N,
    "k": k,
    "maxk":  int(np.sqrt(10 * N)),
    "minc": 50,
    "maxc": int(np.ceil(np.sqrt(N * 10))),
    "tau": 3.0,
    "tau2": 1.0,
    "mu": 0.1,
    }


emb_params = {
    "method": "node2vec",
    "window_length": 10,
    "walk_length": 80,
    "num_walks": 10,
    "dim": 64,
}

 
#"community_table_LFR_n_10000_tau1_3.0_tau2_1.0_mu_0.1_k_50_mincomm_50.npz"

 
# The default for alt_means is using cosine sim for finding knn using FAISS
# And the louvain update scheme uses dot sim update scheme.
# First steps could be to just change louvain update scheme, as FAISS to euclidean could be very slow?



def process_run(run_no, mu):
    # Enforce the specific GPU device within each process if you have multiple GPUs
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change if you want to specify another GPU

    params = params_template.copy()
    params["mu"] = mu

    path_name = f"/nobackup/gogandhi/alt_means_sans_k/data/experiment_mu_change_10000_{k}_3.0_minc50/Run_{run_no}/"
    net, community_table, emb = load_net_and_embedding(params, emb_params, path_name)

    # Compute clustering scores
    result = clustering_method_values(net, community_table, emb, score_keys, device_name=device_name)

    result_values = [result[key] for key in score_keys]
    result_str = f"{run_no},{mu}," + ",".join(map(str, result_values))
    print(f"Completed Run {run_no} with Mu {mu} with esim {result_str}")
    return result_str
    
from tqdm import tqdm
import time

def process_all_combinations_sequential():
    runs_mu_combinations = [(run_no, mu) for run_no in range(1, 11) for mu in mu_values]
    total_combinations = len(runs_mu_combinations)
    start_time = time.time()
    output_file = f"/nobackup/gogandhi/alt_means_sans_k/data/experiment_mu_change_10000_{k}_3.0_minc50/altmeans_clustering_metric_change.txt"
    # Write header to output file
    with open(output_file, "w") as f:
        header = "run_no,mu," + ",".join(score_keys) + "\n"
        f.write(header)

    # Sequentially process each run and mu value with tqdm for progress tracking
    with open(output_file, "a") as f:
        for (run_no, mu) in tqdm(runs_mu_combinations, desc="Processing runs", total=total_combinations):
            try:
                result_str = process_run(run_no, mu)
                f.write(result_str + "\n")
                f.flush()  # Ensures each line is written immediately
            except Exception as e:
                print(f"Error processing Run {run_no}, Mu {mu}: {e}")
    
    print(f"All combinations processed. Total elapsed time: {time.time() - start_time:.2f} seconds.")

# Run the sequential processing
if __name__ == "__main__":
    

    process_all_combinations_sequential()
