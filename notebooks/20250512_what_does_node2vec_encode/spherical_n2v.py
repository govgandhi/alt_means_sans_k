# pipeline_node2vec_embeddings.py

import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from node2vecs import TorchNode2Vec

def load_communities(csv_path):
    """Load node→community mapping from CSV; expects a 'community_id' column."""
    df = pd.read_csv(csv_path)
    return df['community_id'].values

def generate_and_save_embeddings(input_base, output_base, k, runs, mu_values, similarity_measures, device='cuda:0'):
    """
    For each run and mixing parameter:
      - Load the graph and community table
      - Train node2vec embeddings under different similarity objectives
      - Save the embeddings to the output directory
    """
    for run in runs:
        for mu in mu_values:
            mu_str = f"{mu}"
            run_in  = os.path.join(input_base,  f"Run_{run}")
            run_out = os.path.join(output_base, f"Run_{run}")
            os.makedirs(run_out, exist_ok=True)

            net_file  = f"net_LFR_n_10000_tau1_3.0_tau2_1.0_mu_{mu_str}_k_{k}_mincomm_50.npz"
            comm_file = f"community_table_LFR_n_10000_tau1_3.0_tau2_1.0_mu_{mu_str}_k_{k}_mincomm_50.csv"
            net_path  = os.path.join(run_in, net_file)
            comm_path = os.path.join(run_in, comm_file)

            if not os.path.exists(net_path) or not os.path.exists(comm_path):
                print(f"[k={k}, run={run}, μ={mu_str}] Missing input files; skipping.")
                continue

            # Load graph and communities
            A = sp.load_npz(net_path)
            comm = load_communities(comm_path)

            # Train embeddings
            embeddings = {}
            for sim in similarity_measures:
                print(f"[k={k}, run={run}, μ={mu_str}] Training with {sim} objective...")
                model = TorchNode2Vec(
                    vector_size=64,
                    similarity_metric=sim,
                    device=device,
                    num_workers=1
                )
                model.fit(A)
                embeddings[sim] = model.transform()

            # Save embeddings dict to pickle
            out_file = f"embeddings_LFR_n_10000_tau1_3.0_tau2_1.0_mu_{mu_str}_k_{k}_mincomm_50.pkl"
            out_path = os.path.join(run_out, out_file)
            with open(out_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"[k={k}, run={run}, μ={mu_str}] Saved embeddings to {out_path}")

if __name__ == "__main__":
    # Parameters
    ks = [5]
    runs = range(4, 11)
    mu_values = [round(x, 2) for x in np.arange(0.1, 1.01, 0.05)]
    similarity_measures = ["dot", "euclidean", "cosine"]

    base_input_template  = "/l/research/gogandhi.NOBACKUP/alt_means_sans_k/data/experiment_n2v_metric_change_10000_{k}_3.0_minc50_immutable"
    base_output_template = "/l/research/gogandhi.NOBACKUP/alt_means_sans_k/data/experiment_n2v_metric_spherical_10000_{k}_3.0_minc50"

    for k in ks:
        input_base  = base_input_template.format(k=k)
        output_base = base_output_template.format(k=k)
        os.makedirs(output_base, exist_ok=True)
        generate_and_save_embeddings(
            input_base=input_base,
            output_base=output_base,
            k=k,
            runs=runs,
            mu_values=mu_values,
            similarity_measures=similarity_measures,
            device='cuda:0'
        )
