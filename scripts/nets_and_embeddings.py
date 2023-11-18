import lfr
import embcom
from scipy import sparse
import numpy as np



def create_network(params= {
                                "N": 1000,     # number of nodes
                                "k": 25,       # average degree
                                "maxk": 100,   # maximum degree sqrt(10*N)
                                "minc": 5,    # minimum community size
                                "maxc": 100,   # maximum community size sqrt(10*N)
                                "tau": 3.0,    # degree exponent
                                "tau2": 1.0,   # community size exponent
                                "mu": 0.2,     # mixing rate
                            }):
    ng = lfr.NetworkGenerator()
    data = ng.generate(**params)

    net = data["net"]                  # scipy.csr_sparse matrix
    community_table = data["community_table"]  # pandas DataFrame
    seed = data["seed"]                # Seed value
    return net, community_table, seed

def create_embedding(net, emb_params = {
                                            "method": "node2vec",
                                            "window_length": 10,
                                            "walk_length": 80,
                                            "num_walks": 10,
                                            "dim" : 64,
                                        }, default_method = None):
    if default_method is not None:
        emb_params["method"] = default_method

    if emb_params["method"] == 'node2vec':
        model = embcom.embeddings.Node2Vec(window_length = emb_params['window_length'], walk_length=emb_params['walk_length'], num_walks=emb_params['num_walks'])

    if emb_params["method"] == 'deepwalk':
        model = embcom.embeddings.DeepWalk(window_length=emb_params['window_length'], num_walks=emb_params['num_walks'])

    if emb_params["method"] == 'leigenmap':
        model = embcom.embeddings.LaplacianEigenMap()

    if emb_params["method"] == 'modspectralemb':
        model = embcom.embeddings.ModularitySpectralEmbedding()

    if emb_params["method"] == 'nonbackspectralemb':
        model = embcom.embeddings.NonBacktrackingSpectralEmbedding()


    model.fit(net)
    emb = model.transform(dim=emb_params['dim'])

    return emb

def save_files( net, community_table, emb, params, emb_params, path_name = None, file_name = None, embedding_name = None):
    if path_name is None:
        path_name = "/nobackup/gogandhi/alt_means_sans_k/data/LFR_benchmark/"
    file_name = f"LFR_n_{params['N']}_tau1_{params['tau']}_tau2_{params['tau2']}_mu_{params['mu']}_k_{params['k']}_mincomm_{params['minc']}"
    embedding_name = f"_{emb_params['method']}_window_length_{emb_params['window_length']}_dim_{emb_params['dim']}"

    sparse.save_npz(path_name+"net_"+file_name + ".npz" ,net) 
    community_table.to_csv(path_name+"community_table_"+file_name+ ".npz",index=False)
    np.savez_compressed(path_name+"emb_"+file_name + embedding_name+ ".npz",emb = emb)

    return


def create_and_save_network_and_embedding(params, emb_params, path_name=None, save_file=True):
    net, community_table, seed = create_network(params)
    emb = create_embedding(net, emb_params)
    if save_file: 
        save_files(net, community_table, emb, params, emb_params, path_name)
    return net, community_table, emb

"""
# For loading:
# import pandas as pd
# net = sparse.load_npz(path_name+"net_"+file_name + ".npz")
# community_table = pd.read_csv(path_name+"community_table_"+file_name+ ".npz")
# emb = np.load(path_name+"emb_"+file_name + embedding_name+ ".npz")['emb']
"""
