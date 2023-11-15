import lfr
import embcom
from scipy import sparse
import numpy as np



def create_and_save_network_and_embedding(params= None, emb_params = None):
    ng = lfr.NetworkGenerator()
    data = ng.generate(**params)

    net = data["net"]                  # scipy.csr_sparse matrix
    community_table = data["community_table"]  # pandas DataFrame
    seed = data["seed"]                # Seed value

    if params is None:
        params = {
                    "N": 1000,     # number of nodes
                    "k": 25,       # average degree
                    "maxk": 100,   # maximum degree sqrt(10*N)
                    "minc": 5,    # minimum community size
                    "maxc": 100,   # maximum community size sqrt(10*N)
                    "tau": 3.0,    # degree exponent
                    "tau2": 1.0,   # community size exponent
                    "mu": 0.2,     # mixing rate
                }
    if emb_params is None:

        emb_params = {  "method": "node2vec",
                        "window_length": 10,
                        "walk_length": 80,
                        "num_walks": 10,
                        "dim" : 64,
                        }
        
    
    if emb_params["method"] == 'node2vec':
        model = embcom.embeddings.Node2Vec(window_length = emb_params['window_length'], walk_length=emb_params['walk_length'], num_walks=emb_params['num_walks'])

        model.fit(net)

        emb = model.transform(dim=emb_params['dim'])

    file_name = f"LFR_n_{params['N']}_tau1_{params['tau']}_tau2_{params['tau2']}_mu_{params['mu']}_k_{params['k']}_mincomm_{params['minc']}"
    path_name = "/nobackup/gogandhi/alt_means_sans_k/data/LFR_benchmark/"
    embedding_name = f"_node2vec_window_length_{emb_params['window_length']}_dim_{emb_params['dim']}"

    sparse.save_npz(path_name+"net_"+file_name + ".npz" ,net) 
    community_table.to_csv(path_name+"community_table_"+file_name+ ".npz",index=False)
    np.savez_compressed(path_name+"emb_"+file_name + embedding_name+ ".npz",emb = emb)

    return f"Network with mu {params['mu']} created."