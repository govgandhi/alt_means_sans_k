import sys
sys.path.append("/nobackup/gogandhi/alt_means_sans_k/scipts/")

from nets_and_embeddings import create_and_save_network_and_embedding
import time
import concurrent.futures
from functools import partial
import numpy as np
from copy import deepcopy
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
emb_params = {          "method": "node2vec",
                        "window_length": 10,
                        "walk_length": 80,
                        "num_walks": 10,
                        "dim" : 64,
                        }
        
list_of_params=[]
for mu in np.linspace(0,1,6):
    temp_params = deepcopy(params)
    temp_params['mu'] = mu
    list_of_params.append((temp_params,emb_params))

if __name__ == '__main__':
    start = time.perf_counter()
    num_cores = 15
    with concurrent.futures.ProcessPoolExecutor(max_workers = num_cores) as executor:
    
       results = executor.map(create_and_save_network_and_embedding, *zip(*list_of_params))
       for result in results:
           print(result)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds')
