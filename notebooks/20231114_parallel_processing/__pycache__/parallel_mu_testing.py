from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
from copy import deepcopy

import sys

#!TODO: Replace this with parent directory i.e. altmeanssansk independent of gogandhi.

sys.path.append("/nobackup/gogandhi/alt_means_sans_k/")


from scripts.similarity_scores import get_scores
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
        
score_keys = ['kmeans', 'optics', 'dbscan', 'proposed']
path_name = ""

device_name = "cuda:3"# Define the range of mu values


# Use ProcessPoolExecutor to parallelize the computation
list_of_args=[]

# This is the part that you change depending on the test you are running
for mu in np.linspace(0,1,3): # change to 21
    temp_params = deepcopy(params)
    temp_params['mu'] = mu
    list_of_args.append((temp_params,emb_params,score_keys, path_name,device_name))

if __name__ == '__main__':
    start = time.perf_counter()
    num_cores = 15
    with ProcessPoolExecutor(max_workers = num_cores) as executor:
            
        results = executor.map(get_scores, *zip(*list_of_args))
        
        for result in results:
            print(result)

        # Save output in .txt file

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start,2)} seconds')
