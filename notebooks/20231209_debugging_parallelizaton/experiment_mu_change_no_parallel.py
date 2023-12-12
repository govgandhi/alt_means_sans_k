
def main():
    import logging
    import time
    import pandas as pd
    import numpy as np
    from copy import deepcopy
    import sys
    import os
    import csv
    from itertools import cycle
    import warnings
    import matplotlib.pyplot as plt
    import pickle

    sys.path.append("/nobackup/gogandhi/alt_means_sans_k/")
    from scripts.similarity_scores import get_scores
    from scripts.plotting import plotting_mu_change

    # Suppress all warnings
    warnings.filterwarnings("ignore")
    # warnings.resetwarnings() # To change it back (optional)

    

    logging.basicConfig(filename='progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    def process_and_save_result(run_no, mu, path_name, score_keys, device_name, emb_params, params):
        start_time = time.perf_counter()

        if not os.path.isdir(f"{path_name}/Run_{run_no}/"):
            os.mkdir(f"{path_name}/Run_{run_no}/")

        params['mu'] = mu
        result_run_mu = get_scores(params, emb_params, score_keys, f"{path_name}/Run_{run_no}/", device_name)

        elapsed_time = time.perf_counter() - start_time

        logging.info(f"Processed run {run_no} for mu={mu} on {device_name}. Elapsed time: {elapsed_time:.2f} seconds")

        return run_no, mu, result_run_mu


    params = {
        "N": 1000,
        "k": 50,
        "maxk": 100,
        "minc": 5,
        "maxc": 100,
        "tau": 3.0,
        "tau2": 1.0,
        "mu": 0.2,
    }

    emb_params = {
        "method": "node2vec",
        "window_length": 10,
        "walk_length": 80,
        "num_walks": 10,
        "dim": 64,
    }

    # 'belief_prop' avoided for an order of magnitude faster results.
    score_keys = ['kmeans','dbscan', 'optics', 'xmeans', 'infomap', 'flatsbm', 'proposed']

    path_name = f"/nobackup/gogandhi/alt_means_sans_k/data/experiment_mu_change_{params['N']}_{params['k']}_{params['tau']}_testrun"

    #runs = np.arange(1, 11)
    runs = [1]

    #mu_values = np.round(np.arange(0, 1.05, 0.05),decimals=2)
    mu_values = [0.2]

    device_names = [f"cuda:{i}" for i in [1,2,3,4]]  # ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

    #################### End of Params #################

    #if not os.path.isdir(path_name):
    #    os.mkdir(path_name)

    def create_unique_folder(base_folder):
        if os.path.exists(base_folder):
            index = 1
            while True:
                new_folder = f"{base_folder}_{index}"
                if not os.path.exists(new_folder):
                    break
                index += 1
        else:
            new_folder = base_folder

        os.mkdir(new_folder)
        return new_folder

    path_name = create_unique_folder(path_name)
    csv_file_path = path_name + "/result_stream.csv"

    print("Hello, you can find results at:\n",path_name)

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['run_no', 'mu'] + score_keys)

    for run_no in runs:

        start_time = time.perf_counter()
        for mu, device_name in zip(mu_values, cycle(device_names)):

            run_no, mu, result_run_mu = process_and_save_result(run_no, mu, path_name, score_keys, device_name, emb_params, deepcopy(params))
            print(run_no,mu)

            with open(csv_file_path, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([run_no, mu] + [result_run_mu[key] for key in score_keys])

        print(f"Run took: {time.perf_counter() - start_time}, avg time per mu_val: {(time.perf_counter() - start_time)/len(mu_values)}")


    plotting_mu_change(path_name,params)
    
    return True


import sys
if __name__ == "__main__":
    if main():
        print("Success")
        sys.exit(0)