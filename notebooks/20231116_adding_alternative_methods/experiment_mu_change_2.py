# The file is scalable. Ready to run on large
import logging
import time
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from copy import deepcopy
import sys
import os
from itertools import cycle
import matplotlib.pyplot as plt
import os
import pickle

sys.path.append("/nobackup/gogandhi/alt_means_sans_k/")

from scripts.similarity_scores import get_scores
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

def save_accumulated_results(accumulator, pathname):
    for run_no, mu, result_run_mu in accumulator:
        df = pd.DataFrame.from_dict(result_run_mu, orient='index')
        df.reset_index(inplace=True)
        df.columns = ['mu'] + list(df.columns[1:])
        df.to_csv(f"{pathname}/Run_{run_no}/mu_{mu:.2f}_change.csv", index=False)

accumulator = []  # List to accumulate results for each run and mu

params = {
    "N": 10000,
    "k": 50,    # Scaling k, maxk, minc and maxc as sqrt(N)
    "maxk": 1000,
    "minc": 50,
    "maxc": 1000,
    "tau": 2.1,
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

score_keys = ['kmeans', 'optics', 'xmeans', 'belief_prop', 'infomap', 'flatsbm', 'proposed'] # dbscan excluded. Still broke.

path_name = f"/nobackup/gogandhi/alt_means_sans_k/data/experiment_mu_change_{params['N']}_{params['k']}_{params['tau']}"
if not os.path.isdir(path_name):
    os.mkdir(path_name)

num_cores = 24
runs = np.arange(1, 11)
mu_values = np.arange(0, 1.05, 0.05)

#device_names = [f"cuda:{i}" for i in range(1,4)]  # ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
device_names = "cuda:1" # For ember

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []

        for run_no in runs:
            for mu, device_name in zip(mu_values, cycle(device_names)):
                future = executor.submit(process_and_save_result, run_no, mu, path_name, score_keys, device_name, emb_params, deepcopy(params))
                futures.append(future)

        # Wait for all futures to complete
        for future in futures:
            run_no, mu, result_run_mu = future.result()
            accumulator.append((run_no, mu, result_run_mu))

        save_accumulated_results(accumulator, path_name)



# Save the accumulator data
with open(f"{path_name}/final_raw_results.pkl", 'wb') as f:
    pickle.dump(accumulator, f)

# Load the accumulator data
with open(f"{path_name}/final_raw_results.pkl", 'rb') as f:
    accumulator = pickle.load(f)

# Extract mu values and unique labels
mu_values = np.unique([entry[1] for entry in accumulator])
labels = np.unique([label for _, _, scores in accumulator for label in scores])

# Initialize an empty DataFrame with mu_values as the index
df_final = pd.DataFrame(index=mu_values)

# Populate the DataFrame with mean and standard deviation for each label
for label in labels:
    mean_scores = []
    std_scores = []
    for mu in mu_values:
        # Extract scores for the current mu and label
        mu_label_scores = [scores[label] for _, mu_value, scores in accumulator if mu_value == mu]
        # Calculate mean and standard deviation for the current mu and label
        mean_score = np.mean(mu_label_scores)
        std_score = np.std(mu_label_scores)
        mean_scores.append(mean_score)
        std_scores.append(std_score)

    # Add the mean and standard deviation to the DataFrame
    df_final[f'{label}_mean'] = mean_scores
    df_final[f'{label}_std'] = std_scores

# Save the DataFrame to a CSV file
#df_final.to_csv(f"{path_name}/final_experiment_results_with_error_bars.csv", index_label='mu')

# Plot each column as a separate curve with error bars and fill between errors
for label in labels:
    plt.plot(df_final.index, df_final[f'{label}_mean'], '-o', label=label)
    plt.fill_between(df_final.index,
                     df_final[f'{label}_mean'] - df_final[f'{label}_std'],
                     df_final[f'{label}_mean'] + df_final[f'{label}_std'],
                     alpha=0.3)

# Add labels and a legend
plt.xlabel('mu')
plt.ylabel('Score')
plt.legend(title='Algorithm')

# Save the figure
plt.savefig(f"{path_name}/experiment_plot.png")

# Show the plot
plt.show()
