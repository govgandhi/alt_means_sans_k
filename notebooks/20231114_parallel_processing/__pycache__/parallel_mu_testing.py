from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sys
sys.path.append("/nobackup/gogandhi/alt_means_sans_k/scipts/")

from elem_centric_similarity import get_values


def process_single_mu(mu):
    params['mu'] = mu
    file_name = f"mixing_test_LFR_n_{params['N']}_tau1_{params['tau']}_tau2_{params['tau2']}_mu_{params['mu']}_k_{params['k']}_mincomm_{params['minc']}"
    path_name = "/nobackup/gogandhi/kmeans/data/"
    embedding_name = f"sadamori_node2vec_window_length_{window_length}_dim_{dim}"

    # Check if the file exists
    if os.path.isfile(path_name+"net_"+file_name + ".npz"):
        if os.path.isfile(path_name+"community_table_"+file_name+ ".npz"):
            if os.path.isfile(path_name+"emb_"+file_name + embedding_name+ ".npz"):

                net = sparse.load_npz(path_name+"net_"+file_name + ".npz")
                community_table = pd.read_csv(path_name+"community_table_"+file_name+ ".npz")
                emb = np.load(path_name+"emb_"+file_name + embedding_name+ ".npz")['emb']
                print(params['mu'], " there")

    else:
        print(params['mu'], "not there")
        ng = lfr.NetworkGenerator()
        data = ng.generate(**params)

        net = data["net"]                  # scipy.csr_sparse matrix
        community_table = data["community_table"]  # pandas DataFrame
        seed = data["seed"]           # Seed value

        model = embcom.embeddings.Node2Vec(window_length, walk_length, num_walks)
        model.fit(net)
        emb = model.transform(dim=dim)

        file_name = f"mixing_test_LFR_n_{params['N']}_tau1_{params['tau']}_tau2_{params['tau2']}_mu_{params['mu']}_k_{params['k']}_mincomm_{params['minc']}"
        path_name = "/nobackup/gogandhi/kmeans/data/"
        embedding_name = f"sadamori_node2vec_window_length_{window_length}_dim_{dim}"

        sparse.save_npz(path_name+"net_"+file_name + ".npz" ,net) 
        community_table.to_csv(path_name+"community_table_"+file_name+ ".npz",index=False)
        np.savez_compressed(path_name+"emb_"+file_name + embedding_name+ ".npz",emb = emb)

    kmeans_values[mu] = get_values(params)

    with open(write_file_path, 'a') as file:
        content_to_append = ' '.join(map(str, kmeans_values[mu]))

        # Append the string to the file
        file.write(str(mu) + " " + content_to_append + '\n')  # Add a newline to separate entries

# Define the range of mu values
mu_values = np.linspace(0, 1, 21)

# Use ProcessPoolExecutor to parallelize the computation
with ProcessPoolExecutor() as executor:
    executor.map(process_single_mu, mu_values)
