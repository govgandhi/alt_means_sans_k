"""R5: slope of realized context KL (main.tex Eq. graph-context-kl) on log degree in the experimental graph family."""
import networkx as nx, numpy as np
def configuration_graph(nodes, attachment, seed):
    template = nx.barabasi_albert_graph(nodes, attachment, seed=seed)
    g = nx.Graph(nx.configuration_model([d for _, d in template.degree()], seed=seed)); g.remove_edges_from(nx.selfloop_edges(g))
    if not nx.is_connected(g):
        cs = [sorted(c) for c in nx.connected_components(g)]
        for a, b in zip(cs, cs[1:]): g.add_edge(a[0], b[0])
    return nx.convert_node_labels_to_integers(g, ordering="sorted")
for seed in range(5):
    A = nx.to_numpy_array(configuration_graph(600, 3, seed)); d = A.sum(1); vol = d.sum()
    KL = np.log(vol / d) - (A @ np.log(d)) / d
    nb = (A @ np.log(d)) / d
    s, c = np.polyfit(np.log(d), KL, 1); s2, _ = np.polyfit(np.log(d), nb, 1)
    print(f"seed={seed}: slope dKL/dlog d={s:+.3f}; slope of neighbor mean log-degree={s2:+.3f}; corr(KL,log d)={np.corrcoef(KL, np.log(d))[0,1]:+.3f}; KL range=[{KL.min():.2f},{KL.max():.2f}]")
