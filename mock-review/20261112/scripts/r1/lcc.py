import sys
import networkx as nx
g = nx.read_gml(sys.argv[1], label="id")
g = nx.Graph(g)
g.remove_edges_from(nx.selfloop_edges(g))
lcc = max(nx.connected_components(g), key=len)
h = g.subgraph(lcc)
print("nodes", g.number_of_nodes(), "edges", g.number_of_edges(), "LCC nodes", h.number_of_nodes(), "LCC edges", h.number_of_edges())
d = [x for _, x in h.degree()]
print("LCC degree min/max", min(d), max(d), "share deg<3", sum(x < 3 for x in d) / len(d))
