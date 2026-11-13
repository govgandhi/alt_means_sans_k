"""R6 check: upstream Gensim 4.4.0 behaviours the manuscript calls "corrections".

1. BLAS wrapper: with the official wheel, does our_dot return 0 for an exact -1?
2. Integer keys: how far is make_cum_table's negative distribution from the intended
   count**ns_exponent distribution on the paper's own graphs (calibration
   configuration graphs, a direct BA target, NetScience)?
3. Initialisation range of Gensim 4.4.0 input vectors vs the controlled trainer.

Run: PYTHONDONTWRITEBYTECODE=1 r6/.venv/bin/python r6/check_gensim_upstream.py
Reads the worktree's scripts read-only (no bytecode written).
"""

from __future__ import annotations

import ctypes as ct
import sys

sys.dont_write_bytecode = True
WT = "<scratch>/round/wt"
sys.path.insert(0, WT + "/scripts")

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from gensim.models import Word2Vec, word2vec_inner  # noqa: E402
from simulate_norm_frequency import configuration_graph, graph_arrays, sample_walks  # noqa: E402


def blas_minus_one() -> dict[str, float]:
    name = {0: "our_dot_double", 1: "our_dot_float", 2: "our_dot_noblas"}[
        word2vec_inner.FAST_VERSION
    ]
    capsule = word2vec_inner.__pyx_capi__[name]
    get_name = ct.pythonapi.PyCapsule_GetName
    get_name.argtypes, get_name.restype = [ct.py_object], ct.c_char_p
    get_pointer = ct.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes, get_pointer.restype = [ct.py_object, ct.c_char_p], ct.c_void_p
    address = get_pointer(capsule, get_name(capsule))
    integer, real = ct.POINTER(ct.c_int), ct.POINTER(ct.c_float)
    dot = ct.CFUNCTYPE(ct.c_float, integer, real, integer, real, integer)(address)
    n = ct.c_int(1)
    x = (ct.c_float * 1)(1)
    out = {}
    for value in (-2.0, -1.0, -0.5, 0.0, 1.0):
        y = (ct.c_float * 1)(value)
        out[value] = float(dot(ct.byref(n), x, ct.byref(n), y, ct.byref(n)))
    return {"wrapper": name, **{str(k): v for k, v in out.items()}}


def table_probabilities(cum_table: np.ndarray) -> np.ndarray:
    increments = np.diff(np.concatenate(([0], cum_table.astype(np.float64))))
    return increments / increments.sum()


def legacy_vs_intended(
    walks: list[list[int]], n: int, ns_exponent: float
) -> dict[str, float]:
    model = Word2Vec(
        vector_size=8,
        window=1,
        min_count=1,
        sg=1,
        negative=5,
        ns_exponent=ns_exponent,
        workers=1,
        seed=1,
        sample=0.0,
        sorted_vocab=1,
    )
    model.build_vocab(walks)  # make_cum_table runs here (upstream code path)
    slot_prob = table_probabilities(model.cum_table)
    keys = np.asarray(model.wv.index_to_key)
    legacy = np.empty(n)
    legacy[keys] = slot_prob  # probability that node keys[p] is drawn
    counts = model.wv.expandos["count"].astype(np.float64)
    weights = counts**ns_exponent
    intended = np.empty(n)
    intended[keys] = weights / weights.sum()
    ratio = legacy / intended
    return {
        "tv_distance": float(0.5 * np.abs(legacy - intended).sum()),
        "ratio_min": float(ratio.min()),
        "ratio_median": float(np.median(ratio)),
        "ratio_max": float(ratio.max()),
        "spearman_id_vs_count": float(
            np.corrcoef(
                np.argsort(np.argsort(np.arange(n))),
                np.argsort(np.argsort(-counts[np.argsort(keys)])),
            )[0, 1]
        ),
        "identity_slots": float(np.mean(keys == np.arange(n))),
    }


def corpus(graph: nx.Graph, seed: int) -> tuple[list[list[int]], int]:
    _, neighbors, degree = graph_arrays(graph)
    walks, _ = sample_walks(
        neighbors,
        degree / degree.sum(),
        graph.number_of_nodes() * 300,
        5,
        np.random.default_rng(seed),
    )
    return walks, graph.number_of_nodes()


def main() -> None:
    print("1. BLAS wrapper (official gensim 4.4.0 wheel):", blas_minus_one())

    toy = [[2, 2, 2, 2, 0, 0, 1]] * 2  # counts {2:8, 0:4, 1:2}
    m = Word2Vec(
        vector_size=4, min_count=1, negative=5, ns_exponent=1.0, workers=1, sample=0.0
    )
    m.build_vocab(toy)
    print(
        "2a. toy: vocab order",
        m.wv.index_to_key,
        "slot weights",
        np.round(table_probabilities(m.cum_table) * 14).astype(int).tolist(),
        "(intended [8, 4, 2])",
    )

    for seed in range(5):
        walks, n = corpus(configuration_graph(600, 3, seed), seed + 10_000)
        for a in (1.0, 0.75):
            print(
                f"2b. calibration graph {seed} ns_exponent={a}:",
                legacy_vs_intended(walks, n, a),
            )
    ba = nx.convert_node_labels_to_integers(
        nx.barabasi_albert_graph(600, 3, seed=101), ordering="sorted"
    )
    walks, n = corpus(ba, 20_101)
    print("2c. direct BA target 101, ns_exponent=1:", legacy_vs_intended(walks, n, 1.0))
    ns = nx.read_gml(WT + "/are_angles_important/netscience/netscience.gml")
    ns = nx.Graph(ns)
    ns.remove_edges_from(nx.selfloop_edges(ns))
    ns = ns.subgraph(max(nx.connected_components(ns), key=len)).copy()
    ns = nx.convert_node_labels_to_integers(ns, ordering="sorted")
    walks, n = corpus(ns, 40_301)
    print(
        f"2d. NetScience LCC n={n}, ns_exponent=1:", legacy_vs_intended(walks, n, 1.0)
    )

    for d in (24, 64):
        mm = Word2Vec(vector_size=d, min_count=1, workers=1, seed=7)
        mm.build_vocab([[str(i) for i in range(600)]] * 3)
        v = mm.wv.vectors
        print(
            f"3. gensim init d={d}: max|component|*d = {np.abs(v).max() * d:.3f}, "
            f"mean norm = {np.linalg.norm(v, axis=1).mean():.4f}; "
            f"controlled trainer U(-0.5/d,0.5/d) mean norm ~ {0.5 / np.sqrt(3 * d):.4f}; "
            f"syn1neg all zero: {not mm.syn1neg.any()}"
        )


if __name__ == "__main__":
    main()
