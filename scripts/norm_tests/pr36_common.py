"""Shared setup for the PR #36 (issue 21) graphs: corpus, objective and metrics.

Graph seed s: configuration_graph(600, 3, s), 300 walks per node of length 5 started in
proportion to degree (walk RNG seed s + 10,000), window 1, no subsampling, 5 negatives,
exponent 1, d = 64. Exactly the unfiltered PR #36 corpus.
positive[i, j]: pair counts per epoch (input i, output j); negative[i, k]: expected
negative draws per epoch, as in scripts/training_trajectory/metrics.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/training_trajectory"))
import check_subsampling_origin as baseline  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from gensim_training import set_negative_table  # noqa: E402
from metrics import expected_pairs, proposal_probabilities  # noqa: E402
from simulate_norm_frequency import (  # noqa: E402
    balanced_left_norm,
    configuration_graph,
    graph_arrays,
    sample_walks,
)

T = baseline.Config().training
SOURCE = baseline.Config()
MAX_EXP = 6.0


def gensim_model(walks, seed: int, alpha: float, min_alpha: float, epochs: int, cls):
    model = cls(
        vector_size=T.dimension,
        window=1,
        min_count=T.min_count,
        sg=1,
        negative=T.negatives,
        ns_exponent=1.0,
        workers=1,
        epochs=epochs,
        seed=seed + SOURCE.optimizer_seed_offset,
        sample=0.0,
        alpha=alpha,
        min_alpha=min_alpha,
        sorted_vocab=1,
        shrink_windows=False,
    )
    model.build_vocab(walks)
    if set(model.wv.index_to_key) != set(range(T.nodes)):
        raise ValueError("Every node must survive the vocabulary cutoff.")
    set_negative_table(model)
    return model


def setup(seed: int) -> dict:
    graph = configuration_graph(T.nodes, T.attachment, seed)
    _, neighbors, degree = graph_arrays(graph)
    walks, counts = sample_walks(
        neighbors,
        degree / degree.sum(),
        T.nodes * T.walks_per_node,
        T.walk_length,
        np.random.default_rng(seed + SOURCE.walk_seed_offset),
    )
    model = gensim_model(walks, seed, 0.025, 1e-4, 1, Word2Vec)
    proposal = np.empty(T.nodes)
    proposal[np.asarray(model.wv.index_to_key)] = proposal_probabilities(
        model.cum_table
    )
    positive = expected_pairs(np.asarray(walks), np.ones(T.nodes))
    negative = (
        T.negatives * proposal[None, :] * (positive.sum(axis=1)[:, None] - positive)
    )
    return dict(
        walks=walks,
        walks_array=np.asarray(walks, dtype=np.int32),
        degree=degree,
        counts=counts,
        proposal=proposal,
        positive=positive,
        negative=negative,
    )


def metrics(center: np.ndarray, context: np.ndarray, s: dict) -> dict:
    """Per-node readouts. center/context indexed by node id."""
    P, N = s["positive"], s["negative"]
    scores = center @ context.T
    loss = P * np.logaddexp(0, -scores) + N * np.logaddexp(0, scores)
    low, high = scores <= -MAX_EXP, scores >= MAX_EXP
    return dict(
        balanced_norm=balanced_left_norm(center, context),
        raw_norm=np.linalg.norm(center, axis=1),
        node_loss=loss.sum(1) / P.sum(1),
        expected_loss=float(loss.sum() / P.sum()),
        neg_frozen=(N * low).sum(1) / N.sum(1),  # negative mass with score <= -6
        pos_frozen=(P * (low | high)).sum(1) / P.sum(1),
        neg_frozen_total=float((N * low).sum() / N.sum()),
    )
