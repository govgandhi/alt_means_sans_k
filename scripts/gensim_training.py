"""Correct integer-node negative tables and reject defective BLAS before training.

Inputs: an untrained Gensim model and optional counts indexed by integer node ID.
Outputs: a corrected cumulative table; train_checked returns Gensim's train counts.
Used by the standalone experiment scripts; setup is documented in
analysis/negative_sampling_control_protocol.md.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from check_control_gensim import blas_check
from gensim.models import Word2Vec

TABLE_DOMAIN = 2**31 - 1  # Gensim 4.4.0's make_cum_table default.


@lru_cache(maxsize=1)
def require_correct_blas() -> None:
    try:
        blas_check()
    except RuntimeError as error:
        raise RuntimeError(
            "Gensim's BLAS score check failed. In a virtual environment, run "
            "python3 scripts/build_control_gensim.py --install, then restart Python."
        ) from error


def set_negative_table(
    model: Word2Vec, counts_by_node: np.ndarray | None = None
) -> None:
    """Assign counts by vocabulary slot, avoiding integer key/position ambiguity."""
    if not model.negative:
        raise ValueError("This helper requires negative sampling.")
    if counts_by_node is None:
        counts = model.wv.expandos["count"].astype(np.float64)
    else:
        counts_by_node = np.asarray(counts_by_node)
        keys = model.wv.index_to_key
        if (
            counts_by_node.ndim != 1
            or not all(isinstance(key, (int, np.integer)) for key in keys)
            or set(keys) != set(range(len(counts_by_node)))
        ):
            raise ValueError("Counts must cover every contiguous integer node ID.")
        counts = counts_by_node[np.asarray(keys)].astype(np.float64)
    if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("Negative counts must be finite and positive.")
    weights = counts ** float(model.ns_exponent)
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise ValueError("Invalid negative-sampling exponent or weights.")
    cumulative = np.cumsum(weights)
    model.cum_table = np.rint(cumulative / cumulative[-1] * TABLE_DOMAIN).astype(
        np.uint32
    )


def train_checked(model: Word2Vec, walks: list[list[int]]) -> tuple[int, int]:
    """Train only after the active compiled numerical implementation passes."""
    require_correct_blas()
    return model.train(
        corpus_iterable=walks,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        epochs=model.epochs,
        start_alpha=model.alpha,
        end_alpha=model.min_alpha,
    )
