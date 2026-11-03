"""Deterministic expected-pair and logistic-loss diagnostics for window-one SGNS.

Inputs are a fixed integer-node corpus, retention probabilities, the trained
factors and corrected Gensim CDF. No training RNG is read or modified.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def proposal_probabilities(cdf: np.ndarray, bits: int = 32) -> np.ndarray:
    """Uniform-bit integer -> modulo domain -> bisect_left vocabulary slot."""
    cdf = np.asarray(cdf, dtype=np.int64)
    domain = int(cdf[-1])
    if domain <= 0 or np.any(np.diff(cdf) < 0) or domain > 2**bits:
        raise ValueError("Invalid cumulative proposal table.")
    low = np.r_[0, cdf[:-1] + 1]
    high = np.minimum(cdf, domain - 1)
    quotient, remainder = divmod(2**bits, domain)
    counts = quotient * np.maximum(high - low + 1, 0)
    counts += np.maximum(np.minimum(high, remainder - 1) - low + 1, 0)
    result = counts / 2**bits
    np.testing.assert_allclose(result.sum(), 1, rtol=0, atol=1e-15)
    return result


def expected_pairs(walks: np.ndarray, retention: np.ndarray) -> np.ndarray:
    """Integrate over independent deletion before forming symmetric window-one pairs."""
    nodes = len(retention)
    if walks.ndim != 2 or np.any(retention < 0) or np.any(retention > 1):
        raise ValueError("Expected rectangular walks and probabilities in [0,1].")
    result = np.zeros((nodes, nodes))
    keep = retention[walks]
    for left in range(walks.shape[1] - 1):
        deleted_between = np.ones(len(walks))
        for right in range(left + 1, walks.shape[1]):
            weight = keep[:, left] * keep[:, right] * deleted_between
            np.add.at(result, (walks[:, left], walks[:, right]), weight)
            np.add.at(result, (walks[:, right], walks[:, left]), weight)
            deleted_between *= 1 - keep[:, right]
    return result


def evaluate(
    center: np.ndarray,
    context: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    previous_scores: np.ndarray | None,
) -> tuple[dict[str, object], np.ndarray]:
    scores = center @ context.T
    row_mass = positive.sum(axis=1)
    if np.any(row_mass <= 0):
        raise ValueError("Every node must have positive expected exposure.")
    loss_matrix = positive * np.logaddexp(0, -scores) + negative * np.logaddexp(
        0, scores
    )
    outside = (scores <= -6) | (scores >= 6)
    derivative = (
        -positive * expit(-scores) + negative * expit(scores)
    ) / positive.sum()
    input_gradient, output_gradient = derivative @ context, derivative.T @ center
    return {
        "expected_loss": float(loss_matrix.sum() / positive.sum()),
        "node_loss": (loss_matrix.sum(axis=1) / row_mass).tolist(),
        "score_change_rms_by_node": (
            np.zeros(len(center))
            if previous_scores is None
            else np.sqrt(np.mean((scores - previous_scores) ** 2, axis=1))
        ).tolist(),
        "positive_mass_outside_update_window": float(
            positive[outside].sum() / positive.sum()
        ),
        "negative_mass_outside_update_window": float(
            negative[outside].sum() / negative.sum()
        ),
        "idealized_gradient_rms": float(
            np.sqrt(
                (np.square(input_gradient).sum() + np.square(output_gradient).sum())
                / (center.size + context.size)
            )
        ),
    }, scores
