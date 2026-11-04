"""Verify the score-loss boundary symbolically; no embedding training is run.

Inputs: symbolic positive weights a,b and t>0; the manuscript's separable loss.
Outputs: JSON identities on stdout, including an exact two-node SGNS example.
Dependencies: sympy==1.14.0 (analysis/realized_graph_audit_requirements.txt).
Run: python3 scripts/check_sgns_norm_boundary.py
"""

from __future__ import annotations

import json

import sympy as sp


def verify() -> dict[str, str]:
    a, b, t = sp.symbols("a b t", positive=True)
    x = sp.Symbol("x", real=True)
    loss = a * sp.log(1 + sp.exp(-x)) + b * sp.log(1 + sp.exp(x))
    gradient = (b * sp.exp(x) - a) / (1 + sp.exp(x))
    curvature = (a + b) * sp.exp(x) / (1 + sp.exp(x)) ** 2
    assert sp.simplify(sp.diff(loss, x) - gradient) == 0
    assert sp.simplify(sp.diff(loss, x, 2) - curvature) == 0
    assert sp.simplify(gradient.subs(x, sp.log(a / b))) == 0
    assert sp.limit(loss.subs(x, t) / t, t, sp.oo) == b
    assert sp.limit(loss.subs(x, -t) / t, t, sp.oo) == a
    assert sp.limit(loss.subs({a: 0, x: -t}), t, sp.oo) == 0

    scores = sp.Matrix([[0, -t], [-t, 0]])
    center = sp.sqrt(t) * sp.eye(2)
    context = sp.sqrt(t) * sp.Matrix([[0, -1], [-1, 0]])
    assert center * context.T == scores
    assert center.T * center == context.T * context == t * sp.eye(2)
    positive = sp.eye(2) / 2
    negative = sp.ones(2) / 2
    example_loss = sum(
        positive[i, j] * sp.log(1 + sp.exp(-scores[i, j]))
        + negative[i, j] * sp.log(1 + sp.exp(scores[i, j]))
        for i in range(2)
        for j in range(2)
    )
    assert sp.simplify(example_loss - 2 * sp.log(2) - sp.log(1 + sp.exp(-t))) == 0
    assert sp.limit(example_loss, t, sp.oo) == 2 * sp.log(2)
    assert sp.trace(center * center.T) == 2 * t
    return {
        "assumptions": "Fixed finite weights; a,b,t>0 and real scores.",
        "gradient": str(gradient),
        "positive_curvature": str(curvature),
        "unique_scalar_minimizer": "log(a/b)",
        "example": "P=I2, c=q=(1/2,1/2), k=2; M=[[0,-t],[-t,0]]",
        "example_loss": "2*log(2) + log(1+exp(-t))",
        "example_balanced_row_norm": "sqrt(t) on both sides",
        "scope": "Exact identities; no rank-24 optimizer convergence claim.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
