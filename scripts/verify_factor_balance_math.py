# /// script
# requires-python = ">=3.10"
# dependencies = ["sympy==1.14.0"]
# ///
"""Verify the flow and discrete balance identities on symbolic rectangular factors.

Run: uv run scripts/verify_factor_balance_math.py
Or: python3 scripts/verify_factor_balance_math.py in an environment with SymPy
1.14.0. No data files are read or written; the input is symbolic real U, V, G.
The dimension-independent proof and assumptions are in paper/current/factor_balance.tex.
"""

import sympy as sp


def main() -> None:
    u = sp.Matrix(2, 2, sp.symbols("u:4", real=True))
    v = sp.Matrix(3, 2, sp.symbols("v:6", real=True))
    g = sp.Matrix(2, 3, sp.symbols("g:6", real=True))
    eta = sp.Symbol("eta", real=True)
    du, dv = -g * v, -g.T * u
    flow = u.T * du + du.T * u - v.T * dv - dv.T * v
    assert flow.applyfunc(sp.expand) == sp.zeros(2)
    next_u, next_v = u + eta * du, v + eta * dv
    change = next_u.T * next_u - next_v.T * next_v - (u.T * u - v.T * v)
    quadratic = eta**2 * (v.T * g.T * g * v - u.T * g * g.T * u)
    assert (change - quadratic).applyfunc(sp.expand) == sp.zeros(2)
    print("Gradient-flow conservation and simultaneous-step drift verified.")


if __name__ == "__main__":
    main()
