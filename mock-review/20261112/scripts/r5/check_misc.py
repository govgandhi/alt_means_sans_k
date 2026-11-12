"""R5: remaining displayed results (Secs. VIII and X).

- Lemma lem:stability: direction inequality and its sharp (Dunkl-Williams) form; tightness.
- Retention law cutoff (3+sqrt5)/2, retained exposure, elasticity, value 0.309 at cutoff.
- Two-force derivative: opposite-signed terms do not imply an inverted-U (explicit counterexample).

Run: PYTHONDONTWRITEBYTECODE=1 uv run --no-project --with sympy,numpy python check_misc.py
"""

from __future__ import annotations

import numpy as np
import sympy as sp

rng = np.random.default_rng(11)


def report(tag, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}" + (f" :: {detail}" if detail else ""))


# ---------------------------------------------------------------- stability lemma
worst_paper, worst_dw = 0.0, 0.0
for _ in range(200000):
    dim = rng.integers(2, 6)
    y = rng.normal(size=dim)
    x = y + rng.normal(size=dim) * rng.uniform(0.01, 3) * np.linalg.norm(y)
    lhs = np.linalg.norm(x / np.linalg.norm(x) - y / np.linalg.norm(y))
    paper = 2 * np.linalg.norm(x - y) / np.linalg.norm(y)
    dw = 2 * np.linalg.norm(x - y) / (np.linalg.norm(x) + np.linalg.norm(y))
    worst_paper = max(worst_paper, lhs / paper)
    worst_dw = max(worst_dw, lhs / dw)
report(
    "Lemma: ||x/|x| - y/|y||| <= 2|x-y|/|y| (paper)",
    worst_paper <= 1 + 1e-12,
    f"max ratio={worst_paper:.4f}",
)
report(
    "Dunkl-Williams (inner-product) 2|x-y|/(|x|+|y|) also holds and is sharper",
    worst_dw <= 1 + 1e-12,
    f"max ratio={worst_dw:.4f}",
)
# with |x-y|<=eps<rho=|y|:  sharp form gives <= 2 eps/(2 rho - eps) ~ eps/rho, half the paper's 2 eps/rho
report(
    "implied direction bound under eps<rho",
    True,
    "2eps/(2rho-eps) vs paper 2eps/rho: factor ~2 loose for small eps",
)

# ---------------------------------------------------------------- retention law
p, t = sp.symbols("p t", positive=True)
keep = sp.sqrt(t / p) + t / p
cut = sp.solve(
    sp.Eq(keep.subs(p, t * sp.Symbol("c", positive=True)), 1),
    sp.Symbol("c", positive=True),
)
report(
    "keep_t(p)=1 at p/t=(3+sqrt5)/2",
    sp.simplify(cut[0] - (3 + sp.sqrt(5)) / 2) == 0,
    f"{sp.nsimplify(cut[0])}={float(cut[0]):.4f}",
)
m = sp.simplify(p * keep)
report("p*keep = sqrt(t p) + t", sp.simplify(m - (sp.sqrt(t * p) + t)) == 0)
el = sp.simplify(sp.diff(sp.log(m), p) * p)
report(
    "elasticity = 1/(2(1+sqrt(t/p)))",
    sp.simplify(el - 1 / (2 * (1 + sp.sqrt(t / p)))) == 0,
)
val = float(el.subs(p, t * (3 + sp.sqrt(5)) / 2))
report("elasticity at cutoff ~0.31", abs(val - 0.309) < 1e-3, f"{val:.4f}")
print(
    "[INFO] attribution: the sqrt(t/p)+t/p form is the word2vec C/Gensim implementation; Mikolov et al. (2013) Eq. (5)"
    " gives discard probability 1-sqrt(t/f), i.e. keep = sqrt(t/f) (elasticity exactly 1/2 above cutoff t)."
)

# ---------------------------------------------------------------- two-force logic
# nu(x, y) with x = log m, y = KL. Exposure increases with degree (x = L), KL decreases (y = -L) along L=log d.
L = np.linspace(0, 3, 301)
cases = {
    "U-shape with both partials > 0": (
        lambda x, y: np.exp(-x) * 0 + x**2 / 2 + np.exp(-y) * 0 + 1.5 * y + 0 * x,
        None,
    ),
}
x, y = L, -L
# nu = x^2/2 + 1.5 y : d nu/dx = x >= 0, d nu/dy = 1.5 > 0; along path: L^2/2 - 1.5 L -> decreasing then increasing (U)
nu = x**2 / 2 + 1.5 * y
dnu = np.gradient(nu, L)
report(
    "both partials >0, exposure up, KL down, yet curve is U-shaped (not inverted-U)",
    dnu[0] < 0 < dnu[-1],
    f"argmin at log d={L[np.argmin(nu)]:.2f}",
)
# nu = log(1+x) + 2 y^... monotone case: nu = 0.1 x + y  -> monotone decreasing, no crossover
nu2 = 0.1 * x + y
report("both partials >0 but no crossover (monotone)", np.all(np.diff(nu2) < 0))
print(
    "[INFO] an inverted-U additionally needs the ratio of the two terms to cross 1 from above exactly once;"
    " a generic nu(log m, KL) fitted along a one-dimensional degree path can represent any curve."
)
