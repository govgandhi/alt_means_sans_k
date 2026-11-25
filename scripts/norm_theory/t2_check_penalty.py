"""Monte Carlo check of the expected per-example squared gradient (t2_modified_flow.py).

Random factors (sd 0.3) on the seed-0 PR #36 proposal q; three (input, output) pairs;
20,000 negative draws each, gradients by autograd. Prints formula vs Monte Carlo.
Run: uv run --no-project --python 3.13 --with torch --with numpy \
         python -I scripts/norm_theory/t2_check_penalty.py
"""

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
torch.manual_seed(0)
rng = np.random.default_rng(0)
z = np.load(ROOT / "results/norm_theory/t2_setup_s0.npz")
q = torch.tensor(z["q"])
K = 5
n, d = z["W0"].shape
W = torch.tensor(rng.normal(0, 0.3, (n, d)))
C = torch.tensor(rng.normal(0, 0.3, (n, d)))
S = W @ C.T
r, p = torch.sigmoid(-S), torch.sigmoid(S)
v2, u2 = (C * C).sum(1), (W * W).sum(1)
m = (p * q[None, :]) @ C
for a, b in [(0, 5), (17, 300), (250, 3)]:
    # formula
    f_u = ((r[a, b] * C[b] - K * m[a]) ** 2).sum() + K * (
        (q * p[a] ** 2 * v2).sum() - (m[a] ** 2).sum()
    )
    f_v = u2[a] * (
        r[a, b] ** 2
        + K * (q * p[a] ** 2).sum()
        + K * (K - 1) * (q * q * p[a] ** 2).sum()
    )
    # Monte Carlo
    tot = 0.0
    T = 20000
    cdf = np.cumsum(z["q"])
    cdf /= cdf[-1]
    for _ in range(T):
        negs = np.searchsorted(cdf, rng.random(K), side="right")
        Wl = W.clone().requires_grad_(True)
        Cl = C.clone().requires_grad_(True)
        s = Wl[a] @ Cl.T
        loss_k = (
            torch.nn.functional.softplus(-s[b])
            + torch.nn.functional.softplus(s[negs]).sum()
        )
        gW, gC = torch.autograd.grad(loss_k, (Wl, Cl))
        tot += float((gW**2).sum() + (gC**2).sum())
    print(a, b, "formula", float(f_u + f_v), "MC", tot / T)
