# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "numba", "networkx", "matplotlib", "scipy"]
# ///
"""Run one configuration on the BA population corpus; save per-epoch balanced/raw norms."""
import argparse, json, time, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from sgns import train, balanced_left, ba_corpus, graph_walks, SETTINGS, OUT

p = argparse.ArgumentParser()
p.add_argument("name"); p.add_argument("--setting", default="B")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--over", default="{}", help="json overrides")
p.add_argument("--fresh", action="store_true")
p.add_argument("--N", type=int, default=10_000)
p.add_argument("--walks", type=int, default=10)
p.add_argument("--alpha", type=float, default=0.025)
p.add_argument("--min_alpha", type=float, default=1e-4)
p.add_argument("--schedule", default=None, help="json list of [a0,a1] per epoch")
p.add_argument("--freeze_c", action="store_true")
p.add_argument("--shuffle", action="store_true")
p.add_argument("--graph", action="store_true", help="walks on the actual BA graph")
p.add_argument("--fresh_walks", action="store_true")
p.add_argument("--every", type=int, default=1)
a = p.parse_args()
cfg = SETTINGS[a.setting] | json.loads(a.over)
walks, deg, prob = ba_corpus(a.seed, N=a.N, walks_per_node=a.walks)
corpora = None
if a.graph:
    walks, deg, prob = graph_walks(a.seed, 7000 + a.seed, N=a.N, walks_per_node=a.walks)
    if a.fresh_walks:
        corpora = [walks] + [graph_walks(a.seed, 7000 + 100 * e + a.seed, N=a.N, walks_per_node=a.walks)[0]
                             for e in range(1, cfg["epochs"])]
out = {"deg": deg, "cfg": json.dumps(cfg | vars(a))}
t0 = time.time()
def cb(ep, W, C, n_upd, eta2):
    last = ep == (len(sched) if sched else cfg["epochs"]) - 1
    if (ep + 1) % a.every and not last:
        return
    B, s = balanced_left(W, C)
    out[f"bal_{ep+1}"] = np.linalg.norm(B, axis=1)
    out[f"raw_{ep+1}"] = np.linalg.norm(W, axis=1)
    out[f"craw_{ep+1}"] = np.linalg.norm(C, axis=1)
    out[f"sv_{ep+1}"] = s
    out[f"nupd_{ep+1}"] = n_upd.copy(); out[f"eta2_{ep+1}"] = eta2.copy()
    print(f"{a.name} ep{ep+1} {time.time()-t0:.0f}s top sv {s[:3].round(1)}", flush=True)
sched = json.loads(a.schedule) if a.schedule else None
W, C, counts, n_upd, eta2 = train(walks, prob, **cfg, alpha=a.alpha, min_alpha=a.min_alpha,
                                  seed=a.seed, fresh=a.fresh, callback=cb, schedule=sched,
                                  freeze_c=a.freeze_c, shuffle=a.shuffle, corpora=corpora)
out["counts"] = counts
OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT / f"{a.name}.npz", W=W, C=C, **out)
