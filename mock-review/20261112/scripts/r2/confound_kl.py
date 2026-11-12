"""Exposure intervention also changes context specificity: KL(T_i||q) degree slope at beta=0 vs beta=1 (lambda=0), from allocation audit."""
from common import *
import gzip
A = json.load(gzip.open(WT / "results/realized_graph_audit/allocations.json.gz"))
print("conditions:", sorted({(a["beta"], a["lambda"]) for a in A}), "seeds:", sorted({a["seed"] for a in A}))
deg_cache = {}
for lam in (0.0, 1.0):
    for beta in (0.0, 1.0):
        sl_i, sl_r, mean_r = [], [], []
        for a in A:
            if a["beta"] != beta or a["lambda"] != lam: continue
            # degree = intended_support at lambda=0 (row support of one-step transitions)
            nd = a["nodes"]
            kl_r = np.array([n["realized_kl"] for n in nd]); kl_i = np.array([n["intended_kl"] for n in nd])
            if lam == 0.0:
                deg = np.array([n["intended_support"] for n in nd], float)
            else:
                deg = deg_cache[a["seed"]]
            if lam == 0.0 and beta == 0.0: deg_cache[a["seed"]] = deg
            sl_i.append(np.polyfit(np.log(deg), kl_i, 1)[0]); sl_r.append(np.polyfit(np.log(deg), kl_r, 1)[0]); mean_r.append(kl_r.mean())
        print(f"beta={beta} lambda={lam}: slope dKL/dlog d intended {np.mean(sl_i):+.3f} realized {np.mean(sl_r):+.3f} (range {min(sl_r):+.3f}..{max(sl_r):+.3f}); mean realized KL {np.mean(mean_r):.3f}")
