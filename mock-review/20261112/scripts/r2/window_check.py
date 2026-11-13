from common import *
d = json.load(open(WT / "results/window_training_continuation/report/summary.json"))
print(type(d), list(d.keys())[:20] if isinstance(d, dict) else len(d))
def show(o, depth=0, maxd=2):
    if depth > maxd: return
    if isinstance(o, dict):
        for k, v in list(o.items())[:10]:
            print("  " * depth + f"{k}: {type(v).__name__}" + (f" len={len(v)}" if isinstance(v, (list, dict)) else f" = {v}"))
            show(v, depth + 1, maxd)
    elif isinstance(o, list) and o:
        if isinstance(o[0], dict): print("  " * depth + "item keys: " + str(list(o[0].keys())[:25]))
show(d)
print()
E = [e for e in d["endpoints"] if e["balanced_restart"]]
print("example curve keys:", list(E[0]["curve"].keys()) if isinstance(E[0]["curve"], dict) else type(E[0]["curve"]))
gaps = {}
for e in E:
    c = e["curve"]; m = np.array(c["values"])
    gaps[(e["seed"], e["initialization"], e["window"])] = (m[-1] - m[-2], (m[-1] - m[-2]) / m[-2], e["stability_reached"])
for w in (1, 2, 5, 10):
    per_graph = []
    for s in range(5):
        g = [gaps[(s, i, w)][0] for i in (100000 + s, 200000 + s) if (s, i, w) in gaps]
        per_graph.append(np.mean(g))
    pg = np.array(per_graph); print(f"window {w}: upper gap (highest - preceding bin) graph mean {pg.mean():+.4f} +- SE {pg.std(ddof=1)/np.sqrt(5):.4f}; relative {np.mean([np.mean([gaps[(s,i,w)][1] for i in (100000+s,200000+s)]) for s in range(5)])*100:+.2f}%")
w5 = {(s, i): gaps[(s, i, 5)] for s in range(5) for i in (100000 + s, 200000 + s)}
w10 = {(s, i): gaps[(s, i, 10)] for s in range(5) for i in (100000 + s, 200000 + s)}
print("weakening (|gap10| < |gap5|) in", sum(abs(w10[k][0]) < abs(w5[k][0]) for k in w5), "/10 graph-starts")
st = [k for k in w5 if w5[k][2]]
print("restricted to stable W5 fits (n=%d): mean W5 gap %.4f vs mean W10 gap %.4f" % (len(st), np.mean([w5[k][0] for k in st]), np.mean([w10[k][0] for k in st])))
