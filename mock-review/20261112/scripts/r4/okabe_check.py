"""Delta-E under CVD simulation for the palettes used in Figs. 2, 3, 5, 6."""
import itertools
import cvd

sets = {
    "Fig3": {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "pink": "#CC79A7"},
    "Fig6": {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73"},
    "Fig2": {"green": "#33835b", "purple": "#9654a4"},
    "Fig5": {"blue": "#2563a6", "orange": "#c9681a"},
}
for name, cols in sets.items():
    worst = 1e9
    for (n1, c1), (n2, c2) in itertools.combinations(cols.items(), 2):
        a, b = cvd.hexrgb(c1), cvd.hexrgb(c2)
        vals = []
        for m in (cvd.DEUTER, cvd.PROTAN):
            sa, sb = cvd.simulate(a[None], m)[0], cvd.simulate(b[None], m)[0]
            vals.append(cvd.de(sa, sb))
        worst = min(worst, *vals)
        print(name, n1, n2, "normal %.1f deut %.1f prot %.1f" % (cvd.de(a, b), vals[0], vals[1]))
    print(name, "minimum under CVD: %.1f" % worst)
