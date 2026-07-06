#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

infile = "../../ensembleprobs.dat"
outfile = "../../figures/calib.pdf"
nbin = 10

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{courier}\renewcommand{\seriesdefault}{\bfdefault}\boldmath",
    "font.family": "monospace",
    "font.monospace": ["Courier"],
    "font.weight": "bold",
    "font.size": 11,
    "axes.linewidth": 1.6,
    "lines.linewidth": 2.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 6.5, "ytick.major.size": 6.5,
    "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "grid.color": "0.6", "grid.linestyle": ":", "grid.linewidth": 1.0,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

y, p, _ = np.loadtxt(infile, unpack=True)

edges = np.linspace(0, 1, nbin + 1)
conf, freq, ece = [], [], 0.0
for b in range(nbin):
    hi = p <= edges[b+1] if b == nbin-1 else p < edges[b+1]
    inbin = (p >= edges[b]) & hi
    if inbin.sum() == 0:
        continue
    c, f = p[inbin].mean(), y[inbin].mean()
    conf.append(c); freq.append(f)
    ece += inbin.mean() * abs(f - c)

fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1.6)     # perfect calibration
ax.plot(conf, freq, color="k", marker="o", ms=7, mfc="k", mec="k")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
tk = np.arange(0, 1.01, 0.2)
ax.set_xticks(tk); ax.set_yticks(tk)
ax.set_xticklabels(["%.1f" % t for t in tk])   # plain text, not math, so tick labels use the figure font
ax.set_yticklabels(["%.1f" % t for t in tk])
ax.set_xlabel(r"Predicted probability P(DDT)")
ax.set_ylabel(r"Observed detonation fraction")
ax.set_aspect("equal")
ax.grid(True, which="both")
ax.minorticks_on()
ax.text(0.95, 0.08, r"ECE = %.3f" % ece,
        ha="right", va="bottom",
        bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="k", lw=1.3))

fig.savefig(outfile)
fig.savefig(outfile.replace(".pdf", ".png"))
print("wrote", outfile, "  ECE =", round(ece, 3))
