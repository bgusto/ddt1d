#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

infile = "../../ensembleprobs.dat"
outfile = "../../figures/roc.pdf"

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
fpr, tpr, _ = roc_curve(y, p)
auc = roc_auc_score(y, p)

fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.plot([0, 1], [0, 1], color="0.5", ls="--", lw=1.6)     # chance line
ax.plot(fpr, tpr, color="k", solid_joinstyle="round")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
tk = np.arange(0, 1.01, 0.2)
ax.set_xticks(tk); ax.set_yticks(tk)
ax.set_xticklabels(["%.1f" % t for t in tk])   # plain text, not math, so tick labels use the figure font
ax.set_yticklabels(["%.1f" % t for t in tk])
ax.set_xlabel(r"False positive rate")
ax.set_ylabel(r"True positive rate")
ax.set_aspect("equal")
ax.grid(True, which="both")
ax.minorticks_on()
ax.text(0.95, 0.08, r"AUC = %.3f" % auc,
        ha="right", va="bottom",
        bbox=dict(boxstyle="square,pad=0.4", fc="white", ec="k", lw=1.3))

fig.savefig(outfile)
fig.savefig(outfile.replace(".pdf", ".png"))
print("wrote", outfile, "  AUC =", round(auc, 3))
