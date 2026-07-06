#!/usr/bin/env python3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

infile = "../../ensembleprobs.dat"
outfile = "../../figures/confus.pdf"

mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{courier}\renewcommand{\seriesdefault}{\bfdefault}\boldmath",
    "font.family": "monospace",
    "font.monospace": ["Courier"],
    "font.weight": "bold",
    "font.size": 11,
    "axes.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

y, p, _ = np.loadtxt(infile, unpack=True)
yhat = (p >= 0.5).astype(int)

# rows = true (0,1), cols = predicted (0,1)
cm = np.zeros((2, 2), dtype=int)
for t, h in zip(y.astype(int), yhat):
    cm[t, h] += 1

fig, ax = plt.subplots(figsize=(4.8, 4.8))
shade = cm / cm.max() * 0.35                               # keep fills light so black text reads
ax.imshow(shade, cmap="Greys", vmin=0, vmax=1)

for i in range(2):
    for j in range(2):
        ax.text(j, i, r"%d" % cm[i, j], ha="center", va="center", fontsize=22)

ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels([r"no DDT", r"DDT"])
ax.set_yticklabels([r"no DDT", r"DDT"], rotation=90, va="center")
ax.set_xlabel(r"Predicted"); ax.set_ylabel(r"True")
ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()

# bold black cell borders
ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
ax.grid(which="minor", color="k", linewidth=1.6)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)

acc = np.trace(cm) / cm.sum()
ax.set_title(r"accuracy = %.3f" % acc, pad=28)

fig.savefig(outfile)
fig.savefig(outfile.replace(".pdf", ".png"))
print("wrote", outfile, "  acc =", round(acc, 3))
