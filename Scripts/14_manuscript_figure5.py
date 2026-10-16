#!/usr/bin/env python3
"""
14_manuscript_figure5.py -- Figure 5 of the manuscript.

Trajectories of the fourteen alleles that rose from below 5% to above 50% of genomes
between September 2024 and August 2025, with the four highest-slope sites shown last
for comparison. Frequencies are allele-specific and are blanked in months where fewer
than 80% of that month's genomes align at the position.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/gdat/fathi/virus/publication/revision/results"
OUT = "/gdat/fathi/virus/publication/revision/figures/Figure5_Top_Trajectories.png"

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})

d = np.load(os.path.join(RES, "freq_matrix_uncapped.npz"), allow_pickle=True)
bins = [str(b) for b in d["bins"]]
COV, refbase = d["COV"], d["refbase"]
cnt = {b: d[b] for b in "ACGT"}
man = pd.read_csv(os.path.join(RES, "bin_manifest.tsv"), sep="\t").set_index("bin")
nbin = np.array([man.n_genomes.get(b, 0) for b in bins], dtype=float)
t = np.arange(len(bins))

tab = pd.read_csv(os.path.join(RES, "TABLE1_corrected.tsv"), sep="\t")
rising = [(int(m[1:-1]), m[-1], aa.split(" (")[0]) for m, aa in
          zip(tab.Mutation, tab["Amino-acid change"])]
# the four highest overall slopes, for contrast
sl = pd.read_csv(os.path.join(RES, "slopes_with_uncertainty.tsv"), sep="\t")
top = sl.nlargest(4, "Slope_Overall")
SWEPT_AA = {22942: "S:N460K", 22577: "S:G339H\u2020", 23019: "S:F486P\u2020",
            22896: "S:V445H\u2020", 22895: "S:V445H\u2020", 23018: "S:F486P\u2020"}
swept = [(int(r.Position), str(r.Alt), SWEPT_AA.get(int(r.Position), ""))
         for r in top.itertuples()]

panels = [(p, a, aa, False) for p, a, aa in rising] + \
         [(p, a, aa, True) for p, a, aa in swept]

fig, axes = plt.subplots(3, 6, figsize=(11.5, 5.4), sharex=True, sharey=True)
show = [i for i, b in enumerate(bins) if b.endswith("_01")]
for ax, (pos, alt, aa, is_swept) in zip(axes.ravel(), panels):
    i = pos - 1
    ok = (COV[i] >= 50) & (COV[i] / np.maximum(nbin, 1) >= 0.8)
    f = np.where(ok, cnt[alt][i] / np.maximum(COV[i], 1), np.nan)
    ax.plot(t, f, lw=1.4, color="#999999" if is_swept else "#0072B2")
    ax.set_title(f"{refbase[i]}{pos}{alt}\n{aa}", fontsize=7,
                 color="#666666" if is_swept else "black")
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(show)
    ax.set_xticklabels([bins[k][:4] for k in show], fontsize=6, rotation=90)
    for s in ax.spines.values():
        s.set_color("#bbbbbb" if is_swept else "black")
for ax in axes.ravel()[len(panels):]:
    ax.set_visible(False)
fig.supylabel("Allele frequency", fontsize=9)
fig.supxlabel("Sample collection month", fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}  ({len(rising)} rising, {len(swept)} swept shown in grey)")
