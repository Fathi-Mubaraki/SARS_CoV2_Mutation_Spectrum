#!/usr/bin/env python3
"""
06_spectrum.py  --  Reviewer 1 point 4, Reviewer 2 points 1/7.

The manuscript's mutation spectrum counts every alt-allele observation in every
genome. That means a lineage-defining substitution carried by all 80,000 genomes in
a month contributes 80,000 counts, while a genuine one-off de-novo change
contributes one. The spectrum is therefore close to a picture of which lineage was
circulating, not of which mutational process is operating.

This script quantifies that and recomputes the spectrum three ways:
  A. as published                  - weighted by every genome observation
  B. lineage-independent           - each polymorphic SITE counted once per month
  C. de-novo proxy                 - only alleles below 1% in that month, i.e.
                                     variants that have NOT been swept up by a
                                     lineage, weighted by observation

Scheme A here reproduces the published counting exactly, which is what the comparison
needs. Schemes B and C use a fixed 0.1% frequency threshold, which is not a fixed
evidence threshold when monthly depth ranges from 17 to 82,939 genomes; 06b recomputes
both after subsampling every month to a common depth, and those are the figures
reported.
"""
import os
import numpy as np
import pandas as pd

RES = "/gdat/fathi/virus/publication/revision/results"
d = np.load(os.path.join(RES, "freq_matrix.npz"), allow_pickle=True)
bins = [str(b) for b in d["bins"]]
COV = d["COV"]
refbase = d["refbase"]
counts = {b: d[b] for b in "ACGT"}
npos, nb = COV.shape

TYPES = [f"{r}>{a}" for r in "ACGT" for a in "ACGT" if r != a]
rows_A, rows_B, rows_C, frac_rows = [], [], [], []

for j, b in enumerate(bins):
    cov = COV[:, j].astype(float)
    ok = cov > 0
    tot_A = {t: 0.0 for t in TYPES}
    tot_B = {t: 0.0 for t in TYPES}
    tot_C = {t: 0.0 for t in TYPES}
    high_freq_contrib = 0.0
    all_contrib = 0.0

    for r in "ACGT":
        mref = ok & (refbase == r)
        if not mref.any():
            continue
        c = cov[mref]
        for a in "ACGT":
            if a == r:
                continue
            k = counts[a][mref, j].astype(float)
            f = np.divide(k, c, out=np.zeros_like(k), where=c > 0)
            t = f"{r}>{a}"
            tot_A[t] = k.sum()                          # as published
            tot_B[t] = float((f >= 0.001).sum())        # one count per polymorphic site
            tot_C[t] = k[f < 0.01].sum()                # de-novo proxy
            high_freq_contrib += k[f > 0.5].sum()
            all_contrib += k.sum()

    for name, tot, sink in (("A_as_published", tot_A, rows_A),
                            ("B_site_counted", tot_B, rows_B),
                            ("C_low_freq_denovo_proxy", tot_C, rows_C)):
        s = sum(tot.values())
        for t in TYPES:
            sink.append({"TimeBin": b, "MutationType": t,
                         "RelFreq": (tot[t] / s) if s else np.nan,
                         "Count": tot[t]})
    frac_rows.append({
        "TimeBin": b,
        "pct_of_spectrum_from_sites_above_50pct":
            round(100 * high_freq_contrib / all_contrib, 2) if all_contrib else np.nan,
        "n_genomes_max_cov": int(COV[:, j].max()),
    })

for name, rows in (("A_as_published", rows_A), ("B_site_counted", rows_B),
                   ("C_low_freq_denovo_proxy", rows_C)):
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"spectrum_{name}.tsv"),
                              sep="\t", index=False, float_format="%.6g")

frac = pd.DataFrame(frac_rows)
frac.to_csv(os.path.join(RES, "spectrum_lineage_dominance.tsv"), sep="\t",
            index=False)

# --------------------------------------------------- headline comparison table
def pivot(rows):
    return pd.DataFrame(rows).pivot(index="TimeBin", columns="MutationType",
                                    values="RelFreq")


pa, pb, pc = pivot(rows_A), pivot(rows_B), pivot(rows_C)
PERIODS = {"Pre-VOC 2020": ("2020_03", "2020_12"),
           "Alpha/Delta 2021": ("2021_01", "2021_12"),
           "Omicron BA.* 2022": ("2022_01", "2022_12"),
           "XBB 2023": ("2023_01", "2023_12"),
           "JN.1 2024": ("2024_01", "2024_12"),
           "XFG-era 2025": ("2025_01", "2025_09")}
out = []
for label, (lo, hi) in PERIODS.items():
    sel = [b for b in bins if lo <= b <= hi]
    for t in ("C>T", "G>T", "A>G", "T>C"):
        out.append({"Period": label, "Type": t,
                    "A_as_published": round(pa.loc[sel, t].mean(), 4),
                    "B_site_counted": round(pb.loc[sel, t].mean(), 4),
                    "C_denovo_proxy": round(pc.loc[sel, t].mean(), 4)})
cmp = pd.DataFrame(out).pivot(index="Period", columns="Type",
                              values=["A_as_published", "B_site_counted",
                                      "C_denovo_proxy"])
cmp = cmp.reindex(list(PERIODS))
cmp.to_csv(os.path.join(RES, "spectrum_comparison.tsv"), sep="\t")
print(cmp.to_string())
print()
print("Share of the published spectrum contributed by alleles already above 50% "
      "in that month (i.e. lineage-defining, not de-novo):")
print(frac.set_index("TimeBin").loc[
    [b for b in bins if b.endswith(("_01", "_07"))] + bins[-2:]].to_string())
