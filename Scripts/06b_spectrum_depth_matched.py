#!/usr/bin/env python3
"""
06b_spectrum_depth_matched.py  --  replaces the depth-dependent site-counting in 06.

Monthly depth ranges from 17 to 82,939 genomes. Counting a site as polymorphic when
its alt frequency exceeds a fixed 0.1% therefore means "seen in >= 9 genomes" in a deep
month and "seen in >= 1 genome" in a thin one, so the site count tracks sequencing
effort rather than biology (corr(log depth, C>T share) = +0.80 under that rule).

Here every month is subsampled without replacement to a common depth before counting,
using the exact multivariate hypergeometric implied by the observed allele counts, and
a site counts once when the alt allele is seen in at least two subsampled genomes.
Scheme C is computed on the same subsample so all three series cover identical months.
"""
import os
import numpy as np
import pandas as pd

RES = "/gdat/fathi/virus/publication/revision/results"
DEPTH = 250          # common depth; months below this are dropped
MIN_ALT = 2          # a site counts once if the alt is seen in >= 2 subsampled genomes
LOW_FREQ = 0.01      # de-novo proxy cut-off
REPS = 25            # subsampling replicates, averaged
SEED = 20260926

d = np.load(os.path.join(RES, "freq_matrix_uncapped.npz"), allow_pickle=True)
bins = [str(b) for b in d["bins"]]
COV, refbase = d["COV"], d["refbase"]
cnt = {b: d[b] for b in "ACGT"}
DEL = d["DEL"]
TYPES = [f"{r}>{a}" for r in "ACGT" for a in "ACGT" if r != a]
rng = np.random.default_rng(SEED)

rows_A, rows_B, rows_C, kept = [], [], [], []
for j, b in enumerate(bins):
    if COV[:, j].max() < DEPTH:
        continue
    kept.append(b)
    ok = COV[:, j] >= DEPTH
    n = COV[ok, j].astype(np.int64)
    counts = np.stack([cnt[x][ok, j] for x in "ACGT"] + [DEL[ok, j]], axis=1).astype(np.int64)
    rb = refbase[ok]

    accB = {t: 0.0 for t in TYPES}
    accC = {t: 0.0 for t in TYPES}
    for _ in range(REPS):
        # exact multivariate hypergeometric subsample to DEPTH genomes per site
        sub = np.zeros_like(counts)
        remaining = n.copy()
        left = np.full(n.shape, DEPTH, dtype=np.int64)
        for c in range(counts.shape[1] - 1):
            good = counts[:, c]
            draw = rng.hypergeometric(np.maximum(good, 0),
                                      np.maximum(remaining - good, 0),
                                      np.minimum(left, remaining))
            sub[:, c] = draw
            remaining -= good
            left -= draw
        sub[:, -1] = left
        tot = sub.sum(axis=1)
        for ri, r in enumerate("ACGT"):
            m = rb == r
            if not m.any():
                continue
            for ai, a in enumerate("ACGT"):
                if a == r:
                    continue
                k = sub[m, ai]
                accB[f"{r}>{a}"] += float((k >= MIN_ALT).sum())
                f = k / np.maximum(tot[m], 1)
                accC[f"{r}>{a}"] += float(k[f < LOW_FREQ].sum())

    # scheme A on the full (unsubsampled) data, restricted to the same sites
    accA = {}
    for r in "ACGT":
        m = rb == r
        for a in "ACGT":
            if a == r:
                continue
            accA[f"{r}>{a}"] = float(cnt[a][ok, j][m].sum()) if m.any() else 0.0

    for name, acc, sink in (("A", accA, rows_A), ("B", accB, rows_B), ("C", accC, rows_C)):
        s = sum(acc.values())
        for t in TYPES:
            sink.append({"TimeBin": b, "MutationType": t,
                         "RelFreq": acc[t] / s if s else np.nan})
    print(f"[06b] {b}: {int(ok.sum())} sites at >= {DEPTH} genomes", flush=True)

for name, rows in (("A_as_published", rows_A), ("B_site_counted_depth_matched", rows_B),
                   ("C_denovo_depth_matched", rows_C)):
    pd.DataFrame(rows).to_csv(os.path.join(RES, f"spectrum_dm_{name}.tsv"), sep="\t",
                              index=False, float_format="%.6g")


def piv(rows):
    return pd.DataFrame(rows).pivot(index="TimeBin", columns="MutationType", values="RelFreq")


pa, pb, pc = piv(rows_A), piv(rows_B), piv(rows_C)
PERIODS = {"Pre-VOC 2020": ("2020_03", "2020_12"), "Alpha/Delta 2021": ("2021_01", "2021_12"),
           "Omicron BA.* 2022": ("2022_01", "2022_12"), "XBB 2023": ("2023_01", "2023_12"),
           "JN.1 2024": ("2024_01", "2024_12"), "XFG-era 2025": ("2025_01", "2025_08")}
out = []
for label, (lo, hi) in PERIODS.items():
    sel = [b for b in kept if lo <= b <= hi]
    row = {"Period": label, "n_months": len(sel)}
    for t in ("C>T", "G>T"):
        row[f"{t} published"] = round(pa.loc[sel, t].mean(), 4)
        row[f"{t} site-counted"] = round(pb.loc[sel, t].mean(), 4)
        row[f"{t} de-novo"] = round(pc.loc[sel, t].mean(), 4)
    out.append(row)
cmp = pd.DataFrame(out)
cmp.to_csv(os.path.join(RES, "spectrum_comparison_depth_matched.tsv"), sep="\t", index=False)
print("\n" + cmp.to_string(index=False))

g = cmp.set_index("Period")
print("\nG>T fold change 2021 -> 2022:")
for col in ("G>T published", "G>T site-counted", "G>T de-novo"):
    print(f"  {col:22s} {g.loc['Alpha/Delta 2021', col] / g.loc['Omicron BA.* 2022', col]:.2f}-fold")
print(f"\nC>T de-novo range across periods: "
      f"{g['C>T de-novo'].min():.3f} - {g['C>T de-novo'].max():.3f}")
print(f"months retained: {len(kept)} of {len(bins)} (dropped: "
      f"{[b for b in bins if b not in kept]})")

# depth-dependence diagnostic, before and after
dep = np.array([COV[:, bins.index(b)].max() for b in kept], dtype=float)
old = pd.read_csv(os.path.join(RES, "spectrum_B_site_counted.tsv"), sep="\t")
old = old[old.MutationType == "C>T"].set_index("TimeBin").RelFreq.reindex(kept).to_numpy()
new = pb["C>T"].reindex(kept).to_numpy()
print(f"\ncorr(log10 depth, C>T site-counted share):  original {np.corrcoef(np.log10(dep), old)[0,1]:+.2f}"
      f"   depth-matched {np.corrcoef(np.log10(dep), new)[0,1]:+.2f}")
