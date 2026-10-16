#!/usr/bin/env python3
"""
09_allele_specific.py  --  Reviewer 1 points 2 and 3.

Three things the published analysis could not do, because it tracked "frequency of
any non-reference base" rather than the frequency of a named substitution:

  1. ALLELE-SPECIFIC TRAJECTORIES. At a codon that is hit twice in different eras
     the published metric merges two different amino-acid changes into one rising
     curve. Position 22893 is the clear case: A>C (S:K444T) swept in 2022-23, died
     out, and A>G (S:K444R) is what is rising in 2025.
  2. RECURRENCE EPISODES. Counting temporally separated episodes in which a named
     allele exceeds 5% after having been below 1% is an alias-free test of
     independent emergence, and does not depend on Pango naming.
  3. REGION-STRATIFIED SENSITIVITY. Recomputes each trajectory weighting every
     WHO-style region equally instead of weighting by how much each region happened
     to deposit, which is Reviewer 1's requested sensitivity analysis.
"""
import os, gzip, glob
from collections import defaultdict
import numpy as np
import pandas as pd

RES = "/gdat/fathi/virus/publication/revision/results"
ALLE = os.path.join(RES, "per_genome_alleles")

d = np.load(os.path.join(RES, "freq_matrix.npz"), allow_pickle=True)
bins = [str(b) for b in d["bins"]]
COV = d["COV"]
refbase = d["refbase"]
cnt = {b: d[b] for b in "ACGT"}

ann = pd.read_csv(os.path.join(RES, "position_annotation.tsv"), sep="\t") \
        .drop_duplicates("Position").set_index("Position")
FOCAL = sorted(ann.index)

# ----------------------------------------------------------- 1. allele-specific
rows = []
for p in FOCAL:
    i = p - 1
    ref = refbase[i]
    for a in "ACGT":
        if a == ref:
            continue
        f = np.where(COV[i] > 0, cnt[a][i] / np.maximum(COV[i], 1), np.nan)
        ok = COV[i] >= 50
        if not np.any(ok & (f > 0.05)):
            continue
        for j, b in enumerate(bins):
            if ok[j]:
                rows.append({"Position": p, "Ref": ref, "Alt": a, "TimeBin": b,
                             "n_genomes": int(COV[i, j]), "freq": round(f[j], 5)})
traj = pd.DataFrame(rows)
traj.to_csv(os.path.join(RES, "allele_specific_trajectories.tsv"), sep="\t",
            index=False)

# ----------------------------------------------------------- 2. episodes
ep_rows = []
for (p, ref, alt), g in traj.groupby(["Position", "Ref", "Alt"]):
    g = g.set_index("TimeBin").reindex(bins)
    f = g.freq.to_numpy(dtype=float)
    above, episodes, cur = False, [], None
    for j, v in enumerate(f):
        if np.isnan(v):
            continue
        if not above and v >= 0.05:
            above, cur = True, [bins[j], bins[j], v]
        elif above:
            if v > cur[2]:
                cur[2] = v
            cur[1] = bins[j]
            if v < 0.01:
                above = False
                episodes.append(tuple(cur))
                cur = None
    if cur:
        episodes.append(tuple(cur))
    aa = ann.loc[p]
    # amino-acid consequence of THIS specific allele
    ep_rows.append({
        "Position": p, "Ref": ref, "Alt": alt,
        "Gene": aa.Gene, "Domain": aa.Domain if isinstance(aa.Domain, str) else "",
        "N_episodes_above_5pct": len(episodes),
        "Episodes": " | ".join(f"{s}->{e} (peak {pk:.2f})" for s, e, pk in episodes),
        "Recurrent": "YES - separate episodes" if len(episodes) >= 2 else "no",
    })
ep = pd.DataFrame(ep_rows).sort_values(["N_episodes_above_5pct", "Position"],
                                       ascending=[False, True])
ep.to_csv(os.path.join(RES, "recurrence_episodes.tsv"), sep="\t", index=False)
print("=== Allele-specific recurrence (alias-free convergence test) ===")
print(ep[ep.N_episodes_above_5pct >= 1].to_string(index=False), flush=True)

# ----------------------------------------------------------- 3. region-stratified
meta = pd.read_parquet(os.path.join(RES, "analysed_metadata.parquet"))
region = dict(zip(meta.Accession, meta.Region))


def parse_line(bases, ref):
    out, i, n, ref = [], 0, len(bases), ref.upper()
    while i < n:
        c = bases[i]
        if c == "^":
            i += 2; continue
        if c == "$":
            i += 1; continue
        if c in "+-":
            j = i + 1; num = 0
            while j < n and bases[j].isdigit():
                num = num * 10 + int(bases[j]); j += 1
            i = j + num; continue
        out.append(ref if c in ".," else ("-" if c == "*" else c.upper()))
        i += 1
    return out


# position -> bin -> region -> [alt, total]  for the alt that is rising in 2025
RISING_2025 = {28297: "C", 22132: "T", 23021: "G", 12892: "A", 22107: "G",
               16600: "C", 22792: "T", 8818: "T", 7869: "T", 5239: "T",
               8350: "C", 4214: "C", 29409: "T", 21627: "A", 23277: "T",
               22893: "G", 22599: "C", 20178: "T", 23039: "G"}
tab = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for f in sorted(glob.glob(os.path.join(ALLE, "*.tsv.gz"))):
    b = os.path.basename(f).replace(".tsv.gz", "")
    with gzip.open(f, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            _, pos, ref, bases, qn = parts
            pos = int(pos)
            alt = RISING_2025.get(pos)
            if alt is None:
                continue
            al = parse_line(bases, ref)
            names = qn.split(",")
            if len(al) != len(names):
                continue
            for a, q in zip(al, names):
                r = region.get(q, "Other/unknown")
                rec = tab[(pos, b)][r]
                rec[1] += 1
                if a == alt:
                    rec[0] += 1

rs = []
for (pos, b), regs in tab.items():
    tot = sum(v[1] for v in regs.values())
    alt = sum(v[0] for v in regs.values())
    if tot < 50:
        continue
    per = {r: v[0] / v[1] for r, v in regs.items() if v[1] >= 20}
    rs.append({
        "Position": pos, "Alt": RISING_2025[pos], "TimeBin": b,
        "n_genomes": tot,
        "freq_naive_global": round(alt / tot, 4),
        "freq_equal_weight_regions": round(float(np.mean(list(per.values()))), 4)
                                     if per else np.nan,
        "n_regions_with_ge20_genomes": len(per),
        "per_region": "; ".join(f"{r}:{v:.2f}({regs[r][1]})" for r, v in
                                sorted(per.items())),
    })
reg = pd.DataFrame(rs).sort_values(["Position", "TimeBin"])
reg.to_csv(os.path.join(RES, "region_stratified_trajectories.tsv"), sep="\t",
           index=False)

late = reg[reg.TimeBin >= "2025_01"]
print("\n=== Region coverage available for the 2025 emergence claim ===")
print(late.groupby("TimeBin")
          .agg(n_positions=("Position", "nunique"),
               max_regions=("n_regions_with_ge20_genomes", "max"),
               genomes=("n_genomes", "max")).to_string())
print("\n[allele] wrote allele_specific_trajectories.tsv, recurrence_episodes.tsv, "
      "region_stratified_trajectories.tsv")
