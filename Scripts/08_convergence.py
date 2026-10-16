#!/usr/bin/env python3
"""
08_convergence.py  --  Reviewer 1, major point 3 (second half).

Cross-tabulates, for each focal position, the allele carried by every individual
genome against that genome's Pango lineage. This is what separates:

  linkage    - the mutation is essentially fixed inside one lineage and its
               descendants, and rises only because that lineage took over;
  convergence - the mutation is at high frequency in lineages that are not
               descendants of one another, i.e. it was acquired more than once.

The per-lineage carriage table it produces supplies the lineage assignment for Table 1.
Its count of distinct lineage names is not usable as a convergence statistic: without
the Pango alias key, BE, BF and BQ are counted as independent of BA.5, so the count
cannot separate convergence from descent. Convergence is established instead by the
codon-level analysis in 11, which needs no lineage nomenclature.

Outputs:
  results/convergence_by_position.tsv   one row per focal position
  results/lineage_carriage.tsv          position x lineage carriage detail
  results/K444T_timeline.tsv            worked example of a recurrent site
"""
import os, gzip, glob
from collections import defaultdict
import numpy as np
import pandas as pd

RES = "/gdat/fathi/virus/publication/revision/results"
ALLE = os.path.join(RES, "per_genome_alleles")

ann = pd.read_csv(os.path.join(RES, "position_annotation.tsv"), sep="\t")
ann = ann.drop_duplicates("Position").set_index("Position")

meta = pd.read_parquet(os.path.join(RES, "analysed_metadata.parquet"))
lineage = dict(zip(meta.Accession, meta.Pangolin.fillna("unassigned")))
print(f"[conv] lineage lookup for {len(lineage)} genomes", flush=True)


def parse_line(bases: str, ref: str):
    """Strip mpileup markup, returning one allele per aligned genome, in order."""
    out = []
    i, n = 0, len(bases)
    ref = ref.upper()
    while i < n:
        c = bases[i]
        if c == "^":                      # read-start marker + mapping-quality char
            i += 2
            continue
        if c == "$":
            i += 1
            continue
        if c in "+-":                     # indel attached to the PREVIOUS base
            j = i + 1
            num = 0
            while j < n and bases[j].isdigit():
                num = num * 10 + int(bases[j]); j += 1
            i = j + num
            continue
        if c in ".,":
            out.append(ref)
        elif c == "*":
            out.append("-")
        else:
            out.append(c.upper())
        i += 1
    return out


# position -> lineage -> [alt_count, total]
carriage = defaultdict(lambda: defaultdict(lambda: [0, 0]))
# position -> bin -> lineage -> [alt, total]  (only for the K444T example)
K444T_POS = 22893
k_timeline = defaultdict(lambda: defaultdict(lambda: [0, 0]))

alt_of = dict(zip(ann.index, ann.Alt))

for f in sorted(glob.glob(os.path.join(ALLE, "*.tsv.gz"))):
    b = os.path.basename(f).replace(".tsv.gz", "")
    with gzip.open(f, "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            _, pos, ref, bases, qnames = parts
            pos = int(pos)
            alt = alt_of.get(pos)
            if alt is None or alt == "Del":
                continue
            alleles = parse_line(bases, ref)
            names = qnames.split(",")
            if len(alleles) != len(names):
                continue                                  # skip malformed rows
            for a, q in zip(alleles, names):
                lin = lineage.get(q, "unassigned")
                rec = carriage[pos][lin]
                rec[1] += 1
                if a == alt:
                    rec[0] += 1
                if pos == K444T_POS:
                    kr = k_timeline[b][lin]
                    kr[1] += 1
                    if a == alt:
                        kr[0] += 1
    print(f"[conv] parsed {b}", flush=True)


def clade(lin):
    """Coarse clade label so that parent/descendant lineages group together."""
    if not isinstance(lin, str) or lin in ("unassigned", ""):
        return "unassigned"
    if lin.startswith("X"):
        return lin.split(".")[0]                 # XBB, XFG, XEC ... recombinants
    parts = lin.split(".")
    if parts[0] in ("B", "A"):
        return ".".join(parts[:3]) if len(parts) >= 3 else lin
    return parts[0]                              # BA, BQ, JN, KP, NB, LP ...


rows, detail = [], []
for pos, lins in carriage.items():
    tot_all = sum(v[1] for v in lins.values())
    alt_all = sum(v[0] for v in lins.values())
    # lineages where the mutation is essentially fixed, with enough genomes to judge
    fixed, partial = [], []
    for lin, (a, t) in lins.items():
        if t < 30 or lin == "unassigned":
            continue
        fr = a / t
        detail.append({"Position": pos, "Lineage": lin, "n_genomes": t,
                       "n_with_alt": a, "carriage": round(fr, 4),
                       "Clade": clade(lin)})
        if fr >= 0.9:
            fixed.append(lin)
        elif 0.05 <= fr < 0.9:
            partial.append(lin)
    cl_fixed = sorted(set(clade(l) for l in fixed))
    a = ann.loc[pos] if pos in ann.index else None
    rows.append({
        "Position": pos,
        "AA_change": a.AA_change if a is not None else "",
        "Effect": a.Effect if a is not None else "",
        "Alt": alt_of.get(pos, ""),
        "Overall_carriage": round(alt_all / tot_all, 4) if tot_all else np.nan,
        "N_lineages_fixed_ge90pct": len(fixed),
        "N_distinct_clades_fixed": len(cl_fixed),
        "Clades_fixed": ", ".join(cl_fixed[:12]),
        "N_lineages_partial_5to90pct": len(partial),
        "Interpretation": ("multiple independent backgrounds"
                           if len(cl_fixed) >= 3 else
                           ("two backgrounds" if len(cl_fixed) == 2 else
                            "single lineage background")),
    })

conv = pd.DataFrame(rows).sort_values("N_distinct_clades_fixed", ascending=False)
conv.to_csv(os.path.join(RES, "convergence_by_position.tsv"), sep="\t", index=False)
pd.DataFrame(detail).sort_values(["Position", "carriage"], ascending=[True, False]) \
  .to_csv(os.path.join(RES, "lineage_carriage.tsv"), sep="\t", index=False)
print(conv.to_string(index=False), flush=True)

ktl = []
for b, lins in sorted(k_timeline.items()):
    tot = sum(v[1] for v in lins.values())
    alt = sum(v[0] for v in lins.values())
    if tot < 30:
        continue
    carriers = sorted(((a / t, l, t) for l, (a, t) in lins.items()
                       if t >= 20 and a / t >= 0.5), reverse=True)
    ktl.append({"bin": b, "n_genomes": tot,
                "K444T_freq": round(alt / tot, 4),
                "carrying_lineages": "; ".join(f"{l} ({t})" for _, l, t in carriers[:6])})
pd.DataFrame(ktl).to_csv(os.path.join(RES, "K444T_timeline.tsv"), sep="\t", index=False)
print("[conv] wrote K444T_timeline.tsv", flush=True)
