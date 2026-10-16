#!/usr/bin/env python3
"""
13_haplotype_vs_pango.py

Two questions about what the allele-frequency metric is worth.

Does it give earlier warning than lineage tracking? No. Single alleles cross 5% months
earlier, but on unrelated backgrounds -- G22132T crosses in November 2024 carried by
LP.8.1 and LF.7, with no XFG among its carriers -- so comparing a marginal allele
frequency against a joint lineage call is not like for like.

Does it recover the lineage? Yes. Classifying a genome by whether it carries the whole
set reproduces the Pango XFG call almost exactly, with no tree and no nomenclature.
"""
import os, gzip
from collections import defaultdict, Counter
import pandas as pd

RES = "/gdat/fathi/virus/publication/revision/results"
ALLE = os.path.join(RES, "per_genome_alleles")
# the alleles of TABLE1_corrected.tsv that are typed at every focal position
HAPLOTYPE = {28297: "C", 22132: "T", 23021: "G", 12892: "A", 16600: "C",
             22107: "G", 22792: "T", 8818: "T", 7869: "T", 5239: "T",
             8350: "C", 29409: "T", 4214: "C"}
AGREEMENT_MONTHS = ["2025_05", "2025_06", "2025_07", "2025_08"]
ONSET_MONTHS = ["2024_11", "2024_12", "2025_01", "2025_02", "2025_03"]

meta = pd.read_parquet(os.path.join(RES, "analysed_metadata.parquet"))
lineage = dict(zip(meta.Accession, meta.Pangolin.fillna("unassigned")))


def parse_line(bases, ref):
    out, i, n, ref = [], 0, len(bases), ref.upper()
    while i < n:
        c = bases[i]
        if c == "^":
            i += 2
        elif c == "$":
            i += 1
        elif c in "+-":
            j, num = i + 1, 0
            while j < n and bases[j].isdigit():
                num = num * 10 + int(bases[j]); j += 1
            i = j + num
        else:
            out.append(ref if c in ".," else ("-" if c == "*" else c.upper()))
            i += 1
    return out


def genotypes(month):
    """accession -> {position: allele} over the haplotype positions."""
    g = defaultdict(dict)
    with gzip.open(os.path.join(ALLE, f"{month}.tsv.gz"), "rt") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            _, pos, ref, bases, qnames = parts
            pos = int(pos)
            if pos not in HAPLOTYPE:
                continue
            alleles, names = parse_line(bases, ref), qnames.split(",")
            if len(alleles) != len(names):
                continue
            for a, q in zip(alleles, names):
                g[q][pos] = a
    return g


print("Is there an early-warning lead? Who carries the single earliest allele, G22132T?")
onset = []
for m in ONSET_MONTHS:
    g = genotypes(m)
    carriers = [q for q, d in g.items() if d.get(22132) == "T"]
    full = [q for q, d in g.items()
            if len(d) == len(HAPLOTYPE) and all(d[p] == a for p, a in HAPLOTYPE.items())]
    top = Counter(lineage.get(q, "?") for q in carriers).most_common(3)
    onset.append({"month": m, "genomes": len(g), "carry_G22132T": len(carriers),
                  "of_those_XFG": sum(1 for q in carriers
                                      if str(lineage.get(q, "")).startswith("XFG")),
                  "carry_all": len(full),
                  "top_lineages_for_G22132T": "; ".join(f"{k} {v}" for k, v in top)})
on = pd.DataFrame(onset)
on.to_csv(os.path.join(RES, "early_warning_test.tsv"), sep="\t", index=False)
print(on.to_string(index=False), flush=True)
print("\n-> No lead: the early single-allele signal sits on LP.8.1/LF.7, not XFG, and the\n"
      "   full haplotype does not exist before February 2025.\n")

print("Does the tree-free haplotype reproduce the lineage call?")
rows = []
for m in AGREEMENT_MONTHS:
    g = genotypes(m)
    typed = {q: d for q, d in g.items() if len(d) == len(HAPLOTYPE)}
    hap = {q for q, d in typed.items() if all(d[p] == a for p, a in HAPLOTYPE.items())}
    xfg = {q for q in typed if str(lineage.get(q, "")).startswith("XFG")}
    rows.append({"month": m, "genomes_typed_at_all": len(typed),
                 "carry_all": len(hap), "labelled_XFG": len(xfg),
                 "sensitivity_pct": round(100 * len(hap & xfg) / max(len(xfg), 1), 1),
                 "precision_pct": round(100 * len(hap & xfg) / max(len(hap), 1), 1)})
ag = pd.DataFrame(rows)
ag.to_csv(os.path.join(RES, "haplotype_vs_pango_agreement.tsv"), sep="\t", index=False)
print(ag.to_string(index=False), flush=True)

print("\nWhich lineages carry each allele of the table?")
carry = []
for month in AGREEMENT_MONTHS[1:]:
    g = genotypes(month)
    for pos, alt in HAPLOTYPE.items():
        for q, d in g.items():
            if d.get(pos) == alt:
                carry.append((pos, lineage.get(q, "unassigned")))
df = pd.DataFrame(carry, columns=["Position", "Lineage"])
top = (df.groupby("Position").Lineage
         .apply(lambda s: "; ".join(f"{k} {100 * v / len(s):.0f}%"
                                    for k, v in s.value_counts().head(3).items()))
         .rename("Top_lineages").reset_index())
top["N_carriers"] = df.groupby("Position").size().values
top.to_csv(os.path.join(RES, "table1_lineage_assignment.tsv"), sep="\t", index=False)
print(top.to_string(index=False), flush=True)
