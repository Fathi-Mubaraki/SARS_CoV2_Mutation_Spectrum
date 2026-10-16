#!/usr/bin/env python3
"""
03_annotate.py  --  Reviewer 1, major point 3 (first half).

Annotates every genomic position the manuscript highlights with its gene, protein,
codon, amino-acid change and functional domain, so that Table 1 can report
amino-acid changes rather than bare nucleotide coordinates.

Coordinates are for NC_045512.2. ORF1ab is handled with the -1 ribosomal
frameshift at nt 13468 so that nsp12-nsp16 get correct pp1ab residue numbers
(validated against nsp12 P323L = C14408T).
"""
import os
import numpy as np
import pandas as pd

RES = "/gdat/fathi/virus/publication/revision/results"
REF = "/gdat/fathi/virus/publication/NC_045512.2.fasta"

seq = "".join(l.strip() for l in open(REF) if not l.startswith(">")).upper()
assert len(seq) == 29903, len(seq)

GENES = [  # (name, start, end) 1-based inclusive
    ("ORF1a", 266, 13483), ("ORF1b", 13468, 21555), ("S", 21563, 25384),
    ("ORF3a", 25393, 26220), ("E", 26245, 26472), ("M", 26523, 27191),
    ("ORF6", 27202, 27387), ("ORF7a", 27394, 27759), ("ORF7b", 27756, 27887),
    ("ORF8", 27894, 28259), ("N", 28274, 29533), ("ORF10", 29558, 29674),
]
# nsp boundaries in pp1ab residue numbering
NSP = [("nsp1", 1, 180), ("nsp2", 181, 818), ("nsp3", 819, 2763),
       ("nsp4", 2764, 3263), ("nsp5 (3CLpro)", 3264, 3569), ("nsp6", 3570, 3859),
       ("nsp7", 3860, 3942), ("nsp8", 3943, 4140), ("nsp9", 4141, 4253),
       ("nsp10", 4254, 4392), ("nsp12 (RdRp)", 4393, 5324),
       ("nsp13 (helicase)", 5325, 5925), ("nsp14 (ExoN)", 5926, 6452),
       ("nsp15", 6453, 6798), ("nsp16", 6799, 7096)]
# Spike domains, residue numbering
SDOM = [("NTD", 14, 305), ("RBD", 319, 541), ("S1 (other)", 1, 685),
        ("S2", 686, 1273)]
RBM = (438, 506)
FURIN = (681, 685)

CODON = {}
_b = "TCAG"
_aa = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
for i, c1 in enumerate(_b):
    for j, c2 in enumerate(_b):
        for k, c3 in enumerate(_b):
            CODON[c1 + c2 + c3] = _aa[i * 16 + j * 4 + k]


def pp1ab_residue(p):
    """pp1ab residue number and 0-based offset of p within its codon."""
    if 266 <= p <= 13468:
        return (p - 266) // 3 + 1, (p - 266) % 3
    if 13468 < p <= 21555:                       # after the -1 frameshift
        return (p - 13468) // 3 + 4402, (p - 13468) % 3
    return None, None


def annotate(p, alt):
    """Return an annotation dict for 1-based position p mutated to `alt`."""
    out = dict(Position=p, Ref=seq[p - 1], Alt=alt, Gene="intergenic/UTR",
               Protein="", Residue="", AA_change="", Effect="", Domain="",
               Codon_ref="", Codon_alt="")
    if p < 266 or p > 29674:
        out["Gene"] = "5'UTR" if p < 266 else "3'UTR"
        return out

    if 266 <= p <= 21555:
        res, off = pp1ab_residue(p)
        gene = "ORF1a" if p <= 13483 else "ORF1b"
        if 13468 < p <= 13483:
            gene = "ORF1a/ORF1b overlap"
        start = p - off
        codon_ref = seq[start - 1:start + 2]
        prot = next((n for n, a, b in NSP if a <= res <= b), "")
        nsp_res = ""
        for n, a, b in NSP:
            if a <= res <= b:
                nsp_res = res - a + 1
                break
        out.update(Gene=gene, Protein=prot, Residue=f"pp1ab:{res}"
                   + (f" / {prot.split()[0]}:{nsp_res}" if prot else ""))
    else:
        gene = next((n for n, a, b in GENES if a <= p <= b and n not in
                     ("ORF1a", "ORF1b")), None)
        if gene is None:
            return out
        a = dict((n, s) for n, s, e in GENES)[gene]
        off = (p - a) % 3
        res = (p - a) // 3 + 1
        start = p - off
        codon_ref = seq[start - 1:start + 2]
        out.update(Gene=gene, Protein=gene, Residue=f"{gene}:{res}")
        if gene == "S":
            dom = next((n for n, x, y in SDOM if x <= res <= y), "")
            if RBM[0] <= res <= RBM[1]:
                dom = "RBD / receptor-binding motif (RBM)"
            elif dom == "RBD":
                dom = "RBD (core)"
            if FURIN[0] <= res <= FURIN[1]:
                dom = "S1/S2 furin cleavage site"
            out["Domain"] = dom

    if len(codon_ref) != 3:
        return out
    if alt == "Del":
        out.update(Codon_ref=codon_ref, Codon_alt="-",
                   AA_change=out["Residue"].split("/")[-1].strip() + "del",
                   Effect="in-frame deletion" if True else "")
        return out
    codon_alt = list(codon_ref)
    codon_alt[off] = alt
    codon_alt = "".join(codon_alt)
    aa_r, aa_a = CODON.get(codon_ref, "?"), CODON.get(codon_alt, "?")
    tag = out["Residue"].split(":")[-1].split(" ")[0]
    out.update(Codon_ref=codon_ref, Codon_alt=codon_alt,
               AA_change=f"{aa_r}{tag}{aa_a}",
               Effect="synonymous" if aa_r == aa_a else
                      ("nonsense" if aa_a == "*" else "missense"))
    if out["Gene"] == "S":
        out["AA_change"] = f"S:{aa_r}{tag}{aa_a}"
    elif out["Protein"]:
        short = out["Protein"].split(" ")[0]
        nres = out["Residue"].split("/")[-1].strip().split(":")[-1]
        out["AA_change"] = f"{short}:{aa_r}{nres}{aa_a}"
    return out


# ---------------------------------------------------------------- run
slopes = pd.read_csv(os.path.join(RES, "slopes_with_uncertainty.tsv"), sep="\t")
alt_of = dict(zip(slopes.Position, slopes.Alt))

MANUSCRIPT_TOP10 = [22942, 22577, 23019, 22896, 22895, 23018, 22786, 22775, 4184, 22200]
MANUSCRIPT_RECENT4 = [22893, 22107, 12892, 16600]
TOP_RISERS_2025 = [28297, 22132, 23021, 12892, 22107, 16600, 22792, 8818, 7869,
                   5239, 8350, 4214, 29409, 21627, 23277, 21653, 22893, 22599,
                   20178, 23039]

rows = []
seen = set()
for group, plist in (("Manuscript Table 1 (top-10 overall slope)", MANUSCRIPT_TOP10),
                     ("Manuscript Fig. 5 (top-4 recent slope)", MANUSCRIPT_RECENT4),
                     ("Candidate recent risers; Table 1 membership is decided in scripts/12", TOP_RISERS_2025)):
    for p in plist:
        key = (p, group)
        if key in seen:
            continue
        seen.add(key)
        a = annotate(p, alt_of.get(p, "N"))
        a["Group"] = group
        rows.append(a)

ann = pd.DataFrame(rows)[["Group", "Position", "Ref", "Alt", "Gene", "Protein",
                          "Residue", "Codon_ref", "Codon_alt", "AA_change",
                          "Effect", "Domain"]]
ann.to_csv(os.path.join(RES, "position_annotation.tsv"), sep="\t", index=False)
print(ann.to_string(index=False))

# the position list 05_lineage_backgrounds.sh pulls per-genome alleles for:
# every annotated position plus two controls used to validate the annotation
focal = sorted(set(ann.Position) | {23063, 23403})
with open(os.path.join(RES, "focal_positions_list.txt"), "w") as fh:
    for p in focal:
        fh.write(f"NC_045512.2\t{p}\n")
print(f"[annot] wrote focal_positions_list.txt ({len(focal)} positions)")

# sanity checks against textbook mutations
for p, alt, expect in [(23403, "G", "S:D614G"), (23063, "T", "S:N501Y"),
                       (14408, "T", "nsp12:P323L")]:
    got = annotate(p, alt)["AA_change"]
    print(f"[check] {p}{alt}: {got}  (expected {expect})  "
          f"{'OK' if got == expect else '<<< MISMATCH'}")
