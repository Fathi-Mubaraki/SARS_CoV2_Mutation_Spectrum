#!/usr/bin/env python3
"""
01_build_matrices.py

Builds the core reusable data objects for the MDPI revision analyses:

  results/freq_matrix.npz      position x bin matrices of allele counts, coverage,
                               non-reference count and non-reference frequency
  results/bin_manifest.tsv     per-month genome counts actually analysed
  results/accession_bin.tsv    accession -> bin map (reproducible manifest)

Inputs are the per-bin frequencies.tsv files produced by 3_parse_pileup.R and the
NCBI metadata table downloaded alongside the sequences.
"""
import os, glob
import numpy as np
import pandas as pd

BINS_DIR = "/gdat/fathi/virus/temporal_bins"
OUT = "/gdat/fathi/virus/publication/revision/results"
os.makedirs(OUT, exist_ok=True)

GENOME_LEN = 29903

bin_dirs = sorted(glob.glob(os.path.join(BINS_DIR, "BIN_*")))
bins = [os.path.basename(d).replace("BIN_", "") for d in bin_dirs]
print(f"[prep] {len(bins)} temporal bins: {bins[0]} .. {bins[-1]}", flush=True)

nb = len(bins)
# position index 0..GENOME_LEN-1 corresponds to 1-based position 1..GENOME_LEN
A = np.zeros((GENOME_LEN, nb), dtype=np.int32)
C = np.zeros((GENOME_LEN, nb), dtype=np.int32)
G = np.zeros((GENOME_LEN, nb), dtype=np.int32)
T = np.zeros((GENOME_LEN, nb), dtype=np.int32)
DEL = np.zeros((GENOME_LEN, nb), dtype=np.int32)
COV = np.zeros((GENOME_LEN, nb), dtype=np.int32)

refbase = np.full(GENOME_LEN, "N", dtype="<U1")

for j, (d, b) in enumerate(zip(bin_dirs, bins)):
    f = os.path.join(d, "frequencies.tsv")
    if not os.path.exists(f) or os.path.getsize(f) < 10:
        print(f"[prep]   {b}: EMPTY/missing frequencies.tsv", flush=True)
        continue
    df = pd.read_csv(f, sep="\t")
    idx = df["Position"].to_numpy() - 1
    ok = (idx >= 0) & (idx < GENOME_LEN)
    idx = idx[ok]
    A[idx, j] = df.loc[ok, "A"].to_numpy()
    C[idx, j] = df.loc[ok, "C"].to_numpy()
    G[idx, j] = df.loc[ok, "G"].to_numpy()
    T[idx, j] = df.loc[ok, "T"].to_numpy()
    DEL[idx, j] = df.loc[ok, "Del"].to_numpy()
    COV[idx, j] = df.loc[ok, "Coverage"].to_numpy()
    rb = df.loc[ok, "RefBase"].astype(str).to_numpy()
    refbase[idx] = rb
    print(f"[prep]   {b}: {len(idx)} positions, max cov {COV[:, j].max()}", flush=True)

# reference-allele count per position/bin
ref_count = np.zeros_like(COV)
for base, arr in (("A", A), ("C", C), ("G", G), ("T", T)):
    m = (refbase == base)
    ref_count[m, :] = arr[m, :]

nonref = COV - ref_count
with np.errstate(divide="ignore", invalid="ignore"):
    freq = np.where(COV > 0, nonref / np.maximum(COV, 1), np.nan)
    freq[COV == 0] = np.nan

np.savez_compressed(
    os.path.join(OUT, "freq_matrix.npz"),
    bins=np.array(bins), refbase=refbase,
    A=A, C=C, G=G, T=T, DEL=DEL, COV=COV, nonref=nonref, freq=freq,
)
print("[prep] wrote freq_matrix.npz", flush=True)

# ---------------------------------------------------------------- manifest
rows = []
acc_rows = []
for d, b in zip(bin_dirs, bins):
    ids_f = os.path.join(d, "sequence_ids.txt")
    ids = [x.strip() for x in open(ids_f) if x.strip()]
    rows.append({"bin": b, "n_genomes": len(ids)})
    for a in ids:
        acc_rows.append((a, b))

man = pd.DataFrame(rows)
man.to_csv(os.path.join(OUT, "bin_manifest.tsv"), sep="\t", index=False)
print(f"[prep] total genomes actually analysed: {man.n_genomes.sum()}", flush=True)

acc = pd.DataFrame(acc_rows, columns=["accession", "bin"])
acc.to_csv(os.path.join(OUT, "accession_bin.tsv.gz"), sep="\t", index=False,
           compression="gzip")
print(f"[prep] wrote accession_bin.tsv.gz ({len(acc)} rows)", flush=True)
print("[prep] done", flush=True)
