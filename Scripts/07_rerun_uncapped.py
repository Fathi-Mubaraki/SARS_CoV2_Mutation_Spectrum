#!/usr/bin/env python3
"""
07_rerun_uncapped.py  --  Reviewer 1, major point 5.

The published pileups were generated at the default depth setting, so htslib's
per-file cap of 8000 observations applied. In 30 of the 70 monthly bins there were more
genomes than that (up to 82,939 in Jan 2022), so those months were silently
truncated to ~8000 genomes per position. Total per-position coverage across the
study is therefore ~338,000 genome-observations, not 1,134,166.

This re-runs the pileup with the published quality flags and the cap removed
(-d 3000000). Two things change from the published run: the depth cap, and the indel
parser below, which the original R implementation got wrong. -Q 20 and -q 30 are kept
for comparability although neither excludes anything -- assembled genomes carry no base
qualities and every alignment has MAPQ 60.

Writes results/freq_matrix_uncapped.npz with the same structure as the original.
"""
import os, re, sys, subprocess, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

BINS = "/gdat/fathi/virus/temporal_bins"
REF = "/gdat/fathi/virus/publication/NC_045512.2.fasta"
OUT = "/gdat/fathi/virus/publication/revision/results/uncapped"
GENOME_LEN = 29903
os.makedirs(OUT, exist_ok=True)

INDEL = re.compile(rb"[-+](\d+)")
CARET = re.compile(rb"\^.")


def clean(bases: bytes) -> bytes:
    """Strip indel run-lengths and read-start/end markers.

    The published R version used gsub("[-+][0-9]+[ACGTNacgtn]+", ...), which is greedy
    rather than length-exact, so it consumes following bases after a short indel, and
    its character class misses IUPAC ambiguity codes inside one. This reads the
    run-length prefix and skips exactly that many bases.
    """
    bases = CARET.sub(b"", bases).replace(b"$", b"")
    if b"+" in bases or b"-" in bases:
        out = bytearray()
        i, n = 0, len(bases)
        while i < n:
            c = bases[i]
            if c in (43, 45):                      # '+' or '-'
                j = i + 1
                num = 0
                while j < n and 48 <= bases[j] <= 57:
                    num = num * 10 + (bases[j] - 48)
                    j += 1
                i = j + num                        # skip the inserted/deleted bases
            else:
                out.append(c)
                i += 1
        bases = bytes(out)
    return bases


def do_bin(d):
    b = os.path.basename(d).replace("BIN_", "")
    dest = os.path.join(OUT, f"{b}.npz")
    if os.path.exists(dest):
        return b, "cached"
    cmd = ["samtools", "mpileup", "-f", REF, "-B", "-Q", "20", "-q", "30",
           "-d", "3000000", os.path.join(d, "alignments.sorted.bam")]
    A = np.zeros(GENOME_LEN, np.int64); C = np.zeros(GENOME_LEN, np.int64)
    G = np.zeros(GENOME_LEN, np.int64); T = np.zeros(GENOME_LEN, np.int64)
    DL = np.zeros(GENOME_LEN, np.int64)
    RB = np.full(GENOME_LEN, b"N", dtype="S1")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, bufsize=1 << 22)
    for line in p.stdout:
        f = line.rstrip(b"\n").split(b"\t")
        if len(f) < 5:
            continue
        pos = int(f[1])
        if not (1 <= pos <= GENOME_LEN):
            continue
        ref = f[2].upper()
        s = clean(f[4]).upper()
        i = pos - 1
        RB[i] = ref
        nmatch = s.count(b".") + s.count(b",")
        a, c, g, t = s.count(b"A"), s.count(b"C"), s.count(b"G"), s.count(b"T")
        if ref == b"A":
            a += nmatch
        elif ref == b"C":
            c += nmatch
        elif ref == b"G":
            g += nmatch
        elif ref == b"T":
            t += nmatch
        A[i], C[i], G[i], T[i], DL[i] = a, c, g, t, s.count(b"*")
    p.wait()
    np.savez_compressed(dest, A=A, C=C, G=G, T=T, DEL=DL, refbase=RB)
    return b, f"maxdepth={int((A+C+G+T+DL).max())}"


if __name__ == "__main__":
    dirs = sorted(glob.glob(os.path.join(BINS, "BIN_*")))
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    print(f"[uncap] {len(dirs)} bins, {nproc} workers", flush=True)
    with ProcessPoolExecutor(nproc) as ex:
        futs = {ex.submit(do_bin, d): d for d in dirs}
        for k, fu in enumerate(as_completed(futs), 1):
            b, msg = fu.result()
            print(f"[uncap] {k}/{len(dirs)} {b} {msg}", flush=True)

    # assemble
    bins = [os.path.basename(d).replace("BIN_", "") for d in dirs]
    nb = len(bins)
    A = np.zeros((GENOME_LEN, nb), np.int64); C = np.zeros_like(A)
    G = np.zeros_like(A); T = np.zeros_like(A); DL = np.zeros_like(A)
    refbase = np.full(GENOME_LEN, "N", dtype="<U1")
    for j, b in enumerate(bins):
        z = np.load(os.path.join(OUT, f"{b}.npz"))
        A[:, j], C[:, j], G[:, j], T[:, j], DL[:, j] = (z["A"], z["C"], z["G"],
                                                        z["T"], z["DEL"])
        rb = np.char.decode(z["refbase"].astype("S1"))
        m = rb != "N"
        refbase[m] = rb[m]
    COV = A + C + G + T + DL
    ref_count = np.zeros_like(COV)
    for base, arr in (("A", A), ("C", C), ("G", G), ("T", T)):
        mm = (refbase == base)
        ref_count[mm, :] = arr[mm, :]
    nonref = COV - ref_count
    freq = np.where(COV > 0, nonref / np.maximum(COV, 1), np.nan)
    np.savez_compressed(
        "/gdat/fathi/virus/publication/revision/results/freq_matrix_uncapped.npz",
        bins=np.array(bins), refbase=refbase, A=A, C=C, G=G, T=T, DEL=DL,
        COV=COV, nonref=nonref, freq=freq)
    print(f"[uncap] done; total per-position coverage = {COV.max(axis=0).sum()}",
          flush=True)
