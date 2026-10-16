#!/bin/bash
# 05_lineage_backgrounds.sh -- Reviewer 1, major point 3.
#
# For a set of focal positions, pull the allele carried by EVERY individual genome
# (mpileup --output-QNAME gives the genome accession alongside each base), so that
# each mutation can be cross-tabulated against the Pango lineage of the genomes
# carrying it. That is what distinguishes "this mutation arose independently in
# several backgrounds" (convergence) from "this mutation rode one successful
# lineage" (linkage).
set -euo pipefail

BINS=/gdat/fathi/virus/temporal_bins
REF=/gdat/fathi/virus/publication/NC_045512.2.fasta
OUT=/gdat/fathi/virus/publication/revision/results/per_genome_alleles
POSFILE=/gdat/fathi/virus/publication/revision/results/focal_positions_list.txt
mkdir -p "$OUT"

run_bin () {
    local d="$1"
    local b
    b=$(basename "$d" | sed 's/^BIN_//')
    samtools mpileup -f "$REF" -l "$POSFILE" -B -Q 0 -q 0 -d 3000000 \
        --output-QNAME "$d/alignments.sorted.bam" 2>/dev/null \
    | awk -v bin="$b" -F'\t' '{print bin"\t"$2"\t"$3"\t"$5"\t"$7}' \
    | gzip -1 > "$OUT/$b.tsv.gz"
}
export -f run_bin
export BINS REF OUT POSFILE

ls -d "$BINS"/BIN_* | xargs -P 12 -I{} bash -c 'run_bin "$@"' _ {}
echo "done: $(ls "$OUT" | wc -l) bins written"
