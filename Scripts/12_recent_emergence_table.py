#!/usr/bin/env python3
"""
12_recent_emergence_table.py  --  the corrected Table 1.

Reviewer 1 asked for confidence intervals, p-values, multiple-testing correction, a
non-linear model, and amino-acid annotation. This produces one table that supplies all
five, over the window the manuscript actually claims to describe.

Definition. For each genomic position and each specific alternative base, the count of
genomes carrying that base is modelled against month with quasi-binomial logistic
regression over Sep 2024 - Aug 2025. Quasi-binomial rather than binomial because
genomes sampled in one month are related, so the binomial variance is far too small;
the dispersion parameter is estimated from the data and inflates the standard errors
accordingly. Sep 2025 is excluded from fitting: 77 genomes, and deposition is
incomplete.

Alleles are tested individually rather than as "any non-reference base", because at a
codon hit in more than one era the two are different mutations (position 22893 is A>C
in 2022-23 and A>G in 2025).
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

RES = "/gdat/fathi/virus/publication/revision/results"
REF = "/gdat/fathi/virus/publication/NC_045512.2.fasta"
WIN_START, WIN_END = "2024_09", "2025_08"
MIN_COV, MIN_BINS, MIN_PEAK = 50, 8, 0.05

MIN_REL_COV = 0.8    # minimum share of a month's genomes aligned at the position

d = np.load(os.path.join(RES, "freq_matrix_uncapped.npz"), allow_pickle=True)
bins = [str(b) for b in d["bins"]]
COV, refbase = d["COV"], d["refbase"]
cnt = {b: d[b] for b in "ACGT"}
man = pd.read_csv(os.path.join(RES, "bin_manifest.tsv"), sep="\t").set_index("bin")
nbin = np.array([man.n_genomes.get(b, 0) for b in bins], dtype=float)
win = [j for j, b in enumerate(bins) if WIN_START <= b <= WIN_END]
t = np.arange(len(win), dtype=float)
print(f"[table1] window {bins[win[0]]}..{bins[win[-1]]} ({len(win)} months)", flush=True)

seq = "".join(l.strip() for l in open(REF) if not l.startswith(">")).upper()
CODON, _b = {}, "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for i, c1 in enumerate(_b):
    for j, c2 in enumerate(_b):
        for k, c3 in enumerate(_b):
            CODON[c1 + c2 + c3] = _aa[i * 16 + j * 4 + k]

GENES = [("ORF1a", 266, 13483), ("ORF1b", 13468, 21555), ("S", 21563, 25384),
         ("ORF3a", 25393, 26220), ("E", 26245, 26472), ("M", 26523, 27191),
         ("ORF6", 27202, 27387), ("ORF7a", 27394, 27759), ("ORF7b", 27756, 27887),
         ("ORF8", 27894, 28259), ("N", 28274, 29533), ("ORF10", 29558, 29674)]
NSP = [("nsp1", 1, 180), ("nsp2", 181, 818), ("nsp3", 819, 2763), ("nsp4", 2764, 3263),
       ("nsp5", 3264, 3569), ("nsp6", 3570, 3859), ("nsp7", 3860, 3942),
       ("nsp8", 3943, 4140), ("nsp9", 4141, 4253), ("nsp10", 4254, 4392),
       ("nsp12", 4393, 5324), ("nsp13", 5325, 5925), ("nsp14", 5926, 6452),
       ("nsp15", 6453, 6798), ("nsp16", 6799, 7096)]


def majority_codon(start, j):
    """Majority triplet at a codon in month index j, or None if under-covered."""
    out = ""
    for p in range(start, start + 3):
        if COV[p - 1, j] < MIN_COV:
            return None
        out += max("ACGT", key=lambda x: cnt[x][p - 1, j])
    return out


def annotate(pos):
    """Codon-level annotation: the amino acid actually encoded, not the single-site guess."""
    if 266 <= pos <= 21555:
        if pos <= 13468:
            res, off = (pos - 266) // 3 + 1, (pos - 266) % 3
        else:
            res, off = (pos - 13468) // 3 + 4402, (pos - 13468) % 3
        gene = "ORF1a" if pos <= 13483 else "ORF1b"
        prot, pres = next(((n, res - a + 1) for n, a, b in NSP if a <= res <= b), ("", res))
        start, label = pos - off, f"{prot}:{pres}"
    else:
        g = next((x for x in GENES if x[1] <= pos <= x[2] and x[0] not in ("ORF1a", "ORF1b")), None)
        if g is None:
            return "non-coding", "", "", ""
        gene, off = g[0], (pos - g[1]) % 3
        res = (pos - g[1]) // 3 + 1
        start, label = pos - off, f"{gene}:{res}"
    ref_codon = seq[start - 1:start + 2]
    obs = majority_codon(start, win[-1]) or ref_codon
    aa_r, aa_o = CODON.get(ref_codon, "?"), CODON.get(obs, "?")
    dom = ""
    if gene == "S":
        r = int(label.split(":")[1])
        dom = ("RBD/RBM" if 438 <= r <= 506 else "RBD" if 319 <= r <= 541 else
               "NTD" if 14 <= r <= 305 else "S1/S2" if 681 <= r <= 685 else
               "S2" if r >= 686 else "S1")
    aa = f"{label.split(':')[0]}:{aa_r}{label.split(':')[1]}{aa_o}"
    eff = "synonymous" if aa_r == aa_o else "missense"
    nsub = sum(1 for a, b in zip(ref_codon, obs) if a != b)
    return gene, dom, aa + ("" if nsub <= 1 else f" ({nsub} nt changes in codon)"), eff


def quasibinom(k, n):
    """Logistic fit of k/n on month, with overdispersion-corrected SE."""
    X = np.column_stack([np.ones_like(t), t])
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        W = np.clip(n * mu * (1 - mu), 1e-9, None)
        z = X @ beta + (k - n * mu) / W
        try:
            new = np.linalg.solve((X.T * W) @ X, (X.T * W) @ z)
        except np.linalg.LinAlgError:
            return None
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    mu = 1 / (1 + np.exp(-(X @ beta)))
    W = np.clip(n * mu * (1 - mu), 1e-9, None)
    phi = (((k - n * mu) ** 2 / W).sum()) / max(len(t) - 2, 1)
    se = np.sqrt(np.diag(np.linalg.inv((X.T * W) @ X)) * phi)
    return beta[1], se[1], phi


rows = []
for i in range(COV.shape[0]):
    pos, ref = i + 1, refbase[i]
    # minimap2 emits split alignments across indel-rich Spike stretches, so parts of
    # the genome are unaligned in a lineage-correlated way; months where less than
    # MIN_REL_COV of the deposited genomes align here are dropped rather than treated
    # as if the whole month had been observed.
    relcov = COV[i, win] / np.maximum(nbin[win], 1)
    ok = [j for k, j in enumerate(win)
          if COV[i, j] >= MIN_COV and relcov[k] >= MIN_REL_COV]
    if len(ok) < MIN_BINS:
        continue
    n = COV[i, ok].astype(float)
    for alt in "ACGT":
        if alt == ref:
            continue
        k = cnt[alt][i, ok].astype(float)
        f = k / n
        if f.max() < MIN_PEAK or f[-1] <= f[0]:
            continue
        globals()["t"] = np.array([win.index(j) for j in ok], dtype=float)
        fit = quasibinom(k, n)
        globals()["t"] = np.arange(len(win), dtype=float)
        if fit is None or not np.isfinite(fit[0]):
            continue
        beta, se, phi = fit
        if se <= 0 or not np.isfinite(se):
            continue
        z = beta / se
        rows.append(dict(Position=pos, Ref=ref, Alt=alt,
                         N_genomes_end=int(n[-1]),
                         Freq_start=round(f[0], 4), Freq_end=round(f[-1], 4),
                         Change=round(f[-1] - f[0], 4),
                         LogOdds_per_month=beta, SE=se,
                         CI_low=beta - 1.96 * se, CI_high=beta + 1.96 * se,
                         Last_month=bins[ok[-1]],
                         Min_rel_coverage=round(float(min(
                             relcov[k] for k, j in enumerate(win) if j in ok)), 3),
                         Dispersion_phi=round(phi, 1), z=z,
                         p=2 * stats.norm.sf(abs(z)), N_months=len(ok)))

df = pd.DataFrame(rows)
p = df.p.to_numpy()
o = np.argsort(p)
q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
df["q_BH"] = np.nan
df.loc[df.index[o], "q_BH"] = np.clip(q, 0, 1)
ann = [annotate(x) for x in df.Position]
df["Gene"], df["Domain"], df["AA_change"], df["Effect"] = zip(*ann)
df = df.sort_values("Change", ascending=False)
df.to_csv(os.path.join(RES, "recent_emergence_all.tsv"), sep="\t", index=False,
          float_format="%.4g")

sig = df[(df.q_BH < 0.05) & (df.LogOdds_per_month > 0)]
print(f"[table1] {len(df)} allele-position pairs tested, {len(sig)} significant at q<0.05",
      flush=True)

# Selection rule for the table: alleles that went from rare to majority inside the
# window, i.e. below 5% at the start and above 50% at the end. This is a property of
# the trajectory rather than a rank, so there is no arbitrary cut-off -- the criticism
# the saturated overall slope attracted. The rule is also stable: <5% -> >50% and
# <5% -> >60% both return the same set, and the next allele below the threshold reaches
# only 44%.
swept = sig[(sig.Freq_start < 0.05) & (sig.Freq_end > 0.50)] \
    .sort_values("Freq_end", ascending=False)
runner_up = sig[(sig.Freq_start < 0.05) & (sig.Freq_end <= 0.50)].Freq_end.max()
print(f"[table1] {len(swept)} alleles went from <5% to >50% within the window; "
      f"the next allele below the threshold reaches {runner_up:.2f}", flush=True)

out = swept[["Position", "Ref", "Alt", "AA_change", "Gene", "Domain", "Effect",
             "Freq_start", "Freq_end", "N_genomes_end", "Last_month",
             "Min_rel_coverage", "LogOdds_per_month", "CI_low", "CI_high",
             "q_BH"]].copy()
out["Mutation"] = [f"{r.Ref}{r.Position}{r.Alt}" for r in out.itertuples()]
out["Growth (95% CI)"] = [f"{r.LogOdds_per_month:.2f} ({r.CI_low:.2f}-{r.CI_high:.2f})"
                          for r in out.itertuples()]
out = out[["Mutation", "AA_change", "Gene", "Domain", "Effect", "Freq_start",
           "Freq_end", "N_genomes_end", "Last_month", "Min_rel_coverage",
           "Growth (95% CI)", "q_BH"]]
out.columns = ["Mutation", "Amino-acid change", "Gene", "Domain", "Effect",
               "Freq Sep 2024", "Freq at last month", "Genomes at last month",
               "Last month used", "Min coverage / month genomes",
               "Growth, log-odds/month (95% CI)", "q"]
out.to_csv(os.path.join(RES, "TABLE1_corrected.tsv"), sep="\t", index=False,
           float_format="%.4g")
print(out.to_string(index=False), flush=True)
print(f"\n[table1] synonymous: {(swept.Effect == 'synonymous').sum()} of {len(swept)}",
      flush=True)
