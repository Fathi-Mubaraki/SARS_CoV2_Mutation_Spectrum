#!/usr/bin/env python3
"""
11_codon_convergence_and_figures.py

(a) Codon-level convergence: Spike codons whose MAJORITY AMINO ACID changes more than
    once. This is the convergence evidence that survives Reviewer 1's criticism,
    because a codon that changes residue, reverts, and changes again cannot be one
    linked haplotype rising once. Majority state is used rather than counting
    nucleotide changes, because deciding which nucleotide changes belong to the same
    amino-acid event is ambiguous whenever two lineages overlap during a changeover.
(b) The figures supporting the response.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/gdat/fathi/virus/publication/revision/results"
FIG = "/gdat/fathi/virus/publication/revision/figures"
os.makedirs(FIG, exist_ok=True)

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
         "#F0E442", "#666666"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})

d = np.load(os.path.join(RES, "freq_matrix_uncapped.npz"), allow_pickle=True)
bins = [str(b) for b in d["bins"]]
COV, refbase = d["COV"], d["refbase"]
cnt = {b: d[b] for b in "ACGT"}
DEL = d["DEL"]
# recurrent NTD deletions shift under alignment and can masquerade as repeated
# substitution, so each codon carries the largest deletion fraction it ever shows
delfrac = np.nanmax(np.where(COV >= 50, DEL / np.maximum(COV, 1), np.nan), axis=1)
_man = pd.read_csv(os.path.join(RES, "bin_manifest.tsv"), sep="\t").set_index("bin")
_nbin = np.array([_man.n_genomes.get(b, 0) for b in bins], dtype=float)
LAST = bins.index("2025_08")          # every table stops here; Sep 2025 holds 77 genomes
tnum = np.arange(LAST + 1)
slopes = pd.read_csv(os.path.join(RES, "slopes_with_uncertainty.tsv"), sep="\t")


def afreq(pos, alt):
    """Allele frequency, blanked in months failing the same 80% relative-coverage
    rule the codon table applies, so figures and table agree."""
    i = pos - 1
    ok = (COV[i] >= 50) & (COV[i] / np.maximum(_nbin, 1) >= 0.8)
    return np.where(ok, cnt[alt][i] / np.maximum(COV[i], 1), np.nan)


# --------------------------------------------------------------- (a) codon table
def spike_domain(res):
    if 438 <= res <= 506:
        return "RBD / receptor-binding motif (RBM)"
    if 319 <= res <= 541:
        return "RBD (core)"
    if 681 <= res <= 685:
        return "S1/S2 furin cleavage site"
    if 14 <= res <= 305:
        return "NTD"
    return "S2" if res >= 686 else "S1 (other)"


S_START = 21563
CODON_TBL, _bb = {}, "TCAG"
_aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for _i, _c1 in enumerate(_bb):
    for _j, _c2 in enumerate(_bb):
        for _k, _c3 in enumerate(_bb):
            CODON_TBL[_c1 + _c2 + _c3] = _aas[_i * 16 + _j * 4 + _k]

# A codon counts as repeatedly hit when its MAJORITY amino acid changes more than
# once over the series. Majority state avoids having to decide which nucleotide
# changes belong to the same amino-acid event, which is ambiguous whenever two
# lineages overlap during a changeover. A state must hold for two consecutive
# months, on months where at least 80% of that month's genomes align across the
# whole codon, to count.
MIN_REL_COV = 0.8

rows = []
for codon in range(1, 1274):
    start = S_START + (codon - 1) * 3
    ref = "".join(refbase[start - 1 + k] for k in range(3))
    if CODON_TBL.get(ref) is None:
        continue
    seq_states = []
    for j, b in enumerate(bins):
        cov = COV[start - 1:start + 2, j]
        if cov.min() < 50 or cov.min() / max(_nbin[j], 1) < MIN_REL_COV:
            continue
        mc = "".join(max("ACGT", key=lambda x: cnt[x][start - 1 + k, j])
                     for k in range(3))
        seq_states.append((b, CODON_TBL.get(mc, "?"), mc))
    runs = []
    for b, aa, mc in seq_states:
        if runs and runs[-1][1] == aa:
            runs[-1][2] += 1
        else:
            runs.append([b, aa, 1, mc])
    runs = [r for r in runs if r[2] >= 2]
    collapsed = []
    for r in runs:
        if not collapsed or collapsed[-1][1] != r[1]:
            collapsed.append(r)
    if len(collapsed) < 3:                      # reference state plus >= 2 changes
        continue
    eras = sorted(set(r[0][:4] for r in collapsed[1:]))
    # the strict test: a non-reference amino acid that is lost and then acquired
    # again cannot be one linked haplotype rising once
    states = [r[1] for r in collapsed]
    reacquired = sorted({a for k, a in enumerate(states)
                         if a != states[0] and a in states[:k]})
    rows.append({
        "Spike_codon": codon,
        "N_amino_acid_changes": len(collapsed) - 1,
        "State_history": " -> ".join(f"{r[1]} ({r[0]})" for r in collapsed),
        "Reacquired_amino_acids": ",".join(reacquired),
        "N_reacquisitions": len(reacquired),
        "First_years": ", ".join(eras),
        "Spans_multiple_years": "YES" if len(eras) > 1 else "no",
        "Domain": spike_domain(codon),
        "Max_deletion_fraction": round(float(np.nanmax(
            delfrac[start - 1:start + 2])), 3),
    })

cod = pd.DataFrame(rows).sort_values("N_amino_acid_changes", ascending=False)
cod.to_csv(os.path.join(RES, "codon_level_convergence.tsv"), sep="\t", index=False)
print("=== Spike codons whose majority amino acid changed more than once ===")
print(cod.head(25).to_string(index=False), flush=True)
from scipy.stats import hypergeom
clean = cod[cod.Max_deletion_fraction < 0.05]
RBM_LO, RBM_HI, N_CODONS = 438, 506, 1273
n_rbm = RBM_HI - RBM_LO + 1
k_rbm = int(((clean.Spike_codon >= RBM_LO) & (clean.Spike_codon <= RBM_HI)).sum())
p_rbm = hypergeom.sf(k_rbm - 1, N_CODONS, n_rbm, len(clean))
rec = clean[clean.N_reacquisitions > 0]
print(f"[codon] {len(rec)} deletion-free codons lose a non-reference amino acid and "
      f"then acquire it again: "
      + "; ".join(f"S:{r.Spike_codon} ({r.Reacquired_amino_acids})"
                  for r in rec.itertuples()), flush=True)
print(f"[codon] {k_rbm} of {len(clean)} deletion-free codons lie in the receptor-binding "
      f"motif ({n_rbm}/{N_CODONS} = {100*n_rbm/N_CODONS:.1f}% of Spike); "
      f"hypergeometric p = {p_rbm:.2g}", flush=True)
rbd = clean[(clean.Spike_codon >= 319) & (clean.Spike_codon <= 541)]
rbd.to_csv(os.path.join(RES, "rbd_codon_convergence.tsv"), sep="\t", index=False)
print(f"\n[codon] {len(cod)} Spike codons changed majority amino acid more than once "
      f"({(cod.Spans_multiple_years == 'YES').sum()} in separate calendar years); "
      f"deletion-free: {len(clean)}; in the RBD: {len(rbd)}", flush=True)

# --------------------------------------------------------------- Fig 1: ceiling
fig, ax = plt.subplots(figsize=(6.4, 4.0))
s = slopes.dropna(subset=["Slope_Overall", "Slope_as_pct_of_ceiling"])
s = s[s.Slope_Overall > 0]
ax.hist(s.Slope_as_pct_of_ceiling, bins=np.arange(0, 102, 2), color=OKABE[7],
        alpha=0.55, label=f"all {len(s):,} positions with a positive slope")
ax.set_yscale("log")
top = s.nlargest(10, "Slope_Overall")
for k, (_, r) in enumerate(top.iterrows()):
    ax.axvline(r.Slope_as_pct_of_ceiling, color=OKABE[1], lw=1.1, alpha=0.85,
               label="published Table 1 top-10" if k == 0 else None)
ax.set_xlabel("Emergence slope as % of the largest slope arithmetically possible")
ax.set_ylabel("Number of genome positions (log)")
ax.set_title("Emergence slopes relative to their arithmetic maximum")
ax.annotate("all ten between\n94.6% and 99.5%", xy=(96, ax.get_ylim()[1]*0.02),
            xytext=(62, ax.get_ylim()[1]*0.15), fontsize=8, color=OKABE[1],
            arrowprops=dict(arrowstyle="->", color=OKABE[1], lw=1))
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "R1_slope_ceiling.png"))
plt.close(fig)

# --------------------------------------------------------------- Fig 2: who is rising
# Both panels show the manuscript's own metric -- the site-level non-reference
# frequency the emergence slope was fitted to -- so the two are directly comparable.
PUBLISHED = [(22942, "S:460"), (22577, "S:339"), (23018, "S:486"),
             (22775, "S:405"), (22200, "S:213")]
ACTUAL = [(22132, "S:R190S"), (23021, "S:N487D"), (22107, "S:K182R"),
          (29409, "N:T379I"), (4214, "nsp3:L499L")]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
show = [i for i, b in enumerate(bins) if b.endswith(("_01", "_07")) and i <= LAST]
for ax, grp, title in (
        (axes[0], PUBLISHED, "Published Table 1 — swept 2022–2023"),
        (axes[1], ACTUAL, "Currently rising — 2025")):
    for k, (p, lab) in enumerate(grp):
        f = np.where(COV[p - 1] >= 50,
                     d["nonref"][p - 1] / np.maximum(COV[p - 1], 1), np.nan)
        ax.plot(tnum, f[:LAST + 1], lw=1.7, color=OKABE[k], label=f"{lab} ({p})")
    ax.set_xticks(show)
    ax.set_xticklabels([bins[i].replace("_", "-") for i in show], rotation=90,
                       fontsize=6.5)
    ax.set_title(title); ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7.5, loc="upper left")
axes[0].set_ylabel("Non-reference frequency")
fig.suptitle("Overall-slope ranking versus current frequency change",
             fontweight="bold", y=1.0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "R2_published_vs_actual.png"))
plt.close(fig)

# --------------------------------------------------------------- Fig 3: sampling
comp = pd.read_csv(os.path.join(RES, "monthly_region_composition.tsv"), sep="\t",
                   index_col=0)
regs = [c for c in comp.columns if not c.endswith("_pct") and c != "Total"]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.25]})
a1.bar(range(len(comp)), comp["Total"], color=OKABE[0], width=0.85)
a1.set_yscale("log"); a1.set_ylabel("Genomes (log)")
a1.set_title("Monthly genome count and where the genomes came from")
frac = comp[regs].div(comp["Total"], axis=0) * 100
a2.stackplot(range(len(comp)), *[frac[r] for r in regs], labels=regs,
             colors=OKABE[:len(regs)], alpha=0.92)
a2.set_ylim(0, 100); a2.set_ylabel("% of month's genomes")
a2.set_xticks(show)
a2.set_xticklabels([bins[i].replace("_", "-") for i in show], rotation=90, fontsize=6.5)
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=7, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "R3_sampling_composition.png"))
plt.close(fig)

# --------------------------------------------------------------- Fig 4: spectrum
specs = {k: pd.read_csv(os.path.join(RES, f"spectrum_dm_{k}.tsv"), sep="\t")
         for k in ("A_as_published", "B_site_counted_depth_matched",
                   "C_denovo_depth_matched")}
labels = {"A_as_published": "as published (per-genome weighted)",
          "B_site_counted_depth_matched": "each site counted once (depth-matched)",
          "C_denovo_depth_matched": "low-frequency variants only (de-novo proxy)"}
sbins = sorted(set(specs["A_as_published"].TimeBin))
sx = [bins.index(b) for b in sbins]
sshow = [i for i in show if bins[i] in sbins]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharex=True)
for ax, mt in zip(axes, ("C>T", "G>T")):
    for k, (key, df) in enumerate(specs.items()):
        sub = df[df.MutationType == mt].set_index("TimeBin").reindex(sbins)
        ax.plot(sx, sub.RelFreq, lw=1.7, color=OKABE[k], label=labels[key])
    ax.set_title(f"{mt} substitutions")
    ax.set_ylim(0, None)
    ax.set_xticks(sshow)
    ax.set_xticklabels([bins[i].replace("_", "-") for i in sshow], rotation=90,
                       fontsize=6.5)
axes[0].set_ylabel("Share of all substitutions")
axes[0].legend(fontsize=7, loc="lower left")
fig.suptitle("Mutation spectrum under three counting schemes",
             fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "R4_spectrum_three_ways.png"))
plt.close(fig)

# --------------------------------------------------------------- Fig 5: recurrence
# labels give the amino acid the codon reaches when that change is present
# The four codons that lose a non-reference residue and acquire it again AND keep
# >=80% relative coverage throughout. 452 and 446 also re-acquire but sit beside the
# Spike alignment dropout, so their recent months fail the coverage rule.
CODONS = {346: [(22599, "A", "R346K"), (22599, "C", "R346T")],
          213: [(22200, "A", "V213E"), (22200, "G", "V213G")],
          501: [(23063, "T", "N501Y")],
          681: [(23604, "A", "P681H"), (23604, "G", "P681R")]}
fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
for ax, (codon, subs) in zip(axes, CODONS.items()):
    for k, (p, a, aa) in enumerate(subs):
        ax.plot(tnum, afreq(p, a)[:LAST + 1], lw=1.7, color=OKABE[k],
                label=f"{refbase[p-1]}{p}{a} ({aa})")
    ax.set_title(f"Spike codon {codon}")
    ax.set_xticks(show[::2])
    ax.set_xticklabels([bins[i].replace("_", "-") for i in show[::2]], rotation=90,
                       fontsize=6)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(-0.03, 1.32)
axes[0].set_ylabel("Allele frequency")
fig.suptitle("Allele frequencies at repeatedly changed Spike codons",
             fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "R5_codon_recurrence.png"))
plt.close(fig)

print(f"\n[fig] wrote 5 figures to {FIG}")
for f in sorted(os.listdir(FIG)):
    print("   ", f)
