#!/usr/bin/env python3
"""
10_capped_vs_uncapped.py  --  Reviewer 1, major point 5.

Does removing the accidental 8000-genome-per-position depth cap change any published
conclusion? Compares the published (capped, 338,848 observations) analysis against
the re-run (uncapped, 1,134,166 observations) on:

  - per-position overall emergence slope
  - the ranking of the top emerging mutations
  - the monthly mutation spectrum
  - the frequency trajectory of the mutations the manuscript highlights
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

RES = "/gdat/fathi/virus/publication/revision/results"
MIN_COV, MIN_PTS = 50, 3

cap = np.load(os.path.join(RES, "freq_matrix.npz"), allow_pickle=True)
unc = np.load(os.path.join(RES, "freq_matrix_uncapped.npz"), allow_pickle=True)
bins = [str(b) for b in cap["bins"]]
assert bins == [str(b) for b in unc["bins"]]
tidx = np.arange(1, len(bins) + 1, dtype=float)
npos = cap["COV"].shape[0]

print(f"[cmp] capped   total per-position coverage: {cap['COV'].max(axis=0).sum():,}")
print(f"[cmp] uncapped total per-position coverage: {unc['COV'].max(axis=0).sum():,}")


def slopes(z):
    COV, FREQ = z["COV"], z["freq"]
    use = COV >= MIN_COV
    y = np.where(use, FREQ, np.nan)
    m = np.isfinite(y)
    W = m.astype(float)
    sw = W.sum(1)
    yy = np.nan_to_num(y)
    xbar = (W * tidx).sum(1) / np.maximum(sw, 1e-12)
    ybar = (W * yy).sum(1) / np.maximum(sw, 1e-12)
    dx = tidx[None, :] - xbar[:, None]
    dy = yy - ybar[:, None]
    Sxx = (W * dx * dx).sum(1)
    Sxy = (W * dx * dy).sum(1)
    sl = np.divide(Sxy, Sxx, out=np.full(npos, np.nan), where=Sxx > 0)
    sl[m.sum(1) < MIN_PTS] = np.nan
    return sl, m.sum(1)


s_cap, n_cap = slopes(cap)
s_unc, n_unc = slopes(unc)
ok = np.isfinite(s_cap) & np.isfinite(s_unc)
r = stats.pearsonr(s_cap[ok], s_unc[ok])
print(f"[cmp] slope correlation over {ok.sum()} positions: r = {r.statistic:.5f}")
print(f"[cmp] median |difference| = {np.median(np.abs(s_cap[ok]-s_unc[ok])):.3e}")
print(f"[cmp] max    |difference| = {np.abs(s_cap[ok]-s_unc[ok]).max():.3e}")

ann = pd.read_csv(os.path.join(RES, "position_annotation.tsv"), sep="\t") \
        .drop_duplicates("Position").set_index("Position")

order_cap = np.argsort(-np.nan_to_num(s_cap, nan=-9))[:20] + 1
order_unc = np.argsort(-np.nan_to_num(s_unc, nan=-9))[:20] + 1
rank_tbl = pd.DataFrame({
    "rank": np.arange(1, 21),
    "capped_position": order_cap,
    "capped_slope": np.round(s_cap[order_cap - 1], 5),
    "uncapped_position": order_unc,
    "uncapped_slope": np.round(s_unc[order_unc - 1], 5),
})
rank_tbl["capped_AA"] = [ann.AA_change.get(p, "") for p in order_cap]
rank_tbl["uncapped_AA"] = [ann.AA_change.get(p, "") for p in order_unc]
rank_tbl.to_csv(os.path.join(RES, "capped_vs_uncapped_ranking.tsv"), sep="\t",
                index=False)
print("\n[cmp] top-20 ranking, capped vs uncapped:")
print(rank_tbl.to_string(index=False))

overlap = len(set(order_cap[:10]) & set(order_unc[:10]))
print(f"\n[cmp] top-10 overlap: {overlap}/10 positions")

# per-position table for the focal set
foc = sorted(ann.index)
det = pd.DataFrame({
    "Position": foc,
    "AA_change": [ann.AA_change.get(p, "") for p in foc],
    "slope_capped": np.round(s_cap[np.array(foc) - 1], 5),
    "slope_uncapped": np.round(s_unc[np.array(foc) - 1], 5),
    "freq_capped_overall": np.round(
        cap["nonref"][np.array(foc) - 1].sum(1) /
        np.maximum(cap["COV"][np.array(foc) - 1].sum(1), 1), 4),
    "freq_uncapped_overall": np.round(
        unc["nonref"][np.array(foc) - 1].sum(1) /
        np.maximum(unc["COV"][np.array(foc) - 1].sum(1), 1), 4),
    "n_genomes_capped": cap["COV"][np.array(foc) - 1].sum(1),
    "n_genomes_uncapped": unc["COV"][np.array(foc) - 1].sum(1),
})
det["slope_pct_change"] = np.round(
    100 * (det.slope_uncapped - det.slope_capped) / det.slope_capped.abs(), 1)
det.to_csv(os.path.join(RES, "capped_vs_uncapped_focal.tsv"), sep="\t", index=False)
print("\n[cmp] focal mutations, capped vs uncapped:")
print(det.to_string(index=False))

# spectrum comparison, C>T and G>T, per year
def spectrum_year(z):
    COV, refbase = z["COV"], z["refbase"]
    out = {}
    for yr in ["2020", "2021", "2022", "2023", "2024", "2025"]:
        js = [j for j, b in enumerate(bins) if b.startswith(yr)]
        tot = {}
        for rb in "ACGT":
            mref = (refbase == rb)
            for a in "ACGT":
                if a == rb:
                    continue
                tot[f"{rb}>{a}"] = float(z[a][mref][:, js].sum())
        s = sum(tot.values())
        out[yr] = {k: v / s for k, v in tot.items()} if s else {}
    return out


sc, su = spectrum_year(cap), spectrum_year(unc)
spec = pd.DataFrame([
    {"Year": y, "C>T_capped": round(sc[y]["C>T"], 4),
     "C>T_uncapped": round(su[y]["C>T"], 4),
     "G>T_capped": round(sc[y]["G>T"], 4),
     "G>T_uncapped": round(su[y]["G>T"], 4)}
    for y in sc])
spec.to_csv(os.path.join(RES, "capped_vs_uncapped_spectrum.tsv"), sep="\t",
            index=False)
print("\n[cmp] mutation spectrum, capped vs uncapped:")
print(spec.to_string(index=False))
