#!/usr/bin/env python3
"""
02_slope_statistics.py  --  Reviewer 1, major point 1.

Re-estimates every emergence slope in the manuscript WITH uncertainty, and tests
whether the linear model is the right model.

Outputs (revision/results/):
  slopes_with_uncertainty.tsv   per position: OLS slope, SE, 95% CI, p, BH q, the
                                Newey-West HAC standard error and its p-value, n bins
                                used, coverage-weighted slope, and the theoretical
                                maximum slope attainable at that site
  recent_slope_uncertainty.tsv  the 3-month "recent" slope with its CI, blank where
                                the allele is fixed and no slope is estimable
  model_comparison.tsv          linear vs logistic vs segmented fit for the sites
                                the manuscript highlights
  ceiling_demo.tsv              evidence that the top-ranked slopes are pinned at
                                the arithmetic ceiling of the metric
"""
import os, json
import numpy as np
import pandas as pd
from scipy import stats, optimize

RES = "/gdat/fathi/virus/publication/revision/results"
MIN_COVERAGE = 50            # identical to the original 4_analyze_visualize_v4.R
MIN_POINTS_OVERALL = 3       # identical
N_RECENT = 3                 # identical

d = np.load(os.path.join(RES, "freq_matrix.npz"), allow_pickle=True)
bins = [str(b) for b in d["bins"]]
COV, NONREF, FREQ = d["COV"], d["nonref"], d["freq"]
refbase = d["refbase"]
A, C, G, T, DEL = d["A"], d["C"], d["G"], d["T"], d["DEL"]
npos, nb = COV.shape
tidx = np.arange(1, nb + 1, dtype=float)      # 1..70, exactly the original TimeIndex
print(f"[stats] {npos} positions x {nb} bins ({bins[0]}..{bins[-1]})", flush=True)

use = COV >= MIN_COVERAGE                      # the original per-bin filter
n_used = use.sum(axis=1)
print(f"[stats] positions with >= {MIN_POINTS_OVERALL} usable bins: "
      f"{(n_used >= MIN_POINTS_OVERALL).sum()}", flush=True)

# ---------------------------------------------------------------- major alt allele
stack = np.stack([A, C, G, T, DEL], axis=2)            # pos x bin x allele
allele_names = np.array(["A", "C", "G", "T", "Del"])
tot_allele = stack.sum(axis=1)                          # pos x allele, summed over bins
for k, b in enumerate("ACGT"):
    tot_allele[refbase == b, k] = 0                     # zero out the reference allele
alt_k = tot_allele.argmax(axis=1)
alt_base = allele_names[alt_k]
alt_base[tot_allele.max(axis=1) == 0] = "."
# allele-specific counts for the named alt, per bin
ALT = np.take_along_axis(stack, alt_k[:, None, None].repeat(nb, 1), axis=2)[:, :, 0]
# how much of the "non-reference" signal is actually the named substitution
_nr = NONREF.sum(axis=1)
purity = np.divide(ALT.sum(axis=1), _nr, out=np.zeros(npos, float), where=_nr > 0)


def ols_rows(y, x, w=None):
    """Vectorised weighted least squares of y (pos x bin, nan-masked) on x."""
    m = np.isfinite(y)
    W = (m.astype(float) if w is None else np.where(m, w, 0.0))
    sw = W.sum(axis=1)
    yy = np.nan_to_num(y)
    xbar = (W * x).sum(axis=1) / np.maximum(sw, 1e-12)
    ybar = (W * yy).sum(axis=1) / np.maximum(sw, 1e-12)
    dx = x[None, :] - xbar[:, None]
    dy = yy - ybar[:, None]
    Sxx = (W * dx * dx).sum(axis=1)
    Sxy = (W * dx * dy).sum(axis=1)
    slope = np.divide(Sxy, Sxx, out=np.full(npos, np.nan), where=Sxx > 0)
    resid = dy - slope[:, None] * dx
    nobs = m.sum(axis=1)
    dof = nobs - 2
    sse = (W * resid * resid).sum(axis=1)
    sigma2 = np.divide(sse, dof, out=np.full(npos, np.nan), where=dof > 0)
    se = np.sqrt(np.divide(sigma2, Sxx, out=np.full(npos, np.nan), where=Sxx > 0))
    return slope, se, nobs, dof, Sxx


y_all = np.where(use, FREQ, np.nan)
slope, se, nobs, dof, Sxx = ols_rows(y_all, tidx)
with np.errstate(invalid="ignore", divide="ignore"):
    tstat = slope / se
    pval = 2 * stats.t.sf(np.abs(tstat), np.maximum(dof, 1))
pval = np.where(np.isfinite(tstat) & (dof > 0), pval, np.nan)
tcrit = stats.t.ppf(0.975, np.maximum(dof, 1))
ci_lo, ci_hi = slope - tcrit * se, slope + tcrit * se
# an invariant site has zero residual variance: no slope is estimable, and a
# zero-width interval would misreport that as perfect precision
_deg = ~(se > 0)
slope, se, ci_lo, ci_hi = (np.where(_deg, np.nan, v) for v in (slope, se, ci_lo, ci_hi))

# ---------------------------------------------------------------- HAC (Newey-West)
# A swept site leaves strongly autocorrelated residuals around a straight line, so the
# OLS standard error above understates the uncertainty. Bartlett kernel, L=3 by the
# usual 4*(T/100)^(2/9) rule at T=69.
def hac_se(y, x, slope, L=3):
    m = np.isfinite(y)
    xbar = np.where(m, x[None, :], 0).sum(1) / np.maximum(m.sum(1), 1)
    dx = np.where(m, x[None, :] - xbar[:, None], 0.0)
    ybar = np.where(m, np.nan_to_num(y), 0).sum(1) / np.maximum(m.sum(1), 1)
    u = np.where(m, np.nan_to_num(y) - ybar[:, None] - slope[:, None] * dx, 0.0)
    h = dx * u
    S = (h * h).sum(1)
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        S += 2.0 * w * (h[:, l:] * h[:, :-l]).sum(1)
    Sxx_ = (dx * dx).sum(1)
    return np.sqrt(np.divide(np.maximum(S, 0), Sxx_ ** 2,
                             out=np.full(npos, np.nan), where=Sxx_ > 0))


se_hac = hac_se(y_all, tidx, np.nan_to_num(slope))
with np.errstate(invalid="ignore", divide="ignore"):
    p_hac = 2 * stats.norm.sf(np.abs(slope / se_hac))
p_hac = np.where(np.isfinite(se_hac) & (se_hac > 0), p_hac, np.nan)

# ---------------------------------------------------------------- arithmetic ceiling
# For y bounded in [0,1], the largest slope any site could possibly show given the
# bins it has is achieved by y=1 for every bin later than the mean time, else 0.
dxc = np.where(use, tidx[None, :] - np.divide((use * tidx).sum(1),
                                              np.maximum(use.sum(1), 1))[:, None], 0.0)
ceiling = np.divide(np.clip(dxc, 0, None).sum(axis=1), Sxx,
                    out=np.full(npos, np.nan), where=Sxx > 0)

# ---------------------------------------------------------------- coverage-weighted
slope_w, se_w, _, dof_w, _ = ols_rows(y_all, tidx, w=COV.astype(float))

# ---------------------------------------------------------------- BH-FDR
ok = np.isfinite(pval) & (nobs >= MIN_POINTS_OVERALL)
qval = np.full(npos, np.nan)
p_ok = pval[ok]
order = np.argsort(p_ok)
m = p_ok.size
q_sorted = p_ok[order] * m / np.arange(1, m + 1)
q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
q_tmp = np.empty(m)
q_tmp[order] = np.clip(q_sorted, 0, 1)
qval[ok] = q_tmp
print(f"[stats] BH-FDR over {m} tested positions; q<0.05: {(qval < 0.05).sum()}", flush=True)

# ---------------------------------------------------------------- recent slope
recent_mask = np.zeros(nb, bool)
recent_mask[-N_RECENT:] = True
y_rec = np.where(use & recent_mask[None, :], FREQ, np.nan)
slope_r, se_r, nobs_r, dof_r, _ = ols_rows(y_rec, tidx)
with np.errstate(invalid="ignore", divide="ignore"):
    t_r = slope_r / se_r
    p_r = 2 * stats.t.sf(np.abs(t_r), np.maximum(dof_r, 1))
p_r = np.where(np.isfinite(t_r) & (dof_r > 0), p_r, np.nan)
# a site fixed in all recent bins has zero residual variance; its slope is not
# estimable and must not be reported as a zero-width interval
degenerate = ~(se_r > 0)
se_r = np.where(degenerate, np.nan, se_r)
slope_r = np.where(degenerate, np.nan, slope_r)
p_r = np.where(degenerate, np.nan, p_r)
tcrit_r = stats.t.ppf(0.975, np.maximum(dof_r, 1))

pos = np.arange(1, npos + 1)
out = pd.DataFrame({
    "Position": pos, "Ref": refbase, "Alt": alt_base,
    "AltPurity": np.round(purity, 4),
    "N_bins_used": nobs,
    "Slope_Overall": slope, "SE": se, "CI95_low": ci_lo, "CI95_high": ci_hi,
    "t": tstat, "p_value": pval, "q_BH": qval,
    "SE_HAC_NeweyWest": se_hac, "p_value_HAC": p_hac,
    "Slope_ceiling_max_possible": ceiling,
    "Slope_as_pct_of_ceiling": np.round(100 * slope / ceiling, 2),
    "Slope_coverage_weighted": slope_w, "SE_coverage_weighted": se_w,
    "N_bins_recent": nobs_r,
    "Slope_Recent": slope_r, "SE_Recent": se_r,
    "Recent_CI95_low": slope_r - tcrit_r * se_r,
    "Recent_CI95_high": slope_r + tcrit_r * se_r,
    "p_Recent": p_r,
    "Overall_Freq": np.divide(NONREF.sum(1), np.maximum(COV.sum(1), 1)),
    "Total_Coverage": COV.sum(1),
    "Max_bin_freq": np.nanmax(y_all, axis=1),
})
out = out[out.N_bins_used >= MIN_POINTS_OVERALL].copy()
out.sort_values("Slope_Overall", ascending=False, inplace=True)
out.to_csv(os.path.join(RES, "slopes_with_uncertainty.tsv"), sep="\t", index=False,
           float_format="%.6g")
print(f"[stats] wrote slopes_with_uncertainty.tsv ({len(out)} positions)", flush=True)

# ---------------------------------------------------------------- ceiling evidence
top = out.head(50)
ceil_demo = top[["Position", "Ref", "Alt", "Slope_Overall", "N_bins_used",
                 "Slope_ceiling_max_possible", "Slope_as_pct_of_ceiling",
                 "CI95_low", "CI95_high", "q_BH"]]
ceil_demo.to_csv(os.path.join(RES, "ceiling_demo.tsv"), sep="\t", index=False,
                 float_format="%.6g")
print("[stats] top-10 slope vs ceiling:\n",
      ceil_demo.head(10).to_string(index=False), flush=True)

# ---------------------------------------------------------------- model comparison
MANUSCRIPT_TOP = [22942, 22577, 23019, 22896, 22895, 23018, 22786, 22775, 4184, 22200]
MANUSCRIPT_RECENT = [22893, 22107, 12892, 16600]
FOCUS = MANUSCRIPT_TOP + MANUSCRIPT_RECENT


def logistic3(t, r, t0, K):
    return K / (1.0 + np.exp(-r * (t - t0)))


def fit_models(p):
    i = p - 1
    m = use[i]
    if m.sum() < 5:
        return None
    x, y = tidx[m], FREQ[i, m]
    n_i, k_i = COV[i, m].astype(float), NONREF[i, m].astype(float)

    # linear
    lin = stats.linregress(x, y)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_lin = 1 - ((y - (lin.intercept + lin.slope * x)) ** 2).sum() / ss_tot

    # logistic
    r2_log, r_hat, t0_hat, K_hat, t0_date = np.nan, np.nan, np.nan, np.nan, ""
    try:
        p0 = [0.5, x[np.argmin(np.abs(y - y.max() / 2))], max(y.max(), 0.05)]
        popt, _ = optimize.curve_fit(logistic3, x, y, p0=p0, maxfev=40000,
                                     bounds=([0, x.min() - 50, 0.01],
                                             [5, x.max() + 50, 1.0]))
        r_hat, t0_hat, K_hat = popt
        r2_log = 1 - ((y - logistic3(x, *popt)) ** 2).sum() / ss_tot
        ti = int(round(t0_hat))
        t0_date = bins[ti - 1] if 1 <= ti <= nb else f"outside range (index {ti})"
    except Exception:
        pass

    # segmented (one breakpoint, grid search)
    best = (np.inf, None)
    for bp in range(int(x.min()) + 3, int(x.max()) - 2):
        left, right = x <= bp, x > bp
        if left.sum() < 3 or right.sum() < 3:
            continue
        sse = 0.0
        for msk in (left, right):
            lr = stats.linregress(x[msk], y[msk])
            sse += ((y[msk] - (lr.intercept + lr.slope * x[msk])) ** 2).sum()
        if sse < best[0]:
            best = (sse, bp)
    r2_seg = 1 - best[0] / ss_tot if np.isfinite(best[0]) else np.nan
    bp_bin = bins[best[1] - 1] if best[1] else ""


    return dict(
        Position=p, Ref=str(refbase[i]), Alt=str(alt_base[i]), N_bins=int(m.sum()),
        Linear_slope=lin.slope, Linear_R2=r2_lin,
        Logistic_r_per_month=r_hat, Logistic_midpoint_bin=t0_date,
        Logistic_plateau_K=K_hat, Logistic_R2=r2_log,
        Segmented_breakpoint_bin=bp_bin, Segmented_R2=r2_seg,
    )


rows = [r for r in (fit_models(p) for p in FOCUS) if r]
mc = pd.DataFrame(rows)
mc["Manuscript_group"] = np.where(mc.Position.isin(MANUSCRIPT_TOP),
                                  "Top-10 overall slope", "Top-4 recent slope")
mc.to_csv(os.path.join(RES, "model_comparison.tsv"), sep="\t", index=False,
          float_format="%.6g")
print("[stats] model comparison:\n", mc.to_string(index=False), flush=True)

# ---------------------------------------------------------------- recent-slope table
rec = out[out.Position.isin(FOCUS)][
    ["Position", "Ref", "Alt", "N_bins_recent", "Slope_Recent", "SE_Recent",
     "Recent_CI95_low", "Recent_CI95_high", "p_Recent"]]
rec.to_csv(os.path.join(RES, "recent_slope_uncertainty.tsv"), sep="\t", index=False,
           float_format="%.6g")

# how long the highest-slope sites have already been at fixation
fix_rows = []
for p_ in out.head(10).Position:
    i = int(p_) - 1
    m = use[i]
    f = FREQ[i]
    first = next((bins[k] for k in range(nb) if m[k] and f[k] >= 0.99), "")
    after = [f[k] for k in range(bins.index(first), nb) if m[k]] if first else []
    late = [f[k] for k in range(nb) if m[k] and bins[k] >= "2024_01"]
    fix_rows.append({"Position": int(p_), "First_month_above_99pct": first,
                     "Min_freq_since_then": round(min(after), 4) if after else np.nan,
                     "Min_freq_2024_2025": round(min(late), 4) if late else np.nan,
                     "Max_freq_2024_2025": round(max(late), 4) if late else np.nan})
pd.DataFrame(fix_rows).to_csv(os.path.join(RES, "table1_fixation_check.tsv"),
                              sep="\t", index=False)
print("[stats] wrote table1_fixation_check.tsv", flush=True)

summary = dict(
    n_positions_tested=int(ok.sum()),
    n_q_lt_0p05=int(np.nansum(qval < 0.05)),
    n_q_lt_0p05_positive=int(np.nansum((qval < 0.05) & (slope > 0))),
    genomes_last_three_bins={b: None for b in bins[-3:]},
)
with open(os.path.join(RES, "slope_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print("[stats] done", flush=True)
