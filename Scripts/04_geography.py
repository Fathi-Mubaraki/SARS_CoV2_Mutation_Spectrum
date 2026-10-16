#!/usr/bin/env python3
"""
04_geography.py  --  Reviewer 1 point 2, Reviewer 2 point 5, Reviewer 1 point 5.

1. Characterises the geographic composition of the analysed dataset month by month.
2. Reconciles the 1,143,840 downloaded records with the number actually analysed,
   and reports duplicates -- the reproducibility details Reviewer 1 asked for.
3. Writes the per-genome accession manifest and the per-month lineage composition.
   The region-stratified sensitivity analysis itself is in 09_allele_specific.py.
"""
import os
import numpy as np
import pandas as pd

RES = "/gdat/fathi/virus/publication/revision/results"
META = "/gdat/fathi/virus/metadata.tsv"

REGION = {}
_r = {
 "Africa": ["Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon","Cabo Verde","Central African Republic","Chad","Comoros","Congo","Cote d'Ivoire","Democratic Republic of the Congo","Djibouti","Egypt","Equatorial Guinea","Eritrea","Eswatini","Ethiopia","Gabon","Gambia","Ghana","Guinea","Guinea-Bissau","Kenya","Lesotho","Liberia","Libya","Madagascar","Malawi","Mali","Mauritania","Mauritius","Morocco","Mozambique","Namibia","Niger","Nigeria","Rwanda","Senegal","Seychelles","Sierra Leone","Somalia","South Africa","South Sudan","Sudan","Tanzania","Togo","Tunisia","Uganda","Zambia","Zimbabwe","Reunion","Mayotte"],
 "Asia": ["Afghanistan","Armenia","Azerbaijan","Bahrain","Bangladesh","Bhutan","Brunei","Cambodia","China","Cyprus","Georgia","Hong Kong","India","Indonesia","Iran","Iraq","Israel","Japan","Jordan","Kazakhstan","Kuwait","Kyrgyzstan","Laos","Lebanon","Macau","Malaysia","Maldives","Mongolia","Myanmar","Nepal","North Korea","Oman","Pakistan","Palestine","Philippines","Qatar","Saudi Arabia","Singapore","South Korea","Sri Lanka","Syria","Taiwan","Tajikistan","Thailand","Timor-Leste","Turkey","Turkmenistan","United Arab Emirates","Uzbekistan","Viet Nam","Vietnam","West Bank"],
 "Europe": ["Albania","Andorra","Austria","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Croatia","Czech Republic","Czechia","Denmark","Estonia","Faroe Islands","Finland","France","Germany","Gibraltar","Greece","Hungary","Iceland","Ireland","Italy","Kosovo","Latvia","Liechtenstein","Lithuania","Luxembourg","Malta","Moldova","Monaco","Montenegro","Netherlands","North Macedonia","Norway","Poland","Portugal","Romania","Russia","San Marino","Serbia","Slovakia","Slovenia","Spain","Sweden","Switzerland","Ukraine","United Kingdom"],
 "North America": ["Bahamas","Barbados","Belize","Bermuda","Canada","Costa Rica","Cuba","Dominica","Dominican Republic","El Salvador","Greenland","Grenada","Guadeloupe","Guatemala","Haiti","Honduras","Jamaica","Martinique","Mexico","Nicaragua","Panama","Puerto Rico","Saint Lucia","Sint Maarten","Trinidad and Tobago","USA","United States","Aruba","Curacao","Anguilla","Cayman Islands","Saint Kitts and Nevis","Saint Vincent and the Grenadines","Antigua and Barbuda","British Virgin Islands","Montserrat","Turks and Caicos Islands"],
 "South America": ["Argentina","Bolivia","Brazil","Chile","Colombia","Ecuador","French Guiana","Guyana","Paraguay","Peru","Suriname","Uruguay","Venezuela"],
 "Oceania": ["Australia","Fiji","French Polynesia","Guam","Kiribati","Marshall Islands","Micronesia","New Caledonia","New Zealand","Northern Mariana Islands","Palau","Papua New Guinea","Samoa","Solomon Islands","Tonga","Vanuatu","American Samoa","Cook Islands","Wallis and Futuna"],
}
for k, v in _r.items():
    for c in v:
        REGION[c] = k

print("[geo] reading accession -> bin map", flush=True)
acc = pd.read_csv(os.path.join(RES, "accession_bin.tsv.gz"), sep="\t")
acc_set = set(acc.accession)
print(f"[geo]   {len(acc)} analysed accessions, {len(acc_set)} unique", flush=True)

print("[geo] reading NCBI metadata", flush=True)
meta = pd.read_csv(META, sep="\t", usecols=["Accession", "Pangolin", "Country",
                                            "Geo_Location", "Collection_Date",
                                            "Release_Date", "Length", "Host"],
                   dtype=str, low_memory=False)
print(f"[geo]   {len(meta)} metadata records", flush=True)

# ------------------------------------------------ reproducibility reconciliation
n_downloaded = len(meta)
meta["in_analysis"] = meta.Accession.isin(acc_set)
n_analysed = int(meta.in_analysis.sum())
cd = meta.Collection_Date.fillna("")
has_month = cd.str.match(r"^\d{4}-\d{2}")
ym = cd.str.slice(0, 7)
in_window = has_month & (ym >= "2019-12") & (ym <= "2025-09")

full = cd.str.match(r"^\d{4}-\d{2}-\d{2}$")
month_only = cd.str.match(r"^\d{4}-\d{2}$")
year_only = cd.str.match(r"^\d{4}$")
blank = cd == ""
elig = meta[in_window]
recon = pd.DataFrame([
    ("NCBI Virus records downloaded", n_downloaded),
    ("  excluded: year only, no month", int(year_only.sum())),
    ("  excluded: no collection date", int(blank.sum())),
    ("  excluded: collection month outside Dec 2019 - Sep 2025",
     int((has_month & ~in_window).sum())),
    ("Eligible for monthly binning", int(in_window.sum())),
    ("  excluded: month known but no day (binning required YYYY-MM-DD)",
     int((in_window & month_only & ~meta.in_analysis).sum())),
    ("Genomes analysed", n_analysed),
], columns=["Step", "Genomes"])
recon.to_csv(os.path.join(RES, "dataset_reconciliation.tsv"), sep="\t", index=False)
print("[geo] reconciliation:\n", recon.to_string(index=False), flush=True)
assert int(in_window.sum()) + int(year_only.sum()) + int(blank.sum()) \
    + int((has_month & ~in_window).sum()) == n_downloaded
# no full-date eligible genome is missing, so the shortfall is entirely date precision
assert int((in_window & full & ~meta.in_analysis).sum()) == 0
assert int((in_window & month_only & ~meta.in_analysis).sum()) \
    == int(in_window.sum()) - n_analysed
mo_elig = int((in_window & month_only).sum())
print(f"[geo] eligible records with month-only dates: {mo_elig}; "
      f"of these analysed: {int(meta.loc[in_window & month_only, 'in_analysis'].sum())}",
      flush=True)

# Reviewer 1 asked about "duplicate genomes". Accessions are unique, but identical
# consensus sequences are common in surveillance data and are what actually affects a
# frequency estimate, so both are reported.
print(f"[geo] duplicate accessions in metadata: {meta.Accession.duplicated().sum()}; "
      f"in analysed set: {acc.accession.duplicated().sum()}", flush=True)

# ------------------------------------------------ monthly geographic composition
m = meta[meta.in_analysis].merge(acc, left_on="Accession", right_on="accession",
                                 how="left")
m["Region"] = m.Country.map(REGION).fillna("Other/unknown")
print(f"[geo] unmapped countries: "
      f"{sorted(set(m.loc[m.Region == 'Other/unknown', 'Country'].dropna().unique()))[:25]}",
      flush=True)

comp = (m.groupby(["bin", "Region"]).size().unstack(fill_value=0))
comp["Total"] = comp.sum(axis=1)
for r in [c for c in comp.columns if c != "Total"]:
    comp[r + "_pct"] = (100 * comp[r] / comp["Total"]).round(1)
comp = comp.sort_index()
comp.to_csv(os.path.join(RES, "monthly_region_composition.tsv"), sep="\t")

# country-level concentration
def top_countries(g, k=3):
    vc = g.Country.value_counts()
    tot = vc.sum()
    return "; ".join(f"{c} {100*n/tot:.0f}%" for c, n in vc.head(k).items())

def shannon(g):
    p = g.Country.value_counts(normalize=True).to_numpy()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

summ = m.groupby("bin").apply(
    lambda g: pd.Series({
        "n_genomes": len(g),
        "n_countries": g.Country.nunique(),
        "top3_countries": top_countries(g),
        "top1_share_pct": round(100 * g.Country.value_counts(normalize=True).iloc[0], 1)
                          if len(g) else np.nan,
        "shannon_country_diversity": round(shannon(g), 3),
    }), include_groups=False).sort_index()
summ.to_csv(os.path.join(RES, "monthly_sampling_summary.tsv"), sep="\t")
print("[geo] monthly sampling summary (selected months):\n",
      summ.loc[[i for i in summ.index if i.endswith(("_01", "_07"))] +
               list(summ.index[-4:])].to_string(), flush=True)

# ------------------------------------------------ lineage composition per month
lin = (m.assign(Pango=m.Pangolin.fillna("unassigned"))
         .groupby("bin").Pango.value_counts(normalize=True)
         .rename("share").reset_index())
lin_top = lin[lin.share >= 0.05].sort_values(["bin", "share"], ascending=[True, False])
lin_top.to_csv(os.path.join(RES, "monthly_lineage_composition.tsv"), sep="\t",
               index=False)

# accession manifest with the fields needed to rebuild the dataset
manifest = m[["Accession", "bin", "Collection_Date", "Release_Date", "Country",
              "Region", "Pangolin", "Length", "Host"]].sort_values(
                  ["bin", "Accession"])
manifest.to_csv(os.path.join(RES, "accession_manifest.tsv.gz"), sep="\t",
                index=False, compression="gzip")
print(f"[geo] wrote accession_manifest.tsv.gz ({len(manifest)} rows)", flush=True)

meta_small = m[["Accession", "bin", "Country", "Region", "Pangolin",
                "Collection_Date"]]
meta_small.to_parquet(os.path.join(RES, "analysed_metadata.parquet"), index=False)
print("[geo] done", flush=True)
