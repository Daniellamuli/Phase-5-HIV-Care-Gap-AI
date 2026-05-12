"""
scripts/extract_data.py
═══════════════════════════════════════════════════════════════
Production version of: notebooks/01_data_extraction.ipynb
Step 1 in main.py pipeline. Read-only — saves nothing.

Real file structures (confirmed, do not change):
  ART     (47x11) : Period + County col, values "Baringo County"
  HTS     (47x7)  : Period + County col, values "Baringo County"
  HTS_POS (47x~)  : Period + County col, values "Baringo County" — same structure as HTS
  VLT     (47x7)  : County col ONLY, no Period, values "Baringo"
  IIT     (9x13)  : Region col ONLY, no Period, 9 MOH regions
  DHS             : integer county codes 1-47
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from constants import (
    ADULT_ART_FILE, HTS_FILE, HTS_POSITIVE_FILE, VLT_FILE, IIT_FILE, DHS_REDUCED,
    ART_COUNTY_COL, ART_PERIOD_COL, ART_COUNTY_SUFFIX,
    HTS_COUNTY_COL, HTS_PERIOD_COL, HTS_COUNTY_SUFFIX,
    HTS_POS_COUNTY_COL, HTS_POS_PERIOD_COL, HTS_POS_COUNTY_SUFFIX,
    VLT_COUNTY_COL, VLT_HAS_PERIOD,
    IIT_REGION_COL, IIT_HAS_PERIOD,
    COUNTY_NAME_MAP, DHS_COUNTY_MAP,
    IIT_REGION_MAP, REGION_TO_COUNTIES,
)

# ── LOADERS (match notebook exactly) ───────────────────────────

def load_with_period(filepath, period_col, label):
    """ART and HTS — have both Period and County columns."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FILE NOT FOUND: {filepath}")
    df = pd.read_excel(filepath, header=1, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[df[period_col].notna()].reset_index(drop=True)
    print(f"  {label:<6} {df.shape[0]:>3} rows x {df.shape[1]:>3} cols  |  missing: {df.isnull().sum().sum()}")
    return df

def load_snapshot(filepath, county_col, label):
    """VLT — County only, no Period column."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FILE NOT FOUND: {filepath}")
    df = pd.read_excel(filepath, header=1, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[df[county_col].notna()].reset_index(drop=True)
    print(f"  {label:<6} {df.shape[0]:>3} rows x {df.shape[1]:>3} cols  |  missing: {df.isnull().sum().sum()}")
    print(f"         No Period column — single-period snapshot")
    return df

def load_iit(filepath, region_col, label):
    """IIT — Region column only, no Period, 9 rows."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FILE NOT FOUND: {filepath}")
    df = pd.read_excel(filepath, header=1, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df[df[region_col].notna()].reset_index(drop=True)
    print(f"  {label:<6} {df.shape[0]:>3} rows x {df.shape[1]:>3} cols  |  missing: {df.isnull().sum().sum()}")
    print(f"         Region-level only, no Period column")
    return df

# ── VALIDATORS ─────────────────────────────────────────────────

def validate_county_map(raw_values, label, strip_suffix=None):
    check = [v.replace(strip_suffix, "").strip() for v in raw_values] if strip_suffix else list(raw_values)
    unmapped = [n for n in check if n not in COUNTY_NAME_MAP]
    if unmapped:
        print(f"  WARNING {label} — {len(unmapped)} NOT in COUNTY_NAME_MAP: {unmapped}")
    else:
        print(f"  OK {label} — all {len(raw_values)} county values covered")
    return len(unmapped)

def validate_region_map(raw_values, label):
    unmapped = [r for r in raw_values if r not in IIT_REGION_MAP]
    if unmapped:
        print(f"  WARNING {label} — regions NOT in IIT_REGION_MAP: {unmapped}")
    else:
        print(f"  OK {label} — all {len(raw_values)} region values covered")
    return len(unmapped)

def validate_dhs_codes(raw_codes, label):
    codes = [int(float(c)) for c in raw_codes if str(c) not in ("nan", "")]
    unmapped = [c for c in codes if c not in DHS_COUNTY_MAP]
    if unmapped:
        print(f"  WARNING {label} — codes NOT in DHS_COUNTY_MAP: {unmapped}")
    else:
        print(f"  OK {label} — all {len(set(codes))} county codes covered")
    return len(unmapped)

# ── MAIN ───────────────────────────────────────────────────────

def extract_all():
    print("=" * 58)
    print("  STEP 1 — Data Extraction  |  HIV Care Gap AI")
    print("=" * 58)

    # Load
    print("\n  Loading files:")
    art     = load_with_period(ADULT_ART_FILE,    ART_PERIOD_COL, "ART")
    hts     = load_with_period(HTS_FILE,          HTS_PERIOD_COL, "HTS")
    hts_pos = load_with_period(HTS_POSITIVE_FILE, HTS_POS_PERIOD_COL, "HTS+")
    vlt     = load_snapshot(VLT_FILE,             VLT_COUNTY_COL, "VLT")
    iit     = load_iit(IIT_FILE,                  IIT_REGION_COL, "IIT")

    if not os.path.exists(DHS_REDUCED):
        raise FileNotFoundError(f"FILE NOT FOUND: {DHS_REDUCED}")
    dhs = pd.read_csv(DHS_REDUCED, low_memory=False)
    print(f"  {'DHS':<6} {dhs.shape[0]:>5} rows x {dhs.shape[1]:>3} cols")

    # Summary
    print(f"\n  {'File':<28} {'Rows':>5} {'Cols':>5}  {'Period':>8}  Format")
    print(f"  {'-'*75}")
    print(f"  {'Adult_on_ART.xlsx':<28} {art.shape[0]:>5} {art.shape[1]:>5}  {'Yes':>8}  'Baringo County' → strip suffix")
    print(f"  {'Adult_on_HTS.xlsx':<28} {hts.shape[0]:>5} {hts.shape[1]:>5}  {'Yes':>8}  'Baringo County' → strip suffix")
    print(f"  {'HTS_Positive.xlsx':<28} {hts_pos.shape[0]:>5} {hts_pos.shape[1]:>5}  {'Yes':>8}  'Baringo County' → strip suffix")
    print(f"  {'VLT.xlsx':<28} {vlt.shape[0]:>5} {vlt.shape[1]:>5}  {'No':>8}  'Baringo' → clean")
    print(f"  {'IIT.xlsx':<28} {iit.shape[0]:>5} {iit.shape[1]:>5}  {'No':>8}  9 regions → expand to 47 counties")
    print(f"  {'individual_features.csv':<28} {dhs.shape[0]:>5} {dhs.shape[1]:>5}  {'N/A':>8}  integers 1-47")

    # Validate maps
    print("\n  Validating mappings:")
    errors = 0
    errors += validate_county_map(art[ART_COUNTY_COL].dropna().unique(),         "ART",  ART_COUNTY_SUFFIX)
    errors += validate_county_map(hts[HTS_COUNTY_COL].dropna().unique(),         "HTS",  HTS_COUNTY_SUFFIX)
    errors += validate_county_map(hts_pos[HTS_POS_COUNTY_COL].dropna().unique(), "HTS+", HTS_POS_COUNTY_SUFFIX)
    errors += validate_county_map(vlt[VLT_COUNTY_COL].dropna().unique(),         "VLT")
    errors += validate_region_map(iit[IIT_REGION_COL].dropna().unique(),         "IIT")
    errors += validate_dhs_codes(dhs["county"].dropna().unique(),                 "DHS")

    all_in_regions = [c for cs in REGION_TO_COUNTIES.values() for c in cs]
    if len(all_in_regions) != 47:
        print(f"  WARNING REGION_TO_COUNTIES covers {len(all_in_regions)} counties, expected 47")
        errors += 1
    else:
        print(f"  OK REGION_TO_COUNTIES — 47 counties confirmed")

    print(f"\n{'=' * 58}")
    if errors == 0:
        print("  All files loaded. All mappings valid.")
        print("  Next: scripts/clean_data.py")
    else:
        print(f"  {errors} issue(s) found — fix constants.py before cleaning.")
    print()

    return {"art": art, "hts": hts, "hts_pos": hts_pos, "vlt": vlt, "iit": iit, "dhs": dhs}

if __name__ == "__main__":
    extract_all()