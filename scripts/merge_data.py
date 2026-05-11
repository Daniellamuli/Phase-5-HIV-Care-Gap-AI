"""
scripts/merge_data.py
HIV Care Gap AI — NSDCC Data Merge
====================================
Merges the 4 clean NSDCC CSVs into a single county-period level table.
 
Merge strategy
--------------
ART + HTS  → inner join on ['county', 'period']
             (both have 47 counties x 1 period = same shape)
+ VLT      → left join on ['county'] only
             (VLT has no Period — same values broadcast across all periods)
+ IIT      → left join on ['county'] only
             (IIT has no Period — same values broadcast across all periods)
 
Output
------
data/processed/nsdcc_clean.csv  (NSDCC_CLEAN from constants.py)
 
Validation
----------
- 47 unique counties
- No rows lost vs ART baseline
- 0 missing values
 
Usage
-----
    # From project root:
    python scripts/merge_data.py
 
    # From a notebook:
    import sys, os
    sys.path.append(os.path.abspath('..'))
    from scripts.merge_data import run_merge
    merged = run_merge()
 
Author : Eve Michelle
Project: HIV Care Gap AI — Phase 5 Capstone
"""
 
import sys
import os
import warnings
warnings.filterwarnings('ignore')
 
import pandas as pd
import numpy as np
 
# ── Root imports
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)
 
from constants import (
    ART_CLEAN,
    HTS_CLEAN,
    VLT_CLEAN,
    IIT_CLEAN,
    NSDCC_CLEAN,
    PROCESSED_DIR,
)
 
 
# ============================================================
# HELPERS
# ============================================================
 
def load_csv(path, label):
    """Load a processed CSV and print shape."""
    df = pd.read_csv(path)
    print(f'  Loaded {label}: {df.shape}')
    return df
 
 
def validate_merge(df, label, expected_counties=47, baseline_rows=None):
    """
    Validate merged DataFrame.
 
    Checks
    ------
    - Correct number of unique counties
    - No rows lost vs baseline
    - Zero missing values
    """
    n_counties = df['county'].nunique()
    n_missing  = df.isnull().sum().sum()
    n_rows     = len(df)
 
    ok_counties = n_counties == expected_counties
    ok_rows     = (baseline_rows is None) or (n_rows == baseline_rows)
    ok_missing  = n_missing == 0
 
    ok = ok_counties and ok_rows and ok_missing
 
    status = 'OK' if ok else 'WARNING'
    print(f'  {status} {label}:')
    print(f'    Shape      : {df.shape}')
    print(f'    Counties   : {n_counties} (expected {expected_counties})')
    print(f'    Missing    : {n_missing}')
 
    if baseline_rows is not None:
        row_status = 'OK' if ok_rows else f'WARNING - expected {baseline_rows}'
        print(f'    Rows vs ART: {n_rows} ({row_status})')
 
    if not ok_counties:
        from constants import COUNTY_NAME_MAP
        canonical = set(COUNTY_NAME_MAP.values())
        missing_c = sorted(canonical - set(df['county'].unique()))
        extra_c   = sorted(set(df['county'].unique()) - canonical)
        if missing_c:
            print(f'    Missing counties: {missing_c}')
        if extra_c:
            print(f'    Extra counties  : {extra_c}')
 
    return ok
 
 
# ============================================================
# CORE MERGE
# ============================================================
 
def run_merge(save=True):
    """
    Load, merge, validate and optionally save the NSDCC master table.
 
    Parameters
    ----------
    save : bool
        If True, save merged DataFrame to NSDCC_CLEAN (constants.py).
 
    Returns
    -------
    pd.DataFrame — merged county-period level table
    """
 
    print('=' * 55)
    print('NSDCC Data Merge')
    print('=' * 55)
 
    # ── Step 1: Load all 4 clean files
    print('\n[1] Loading clean files...')
    art = load_csv(ART_CLEAN, 'ART')
    hts = load_csv(HTS_CLEAN, 'HTS')
    vlt = load_csv(VLT_CLEAN, 'VLT')
    iit = load_csv(IIT_CLEAN, 'IIT')
 
    # ── Step 2: Normalise join keys
    print('\n[2] Normalising join keys...')
 
    for df, label in [(art, 'ART'), (hts, 'HTS')]:
 
        df['county'] = df['county'].astype(str).str.strip()
 
        # ART period is saved as "December 2025" string in the CSV.
        # Extract the 4-digit year with regex before converting to int.
        df['period'] = (
            df['period']
            .astype(str)
            .str.extract(r'(\d{4})')[0]
        )
 
        df['period'] = pd.to_numeric(df['period'], errors='coerce')
 
        n_before = len(df)
        df.dropna(subset=['period'], inplace=True)
        n_dropped = n_before - len(df)
 
        if n_dropped:
            print(f'  {label}: dropped {n_dropped} blank rows')
 
        df['period'] = df['period'].astype(int)
        df.reset_index(drop=True, inplace=True)
 
        print(
            f'  {label}: '
            f'county={df["county"].dtype}, '
            f'period={df["period"].dtype}, '
            f'shape={df.shape}'
        )
 
    for df, label in [(vlt, 'VLT'), (iit, 'IIT')]:
        df['county'] = df['county'].astype(str).str.strip()
        print(f'  {label}: county={df["county"].dtype}, shape={df.shape}')
 
    # Baseline row count — ART drives the merge
    baseline_rows = len(art)
 
    # ── Step 3: ART + HTS on county + period
    print('\n[3] Merging ART + HTS on [county, period]...')
    merged = pd.merge(
        art,
        hts,
        on=['county', 'period'],
        how='inner',
        suffixes=('_art', '_hts'),
    )
    print(f'  After ART + HTS: {merged.shape}')
 
    if len(merged) != baseline_rows:
        print(
            f'  WARNING: row count changed — '
            f'{baseline_rows} -> {len(merged)}'
        )
 
    # ── Step 4: + VLT on county only
    print('\n[4] Merging + VLT on [county]...')
    merged = pd.merge(
        merged,
        vlt,
        on='county',
        how='left',
    )
    print(f'  After + VLT: {merged.shape}')
 
    # ── Step 5: + IIT on county only
    print('\n[5] Merging + IIT on [county]...')
    merged = pd.merge(
        merged,
        iit,
        on='county',
        how='left',
    )
    print(f'  After + IIT: {merged.shape}')
 
    # ── Step 6: Sort
    merged = merged.sort_values(
        ['county', 'period']
    ).reset_index(drop=True)
 
    # ── Step 7: Validate
    print('\n[6] Validating merged table...')
    ok = validate_merge(
        merged,
        'NSDCC merged',
        baseline_rows=baseline_rows,
    )
 
    # ── Step 8: Save to data/processed/nsdcc_clean.csv
    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        merged.to_csv(NSDCC_CLEAN, index=False)
        size_kb = os.path.getsize(NSDCC_CLEAN) / 1024
        print(f'\n[7] Saved -> {NSDCC_CLEAN}  ({size_kb:.1f} KB)')
    else:
        print('\n[7] save=False — not saved to disk')
 
    print('\n' + ('All done ✓' if ok else 'Completed with warnings ⚠'))
    print('=' * 55)
 
    return merged
 
 
# ============================================================
# COLUMN SUMMARY HELPER
# ============================================================
 
def column_summary(df):
    """
    Return a tidy summary of every column in the merged table.
    Useful for handoff to Verah (feature engineering).
    """
    rows = []
    for col in df.columns:
        rows.append({
            'column'  : col,
            'dtype'   : str(df[col].dtype),
            'non_null': df[col].notna().sum(),
            'null'    : df[col].isna().sum(),
            'sample'  : df[col].dropna().iloc[0]
                        if df[col].notna().any() else None,
        })
    return pd.DataFrame(rows)
 
 
# ============================================================
# ENTRYPOINT
# ============================================================
 
if __name__ == '__main__':
    merged = run_merge(save=True)
    print('\nColumn summary:')
    print(column_summary(merged).to_string(index=False))