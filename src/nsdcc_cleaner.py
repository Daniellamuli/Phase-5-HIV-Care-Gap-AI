"""
src/nsdcc_cleaner.py

HIV Care Gap AI — NSDCC Cleaning Utilities
==========================================

Reusable preprocessing utilities for:
- ART
- HTS
- HTS_Positive
- VLT
- IIT

Scope: Loading, cleaning, imputing, validating and visualising.
       No feature engineering — that belongs to src/feature_engineering.py (Verah).

Author : Eve Michelle
Project: HIV Care Gap AI — Phase 5 Capstone
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os

# ============================================================
# FIX ROOT IMPORTS
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

# ============================================================
# LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

# ============================================================
# IMPORT CONSTANTS
# ============================================================

from constants import (
    COUNTY_NAME_MAP,
    IIT_REGION_MAP,
    REGION_TO_COUNTIES,
)

# ============================================================
# LOADERS
# ============================================================

def load_with_period(filepath, period_col, label):
    """
    Load datasets with Period and County columns.
    Used for: ART, HTS, HTS_Positive
    """
    df = pd.read_excel(filepath, header=1, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df[df[period_col].notna()].reset_index(drop=True)
    print(f'Loaded {label}: {df.shape}')
    return df


def load_snapshot(filepath, county_col, label):
    """
    Load snapshot datasets without Period column.
    Used for: VLT
    """
    df = pd.read_excel(filepath, header=1, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df[df[county_col].notna()].reset_index(drop=True)
    print(f'Loaded {label}: {df.shape}')
    return df


def load_iit(filepath, region_col, label):
    """
    Load IIT regional dataset.
    """
    df = pd.read_excel(filepath, header=1, dtype=str)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df[df[region_col].notna()].reset_index(drop=True)
    print(f'Loaded {label}: {df.shape}')
    return df


# ============================================================
# COUNTY CLEANING
# ============================================================

def strip_suffix(df, county_col, suffix, label):
    """
    Remove suffixes like ' County' from county column.
    e.g. 'Baringo County' -> 'Baringo'
    """
    df = df.copy()
    df[county_col] = (
        df[county_col]
        .str.replace(suffix, '', regex=False)
        .str.strip()
    )
    print(
        f'{label}: '
        f'{df[county_col].nunique()} unique counties'
    )
    return df


def standardise(df, county_col, label):
    """
    Standardise county names using COUNTY_NAME_MAP from constants.py.
    """
    df = df.copy()
    df[county_col] = (
        df[county_col]
        .map(COUNTY_NAME_MAP)
        .fillna(df[county_col])
    )
    bad = [
        c for c in df[county_col].unique()
        if c not in COUNTY_NAME_MAP.values()
    ]
    if bad:
        print(
            f'WARNING {label}: '
            f'{len(bad)} unmatched counties -> {bad}'
        )
    else:
        print(f'OK {label}: county names standardised')
    return df


# ============================================================
# NUMERIC CLEANING + IMPUTATION
# ============================================================

def to_numeric_impute(df, county_col, period_col=None):
    """
    Convert numeric columns to float and impute missing values.

    Strategy: county mean -> column median fallback
    """
    df = df.copy()
    skip = [county_col]
    if period_col and period_col in df.columns:
        skip.append(period_col)
    data_cols = [
        c for c in df.columns
        if c not in skip
    ]
    for col in data_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if period_col and period_col in df.columns:
        df[data_cols] = (
            df.groupby(county_col)[data_cols]
            .transform(lambda x: x.fillna(x.mean()))
        )
    df[data_cols] = (
        df[data_cols]
        .fillna(df[data_cols].median())
    )
    return df


# ============================================================
# VALIDATION
# ============================================================

def validate(df, label, county_col='county', period_col=None):
    """
    Validate: county coverage, duplicates, missing values.
    """
    n_counties   = df[county_col].nunique()
    missing_vals = df.isnull().sum().sum()
    dup_cols = (
        [county_col]
        if not period_col
        else [county_col, period_col]
    )
    duplicates = df.duplicated(subset=dup_cols).sum()
    ok = n_counties == 47 and duplicates == 0
    print(
        f'{"OK" if ok else "WARNING"} {label}: '
        f'{n_counties} counties, '
        f'{duplicates} duplicates, '
        f'{missing_vals} missing values'
    )
    if n_counties != 47:
        missing = (
            set(COUNTY_NAME_MAP.values())
            - set(df[county_col].unique())
        )
        if missing:
            print(f'Missing counties: {sorted(missing)}')


# ============================================================
# IIT REGION EXPANSION
# ============================================================

def expand_iit_regions(iit_raw, region_col):
    """
    Expand IIT from 8 regions to 47 counties.

    Rate/pct cols : copied as-is (apply region-wide)
    Count cols    : divided equally across counties in each region
    """
    rows = []
    for _, row in iit_raw.iterrows():
        region     = row[region_col]
        std_region = IIT_REGION_MAP.get(region)
        if not std_region or std_region not in REGION_TO_COUNTIES:
            continue
        counties = REGION_TO_COUNTIES[std_region]
        n = len(counties)
        for county in counties:
            new_row = {'county': county}
            for col in iit_raw.columns:
                if col == region_col:
                    continue
                val = row[col]
                try:
                    fval = float(val)
                    new_row[col] = (
                        fval if ('pct' in col or 'rate' in col)
                        else fval / n
                    )
                except (ValueError, TypeError):
                    new_row[col] = np.nan
            rows.append(new_row)
    expanded = pd.DataFrame(rows).reset_index(drop=True)
    print(
        f'IIT expanded: '
        f'{iit_raw.shape[0]} regions -> {expanded.shape[0]} counties'
    )
    return expanded


# ============================================================
# IIT MISSING FIX
# ============================================================

def fix_iit_missing(iit):
    """
    Fix remaining IIT missing values after region expansion.
    North Eastern '-' strings cause NaN in pct cols.

    Rate/pct cols : filled with column mean
    Count cols    : filled with 0
    """
    iit = iit.copy()
    rate_cols = [
        c for c in iit.columns
        if 'pct' in c or 'rate' in c
    ]
    count_cols = [
        c for c in iit.columns
        if c not in rate_cols + ['county']
    ]
    for col in rate_cols:
        iit[col] = iit[col].fillna(iit[col].mean())
    for col in count_cols:
        iit[col] = iit[col].fillna(0)
    print(
        f'IIT remaining missing values: '
        f'{iit.isnull().sum().sum()}'
    )
    return iit


# ============================================================
# VLT MISSING FIX
# ============================================================

def fix_vlt_missing(vlt):
    """
    Fix NaN values in VLT after numeric conversion.
    Fills with column median.
    """
    num_cols = vlt.select_dtypes(include='number').columns
    for col in num_cols:
        vlt[col] = vlt[col].fillna(vlt[col].median())
    print(
        f'VLT remaining missing values: '
        f'{vlt.isnull().sum().sum()}'
    )
    return vlt


# ============================================================
# MISSING VALUE HEATMAP
# ============================================================

def missing_heatmap(
    df_before,
    df_after,
    label,
    id_cols=None,
    figsize=(12, 4)
):
    """
    Plot before/after missing-value heatmaps (dark theme).
    Green = present | Red = missing.
    """
    if id_cols is None:
        id_cols = []

    def to_matrix(df):
        cols = [c for c in df.columns if c not in id_cols]
        num  = df[cols].copy()
        for c in cols:
            num[c] = pd.to_numeric(num[c], errors='coerce')
        return num.isnull().astype(int), cols

    mat_b, cols_b = to_matrix(df_before)
    mat_a, cols_a = to_matrix(df_after)

    MISSING  = '#E74C3C'
    PRESENT  = '#1ABC9C'
    BG       = '#0F1117'
    PANEL_BG = '#1E2130'
    TITLE_C  = '#ECF0F1'
    LABEL_C  = '#BDC3C7'
    cmap = mcolors.ListedColormap([PRESENT, MISSING])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor(BG)

    for ax, mat, cols, phase, accent in [
        (ax1, mat_b, cols_b, 'BEFORE Imputation', MISSING),
        (ax2, mat_a, cols_a, 'AFTER Imputation',  PRESENT),
    ]:
        ax.set_facecolor(PANEL_BG)
        ax.imshow(
            mat.values, aspect='auto',
            cmap=cmap, vmin=0, vmax=1,
            interpolation='nearest'
        )
        n_miss = int(mat.values.sum())
        total  = mat.size
        pct    = n_miss / total * 100 if total else 0
        ax.set_title(
            f'{phase}\n{n_miss}/{total} cells missing  ({pct:.1f}%)',
            fontsize=9, color=accent, fontweight='bold', pad=6
        )
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(
            cols, rotation=35, ha='right',
            fontsize=7.5, color=LABEL_C
        )
        ax.set_yticks([])
        ax.set_ylabel('rows ->', fontsize=7, color=LABEL_C)
        ax.tick_params(axis='x', colors=LABEL_C, length=0)
        for sp in ax.spines.values():
            sp.set_edgecolor('#2C3E50')

    fig.legend(
        handles=[
            mpatches.Patch(facecolor=MISSING, label='Missing'),
            mpatches.Patch(facecolor=PRESENT, label='Present'),
        ],
        loc='lower center', ncol=2, fontsize=8,
        facecolor=PANEL_BG, edgecolor='#2C3E50',
        labelcolor=TITLE_C, framealpha=1,
        bbox_to_anchor=(0.5, -0.04)
    )
    fig.suptitle(
        f'{label} - Missing Value Heatmap',
        fontsize=11, color=TITLE_C,
        fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.show()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    'load_with_period',
    'load_snapshot',
    'load_iit',
    'strip_suffix',
    'standardise',
    'to_numeric_impute',
    'validate',
    'expand_iit_regions',
    'fix_iit_missing',
    'fix_vlt_missing',
    'missing_heatmap',
]


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":
    print("NSDCC Cleaner Module Loaded Successfully")