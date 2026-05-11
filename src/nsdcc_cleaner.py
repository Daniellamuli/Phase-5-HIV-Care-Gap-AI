"""
src/nsdcc_cleaner.py

HIV Care Gap AI — NSDCC Cleaning Utilities
==========================================

Reusable preprocessing utilities for:
- ART
- HTS
- VLT
- IIT

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
    Load datasets with:
    - County
    - Period
    """

    df = pd.read_excel(filepath, header=1, dtype=str)

    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    df = df[df[period_col].notna()].reset_index(drop=True)

    print(f'Loaded {label}: {df.shape}')

    return df


def load_snapshot(filepath, county_col, label):
    """
    Load snapshot datasets without Period column.
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
    Remove suffixes like ' County'
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
    Standardise county names using COUNTY_NAME_MAP.
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
    Convert numeric columns and impute missing values.

    Strategy:
    county mean -> column median fallback
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
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

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
    Validate:
    - county coverage
    - duplicates
    - missing values
    """

    n_counties = df[county_col].nunique()

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

def expand_iit_regions(iit, region_col='Region'):
    """
    Expand IIT regions into county-level rows.
    """

    iit = iit.copy()

    iit[region_col] = (
        iit[region_col]
        .map(IIT_REGION_MAP)
        .fillna(iit[region_col])
    )

    rows = []

    for _, row in iit.iterrows():

        region = row[region_col]

        if region in REGION_TO_COUNTIES:

            counties = REGION_TO_COUNTIES[region]

            for county in counties:

                new_row = row.copy()

                new_row['county'] = county

                rows.append(new_row)

    expanded = pd.DataFrame(rows)

    print(
        f'IIT expanded to '
        f'{expanded["county"].nunique()} counties'
    )

    return expanded


# ============================================================
# VLS ENGINEERING
# ============================================================

def engineer_vls_rates(vlt):
    """
    Engineer Viral Load Suppression rates.
    """

    vlt = vlt.copy()

    vlt['vls_rate_male15plus'] = (
        vlt['vls_suppressed_male_15plus']
        / vlt['vlt_valid_male_15plus']
    ).clip(0, 1)

    vlt['vls_rate_female15plus'] = (
        vlt['vls_suppressed_female_15plus']
        / vlt['vlt_valid_female_15plus']
    ).clip(0, 1)

    vlt['vls_rate_under15'] = (
        vlt['vls_suppressed_under15']
        / vlt['vlt_valid_under15']
    ).clip(0, 1)

    total_valid = (
        vlt['vlt_valid_male_15plus']
        + vlt['vlt_valid_female_15plus']
    )

    total_suppressed = (
        vlt['vls_suppressed_male_15plus']
        + vlt['vls_suppressed_female_15plus']
    )

    vlt['vls_rate_adult'] = (
        total_suppressed / total_valid
    ).clip(0, 1)

    print(
        f'National adult VLS rate: '
        f'{vlt["vls_rate_adult"].mean():.3f}'
    )

    return vlt


def fix_vlt_missing(vlt):
    """
    Fix NaN values after VLS engineering.
    """

    rate_cols = [
        'vls_rate_adult',
        'vls_rate_male15plus',
        'vls_rate_female15plus',
        'vls_rate_under15',
    ]

    for col in rate_cols:

        vlt[col] = (
            vlt[col]
            .fillna(vlt[col].median())
        )

    print(
        f'VLT remaining missing values: '
        f'{vlt.isnull().sum().sum()}'
    )

    return vlt


# ============================================================
# IIT MISSING FIXES
# ============================================================

def fix_iit_missing(iit):
    """
    Fix remaining IIT missing values.
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

        iit[col] = (
            iit[col]
            .fillna(iit[col].mean())
        )

    for col in count_cols:

        iit[col] = (
            iit[col]
            .fillna(0)
        )

    print(
        f'IIT remaining missing values: '
        f'{iit.isnull().sum().sum()}'
    )

    return iit


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
    Plot before/after missing-value heatmaps.
    """

    if id_cols is None:
        id_cols = []

    def to_matrix(df):

        cols = [
            c for c in df.columns
            if c not in id_cols
        ]

        num = df[cols].copy()

        for c in cols:
            num[c] = pd.to_numeric(
                num[c],
                errors='coerce'
            )

        return num.isnull().astype(int), cols

    mat_b, cols_b = to_matrix(df_before)
    mat_a, cols_a = to_matrix(df_after)

    MISSING = '#E74C3C'
    PRESENT = '#1ABC9C'

    cmap = mcolors.ListedColormap([
        PRESENT,
        MISSING
    ])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    ax1.imshow(
        mat_b.values,
        aspect='auto',
        cmap=cmap
    )

    ax1.set_title('Before Imputation')

    ax2.imshow(
        mat_a.values,
        aspect='auto',
        cmap=cmap
    )

    ax2.set_title('After Imputation')

    fig.suptitle(
        f'{label} Missing Value Heatmap'
    )

    missing_patch = mpatches.Patch(
        facecolor=MISSING,
        label='Missing'
    )

    present_patch = mpatches.Patch(
        facecolor=PRESENT,
        label='Present'
    )

    fig.legend(
        handles=[missing_patch, present_patch],
        loc='lower center',
        ncol=2
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
    'engineer_vls_rates',
    'fix_vlt_missing',
    'fix_iit_missing',
    'missing_heatmap',
]


# ============================================================
# OPTIONAL TEST
# ============================================================

if __name__ == "__main__":

    print("NSDCC Cleaner Module Loaded Successfully")