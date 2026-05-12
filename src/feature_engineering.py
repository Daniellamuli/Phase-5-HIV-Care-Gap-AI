
import warnings
warnings.filterwarnings('ignore')

import sys
import os
import pandas as pd
import numpy as np

# ============================================================
# FIX ROOT IMPORTS
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )

)

from constants import (

    # Shared column names
    NSDCC_COUNTY_COL,
    NSDCC_PERIOD_COL,
    COUNTY_PROF,
    TIER_TS,
    IIT_WEIGHT,
    VLS_WEIGHT,
    HTS_WEIGHT,
    CGI_SCALE_MAX
)

# ============================================================
# YEAR-ON-YEAR CHANGE
# ============================================================

def calculate_yoy_change(df, value_col):
    """
    Calculate year-on-year percentage change
    for a metric within each county.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    value_col : str
        Column to calculate YoY change for

    Returns
    -------
    pd.DataFrame
        Dataframe with new YoY column
    """

    df = df.copy()

    # Sort properly before pct_change
    df = df.sort_values(
        [NSDCC_COUNTY_COL, NSDCC_PERIOD_COL]
    )

    # New column name
    yoy_col = f"{value_col}_yoy_change"

    # Calculate YoY percentage change
    df[yoy_col] = (
        df.groupby(NSDCC_COUNTY_COL)[value_col]
        .pct_change()
    )

    return df

# ============================================================
# ENGINEER IIT + VLS YOY FEATURES
# ============================================================

def engineer_yoy_features(df):
    """
    Add IIT and VLS year-on-year features.

    Returns
    -------
    pd.DataFrame
    """

    df = calculate_yoy_change(
        df,
        value_col="iit_rate"
    )

    df = calculate_yoy_change(
        df,
        value_col="vls_rate"
    )

    return df


# ============================================================
# CARE GAP INDEX (CGI)
# ============================================================

def calculate_care_gap_index(df):
    """
    Calculate Care Gap Index scaled 0-100.

    Formula:
        CGI =
        (0.4 × IIT_rate)
      + (0.4 × (1 - VLS_rate))
      + (0.2 × HTS_positivity_rate)

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    required_cols = [
        "iit_rate",
        "vls_rate",
        "hts_positivity_rate"
    ]

    # Validate columns
    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # CGI calculation
    raw_cgi = (
        (IIT_WEIGHT * df["iit_rate"]) +
        (VLS_WEIGHT * (1 - df["vls_rate"])) +
        (HTS_WEIGHT * df["hts_positivity_rate"])
    )

    # Scale 0-100
    df["care_gap_index"] = (
        raw_cgi * CGI_SCALE_MAX
    )

    return df

# ============================================================
# COUNTY PROFILES
# ============================================================

def build_county_profiles(df, save=True):
    """
    Create one-row-per-county profile dataset
    using latest available period.

    Parameters
    ----------
    df : pd.DataFrame

    save : bool
        Whether to save CSV

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # Get latest record per county
    county_profiles = (
        df.sort_values(NSDCC_PERIOD_COL)
        .groupby(NSDCC_COUNTY_COL)
        .last()
        .reset_index()
    )

    # Important engineered features
    keep_cols = [
        NSDCC_COUNTY_COL,
        NSDCC_PERIOD_COL,
        "iit_rate",
        "vls_rate",
        "hts_positivity_rate",
        "iit_rate_yoy_change",
        "vls_rate_yoy_change",
        "care_gap_index"
    ]

    # Keep only available columns
    existing_cols = [
        col for col in keep_cols
        if col in county_profiles.columns
    ]

    county_profiles = county_profiles[
        existing_cols
    ]

    # Save
    if save:
        county_profiles.to_csv(
            COUNTY_PROF,
            index=False
        )

    return county_profiles

# ============================================================
# TIER TIME SERIES
# ============================================================

def build_tier_timeseries(df, save=True):
    """
    Aggregate IIT/VLS/CGI metrics
    by tier and year.

    Prophet-ready dataset.

    Parameters
    ----------
    df : pd.DataFrame

    save : bool

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # Validate tier column
    if "tier" not in df.columns:
        raise ValueError(
            "'tier' column missing from dataframe."
        )

    # Aggregate metrics
    tier_ts = (
        df.groupby(
            ["tier", NSDCC_PERIOD_COL]
        )
        .agg({
            "iit_rate": "mean",
            "vls_rate": "mean",
            "care_gap_index": "mean"
        })
        .reset_index()
    )

    # Save
    if save:
        tier_ts.to_csv(
            TIER_TS,
            index=False
        )

    return tier_ts


# ============================================================
# FULL FEATURE ENGINEERING PIPELINE
# ============================================================

def run_feature_engineering(df):
    """
    Full reusable feature engineering pipeline.

    Steps:
    1. Engineer YoY changes
    2. Engineer Care Gap Index

    Returns
    -------
    pd.DataFrame
    """

    # YoY features
    df = engineer_yoy_features(df)

    # Care Gap Index
    df = calculate_care_gap_index(df)

    return df

if __name__ == "__main__":

    print("feature_engineering.py loaded successfully")

