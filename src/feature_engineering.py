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

    NSDCC_COUNTY_COL,
    NSDCC_PERIOD_COL,

    COUNTY_PROF,

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
    Placeholder YoY calculation.

    Dataset currently contains only one year (2025),
    therefore YoY values are set to 0.

    Parameters
    ----------
    df : pd.DataFrame

    value_col : str

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    yoy_col = f"{value_col}_yoy_change"

    # Placeholder values
    df[yoy_col] = 0

    return df

# ============================================================
# ENGINEER YOY FEATURES
# ============================================================

def engineer_yoy_features(df):
    """
    Create IIT and VLS YoY placeholder features.
    """

    df = calculate_yoy_change(
        df,
        "iit_rate"
    )

    df = calculate_yoy_change(
        df,
        "vls_rate_adult"
    )

    return df

# ============================================================
# CARE GAP INDEX
# ============================================================

def calculate_care_gap_index(df):
    """
    Calculate Care Gap Index (CGI).

    Formula:
        (0.4 × IIT_rate)
      + (0.4 × (1 - VLS_rate_adult))
      + (0.2 × HTS_positivity_rate)

    Scaled 0-100.
    """

    df = df.copy()

    required_cols = [
        "iit_rate",
        "vls_rate_adult",
        "hts_positivity_rate"
    ]

    # Validate required columns
    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:

        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # Convert percentages if values exceed 1
    for col in required_cols:

        if df[col].max() > 1:
            df[col] = df[col] / 100

    raw_cgi = (
        (IIT_WEIGHT * df["iit_rate"]) +
        (VLS_WEIGHT * (1 - df["vls_rate_adult"])) +
        (HTS_WEIGHT * df["hts_positivity_rate"])
    )

    # Scale 0-100
    df["care_gap_index"] = (
        raw_cgi * CGI_SCALE_MAX
    )

    return df

# ============================================================
# ART COVERAGE
# ============================================================

def calculate_art_coverage(df):
    """
    Calculate ART coverage.

    Formula:
        adults_on_art / plhiv_estimate

    Returns placeholder 0.5 if denominator missing.
    """

    df = df.copy()

    if (
        "adults_on_art" in df.columns and
        "plhiv_estimate" in df.columns
    ):

        df["art_coverage"] = (
            df["adults_on_art"] /
            df["plhiv_estimate"]
        ).clip(0, 1).fillna(0)

    else:

        print(
            "WARNING: Placeholder ART coverage used"
        )

        df["art_coverage"] = 0.5

    return df

# ============================================================
# COUNTY PROFILES
# ============================================================

def build_county_profiles(df, save=True):
    """
    Create one row per county
    using latest available period.

    Parameters
    ----------
    df : pd.DataFrame

    save : bool

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # Handle both 'period' and 'Period' column names
    period_col = NSDCC_PERIOD_COL if NSDCC_PERIOD_COL in df.columns else "period"
    county_col = NSDCC_COUNTY_COL if NSDCC_COUNTY_COL in df.columns else "county"

    county_profiles = (
        df.sort_values(period_col)
        .groupby(county_col)
        .last()
        .reset_index()
    )

    # Save dataset
    if save:

        county_profiles.to_csv(
            COUNTY_PROF,
            index=False
        )

    return county_profiles

# ============================================================
# MASTER FEATURE ENGINEERING PIPELINE
# ============================================================

def run_feature_engineering(df, save=True):
    """
    Master feature engineering pipeline.

    Steps:
    ------
    1. Create HTS positivity rate
    2. Engineer YoY placeholder features
    3. Calculate Care Gap Index
    4. Calculate ART coverage
    5. Build county profiles

    Parameters
    ----------
    df : pd.DataFrame

    save : bool

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    # ========================================================
    # CREATE HTS POSITIVITY RATE
    # ========================================================

    if "hts_positivity_rate" not in df.columns:

        if (
            "hts_positive" in df.columns and
            "hts_tested" in df.columns
        ):

            df["hts_positivity_rate"] = (
                df["hts_positive"] /
                df["hts_tested"]
            ).fillna(0)

        else:

            raise ValueError(
                "Missing HTS columns"
            )

    # ========================================================
    # COMPUTE IIT RATE
    # iit_rate_pct exists in merged data as percentage (0-100)
    # Convert to decimal (0-1) for CGI formula
    # ========================================================

    if "iit_rate" not in df.columns:
        if "iit_rate_pct" in df.columns:
            df["iit_rate"] = df["iit_rate_pct"] / 100
            print("  Computed iit_rate from iit_rate_pct")
        elif "iit_count" in df.columns and "adults_on_treatment" in df.columns:
            df["iit_rate"] = (
                df["iit_count"] / df["adults_on_treatment"]
            ).fillna(0)
            print("  Computed iit_rate from iit_count / adults_on_treatment")
        else:
            raise ValueError(
                "Cannot compute iit_rate: neither iit_rate_pct nor "
                "iit_count/adults_on_treatment found in data."
            )

    # ========================================================
    # YOY FEATURES
    # ========================================================

    df = engineer_yoy_features(df)

    # ========================================================
    # CARE GAP INDEX
    # ========================================================

    df = calculate_care_gap_index(df)

    # ========================================================
    # ART COVERAGE
    # ========================================================

    df = calculate_art_coverage(df)

    # ========================================================
    # COUNTY PROFILES
    # ========================================================

    if save:

        build_county_profiles(
            df,
            save=True
        )

    return df

# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "feature_engineering.py loaded successfully"
    )
