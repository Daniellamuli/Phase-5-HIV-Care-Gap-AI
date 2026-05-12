
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
    IIT_WEIGHT,
    VLS_WEIGHT,
    HTS_WEIGHT,
    CGI_SCALE_MAX,
    COUNTY_PROF,
    TIER_TS
)


def calculate_yoy_change(
    df,
    value_col,
    group_col="county",
    year_col="period"
):
    """
    Calculate year-on-year percentage change.
    """

    df = df.copy()

    df = df.sort_values([group_col, year_col])

    df[f"{value_col}_yoy_change"] = (
        df.groupby(group_col)[value_col]
        .pct_change()
        .fillna(0)
    )

    return df


def calculate_care_gap_index(df):
    """
    Calculate Care Gap Index (0-100 scale).
    """

    df = df.copy()

    # Create HTS positivity rate if missing
    if "hts_positivity_rate" not in df.columns:

        if (
            "hts_tst_pos" in df.columns
            and "hts_tst" in df.columns
        ):

            df["hts_positivity_rate"] = (
                df["hts_tst_pos"] / df["hts_tst"]
            ).fillna(0)

        else:
            raise ValueError(
                "Missing columns for hts positivity calculation"
            )

    df["care_gap_index"] = (
        (IIT_WEIGHT * df["iit_rate"])
        + (
            VLS_WEIGHT
            * (1 - df["vls_rate_adult"])
        )
        + (
            HTS_WEIGHT
            * df["hts_positivity_rate"]
        )
    ) * CGI_SCALE_MAX

    return df


def build_county_profiles(df):
    """
    Build county_profiles.csv
    using latest county records.
    """

    county_profiles = (
        df.sort_values("period")
        .groupby("county")
        .tail(1)
        .reset_index(drop=True)
    )

    county_profiles.to_csv(
        COUNTY_PROF,
        index=False
    )

    print(f"Saved county profiles to: {COUNTY_PROF}")

    return county_profiles


def build_tier_timeseries(df):
    """
    Build tier_timeseries.csv
    for Prophet forecasting.
    """

    tier_ts = (
        df.groupby(["tier", "period"])
        .agg({
            "iit_rate": "mean",
            "vls_rate_adult": "mean"
        })
        .reset_index()
    )

    tier_ts.to_csv(
        TIER_TS,
        index=False
    )

    print(f"Saved tier time series to: {TIER_TS}")

    return tier_ts


if __name__ == "__main__":

    print("feature_engineering.py loaded successfully")




