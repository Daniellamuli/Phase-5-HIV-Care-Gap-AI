
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
    group_col="County",
    year_col="Period"
):

    df = df.copy()

    df = df.sort_values([group_col, year_col])

    df[f"{value_col}_yoy_change"] = (
        df.groupby(group_col)[value_col]
        .pct_change()
        .fillna(0)
    )

    return df


def calculate_care_gap_index(df):

    df = df.copy()

    df["care_gap_index"] = (
        (IIT_WEIGHT * df["iit_rate"])
        + (VLS_WEIGHT * (1 - df["vls_rate"]))
        + (HTS_WEIGHT * df["hts_positivity"])
    ) * CGI_SCALE_MAX

    return df


def build_county_profiles(df):

    county_profiles = (
        df.sort_values("Period")
        .groupby("County")
        .tail(1)
        .reset_index(drop=True)
    )

    county_profiles.to_csv(COUNTY_PROF, index=False)

    return county_profiles


def build_tier_timeseries(df):

    tier_ts = (
        df.groupby(["tier", "Period"])
        .agg({
            "iit_rate": "mean",
            "vls_rate": "mean"
        })
        .reset_index()
    )

    tier_ts.to_csv(TIER_TS, index=False)

    return tier_ts


if __name__ == "__main__":

    print("feature_engineering.py loaded successfully")