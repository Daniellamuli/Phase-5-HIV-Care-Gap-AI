import warnings
warnings.filterwarnings("ignore")

import sys
import os

import pandas as pd
import numpy as np

# ============================================================
# ROOT IMPORTS
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from constants import (
    COUNTY_PROF,                   
    IIT_ALERTS,                    
    IIT_ALERT_FALLBACK_THRESHOLD,  
    CGI_IIT_COL,                   
    CGI_VLS_COL,                   
    CGI_HTS_COL,                  
    CLEAN_COUNTY_COL,              
    TIER_LABELS,                   
    BRIDGED_TIERS,                 
)

# ============================================================
# LOAD COUNTY PROFILES
# Never hardcode paths — COUNTY_PROF from constants.py
# ============================================================

county_profiles = pd.read_csv(COUNTY_PROF)

print(
    f"Loaded county_profiles.csv: "
    f"{county_profiles.shape[0]} counties"
)

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

REQUIRED_COLS = [
    CLEAN_COUNTY_COL,   
    "tier",
    CGI_IIT_COL,       
    CGI_VLS_COL,        
    CGI_HTS_COL,        
    "care_gap_index",
]

missing_cols = [
    col for col in REQUIRED_COLS
    if col not in county_profiles.columns
]

if missing_cols:
    raise ValueError(
        f"Missing required columns: {missing_cols}\n"
        f"Re-run notebook 04 (feature_engineering) "
        f"to rebuild county_profiles.csv."
    )

# ============================================================
# COMPUTE NATIONAL AVERAGE IIT RATE
# ============================================================

national_avg_iit = (
    county_profiles[CGI_IIT_COL]
    .mean()
)

print(
    f"\nNational Average IIT Rate : "
    f"{national_avg_iit:.4f} "
    f"({national_avg_iit * 100:.2f}%)"
)

print(
    f"Alert Fallback Threshold  : "
    f"{IIT_ALERT_FALLBACK_THRESHOLD:.4f} "
    f"({IIT_ALERT_FALLBACK_THRESHOLD * 100:.0f}%)"
)

# ============================================================
# FLAG ALERT COUNTIES
# Two conditions — either triggers an alert:
#   1. iit_rate > national average  (relative comparison)
#   2. iit_rate > 0.15 fallback     (absolute threshold)
# ============================================================

above_national = (
    county_profiles[CGI_IIT_COL]
    > national_avg_iit
)

above_threshold = (
    county_profiles[CGI_IIT_COL]
    > IIT_ALERT_FALLBACK_THRESHOLD
)

alerts_df = county_profiles[
    above_national | above_threshold
].copy()

# ============================================================
# ADD ALERT REASON
# Threshold breach labelled first — more severe condition
# ============================================================

alerts_df["alert_reason"] = np.where(
    alerts_df[CGI_IIT_COL]
    > IIT_ALERT_FALLBACK_THRESHOLD,
    f"Above fallback threshold "
    f"({IIT_ALERT_FALLBACK_THRESHOLD * 100:.0f}%)",
    f"Above national average "
    f"({national_avg_iit * 100:.2f}%)"
)

# ============================================================
# ADD ALERT SEVERITY
# Maps to BRIDGED_TIERS from constants —
# same tiers that receive intervention in Model 3
# Critical + High = IMMEDIATE ACTION
# Moderate + Low  = MONITOR
# ============================================================

alerts_df["severity"] = np.where(
    alerts_df["tier"].isin(BRIDGED_TIERS),
    "IMMEDIATE ACTION",
    "MONITOR"
)

# ============================================================
# FINAL ALERT TABLE
# Sorted by iit_rate descending — worst county first
# ============================================================

final_alerts = alerts_df[
    [
        CLEAN_COUNTY_COL, 
        "tier",
        "severity",
        CGI_IIT_COL,        
        CGI_VLS_COL,        
        CGI_HTS_COL,        
        "care_gap_index",
        "alert_reason",
    ]
].sort_values(
    CGI_IIT_COL,
    ascending=False
).reset_index(drop=True)

# ============================================================
# PRINT ALERT SUMMARY
# ============================================================

total     = len(final_alerts)
immediate = len(
    final_alerts[
        final_alerts["severity"] == "IMMEDIATE ACTION"
    ]
)
monitor   = len(
    final_alerts[
        final_alerts["severity"] == "MONITOR"
    ]
)

print(f"\nIIT ALERT COUNTIES\n")
print(
    f"Total flagged    : {total} / "
    f"{len(county_profiles)}"
)
print(f"Immediate action : {immediate}")
print(f"Monitor          : {monitor}")

print("\n--- IMMEDIATE ACTION ---")
print(
    final_alerts[
        final_alerts["severity"] == "IMMEDIATE ACTION"
    ][[
        CLEAN_COUNTY_COL,
        "tier",
        CGI_IIT_COL,
        "care_gap_index",
        "alert_reason",
    ]].to_string(index=False)
)

print("\n--- MONITOR ---")
print(
    final_alerts[
        final_alerts["severity"] == "MONITOR"
    ][[
        CLEAN_COUNTY_COL,
        "tier",
        CGI_IIT_COL,
        "care_gap_index",
        "alert_reason",
    ]].to_string(index=False)
)

# ============================================================
# TIER SUMMARY
# ============================================================

print("\n--- FLAGGED COUNTIES BY TIER ---")

tier_summary = (
    final_alerts
    .groupby("tier")
    .agg(
        counties_flagged=(CLEAN_COUNTY_COL, "count"),
        mean_iit_rate   =(CGI_IIT_COL,     "mean"),
        max_iit_rate    =(CGI_IIT_COL,     "max"),
    )
    .reindex(TIER_LABELS)
    .dropna()
    .round(4)
    .reset_index()
)

print(tier_summary.to_string(index=False))

# ============================================================
# SAVE ALERTS CSV
# Uses IIT_ALERTS from constants
# Saves to data/processed/iit_alerts.csv
# ============================================================

final_alerts.to_csv(
    IIT_ALERTS,
    index=False
)

print(
    f"\nIIT alerts saved → {IIT_ALERTS}"
)

# ============================================================
# VALIDATION
# ============================================================

assert os.path.exists(IIT_ALERTS), (
    f"iit_alerts.csv not found at {IIT_ALERTS}"
)

assert len(final_alerts) > 0, (
    "No counties flagged — check iit_rate in "
    "county_profiles.csv"
)

assert final_alerts[CGI_IIT_COL].max() <= 1.0, (
    "iit_rate > 1.0 detected — "
    "re-run notebook 04 to fix"
)

print(f"\nAll validation checks passed")
print(f"Counties flagged : {total}")
print(f"Immediate action : {immediate}")
print(f"Monitor          : {monitor}")

