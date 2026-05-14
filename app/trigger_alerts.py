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

    IIT_ALERT_FALLBACK_THRESHOLD,

    IIT_ALERTS
)

# ============================================================
# LOAD COUNTY PROFILES
# ============================================================

county_profiles = pd.read_csv(
    COUNTY_PROF
)

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_cols = [

    "county",

    "tier",

    "iit_rate"
]

missing_cols = [

    col for col in required_cols
    if col not in county_profiles.columns
]

if missing_cols:

    raise ValueError(

        f"Missing required columns: "
        f"{missing_cols}"
    )

# ============================================================
# COMPUTE NATIONAL AVERAGE IIT RATE
# ============================================================

national_avg_iit = (

    county_profiles["iit_rate"]
    .mean()
)

print(
    f"\nNational Average IIT Rate: "
    f"{national_avg_iit:.4f}"
)

# ============================================================
# FLAG ALERT COUNTIES
# ============================================================

alerts_df = county_profiles[

    (
        county_profiles["iit_rate"]
        > national_avg_iit
    )

    |

    (
        county_profiles["iit_rate"]
        > IIT_ALERT_FALLBACK_THRESHOLD
    )
].copy()

# ============================================================
# ADD ALERT REASONS
# ============================================================

alerts_df["alert_reason"] = np.where(

    alerts_df["iit_rate"]
    > IIT_ALERT_FALLBACK_THRESHOLD,

    "Above fallback threshold",

    "Above national average"
)

# ============================================================
# FINAL ALERT TABLE
# ============================================================

final_alerts = alerts_df[

    [
        "county",
        "tier",
        "iit_rate",
        "alert_reason"
    ]
].sort_values(

    "iit_rate",
    ascending=False
)

# ============================================================
# PRINT ALERTS
# ============================================================

print("\nIIT ALERT COUNTIES\n")

print(final_alerts)

# ============================================================
# SAVE ALERTS CSV
# ============================================================

final_alerts.to_csv(

    IIT_ALERTS,

    index=False
)

print(
    "\nIIT alerts CSV saved successfully"
)