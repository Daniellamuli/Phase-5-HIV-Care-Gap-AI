"""
==============================================
HIV Care Gap AI — Feature Engineering Wrapper
==============================================
Loads nsdcc_clean.csv and runs feature engineering.

"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from constants import NSDCC_CLEAN, COUNTY_PROF
from src.feature_engineering import run_feature_engineering


def run_prepare():
    """Load data, run feature engineering, save county_profiles.csv."""
    print("=" * 50)
    print("Feature Engineering Pipeline")
    print("=" * 50)

    # Step 1: Load merged data
    print("\n[1] Loading merged NSDCC data...")
    df = pd.read_csv(NSDCC_CLEAN)
    print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Step 2: Run feature engineering
    print("\n[2] Running feature engineering...")
    df_features = run_feature_engineering(df, save=True)
    print(f"    Done. New shape: {df_features.shape[0]:,} rows x {df_features.shape[1]} columns")

    # Step 3: Save county profiles
    print(f"\n[3] County profiles saved to: {COUNTY_PROF}")

    print("\n" + "=" * 50)
    print("Feature Engineering Complete ✓")
    print("=" * 50)

    return df_features


if __name__ == "__main__":
    run_prepare()