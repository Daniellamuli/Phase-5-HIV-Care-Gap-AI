"""
Run this script ONCE before opening Tableau.
It prepares all 3 CSV files you need.
Place this file in your project root folder (same level as constants.py).
Run: python prep_tableau_data.py
"""

import json
import pandas as pd
import os

PROCESSED = "data/processed"

print("Step 1: Converting odds_ratios_with_ci.json → odds_ratios_flat.csv ...")
with open(f"{PROCESSED}/odds_ratios_with_ci.json") as f:
    d = json.load(f)
odds_df = pd.DataFrame(d["features"])
odds_df["direction"] = odds_df["Odds_Ratio"].apply(lambda x: "Risk" if x > 1 else "Protective")
odds_df["OR_label"] = odds_df["Odds_Ratio"].round(3).astype(str)
odds_df.to_csv(f"{PROCESSED}/odds_ratios_flat.csv", index=False)
print(f"  Saved odds_ratios_flat.csv — {len(odds_df)} rows")

print("Step 2: Stacking 4 tier forecasts → forecast_all_tiers.csv ...")
dfs = []
for tier in ["critical", "high", "moderate", "low"]:
    path = f"{PROCESSED}/forecast_{tier}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["tier"] = tier.capitalize()
        dfs.append(df)
    else:
        print(f"  WARNING: {path} not found, skipping")
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined["iit_pct"] = (combined["iit_rate"] * 100).round(4)
    combined["vls_pct"] = (combined["vls_rate_adult"] * 100).round(4)
    combined.to_csv(f"{PROCESSED}/forecast_all_tiers.csv", index=False)
    print(f"  Saved forecast_all_tiers.csv — {len(combined)} rows")

print("Step 3: Adding % columns to county_profiles.csv → county_profiles_tableau.csv ...")
cp = pd.read_csv(f"{PROCESSED}/county_profiles.csv")
cp["iit_pct"] = (cp["iit_rate"] * 100).round(2)
cp["vls_pct"] = (cp["vls_rate_adult"] * 100).round(2)
cp["hts_pos_pct"] = (cp["hts_positivity_rate"] * 100).round(2)
cp["tier_rank"] = cp["tier"].map({"Critical": 1, "High": 2, "Moderate": 3, "Low": 4})
cp.to_csv(f"{PROCESSED}/county_profiles_tableau.csv", index=False)
print(f"  Saved county_profiles_tableau.csv — {len(cp)} rows")

print("\nAll done. Your 3 Tableau-ready files are in data/processed/:")
print("  county_profiles_tableau.csv")
print("  odds_ratios_flat.csv")
print("  forecast_all_tiers.csv")