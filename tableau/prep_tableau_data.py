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

# Add county coordinates for Tableau map
coords = {
    'Baringo':[0.4667,35.9667],'Bomet':[-0.7833,35.3333],'Bungoma':[0.5667,34.5667],
    'Busia':[0.4667,34.1167],'Elgeyo Marakwet':[0.8333,35.5833],'Embu':[-0.5333,37.45],
    'Garissa':[-0.45,39.65],'Homa Bay':[-0.5167,34.45],'Isiolo':[0.35,37.5833],
    'Kajiado':[-1.85,36.7833],'Kakamega':[0.2833,34.75],'Kericho':[-0.3667,35.2833],
    'Kiambu':[-1.1667,36.8333],'Kilifi':[-3.6333,39.85],'Kirinyaga':[-0.5,37.2833],
    'Kisii':[-0.6833,34.7667],'Kisumu':[-0.1,34.75],'Kitui':[-1.3667,38.0167],
    'Kwale':[-4.1667,39.45],'Laikipia':[0.1833,36.95],'Lamu':[-2.2667,40.9],
    'Machakos':[-1.5167,37.2667],'Makueni':[-2.2833,37.8333],'Mandera':[3.9333,41.8667],
    'Marsabit':[2.3333,37.9833],'Meru':[0.05,37.65],'Migori':[-1.0667,34.4667],
    'Mombasa':[-4.05,39.6667],"Murang'a":[-0.7167,37.15],'Nairobi':[-1.2833,36.8167],
    'Nakuru':[-0.3,36.0667],'Nandi':[0.1667,35.1167],'Narok':[-1.0833,35.8667],
    'Nyamira':[-0.5667,34.9333],'Nyandarua':[-0.1667,36.6333],'Nyeri':[-0.4167,36.95],
    'Samburu':[1.1667,36.6667],'Siaya':[0.0667,34.2833],'Taita Taveta':[-3.4,38.3667],
    'Tana River':[-1.7333,39.6667],'Tharaka Nithi':[-0.3,37.95],'Trans Nzoia':[1.0167,34.9667],
    'Turkana':[3.3167,35.5667],'Uasin Gishu':[0.5167,35.2833],'Vihiga':[0.0833,34.7167],
    'Wajir':[1.75,40.0667],'West Pokot':[1.1667,35.1167]
}
cp['Latitude'] = cp['county'].map(lambda x: coords.get(x, [None,None])[0])
cp['Longitude'] = cp['county'].map(lambda x: coords.get(x, [None,None])[1])
cp.to_csv(f"{PROCESSED}/county_profiles_tableau.csv", index=False)
print("  Added Latitude/Longitude columns to county_profiles_tableau.csv")