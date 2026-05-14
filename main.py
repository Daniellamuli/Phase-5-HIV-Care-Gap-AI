"""
main.py — HIV Care Gap AI Pipeline
====================================
Entry point for the full end-to-end pipeline.
Run this file to execute all steps in sequence.

Usage:
    python main.py                     # Run full pipeline
    python main.py --step extract      # Step 1 only
    python main.py --step clean_nsdcc  # Step 2a only
    python main.py --step clean_dhs    # Step 2b only
    python main.py --step merge        # Step 3 only
    python main.py --step features     # Step 4 only
    python main.py --step model1       # Step 5 only
    python main.py --step model2       # Step 6 only
    python main.py --step model3       # Step 7 only
    python main.py --step evaluate     # Step 8 only
    python main.py --dry-run           # Validate all paths, no processing

NOTE — constants.py gap (fix before running):
    The following constants are used by notebook 02 and this file
    but are NOT yet defined in constants.py. Add them before running:

        HTS_POS_COUNTY_COL    = "County"
        HTS_POS_PERIOD_COL    = "Period"
        HTS_POS_COUNTY_SUFFIX = " County"
        HTS_POS_CLEAN         = os.path.join(PROCESSED_DIR, "hts_positive_clean.csv")
        HTS_POSITIVE_RENAME   = {
            "Total":        "hts_positive",
            "Total_Males":  "hts_positive_males",
            "Total_Females":"hts_positive_females",
        }

Team: Daniella · Eve · Verah · Naomi · Lorenah · Dennis
Methodology: CRISP-DM
"""

import argparse
import sys
import os
import time

# ── Make repo root importable ──────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from constants import (
    # ── Raw file paths
    ADULT_ART_FILE, HTS_FILE, HTS_POSITIVE_FILE,
    VLT_FILE, IIT_FILE, DHS_REDUCED,
    # ── Processed file paths
    ART_CLEAN, HTS_CLEAN, VLT_CLEAN, IIT_CLEAN,
    DHS_CLEAN, NSDCC_CLEAN, COUNTY_PROF, TIER_TS,
    # ── Column constants — ART
    ART_COUNTY_COL, ART_PERIOD_COL, ART_COUNTY_SUFFIX,
    # ── Column constants — HTS
    HTS_COUNTY_COL, HTS_PERIOD_COL, HTS_COUNTY_SUFFIX,
    # ── Column constants — VLT and IIT
    VLT_COUNTY_COL, IIT_REGION_COL,
    # ── Rename maps
    ART_RENAME, HTS_RENAME, VLT_RENAME, IIT_RENAME,
    # ── Region map
    IIT_REGION_MAP,
    # ── Model file paths
    KMEANS_MODEL, XGBOOST_MODEL,
    # ── Forecast file paths
    FORECAST_CRITICAL, FORECAST_HIGH,
    FORECAST_MODERATE, FORECAST_LOW, FORECAST_NATIONAL,
    # ── Model parameters
    KMEANS_K, KMEANS_RANDOM_STATE, KMEANS_FEATURES,
    TIER_LABELS, TIER_COLORS,
    MODEL2_FEATURES, MODEL2_TARGET,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SCALE_POS_WEIGHT, TEST_SIZE, RANDOM_STATE,
    # ── Care Gap Index weights
    IIT_WEIGHT, VLS_WEIGHT, HTS_WEIGHT, CGI_SCALE_MAX,
    # ── Scenario projection parameters
    FORECAST_YEAR_END, IIT_REDUCTION_RATE,
    BRIDGED_TIERS, BRIDGED_START_YEAR,
    # ── Directories
    PROCESSED_DIR, MODELS_DIR,
)

# HTS_Positive constants — imported separately so that a missing-constant
# error here is caught with a clear message rather than crashing the whole
# import block above.
try:
    from constants import (
        HTS_POS_COUNTY_COL,
        HTS_POS_PERIOD_COL,
        HTS_POS_COUNTY_SUFFIX,
        HTS_POS_CLEAN,
        HTS_POSITIVE_RENAME,
    )
    _HTS_POS_CONSTANTS_OK = True
except ImportError:
    _HTS_POS_CONSTANTS_OK = False
    print(
        "\nWARNING: HTS_Positive constants missing from constants.py.\n"
        "         Steps 2a and 3 (NSDCC cleaning and merging) will fail.\n"
        "         Add HTS_POS_COUNTY_COL, HTS_POS_PERIOD_COL,\n"
        "         HTS_POS_COUNTY_SUFFIX, HTS_POS_CLEAN, HTS_POSITIVE_RENAME\n"
        "         to constants.py before running the full pipeline.\n"
    )


# ══════════════════════════════════════════════════════════════════════════
# STEP 0 — DRY RUN: validate all required paths exist
# ══════════════════════════════════════════════════════════════════════════

def dry_run() -> bool:
    """
    Check that every raw input file exists on disk.
    Prints a status line per file. Returns True if all present.
    Does not execute any processing.
    """
    print("\n[DRY RUN] Validating required file paths")
    print("─" * 52)

    required = {
        "ART raw":              ADULT_ART_FILE,
        "HTS raw":              HTS_FILE,
        "HTS_Positive raw":     HTS_POSITIVE_FILE,
        "VLT raw":              VLT_FILE,
        "IIT raw":              IIT_FILE,
        "DHS individual CSV":   DHS_REDUCED,
    }

    all_ok = True
    for label, path in required.items():
        exists = os.path.exists(path)
        status = "OK    " if exists else "MISSING"
        print(f"  {status}  {label:<25}  {path}")
        if not exists:
            all_ok = False

    print()
    if all_ok:
        print("  All raw files present. Pipeline is ready to run.")
    else:
        print("  One or more files missing. Copy them to data/raw/ before running.")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA EXTRACTION
# Owner: Naomi  |  scripts/extract_data.py  |  notebook 01
# ══════════════════════════════════════════════════════════════════════════

def run_extraction() -> dict:
    """
    Load all 6 raw files exactly as downloaded.
    Read-only — saves nothing to disk.

    Validates:
      - All county name maps cover every value in the actual files
      - All region maps cover every IIT region row
      - DHS county codes 1-47 all resolve

    Returns:
        dict with keys: art, hts, hts_pos, vlt, iit, dhs
    """
    print("\n[STEP 1] Data Extraction")
    print("─" * 52)

    from scripts.extract_data import extract_all
    raw_data = extract_all()
    return raw_data


# ══════════════════════════════════════════════════════════════════════════
# STEP 2a — NSDCC DATA CLEANING
# Owner: Eve  |  src/nsdcc_cleaner.py  |  notebook 02
#
# Confirmed clean file structures (47 rows each, period = 2025):
#   ART:          county, period, adults_on_art, art_total_males, art_total_females
#   HTS:          county, period, hts_tested, hts_tested_males, hts_tested_females
#   HTS_Positive: county, period, hts_positive, hts_positive_males, hts_positive_females
#   VLT:          county, vlt_valid_under15, vls_suppressed_under15,
#                         vlt_valid_male_15plus, vls_suppressed_male_15plus,
#                         vlt_valid_female_15plus, vls_suppressed_female_15plus,
#                         vls_rate_adult, vls_rate_male15plus,
#                         vls_rate_female15plus, vls_rate_under15
#   IIT:          county, adults_on_treatment, iit_count, iit_rate_pct,
#                         iit_children, iit_rate_children_pct,
#                         iit_male, iit_rate_male_pct,
#                         iit_female, iit_rate_female_pct
# ══════════════════════════════════════════════════════════════════════════

def run_nsdcc_cleaning() -> None:
    """
    Clean all 5 NSDCC raw files and save to data/processed/.

    Operations per file:
      ART / HTS / HTS_Positive:
        strip ' County' suffix → standardise county names →
        rename MOH cols → to_numeric_impute (county mean → median fallback)
      VLT:
        standardise county names → rename cols →
        to_numeric_impute → engineer VLS rates (vls_rate_adult etc.)
      IIT:
        drop Kenya national total row → rename cols →
        expand 9 regions → 47 counties (equal-share count split) →
        impute rate cols with mean, count cols with 0

    Reads:  ADULT_ART_FILE, HTS_FILE, HTS_POSITIVE_FILE,
            VLT_FILE, IIT_FILE
    Writes: ART_CLEAN, HTS_CLEAN, HTS_POS_CLEAN,
            VLT_CLEAN, IIT_CLEAN
    """
    print("\n[STEP 2a] NSDCC Data Cleaning")
    print("─" * 52)

    if not _HTS_POS_CONSTANTS_OK:
        print(
            "  SKIPPED — HTS_Positive constants missing from constants.py.\n"
            "  Add them and re-run."
        )
        return

    import numpy as np
    from src.nsdcc_cleaner import (
        load_with_period,
        load_snapshot,
        load_iit,
        strip_suffix,
        standardise,
        to_numeric_impute,
        fix_vlt_missing,
        expand_iit_regions,
        fix_iit_missing,
        validate,
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── ART ──────────────────────────────────────────────────────────────
    art = load_with_period(ADULT_ART_FILE, ART_PERIOD_COL, "ART")
    art = strip_suffix(art, ART_COUNTY_COL, ART_COUNTY_SUFFIX, "ART")
    art = standardise(art, ART_COUNTY_COL, "ART")
    art = art.rename(columns=ART_RENAME)
    art = to_numeric_impute(art, ART_COUNTY_COL, ART_PERIOD_COL)
    art = art.rename(columns={ART_COUNTY_COL: "county", ART_PERIOD_COL: "period"})
    validate(art, "ART", county_col="county", period_col="period")
    art.to_csv(ART_CLEAN, index=False)
    print(f"  Saved → {ART_CLEAN}")

    # ── HTS ──────────────────────────────────────────────────────────────
    hts = load_with_period(HTS_FILE, HTS_PERIOD_COL, "HTS")
    hts = strip_suffix(hts, HTS_COUNTY_COL, HTS_COUNTY_SUFFIX, "HTS")
    hts = standardise(hts, HTS_COUNTY_COL, "HTS")
    hts = hts.rename(columns=HTS_RENAME)
    hts = to_numeric_impute(hts, HTS_COUNTY_COL, HTS_PERIOD_COL)
    hts = hts.rename(columns={HTS_COUNTY_COL: "county", HTS_PERIOD_COL: "period"})
    validate(hts, "HTS", county_col="county", period_col="period")
    hts.to_csv(HTS_CLEAN, index=False)
    print(f"  Saved → {HTS_CLEAN}")

    # ── HTS_Positive ─────────────────────────────────────────────────────
    hts_pos = load_with_period(HTS_POSITIVE_FILE, HTS_POS_PERIOD_COL, "HTS_Positive")
    hts_pos = strip_suffix(hts_pos, HTS_POS_COUNTY_COL, HTS_POS_COUNTY_SUFFIX, "HTS_Positive")
    hts_pos = standardise(hts_pos, HTS_POS_COUNTY_COL, "HTS_Positive")
    hts_pos = hts_pos.rename(columns=HTS_POSITIVE_RENAME)
    hts_pos = to_numeric_impute(hts_pos, HTS_POS_COUNTY_COL, HTS_POS_PERIOD_COL)
    hts_pos = hts_pos.rename(
        columns={HTS_POS_COUNTY_COL: "county", HTS_POS_PERIOD_COL: "period"}
    )
    validate(hts_pos, "HTS_Positive", county_col="county", period_col="period")
    hts_pos.to_csv(HTS_POS_CLEAN, index=False)
    print(f"  Saved → {HTS_POS_CLEAN}")

    # ── VLT ──────────────────────────────────────────────────────────────
    # VLT has no Period column — single county-level snapshot.
    # VLS rates are engineered here from raw suppressed / valid counts.
    vlt = load_snapshot(VLT_FILE, VLT_COUNTY_COL, "VLT")
    vlt = standardise(vlt, VLT_COUNTY_COL, "VLT")
    vlt = vlt.rename(columns=VLT_RENAME)
    vlt = to_numeric_impute(vlt, VLT_COUNTY_COL)

    # Engineer VLS suppression rates from raw counts
    vlt["vls_rate_male15plus"]   = (
        vlt["vls_suppressed_male_15plus"] / vlt["vlt_valid_male_15plus"]
    ).clip(0, 1).round(4)
    vlt["vls_rate_female15plus"] = (
        vlt["vls_suppressed_female_15plus"] / vlt["vlt_valid_female_15plus"]
    ).clip(0, 1).round(4)
    vlt["vls_rate_under15"]      = (
        vlt["vls_suppressed_under15"] / vlt["vlt_valid_under15"]
    ).clip(0, 1).round(4)
    vlt["vls_rate_adult"]        = (
        (vlt["vls_suppressed_male_15plus"] + vlt["vls_suppressed_female_15plus"])
        / (vlt["vlt_valid_male_15plus"] + vlt["vlt_valid_female_15plus"])
    ).clip(0, 1).round(4)

    # Fix any NaN introduced by division (zero denominators in small counties)
    vlt = fix_vlt_missing(vlt)
    vlt = vlt.rename(columns={VLT_COUNTY_COL: "county"})
    validate(vlt, "VLT", county_col="county")
    vlt.to_csv(VLT_CLEAN, index=False)
    print(f"  Saved → {VLT_CLEAN}")

    # ── IIT ──────────────────────────────────────────────────────────────
    # IIT is region-level (9 rows). Drop national total, then expand to
    # 47 counties using REGION_TO_COUNTIES from constants.py.
    # Rate/pct columns are copied as-is (apply region-wide).
    # Count columns are divided equally across counties in the region.
    iit_raw = load_iit(IIT_FILE, IIT_REGION_COL, "IIT")
    iit_raw["_std"] = iit_raw[IIT_REGION_COL].map(IIT_REGION_MAP)
    iit_raw = iit_raw[iit_raw["_std"].notna()].drop(columns=["_std"])
    iit_raw = iit_raw.rename(columns=IIT_RENAME)

    # Convert to numeric before expansion
    data_cols = [c for c in iit_raw.columns if c != IIT_REGION_COL]
    for col in data_cols:
        iit_raw[col] = pd.to_numeric(iit_raw[col], errors="coerce")

    iit = expand_iit_regions(iit_raw, region_col=IIT_REGION_COL)
    iit = fix_iit_missing(iit)
    validate(iit, "IIT", county_col="county")
    iit.to_csv(IIT_CLEAN, index=False)
    print(f"  Saved → {IIT_CLEAN}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2b — DHS DATA CLEANING
# Owner: Lorenah  |  src/dhs_cleaner.py  |  notebook 03
#
# Output: individual_features_clean.csv
#   Class balance: 32,130 retained (99.92%) / 26 dropout (0.08%)
#   XGB_SCALE_POS_WEIGHT = 1236 (set in constants.py)
#
#   Columns dropped (100% missing):
#     - knows_aids_death
#     - has_health_insurance (imputed to 0, no signal)
#   Column absent from dataset:
#     - anc_visits (does not exist — not imputed)
# ══════════════════════════════════════════════════════════════════════════

def run_dhs_cleaning() -> None:
    """
    Clean DHS individual features and save to data/processed/.

    Reads:  DHS_REDUCED  (data/raw/individual_features.csv)
    Writes: DHS_CLEAN    (data/processed/individual_features_clean.csv)
    """
    print("\n[STEP 2b] DHS Data Cleaning")
    print("─" * 52)

    import constants as c
    from src.dhs_cleaner import DHSCleaner

    cleaner = DHSCleaner(c)
    cleaner.load_data(DHS_REDUCED)

    # Detect county column — raw file uses lowercase 'county'
    county_col = next(
        (col for col in cleaner.raw_df.columns if col.lower() in ["county", "hv024"]),
        "county",
    )
    cleaner.decode_county(county_col=county_col)
    cleaner.decode_demographics()
    cleaner.impute_binary_flags()
    # anc_visits does not exist in this dataset — only impute num_sexual_partners
    cleaner.impute_numeric_columns(numeric_cols=["num_sexual_partners"])
    cleaner.engineer_dropout_target()
    cleaner.one_hot_encode(["education_level", "wealth_index"])
    cleaner.save_clean_data(DHS_CLEAN)
    print(f"  Saved → {DHS_CLEAN}")

    ratios = cleaner.get_class_ratio()
    print(f"  Dropout class balance: {ratios}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — DATA MERGING
# Owner: Eve  |  scripts/merge_data.py
#
# Merge strategy (confirmed from merge_data.py):
#   ART + HTS + HTS_Positive : inner join on ['county', 'period']
#   + VLT                    : left join on ['county'] only
#   + IIT                    : left join on ['county'] only
#   Period normalisation     : 4-digit year extracted from "December 2025"
#   Expected output          : 47 rows, 0 missing values
# ══════════════════════════════════════════════════════════════════════════

def run_merge() -> None:
    """
    Merge all 5 cleaned NSDCC files into one county-level feature table.

    Reads:  ART_CLEAN, HTS_CLEAN, HTS_POS_CLEAN, VLT_CLEAN, IIT_CLEAN
    Writes: NSDCC_CLEAN  (data/processed/nsdcc_clean.csv)
    """
    print("\n[STEP 3] Data Merging → nsdcc_clean.csv")
    print("─" * 52)

    from scripts.merge_data import run_merge as _merge
    merged = _merge(save=True)
    print(f"  Merged shape: {merged.shape}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING
# Owner: Verah  |  src/feature_engineering.py  |  notebook 04
#
# Confirmed outputs in county_profiles.csv:
#   - iit_rate           : alias for iit_rate_pct (added here for Model 1)
#   - hts_positivity_rate: hts_positive / hts_tested
#   - care_gap_index     : (IIT_WEIGHT × iit_rate)
#                          + (VLS_WEIGHT × (1 − vls_rate_adult))
#                          + (HTS_WEIGHT × hts_positivity_rate) × 100
#   - art_coverage       : placeholder 0.5 (PLHIV estimates unavailable)
#   - iit_rate_yoy_change: 0 (single year — no prior period)
#   - vls_rate_adult_yoy_change: 0 (same reason)
# ══════════════════════════════════════════════════════════════════════════

def run_feature_engineering() -> None:
    """
    Build all features required by Models 1, 2, and 3.

    Reads:  NSDCC_CLEAN
    Writes: COUNTY_PROF (data/processed/county_profiles.csv)
    """
    print("\n[STEP 4] Feature Engineering")
    print("─" * 52)

    import pandas as pd
    from src.feature_engineering import run_feature_engineering as _fe

    df = pd.read_csv(NSDCC_CLEAN)
    print(f"  Loaded {NSDCC_CLEAN}: {df.shape}")

    # Add iit_rate alias so that KMEANS_FEATURES = ['iit_rate', 'vls_rate_adult']
    # works consistently in notebook 05 and main.py Step 5.
    # iit_rate_pct is the raw IIT percentage from IIT_RENAME.
    if "iit_rate" not in df.columns and "iit_rate_pct" in df.columns:
        import pandas as _pd
        df["iit_rate"] = _pd.to_numeric(df["iit_rate_pct"], errors="coerce")
        print("  Added iit_rate alias from iit_rate_pct")

    df_features = _fe(df, save=True)
    print(f"  Feature engineering complete. Shape: {df_features.shape}")
    print(f"  Saved → {COUNTY_PROF}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — MODEL 1: COUNTY CLUSTERING (KMeans)
# Owner: Naomi  |  notebook 05
#
# Confirmed from notebook 05:
#   Features used : ['iit_rate', 'vls_rate_adult']  ← KMEANS_FEATURES
#   k             : 4 (project decision — 4 named tiers)
#   Silhouette    : 0.6359 (k=4), note: k=3 scores 0.6582 but 4 tiers required
#   Tier counts   : Critical (14), High (25), Moderate (7), Low (1)
#   Tier ranking  : by mean iit_rate descending (highest IIT = Critical)
# ══════════════════════════════════════════════════════════════════════════

def run_model1() -> None:
    """
    Model 1 — County Care Gap Map.

    Algorithm : KMeans (k=4) on KMEANS_FEATURES = ['iit_rate', 'vls_rate_adult']
    Output:
      - Tier label per county: Critical / High / Moderate / Low
      - Ranked bar chart of all 47 counties by Care Gap Index
      - Silhouette score validation
      - Saves: KMEANS_MODEL (models/kmeans_county_tiers.pkl)
      - Updates county_profiles.csv with cluster and tier columns

    Reads:  COUNTY_PROF
    Writes: KMEANS_MODEL, updated COUNTY_PROF
    """
    print("\n[STEP 5] Model 1 — County Clustering (KMeans)")
    print("─" * 52)

    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df = pd.read_csv(COUNTY_PROF)
    print(f"  Loaded {COUNTY_PROF}: {df.shape}")

    # Validate KMEANS_FEATURES are present
    # KMEANS_FEATURES = ['iit_rate', 'vls_rate_adult'] per constants.py
    missing_cols = [f for f in KMEANS_FEATURES if f not in df.columns]
    if missing_cols:
        raise ValueError(
            f"KMEANS_FEATURES missing from county_profiles.csv: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}\n"
            f"Ensure Step 4 completed successfully."
        )

    missing_vals = df[KMEANS_FEATURES].isnull().sum().sum()
    if missing_vals > 0:
        raise ValueError(
            f"NaN values in KMEANS_FEATURES — impute before clustering.\n"
            f"{df[KMEANS_FEATURES].isnull().sum().to_dict()}"
        )

    # Scale features
    X = df[KMEANS_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Features: {KMEANS_FEATURES}")
    print(f"  Feature matrix shape: {X_scaled.shape}")

    # Fit KMeans
    km = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    print(f"  KMeans fitted | inertia: {km.inertia_:.2f}")

    # Assign tier labels ranked by mean iit_rate descending
    # Highest IIT rate cluster → Critical, lowest → Low
    cluster_iit_mean = (
        df.groupby("cluster")["iit_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    cluster_to_tier = {
        cluster_id: TIER_LABELS[rank]
        for rank, cluster_id in enumerate(cluster_iit_mean.index)
    }
    df["tier"] = df["cluster"].map(cluster_to_tier)

    print("\n  Cluster → Tier mapping:")
    for cluster_id, tier in cluster_to_tier.items():
        mean_iit = cluster_iit_mean[cluster_id]
        n        = (df["cluster"] == cluster_id).sum()
        print(f"    Cluster {cluster_id} → {tier:<10} | mean IIT: {mean_iit:.4f} | counties: {n}")

    # Silhouette score validation
    sil_score = silhouette_score(X_scaled, km.labels_)
    print(f"\n  Silhouette score (k={KMEANS_K}): {sil_score:.4f}")
    if sil_score > 0.5:
        print("  Interpretation: Strong cluster structure — tiers are well-separated.")
    elif sil_score > 0.25:
        print("  Interpretation: Fair structure — some overlap between adjacent tiers.")
    else:
        print("  Interpretation: Weak structure — consider re-scaling or reviewing features.")

    # Tier breakdown
    print("\n  Tier breakdown:")
    for tier in TIER_LABELS:
        n        = len(df[df["tier"] == tier])
        counties = df[df["tier"] == tier]["county"].tolist()
        print(f"    {tier:<10}: {n} counties")
        print(f"               {counties}")

    # Save updated county_profiles.csv
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(COUNTY_PROF, index=False)
    print(f"\n  Updated → {COUNTY_PROF}")

    # Save model bundle: KMeans + scaler + metadata
    import pickle
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_bundle = {
        "kmeans":           km,
        "scaler":           scaler,
        "features":         KMEANS_FEATURES,
        "cluster_to_tier":  cluster_to_tier,
        "silhouette_score": sil_score,
        "n_clusters":       KMEANS_K,
    }
    with open(KMEANS_MODEL, "wb") as f:
        pickle.dump(model_bundle, f)
    size_kb = os.path.getsize(KMEANS_MODEL) / 1024
    print(f"  Saved model → {KMEANS_MODEL} ({size_kb:.1f} KB)")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL 2: DROPOUT RISK FACTOR ANALYSIS (Logistic Regression)
# Owner: Lorenah  |  notebook 06
#
# Approach: Risk factor identification — NOT outcome prediction.
# Language: "associated with dropout risk", not "predicts dropout".
# Primary output: odds ratios from logistic regression coefficients.
# Justification: only 26 dropout cases in 32,156 records (0.08%).
#   Odds ratios from balanced logistic regression are interpretable
#   and policy-relevant. XGBoost runs as secondary output.
# ══════════════════════════════════════════════════════════════════════════

def run_model2() -> None:
    """
    Model 2 — Individual Dropout Risk Factor Analysis.

    Algorithm : Logistic Regression (primary, balanced class weights)
                XGBoost (secondary, scale_pos_weight=1236)
    Output:
      - Odds ratios per feature (forest plot)
      - AUC-ROC, Recall, Precision, F1
      - Confusion matrix and ROC curve figures
      - Saves: XGBOOST_MODEL, logreg_baseline.json

    Reads:  DHS_CLEAN
    Writes: XGBOOST_MODEL, data/processed/logreg_baseline.json
    """
    print("\n[STEP 6] Model 2 — Dropout Risk Factor Analysis")
    print("─" * 52)

    import pandas as pd
    import json
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score, recall_score,
        precision_score, f1_score,
        confusion_matrix,
    )

    df = pd.read_csv(DHS_CLEAN)
    print(f"  Loaded {DHS_CLEAN}: {df.shape}")

    # Select features that actually exist post-cleaning
    available_features = [col for col in MODEL2_FEATURES if col in df.columns]
    missing_features   = [col for col in MODEL2_FEATURES if col not in df.columns]
    if missing_features:
        print(f"  NOTE: {len(missing_features)} MODEL2_FEATURES not in clean data "
              f"(expected if 100% missing were dropped): {missing_features}")

    X = df[available_features].copy()
    y = df[MODEL2_TARGET].copy()

    # Encode any remaining object/categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    if categorical_cols:
        print(f"  Label-encoded {len(categorical_cols)} categorical columns: {categorical_cols}")

    # Train / test split — stratified to preserve 0.08% dropout class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Dropout in test set: {y_test.sum()} positive cases")

    # ── Logistic Regression (primary: odds ratios) ────────────────────
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="liblinear",
    )
    lr_model.fit(X_train, y_train)

    y_pred      = lr_model.predict(X_test)
    y_pred_prob = lr_model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_pred_prob)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  Logistic Regression — Risk Factor Analysis:")
    print(f"    AUC-ROC  : {auc_score:.4f}")
    print(f"    Recall   : {recall:.4f}   ← primary metric (minimise missed cases)")
    print(f"    Precision: {precision:.4f}")
    print(f"    F1       : {f1:.4f}")

    # Odds ratios from logistic regression coefficients
    import numpy as np
    odds_ratios = np.exp(lr_model.coef_[0])
    or_df = (
        pd.DataFrame({
            "feature":     available_features,
            "odds_ratio":  odds_ratios,
            "coefficient": lr_model.coef_[0],
        })
        .sort_values("odds_ratio", ascending=False)
    )
    print("\n  Top 5 risk factors (highest odds ratio):")
    print(or_df.head(5).to_string(index=False))

    # Save logistic regression baseline results
    import os
    logreg_path = os.path.join(PROCESSED_DIR, "logreg_baseline.json")
    results = {
        "model":       "LogisticRegression",
        "auc_roc":     round(auc_score, 4),
        "recall":      round(recall, 4),
        "precision":   round(precision, 4),
        "f1_score":    round(f1, 4),
        "n_features":  len(available_features),
        "n_positive":  int(y.sum()),
        "n_total":     len(y),
        "odds_ratios": {
            row["feature"]: round(row["odds_ratio"], 4)
            for _, row in or_df.iterrows()
        },
    }
    with open(logreg_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {logreg_path}")

    # ── XGBoost (secondary) ───────────────────────────────────────────
    try:
        from src.model_training import train_xgboost, save_model
        import joblib

        xgb_model = train_xgboost(X_train, y_train)
        save_model(xgb_model, XGBOOST_MODEL)
        print(f"  XGBoost saved → {XGBOOST_MODEL}")
    except Exception as e:
        print(f"  XGBoost training skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — MODEL 3: SCENARIO PROJECTION 2025 → 2030
# Owner: Lorenah  |  src/projection.py  |  notebook 07
#
# Approach: Scenario-based projection (replaced Prophet — only 2025 data).
#   Scenario A (BAU):      flat projection — 2025 rates unchanged to 2030.
#   Scenario B (Bridged):  30% IIT reduction in Critical + High tiers
#                          from 2026. VLS improves proportionally.
#
# Note on patients_retained:
#   src/projection.py uses actual adults_on_art counts as denominator.
#   This is correct. Notebook 07's ×10,000 scaling is a placeholder —
#   use projection.py output for the dashboard and presentation.
# ══════════════════════════════════════════════════════════════════════════

def run_model3() -> None:
    """
    Model 3 — 2030 Dual Scenario Projection.

    Algorithm : Scenario-based projection (BAU flat + Bridged Gap 30% reduction)
    Output:
      - forecast_critical/high/moderate/low.csv (both scenarios per tier)
      - forecast_national.csv
      - county_comparison.csv
      - patients_retained.csv (county-level, using actual ART counts)

    Reads:  COUNTY_PROF (must have tier column from Step 5)
    Writes: FORECAST_CRITICAL, FORECAST_HIGH, FORECAST_MODERATE,
            FORECAST_LOW, FORECAST_NATIONAL,
            data/processed/county_comparison.csv,
            data/processed/patients_retained.csv
    """
    print("\n[STEP 7] Model 3 — 2030 Scenario Projection")
    print("─" * 52)

    from src.projection import run_projection_pipeline
    run_projection_pipeline(save=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — EVALUATION
# Owner: Dennis  |  src/evaluation.py  |  notebook 08
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation() -> None:
    """
    Evaluate all three models.

    Model 1 : Silhouette score (already printed in Step 5)
    Model 2 : AUC-ROC, Recall, Precision, F1 (already printed in Step 6)
    Model 3 : Cross-check that forecast CSVs exist and have correct shape
    """
    print("\n[STEP 8] Model Evaluation")
    print("─" * 52)

    import pandas as pd

    # ── Model 1 validation ───────────────────────────────────────────
    print("\n  Model 1 — County Clustering:")
    try:
        profiles = pd.read_csv(COUNTY_PROF)
        n_tiers  = profiles["tier"].nunique()
        for tier in TIER_LABELS:
            n = len(profiles[profiles["tier"] == tier])
            print(f"    {tier:<10}: {n} counties")
        if n_tiers == 4:
            print("    OK — 4 tiers present as expected.")
        else:
            print(f"    WARNING — expected 4 tiers, found {n_tiers}.")
    except Exception as e:
        print(f"    Could not load county_profiles.csv: {e}")

    # ── Model 3 forecast files ───────────────────────────────────────
    print("\n  Model 3 — Forecast files:")
    forecast_files = {
        "Critical": FORECAST_CRITICAL,
        "High":     FORECAST_HIGH,
        "Moderate": FORECAST_MODERATE,
        "Low":      FORECAST_LOW,
        "National": FORECAST_NATIONAL,
    }
    for label, path in forecast_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"    OK   {label:<10}: {df.shape[0]} rows, "
                  f"years {df['year'].min()}–{df['year'].max()}")
        else:
            print(f"    MISSING {label}: {path}")

    # ── logreg_baseline.json ─────────────────────────────────────────
    print("\n  Model 2 — Baseline results:")
    import json
    logreg_path = os.path.join(PROCESSED_DIR, "logreg_baseline.json")
    if os.path.exists(logreg_path):
        with open(logreg_path) as f:
            results = json.load(f)
        print(f"    AUC-ROC  : {results.get('auc_roc', 'N/A')}")
        print(f"    Recall   : {results.get('recall', 'N/A')}")
        print(f"    Precision: {results.get('precision', 'N/A')}")
        print(f"    F1       : {results.get('f1_score', 'N/A')}")
        print(f"    Dropout cases: {results.get('n_positive', 'N/A')} "
              f"/ {results.get('n_total', 'N/A')} total")
    else:
        print(f"    logreg_baseline.json not found at {logreg_path}")


# ══════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════

STEPS = {
    "extract":      run_extraction,
    "clean_nsdcc":  run_nsdcc_cleaning,
    "clean_dhs":    run_dhs_cleaning,
    "merge":        run_merge,
    "features":     run_feature_engineering,
    "model1":       run_model1,
    "model2":       run_model2,
    "model3":       run_model3,
    "evaluate":     run_evaluation,
}

STEP_ORDER = [
    "extract",
    "clean_nsdcc",
    "clean_dhs",
    "merge",
    "features",
    "model1",
    "model2",
    "model3",
    "evaluate",
]


def run_full_pipeline() -> None:
    """Run all steps in order."""
    print("=" * 52)
    print("  HIV Care Gap AI — Full Pipeline")
    print("  CRISP-DM  |  Team: Daniella · Eve · Verah")
    print("            |         Naomi · Lorenah · Dennis")
    print("=" * 52)

    start = time.time()
    failed = []

    for step_name in STEP_ORDER:
        try:
            STEPS[step_name]()
        except Exception as e:
            print(f"\n  ERROR in step '{step_name}': {e}")
            failed.append(step_name)
            print(f"  Skipping to next step.\n")

    elapsed = time.time() - start
    print("\n" + "=" * 52)
    if not failed:
        print(f"  Pipeline complete. Elapsed: {elapsed:.1f}s")
    else:
        print(f"  Pipeline finished with errors in: {failed}")
        print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 52)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HIV Care Gap AI — Pipeline Entry Point"
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()),
        help="Run a single pipeline step instead of the full pipeline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all raw file paths without running any processing.",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
    elif args.step:
        print(f"\nRunning single step: {args.step}")
        print("=" * 52)
        try:
            STEPS[args.step]()
        except Exception as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
    else:
        run_full_pipeline()