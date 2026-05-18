"""
main.py — HIV Care Gap AI Pipeline
====================================
Entry point for the full end-to-end pipeline.
Run this file from the project root to execute all steps in sequence.

Usage:
    python main.py                     # Full pipeline
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

Team:  Daniella (Lead) · Eve · Verah · Naomi · Lorenah · Dennis
Method: CRISP-DM
"""

import argparse
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ── Make repo root importable from any working directory ──────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Constants import ──────────────────────────────────────────────────────
# All names imported here exist in constants.py (v6, verified).
# KMEANS_FEATURES = ["iit_rate", "vls_rate"]  — confirmed from NB05 cell 6.
# CGI_IIT_COL / CGI_VLS_COL / CGI_HTS_COL added in constants v6.
from constants import (
    # Raw file paths
    ADULT_ART_FILE, HTS_FILE, HTS_POSITIVE_FILE,
    VLT_FILE, IIT_FILE, DHS_REDUCED,
    # Processed file paths
    ART_CLEAN, HTS_CLEAN, HTS_POS_CLEAN,
    VLT_CLEAN, IIT_CLEAN,
    DHS_CLEAN, NSDCC_CLEAN, COUNTY_PROF,
    TIER_TS, IIT_ALERTS,
    # Model file paths
    KMEANS_MODEL, XGBOOST_MODEL, MODEL3_BUNDLE,
    # Forecast file paths
    FORECAST_CRITICAL, FORECAST_HIGH,
    FORECAST_MODERATE, FORECAST_LOW, FORECAST_NATIONAL,
    # ART column constants
    ART_COUNTY_COL, ART_PERIOD_COL, ART_COUNTY_SUFFIX,
    # HTS column constants
    HTS_COUNTY_COL, HTS_PERIOD_COL, HTS_COUNTY_SUFFIX,
    # HTS_Positive column constants (added in constants v6)
    HTS_POS_COUNTY_COL, HTS_POS_PERIOD_COL, HTS_POS_COUNTY_SUFFIX,
    # VLT / IIT column constants
    VLT_COUNTY_COL, IIT_REGION_COL,
    # Rename maps
    ART_RENAME, HTS_RENAME, HTS_POSITIVE_RENAME, VLT_RENAME, IIT_RENAME,
    # Region maps
    IIT_REGION_MAP,
    # CGI column name aliases (added in constants v6)
    CGI_IIT_COL, CGI_VLS_COL, CGI_HTS_COL,
    # CGI weights
    IIT_WEIGHT, VLS_WEIGHT, HTS_WEIGHT,
    CGI_SCALE_MIN, CGI_SCALE_MAX,
    # KMeans parameters
    # KMEANS_FEATURES = ["iit_rate", "vls_rate"] per constants v6 and NB05 cell 6.
    # NB05 cell 6 maps: iit_rate_pct → iit_rate, vls_rate_adult → vls_rate.
    # Step 5 here applies the same renames before clustering.
    KMEANS_K, KMEANS_RANDOM_STATE, KMEANS_FEATURES,
    TIER_LABELS, TIER_COLORS,
    # Model 2 parameters
    MODEL2_FEATURES, MODEL2_TARGET,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SCALE_POS_WEIGHT, TEST_SIZE, RANDOM_STATE,
    # Scenario projection parameters
    FORECAST_YEAR_END, IIT_REDUCTION_RATE,
    BRIDGED_TIERS, BRIDGED_START_YEAR,
    # Alert threshold
    IIT_ALERT_FALLBACK_THRESHOLD,
    # Directories
    PROCESSED_DIR, MODELS_DIR,
)


# ══════════════════════════════════════════════════════════════════════════
# STEP 0 — DRY RUN
# ══════════════════════════════════════════════════════════════════════════

def dry_run() -> bool:
    """
    Validate all raw input file paths without running any processing.
    Prints OK / MISSING per file. Returns True if all present.
    """
    print("\n[DRY RUN] Validating required file paths")
    print("─" * 56)

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
        status = "OK     " if exists else "MISSING"
        print(f"  {status}  {label:<25}  {path}")
        if not exists:
            all_ok = False

    print()
    if all_ok:
        print("  All raw files present — pipeline is ready to run.")
    else:
        print("  One or more files missing. Copy them to data/raw/ before running.")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA EXTRACTION
# Owner: Naomi  |  scripts/extract_data.py  |  notebook 01
# Read-only — saves nothing to disk.
# ══════════════════════════════════════════════════════════════════════════

def run_extraction() -> dict:
    """
    Load all 6 raw source files and validate every mapping.

    Validates:
      - county name maps cover ART / HTS / HTS_Positive / VLT
      - IIT_REGION_MAP covers every region row
      - DHS county codes 1-47 all resolve via DHS_COUNTY_MAP
      - REGION_TO_COUNTIES covers exactly 47 counties

    Returns:
        dict with keys: art, hts, hts_pos, vlt, iit, dhs
    """
    print("\n[STEP 1] Data Extraction")
    print("─" * 56)
    from scripts.extract_data import extract_all
    return extract_all()


# ══════════════════════════════════════════════════════════════════════════
# STEP 2a — NSDCC DATA CLEANING
# Owner: Eve  |  src/nsdcc_cleaner.py  |  notebook 02
#
# Per-file operations (confirmed from NB02 + nsdcc_cleaner.py):
#   ART / HTS / HTS_Positive:
#     strip ' County' → standardise county names → rename MOH cols →
#     to_numeric_impute (county mean → median fallback)
#   VLT:
#     standardise → rename → to_numeric_impute →
#     engineer VLS rates from raw suppressed / valid counts
#   IIT:
#     drop Kenya total row → rename → to_numeric →
#     expand 9 regions → 47 counties → impute
#
# Writes: ART_CLEAN, HTS_CLEAN, HTS_POS_CLEAN, VLT_CLEAN, IIT_CLEAN
# ══════════════════════════════════════════════════════════════════════════

def run_nsdcc_cleaning() -> None:
    """Clean all 5 NSDCC raw files and save to data/processed/."""
    print("\n[STEP 2a] NSDCC Data Cleaning")
    print("─" * 56)

    import pandas as pd
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
    print("\n  Cleaning ART...")
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
    print("\n  Cleaning HTS...")
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
    print("\n  Cleaning HTS_Positive...")
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
    # VLT has no Period column (single county snapshot).
    # VLS rates are engineered here from raw suppressed / valid counts.
    print("\n  Cleaning VLT...")
    vlt = load_snapshot(VLT_FILE, VLT_COUNTY_COL, "VLT")
    vlt = standardise(vlt, VLT_COUNTY_COL, "VLT")
    vlt = vlt.rename(columns=VLT_RENAME)
    vlt = to_numeric_impute(vlt, VLT_COUNTY_COL)

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

    vlt = fix_vlt_missing(vlt)
    vlt = vlt.rename(columns={VLT_COUNTY_COL: "county"})
    validate(vlt, "VLT", county_col="county")
    vlt.to_csv(VLT_CLEAN, index=False)
    print(f"  Saved → {VLT_CLEAN}")

    # ── IIT ──────────────────────────────────────────────────────────────
    # IIT is region-level (9 rows). Drop national total, expand to 47 counties.
    # Rate/pct cols copied region-wide; count cols divided equally per county.
    print("\n  Cleaning IIT...")
    iit_raw = load_iit(IIT_FILE, IIT_REGION_COL, "IIT")
    iit_raw["_std"] = iit_raw[IIT_REGION_COL].map(IIT_REGION_MAP)
    iit_raw = iit_raw[iit_raw["_std"].notna()].drop(columns=["_std"])
    iit_raw = iit_raw.rename(columns=IIT_RENAME)
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
# Output in individual_features_clean.csv:
#   Dropped (100% missing): knows_aids_death, has_health_insurance
#   Not in dataset:         anc_visits
#   Imputed with median:    num_sexual_partners, distance_to_facility
#   One-hot encoded:        education_level → edu_*, wealth_index → wealth_*
#   Target:                 dropout = 1 if told_hiv_positive=1 AND
#                                         tested_hiv_last_12months=0
#   Class balance:          32,130 retained / 26 dropout (XGB_SCALE_POS_WEIGHT=1236)
# ══════════════════════════════════════════════════════════════════════════

def run_dhs_cleaning() -> None:
    """
    Clean DHS individual features and save to data/processed/.

    Reads:  DHS_REDUCED  (data/raw/individual_features.csv)
    Writes: DHS_CLEAN    (data/processed/individual_features_clean.csv)
    """
    print("\n[STEP 2b] DHS Data Cleaning")
    print("─" * 56)

    import constants as c
    from src.dhs_cleaner import DHSCleaner

    cleaner = DHSCleaner(c)
    cleaner.load_data(DHS_REDUCED)

    county_col = next(
        (col for col in cleaner.raw_df.columns if col.lower() == "county"),
        "county",
    )
    cleaner.decode_county(county_col=county_col)
    cleaner.decode_demographics()
    cleaner.impute_binary_flags()
    cleaner.impute_numeric_columns(numeric_cols=["num_sexual_partners"])
    cleaner.engineer_dropout_target()
    cleaner.one_hot_encode(["education_level", "wealth_index"])
    cleaner.save_clean_data(DHS_CLEAN)
    print(f"  Saved → {DHS_CLEAN}")

    ratios = cleaner.get_class_ratio()
    print(f"  Dropout class balance: {ratios}")
    print(f"  NOTE: Only 26 dropout cases — findings are descriptive risk factors,")
    print(f"        not predictive scores. See NB06 for full interpretation.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — DATA MERGING
# Owner: Eve  |  scripts/merge_data.py
#
# Strategy (from merge_data.py — confirmed correct):
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
    print("─" * 56)
    from scripts.merge_data import run_merge as _run_merge
    merged = _run_merge(save=True)
    print(f"  Merged shape: {merged.shape}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — FEATURE ENGINEERING
# Owner: Verah  |  src/feature_engineering.py  |  notebook 04
#
# Column handling (confirmed from NB04 cells 3, 6 + constants CGI_* names):
#   hts_positivity_rate computed: hts_positive / hts_tested
#   iit_rate_pct normalised to decimal if > 1, then aliased → CGI_IIT_COL ("iit_rate")
#   care_gap_index = (IIT_WEIGHT × iit_rate)
#                  + (VLS_WEIGHT × (1 − vls_rate_adult))
#                  + (HTS_WEIGHT × hts_positivity_rate) × CGI_SCALE_MAX
#   art_coverage   = 0.5 placeholder (PLHIV estimates unavailable)
#   *_yoy_change   = 0 (single year — 2025 only)
#
# NB05 cell 6 additionally renames:
#   iit_rate_pct  → iit_rate   (same as CGI_IIT_COL)
#   vls_rate_adult → vls_rate  (required by KMEANS_FEATURES)
# Step 5 applies those same renames so it is self-contained.
# ══════════════════════════════════════════════════════════════════════════

def run_feature_engineering() -> None:
    """
    Build all features needed by Models 1, 2, and 3.

    Reads:  NSDCC_CLEAN
    Writes: COUNTY_PROF  (data/processed/county_profiles.csv)
    """
    print("\n[STEP 4] Feature Engineering")
    print("─" * 56)

    import pandas as pd

    df = pd.read_csv(NSDCC_CLEAN)
    print(f"  Loaded {NSDCC_CLEAN}: {df.shape}")

    # ── HTS positivity rate ───────────────────────────────────────────
    if CGI_HTS_COL not in df.columns:
        if "hts_positive" in df.columns and "hts_tested" in df.columns:
            df[CGI_HTS_COL] = (df["hts_positive"] / df["hts_tested"]).fillna(0)
            print(f"  Computed {CGI_HTS_COL}")
        else:
            raise ValueError(
                "Cannot compute hts_positivity_rate — hts_positive or hts_tested missing.\n"
                "Ensure Step 3 (merge) completed successfully."
            )

    # ── YoY placeholders (2025 is the only available period) ─────────
    df["iit_rate_yoy_change"]       = 0
    df["vls_rate_adult_yoy_change"] = 0

    # ── Normalise rates to decimal if stored as percentage ────────────
    for col in ["iit_rate_pct", "vls_rate_adult", CGI_HTS_COL]:
        if col in df.columns and df[col].max() > 1:
            df[col] = df[col] / 100

    # ── iit_rate alias (CGI_IIT_COL = "iit_rate") ────────────────────
    # NB04 cell 6 maps iit_rate_pct → CGI_IIT_COL before the CGI formula.
    # We apply the same alias here so Step 5 KMEANS_FEATURES works correctly.
    if CGI_IIT_COL not in df.columns and "iit_rate_pct" in df.columns:
        df[CGI_IIT_COL] = pd.to_numeric(df["iit_rate_pct"], errors="coerce")
        print(f"  Mapped iit_rate_pct → {CGI_IIT_COL}")

    # ── Care Gap Index ────────────────────────────────────────────────
    missing_cgi = [c for c in [CGI_IIT_COL, CGI_VLS_COL, CGI_HTS_COL] if c not in df.columns]
    if missing_cgi:
        raise ValueError(f"Cannot compute CGI — missing columns: {missing_cgi}")

    df["care_gap_index"] = (
        (IIT_WEIGHT * df[CGI_IIT_COL]) +
        (VLS_WEIGHT * (1 - df[CGI_VLS_COL])) +
        (HTS_WEIGHT * df[CGI_HTS_COL])
    ) * CGI_SCALE_MAX

    print(f"  CGI range: {df['care_gap_index'].min():.4f} – {df['care_gap_index'].max():.4f}")

    # ── ART coverage (placeholder — PLHIV estimates unavailable) ──────
    if "adults_on_art" in df.columns and "plhiv_estimate" in df.columns:
        df["art_coverage"] = (
            df["adults_on_art"] / df["plhiv_estimate"]
        ).clip(0, 1).fillna(0)
    else:
        print("  WARNING: PLHIV estimates unavailable — art_coverage set to 0.5 placeholder")
        df["art_coverage"] = 0.5

    # ── County profiles: one row per county using latest period ───────
    county_profiles = (
        df.sort_values("period")
        .groupby("county")
        .last()
        .reset_index()
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    county_profiles.to_csv(COUNTY_PROF, index=False)
    print(f"  Saved → {COUNTY_PROF}  ({county_profiles.shape[0]} counties)")
    print(f"  Columns: {county_profiles.columns.tolist()}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — MODEL 1: COUNTY CLUSTERING (KMeans)
# Owner: Naomi  |  scripts/train_model1.py  |  notebook 05
#
# KMEANS_FEATURES = ["iit_rate", "vls_rate"] per constants v6.
# NB05 cell 6 renames iit_rate_pct → iit_rate and vls_rate_adult → vls_rate
# before clustering. This step applies the same renames so it is
# self-contained regardless of notebook run order.
#
# Tier assignment: clusters ranked by mean iit_rate descending.
#   Highest IIT → Critical, lowest → Low  (matches NB05 cell 14).
# county_profiles.csv updated with: cluster, tier, iit_rate, vls_rate.
# ══════════════════════════════════════════════════════════════════════════

def run_model1() -> None:
    """
    Model 1 — County Care Gap Map (KMeans, k=4).

    Reads:  COUNTY_PROF
    Writes: KMEANS_MODEL, updated COUNTY_PROF
    """
    print("\n[STEP 5] Model 1 — County Clustering (KMeans)")
    print("─" * 56)

    import pickle
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df = pd.read_csv(COUNTY_PROF)
    print(f"  Loaded {COUNTY_PROF}: {df.shape}")

    # ── Apply NB05 cell 6 renames ─────────────────────────────────────
    # KMEANS_FEATURES = ["iit_rate", "vls_rate"]
    # county_profiles.csv after Step 4 has "iit_rate" (from CGI_IIT_COL alias)
    # and "vls_rate_adult". NB05 cell 6 also creates "vls_rate" from "vls_rate_adult".
    if "iit_rate" not in df.columns and "iit_rate_pct" in df.columns:
        df["iit_rate"] = pd.to_numeric(df["iit_rate_pct"], errors="coerce")
        print("  Mapped iit_rate_pct → iit_rate")

    if "vls_rate" not in df.columns and "vls_rate_adult" in df.columns:
        df["vls_rate"] = pd.to_numeric(df["vls_rate_adult"], errors="coerce")
        print("  Mapped vls_rate_adult → vls_rate")

    # ── Validate KMEANS_FEATURES ──────────────────────────────────────
    missing_cols = [f for f in KMEANS_FEATURES if f not in df.columns]
    if missing_cols:
        raise ValueError(
            f"KMEANS_FEATURES {missing_cols} missing from county_profiles.csv.\n"
            f"Available columns: {df.columns.tolist()}\n"
            "Run Step 4 (feature engineering) first."
        )
    if df[KMEANS_FEATURES].isnull().sum().sum() > 0:
        raise ValueError(
            f"NaN values in KMEANS_FEATURES:\n"
            f"{df[KMEANS_FEATURES].isnull().sum().to_dict()}"
        )

    # ── Scale + fit ───────────────────────────────────────────────────
    X        = df[KMEANS_FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Features:     {KMEANS_FEATURES}")
    print(f"  Matrix shape: {X_scaled.shape}")

    km = KMeans(n_clusters=KMEANS_K, random_state=KMEANS_RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    print(f"  KMeans fitted | inertia: {km.inertia_:.4f}")

    # ── Tier labels ranked by mean iit_rate descending ────────────────
    cluster_iit_mean = (
        df.groupby("cluster")["iit_rate"]
        .mean()
        .sort_values(ascending=False)
    )
    cluster_to_tier = {
        cid: TIER_LABELS[rank]
        for rank, cid in enumerate(cluster_iit_mean.index)
    }
    df["tier"] = df["cluster"].map(cluster_to_tier)

    print("\n  Cluster → Tier:")
    for cid, tier in cluster_to_tier.items():
        n = (df["cluster"] == cid).sum()
        print(f"    Cluster {cid} → {tier:<10} | mean IIT: {cluster_iit_mean[cid]:.4f} | n={n}")

    print("\n  Tier breakdown:")
    for tier in TIER_LABELS:
        counties = df[df["tier"] == tier]["county"].tolist()
        print(f"    {tier:<10}: {len(counties):>2} counties")
        print(f"               [{', '.join(sorted(counties))}]")

    # ── Silhouette score ──────────────────────────────────────────────
    sil = silhouette_score(X_scaled, km.labels_)
    print(f"\n  Silhouette (k={KMEANS_K}): {sil:.4f}")
    print("  Note: k=3 scores ~0.66 (higher), but k=4 is required for the")
    print("        4-tier intervention framework. Documented and accepted.")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(COUNTY_PROF, index=False)
    print(f"\n  Updated → {COUNTY_PROF}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    bundle = {
        "kmeans":           km,
        "scaler":           scaler,
        "features":         KMEANS_FEATURES,
        "cluster_to_tier":  cluster_to_tier,
        "silhouette_score": sil,
        "n_clusters":       KMEANS_K,
    }
    with open(KMEANS_MODEL, "wb") as f:
        pickle.dump(bundle, f)
    size_kb = os.path.getsize(KMEANS_MODEL) / 1024
    print(f"  Saved model → {KMEANS_MODEL}  ({size_kb:.1f} KB)")


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — MODEL 2: DROPOUT RISK FACTOR ANALYSIS
# Owner: Lorenah  |  notebook 06
#
# Approach (confirmed from NB06): Risk factor identification.
# Language: "associated with dropout risk" — NOT "predicts dropout".
# Primary output: odds ratios from balanced logistic regression.
# 26 positive cases / 32,156 total (0.08%) — too few for reliable prediction.
# Solver: liblinear (confirmed from NB06 cell 13).
#
# Outputs (confirmed from NB06 cell 30):
#   data/processed/logreg_baseline.json      ← metrics + top/protective factors
#   data/processed/dropout_risk_factors.csv  ← full OR table (Streamlit Tab 2)
#   figures/odds_ratios_forest.png
#   figures/confusion_matrix_logreg.png
#   figures/roc_curve_logreg.png
# ══════════════════════════════════════════════════════════════════════════

def run_model2() -> None:
    """
    Model 2 — Individual Dropout Risk Factor Analysis.

    Algorithm: Logistic Regression (balanced, liblinear)
    Output:    odds ratios, metrics, dropout_risk_factors.csv,
               logreg_baseline.json
    """
    print("\n[STEP 6] Model 2 — Dropout Risk Factor Analysis")
    print("─" * 56)

    import json
    import numpy as np
    import pandas as pd
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

    # ── Features ──────────────────────────────────────────────────────
    available   = [col for col in MODEL2_FEATURES if col in df.columns]
    not_present = [col for col in MODEL2_FEATURES if col not in df.columns]
    if not_present:
        print(f"  NOTE: {len(not_present)} MODEL2_FEATURES absent "
              f"(expected — 100% missing cols dropped):")
        print(f"        {not_present}")

    X = df[available].copy()
    y = df[MODEL2_TARGET].copy()
    print(f"  Features: {len(available)} | Target: {MODEL2_TARGET}")
    print(f"  Positive cases: {int(y.sum())} / {len(y)} ({y.mean():.4%})")

    # ── Encode any remaining object columns ───────────────────────────
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    if cat_cols:
        print(f"  Label-encoded: {cat_cols}")

    # ── Train / test split (stratified) ───────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Positive cases in test: {y_test.sum()}")

    # ── Logistic Regression (liblinear — confirmed from NB06 cell 13) ─
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
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Results:")
    print(f"    AUC-ROC  : {auc_score:.4f}")
    print(f"    Recall   : {recall:.4f}   ← primary metric")
    print(f"    Precision: {precision:.4f}")
    print(f"    F1       : {f1:.4f}")
    print(f"    Caught: {tp} | Missed: {fn}")

    # ── Odds ratios ───────────────────────────────────────────────────
    odds_ratios = np.exp(lr_model.coef_[0])
    risk_df = (
        pd.DataFrame({
            "Feature":     available,
            "Coefficient": lr_model.coef_[0],
            "Odds_Ratio":  odds_ratios.round(3),
        })
        .sort_values("Odds_Ratio", ascending=False)
    )
    risk_df["Risk_Direction"] = risk_df["Odds_Ratio"].apply(
        lambda x: "Higher" if x > 1 else ("Lower" if x < 1 else "Neutral")
    )
    print("\n  Top 5 risk factors:")
    print(risk_df.head(5)[["Feature", "Odds_Ratio", "Risk_Direction"]].to_string(index=False))

    # ── Save outputs ──────────────────────────────────────────────────
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # dropout_risk_factors.csv — consumed by Streamlit Tab 2
    risk_path = os.path.join(PROCESSED_DIR, "dropout_risk_factors.csv")
    risk_df.to_csv(risk_path, index=False)
    print(f"\n  Saved → {risk_path}")

    # logreg_baseline.json — consumed by Step 8 evaluation
    high_risk  = risk_df[risk_df["Odds_Ratio"] > 1.5]
    protective = risk_df[risk_df["Odds_Ratio"] < 0.7]
    results = {
        "model":              "LogisticRegression",
        "auc_roc":            round(auc_score, 4),
        "recall":             round(recall, 4),
        "precision":          round(precision, 4),
        "f1_score":           round(f1, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_features":         len(available),
        "n_positive":         int(y.sum()),
        "n_total":            len(y),
        "top_risk_factors":   high_risk.head(10)["Feature"].tolist(),
        "top_risk_ors":       high_risk.head(10)["Odds_Ratio"].tolist(),
        "protective_factors": protective.head(10)["Feature"].tolist(),
    }
    logreg_path = os.path.join(PROCESSED_DIR, "logreg_baseline.json")
    with open(logreg_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {logreg_path}")

    # ── XGBoost (NOT IMPLEMENTED) ─────────────────────────────────────
    # Due to extreme class imbalance (26 dropout cases / 32,156 total),
    # XGBoost is not used. The model cannot learn meaningful patterns
    # from only 26 positive cases. Logistic Regression is used instead
    # for risk factor identification (odds ratios).
    print(f"\n  NOTE: XGBoost skipped — insufficient positive cases ({int(y.sum())})")
    print(f"        Logistic Regression used for risk factor analysis only.")


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — MODEL 3: SCENARIO PROJECTION 2025 → 2030
# Owner: Lorenah  |  scripts/train_model3.py  |  src/projection.py
#
# Approach: Scenario-based projection (replaced Prophet — single year 2025).
#   Scenario A (BAU):      2025 IIT + VLS rates held flat to 2030.
#   Scenario B (Bridged):  30% IIT reduction in Critical + High from 2026.
#                          VLS improvement: ΔVLS = −0.5 × ΔIIT.
#
# Uses scripts/train_model3.py which wraps src/projection.py.
# Both files are confirmed correct — no changes needed.
# Requires county_profiles.csv to have the 'tier' column (from Step 5).
#
# Writes:
#   forecast_critical/high/moderate/low.csv  (both scenarios A + B per tier)
#   forecast_national.csv                    (national aggregate)
#   county_comparison.csv                    (cross-sectional rankings)
#   patients_retained.csv                    (ART patients saved under B)
#   models/model3_scenario.pkl               (full bundle)
# ══════════════════════════════════════════════════════════════════════════

def run_model3() -> None:
    """
    Model 3 — 2030 Dual Scenario Projection.

    Delegates to scripts/train_model3.py → src/projection.py.
    """
    print("\n[STEP 7] Model 3 — 2030 Scenario Projection")
    print("─" * 56)
    from scripts.train_model3 import run_model3 as _run_model3
    _run_model3(save=True)


# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — EVALUATION
# Owner: Dennis  |  src/evaluation.py  |  notebook 08
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation() -> None:
    """
    Validate all pipeline outputs.

    Model 1: Tier distribution from county_profiles.csv
    Model 2: Metrics from logreg_baseline.json + dropout_risk_factors.csv
    Model 3: Shape + year range of all forecast CSVs
    Other:   county_comparison.csv, patients_retained.csv, model bundles
    """
    print("\n[STEP 8] Model Evaluation")
    print("─" * 56)

    import json
    import pandas as pd

    # ── Model 1 ───────────────────────────────────────────────────────
    print("\n  Model 1 — County Clustering:")
    try:
        profiles   = pd.read_csv(COUNTY_PROF)
        n_counties = profiles["county"].nunique()
        n_tiers    = profiles["tier"].nunique()
        print(f"    Counties: {n_counties}  Tiers: {n_tiers}")
        for tier in TIER_LABELS:
            n = len(profiles[profiles["tier"] == tier])
            print(f"    {tier:<10}: {n} counties")
        print(f"    {'OK' if n_tiers == 4 and n_counties == 47 else 'WARNING'}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # ── Model 2 ───────────────────────────────────────────────────────
    print("\n  Model 2 — Risk Factor Analysis:")
    logreg_path = os.path.join(PROCESSED_DIR, "logreg_baseline.json")
    risk_path   = os.path.join(PROCESSED_DIR, "dropout_risk_factors.csv")
    if os.path.exists(logreg_path):
        with open(logreg_path) as f:
            res = json.load(f)
        print(f"    AUC-ROC  : {res.get('auc_roc')}")
        print(f"    Recall   : {res.get('recall')}   ← primary metric")
        print(f"    Precision: {res.get('precision')}")
        print(f"    F1       : {res.get('f1_score')}")
        print(f"    Cases    : {res.get('n_positive')} dropout / {res.get('n_total')} total")
        print(f"    Top risk : {res.get('top_risk_factors', [])[:3]}")
    else:
        print(f"    MISSING: {logreg_path}")
    if os.path.exists(risk_path):
        rf = pd.read_csv(risk_path)
        print(f"    dropout_risk_factors.csv: {rf.shape[0]} features")
    else:
        print(f"    MISSING: {risk_path}")

    # ── Model 3 ───────────────────────────────────────────────────────
    print("\n  Model 3 — Forecast files:")
    for label, path in {
        "Critical": FORECAST_CRITICAL,
        "High":     FORECAST_HIGH,
        "Moderate": FORECAST_MODERATE,
        "Low":      FORECAST_LOW,
        "National": FORECAST_NATIONAL,
    }.items():
        if os.path.exists(path):
            fdf   = pd.read_csv(path)
            yr_r  = f"{fdf['year'].min()}–{fdf['year'].max()}" if "year" in fdf.columns else "?"
            scens = sorted(fdf["scenario"].unique()) if "scenario" in fdf.columns else "?"
            print(f"    OK   {label:<10}: {fdf.shape[0]:>3} rows  {yr_r}  scenarios={scens}")
        else:
            print(f"    MISSING {label}: {path}")

    # ── Other outputs ─────────────────────────────────────────────────
    print("\n  Other outputs:")
    for label, path in {
        "county_comparison.csv":  os.path.join(PROCESSED_DIR, "county_comparison.csv"),
        "patients_retained.csv":  os.path.join(PROCESSED_DIR, "patients_retained.csv"),
        "model3_scenario.pkl":    MODEL3_BUNDLE,
        "kmeans_county_tiers.pkl":KMEANS_MODEL,
        "xgboost_dropout.pkl":    XGBOOST_MODEL,
    }.items():
        exists  = os.path.exists(path)
        size_kb = f"({os.path.getsize(path)/1024:.1f} KB)" if exists else ""
        print(f"    {'OK  ' if exists else 'MISS'} {label:<32} {size_kb}")


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
    """Run all steps in order, continuing past individual step errors."""
    print("=" * 56)
    print("  HIV Care Gap AI — Full Pipeline")
    print("  CRISP-DM  |  Daniella · Eve · Verah")
    print("            |  Naomi · Lorenah · Dennis")
    print("=" * 56)

    start  = time.time()
    failed = []

    for step_name in STEP_ORDER:
        try:
            STEPS[step_name]()
        except Exception as e:
            print(f"\n  ERROR in step '{step_name}': {e}")
            failed.append(step_name)
            print("  Continuing to next step.\n")

    elapsed = time.time() - start
    print("\n" + "=" * 56)
    if not failed:
        print(f"  Pipeline complete. Elapsed: {elapsed:.1f}s")
        print("  Next: streamlit run app/streamlit_app.py")
    else:
        print(f"  Pipeline finished with errors in: {failed}")
        print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 56)


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
        print("=" * 56)
        try:
            STEPS[args.step]()
        except Exception as e:
            print(f"\nERROR: {e}")
            sys.exit(1)
    else:
        run_full_pipeline()
