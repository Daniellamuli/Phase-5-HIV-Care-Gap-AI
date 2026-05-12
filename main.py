"""
main.py — HIV Care Gap AI Pipeline
====================================
Entry point for the full end-to-end pipeline.
Run this file to execute all steps in sequence.
 
Usage:
    python main.py                    # Run full pipeline
    python main.py --step extract     # Run one step only
    python main.py --step clean_nsdcc
    python main.py --step clean_dhs
    python main.py --step merge
    python main.py --step features
    python main.py --step model1
    python main.py --step model2
    python main.py --step model3
    python main.py --step evaluate
    python main.py --dry-run          # Validate paths only, no processing
 
Team: Daniella (Lead) · Eve · Verah · Naomi · Lorenah · Dennis
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
    # Raw file paths
    ADULT_ART_FILE, HTS_FILE, VLT_FILE, IIT_FILE, DHS_REDUCED,
    # Processed file paths
    ART_CLEAN, HTS_CLEAN, VLT_CLEAN, IIT_CLEAN, DHS_CLEAN,
    NSDCC_CLEAN, COUNTY_PROF, TIER_TS,
    # Column constants needed for cleaning calls
    ART_COUNTY_COL, ART_PERIOD_COL, ART_COUNTY_SUFFIX,
    HTS_COUNTY_COL, HTS_PERIOD_COL, HTS_COUNTY_SUFFIX,
    VLT_COUNTY_COL, IIT_REGION_COL,
    ART_RENAME, HTS_RENAME, VLT_RENAME, IIT_RENAME,
    IIT_REGION_MAP,
    # Model file paths
    KMEANS_MODEL, XGBOOST_MODEL,
    # Forecast file paths
    FORECAST_CRITICAL, FORECAST_HIGH, FORECAST_MODERATE,
    FORECAST_LOW, FORECAST_NATIONAL,
    # Model parameters
    KMEANS_K, KMEANS_RANDOM_STATE, KMEANS_FEATURES, TIER_LABELS,
    MODEL2_FEATURES, MODEL2_TARGET,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SCALE_POS_WEIGHT, TEST_SIZE, RANDOM_STATE,
    IIT_WEIGHT, VLS_WEIGHT, HTS_WEIGHT, CGI_SCALE_MAX,
    FORECAST_YEAR_END, IIT_REDUCTION_RATE,
    BRIDGED_TIERS, BRIDGED_START_YEAR,
    PROPHET_CHANGEPOINT_PRIOR, PROPHET_SEASONALITY_MODE,
    PROCESSED_DIR,
)
 
 
# ══════════════════════════════════════════════════════════════════════
# STEP 1 — DATA EXTRACTION
# Owner: Naomi  |  scripts/extract_data.py  |  notebook 01
# ══════════════════════════════════════════════════════════════════════
 
def run_extraction() -> dict:
    """
    Load all 5 raw files exactly as downloaded. Read-only — saves nothing.
    Validates that all county name maps and region maps cover every value
    found in the actual files.
 
    Returns:
        dict with keys: art, hts, vlt, iit, dhs — each a raw DataFrame.
 
    Raises:
        FileNotFoundError if any raw file is missing from data/raw/.
    """
    print("\n[STEP 1] Data Extraction")
    print("─" * 50)
 
    from scripts.extract_data import extract_all
    raw_data = extract_all()
    return raw_data
 
 
# ══════════════════════════════════════════════════════════════════════
# STEP 2a — NSDCC DATA CLEANING
# Owner: Eve  |  src/nsdcc_cleaner.py  |  notebook 02
#
# nsdcc_clean.csv confirmed structure (47 rows x 38 cols, period=2025):
#   county, period,
#   adults_on_art, art_total_males, art_total_females,      <- ART_RENAME
#   hts_tested, hts_tested_males, hts_tested_females,       <- HTS_RENAME
#   vlt_valid_under15, vls_suppressed_under15,               <- VLT_RENAME
#   vlt_valid_male_15plus, vls_suppressed_male_15plus,
#   vlt_valid_female_15plus, vls_suppressed_female_15plus,
#   vls_rate_male15plus, vls_rate_female15plus,              <- engineered
#   vls_rate_under15, vls_rate_adult,
#   adults_on_treatment, iit_count, iit_rate_pct,           <- IIT_RENAME
#   iit_children, iit_rate_children_pct,
#   iit_male, iit_rate_male_pct,
#   iit_female, iit_rate_female_pct
# ══════════════════════════════════════════════════════════════════════
 
def run_nsdcc_cleaning() -> None:
    """
    Clean all 4 NSDCC raw files and save to data/processed/.
 
    Operations per file:
      ART/HTS : strip ' County' suffix -> standardise county names ->
                rename MOH cols (ART_RENAME / HTS_RENAME) ->
                to_numeric_impute (county mean -> column median fallback)
      VLT     : standardise county names -> rename cols (VLT_RENAME) ->
                to_numeric_impute -> engineer VLS rates (vls_rate_adult etc.)
      IIT     : drop Kenya national total row -> rename cols (IIT_RENAME) ->
                expand 9 regions -> 47 counties (equal-share count split) ->
                impute rate cols with mean, count cols with 0
 
    Reads:  ADULT_ART_FILE, HTS_FILE, VLT_FILE, IIT_FILE  (data/raw/)
    Writes: ART_CLEAN, HTS_CLEAN, VLT_CLEAN, IIT_CLEAN    (data/processed/)
    """
    print("\n[STEP 2a] NSDCC Data Cleaning")
    print("─" * 50)
 
    from src.nsdcc_cleaner import (
        load_with_period,
        load_snapshot,
        load_iit,
        strip_suffix,
        standardise,
        to_numeric_impute,
        engineer_vls_rates,
        fix_vlt_missing,
        expand_iit_regions,
        fix_iit_missing,
        validate,
    )
    import os
 
    os.makedirs(PROCESSED_DIR, exist_ok=True)
 
    # ── ART ───────────────────────────────────────────────────────────
    art = load_with_period(ADULT_ART_FILE, ART_PERIOD_COL, 'ART')
    art = strip_suffix(art, ART_COUNTY_COL, ART_COUNTY_SUFFIX, 'ART')
    art = standardise(art, ART_COUNTY_COL, 'ART')
    art = art.rename(columns=ART_RENAME)
    art = to_numeric_impute(art, ART_COUNTY_COL, ART_PERIOD_COL)
    art = art.rename(columns={ART_COUNTY_COL: 'county', ART_PERIOD_COL: 'period'})
    validate(art, 'ART', county_col='county', period_col='period')
    art.to_csv(ART_CLEAN, index=False)
    print(f"  Saved -> {ART_CLEAN}")
 
    # ── HTS ───────────────────────────────────────────────────────────
    hts = load_with_period(HTS_FILE, HTS_PERIOD_COL, 'HTS')
    hts = strip_suffix(hts, HTS_COUNTY_COL, HTS_COUNTY_SUFFIX, 'HTS')
    hts = standardise(hts, HTS_COUNTY_COL, 'HTS')
    hts = hts.rename(columns=HTS_RENAME)
    hts = to_numeric_impute(hts, HTS_COUNTY_COL, HTS_PERIOD_COL)
    hts = hts.rename(columns={HTS_COUNTY_COL: 'county', HTS_PERIOD_COL: 'period'})
    validate(hts, 'HTS', county_col='county', period_col='period')
    hts.to_csv(HTS_CLEAN, index=False)
    print(f"  Saved -> {HTS_CLEAN}")
 
    # ── VLT ───────────────────────────────────────────────────────────
    vlt = load_snapshot(VLT_FILE, VLT_COUNTY_COL, 'VLT')
    vlt = standardise(vlt, VLT_COUNTY_COL, 'VLT')
    vlt = vlt.rename(columns=VLT_RENAME)
    vlt = to_numeric_impute(vlt, VLT_COUNTY_COL)
    vlt = engineer_vls_rates(vlt)
    vlt = fix_vlt_missing(vlt)
    vlt = vlt.rename(columns={VLT_COUNTY_COL: 'county'})
    validate(vlt, 'VLT', county_col='county')
    vlt.to_csv(VLT_CLEAN, index=False)
    print(f"  Saved -> {VLT_CLEAN}")
 
    # ── IIT ───────────────────────────────────────────────────────────
    iit_raw = load_iit(IIT_FILE, IIT_REGION_COL, 'IIT')
    # Drop Kenya national total row (IIT_REGION_MAP maps it to None)
    iit_raw['_std'] = iit_raw[IIT_REGION_COL].map(IIT_REGION_MAP)
    iit_raw = iit_raw[iit_raw['_std'].notna()].drop(columns=['_std'])
    iit_raw = iit_raw.rename(columns=IIT_RENAME)
    iit = expand_iit_regions(iit_raw, region_col=IIT_REGION_COL)
    iit = fix_iit_missing(iit)
    validate(iit, 'IIT', county_col='county')
    iit.to_csv(IIT_CLEAN, index=False)
    print(f"  Saved -> {IIT_CLEAN}")
 
 
# ══════════════════════════════════════════════════════════════════════
# STEP 2b — DHS DATA CLEANING
# Owner: Lorenah  |  src/dhs_cleaner.py  |  notebook 03
#
# Output confirmed: individual_features_clean.csv
#   Columns post one-hot encoding (from constants.py MODEL2_FEATURES):
#   county, age_group, marital_status, distance_to_facility,
#   ever_tested_hiv, tested_hiv_last_12months, num_sexual_partners,
#   worked_last_12months, currently_in_union,
#   edu_Higher, edu_No education, edu_Primary, edu_Secondary,
#   wealth_Middle, wealth_Poorer, wealth_Poorest, wealth_Richer, wealth_Richest,
#   dropout  (target: told_hiv_positive=1 AND tested_hiv_last_12months=0)
#
#   Class balance confirmed: 32,130 retained (99.92%) / 26 dropout (0.08%)
#   XGB_SCALE_POS_WEIGHT = 1236  (set in constants.py by Naomi)
#
#   Columns NOT present / dropped:
#   - anc_visits           : does not exist in this dataset
#   - knows_aids_death     : 100% missing -> dropped
#   - has_health_insurance : 100% missing -> imputed to 0, no signal
# ══════════════════════════════════════════════════════════════════════
 
def run_dhs_cleaning() -> None:
    """
    Clean DHS individual features and save to data/processed/.
 
    Operations:
      - Decode county codes 1-47 -> county names (DHS_COUNTY_MAP)
      - Decode age_group, education_level, wealth_index,
        marital_status, distance_to_facility (DHS_*_MAP constants)
      - Impute binary flags (ever_tested_hiv, tested_hiv_last_12months,
        told_hiv_positive, has_health_insurance) with 0
      - Impute num_sexual_partners with median
        (anc_visits does NOT exist in this dataset)
      - Engineer dropout target:
            dropout = 1 if told_hiv_positive=1 AND tested_hiv_last_12months=0
      - One-hot encode education_level -> edu_* columns
      - One-hot encode wealth_index    -> wealth_* columns
 
    Reads:  DHS_REDUCED  (data/raw/individual_features.csv)
    Writes: DHS_CLEAN    (data/processed/individual_features_clean.csv)
    """
    print("\n[STEP 2b] DHS Data Cleaning")
    print("─" * 50)
 
    import constants as c
    from src.dhs_cleaner import DHSCleaner
 
    cleaner = DHSCleaner(c)
    cleaner.load_data(DHS_REDUCED)
 
    # Detect county column (raw file uses lowercase 'county')
    county_col = next(
        (col for col in cleaner.raw_df.columns
         if col.lower() in ['county', 'hv024']),
        'county'
    )
    cleaner.decode_county(county_col=county_col)
    cleaner.decode_demographics()
    cleaner.impute_binary_flags()
    # anc_visits absent in this dataset — only impute num_sexual_partners
    cleaner.impute_numeric_columns(numeric_cols=['num_sexual_partners'])
    cleaner.engineer_dropout_target()
    cleaner.one_hot_encode(['education_level', 'wealth_index'])
    cleaner.save_clean_data(DHS_CLEAN)
    print(f"  Saved -> {DHS_CLEAN}")
 
    ratios = cleaner.get_class_ratio()
    print(f"  Dropout class balance -> {ratios}")
 
 
# ══════════════════════════════════════════════════════════════════════
# STEP 3 — DATA MERGING
# Owner: Eve  |  scripts/merge_data.py  |  Day 2
#
# Confirmed merge strategy (Eve's merge_data.py + nsdcc_clean.csv output):
#   ART + HTS  : inner join on ['county', 'period']
#   + VLT      : left join on ['county'] only  (single snapshot, no period)
#   + IIT      : left join on ['county'] only  (single snapshot, no period)
#   Period col : 4-digit year integer extracted from "December 2025" string
#   Result     : (47, 38), 0 missing values  <- confirmed from nsdcc_clean.csv
# ══════════════════════════════════════════════════════════════════════
 
def run_merge() -> None:
    """
    Merge all 4 cleaned NSDCC files into one county-level feature table.
 
    Reads:  ART_CLEAN, HTS_CLEAN, VLT_CLEAN, IIT_CLEAN
    Writes: NSDCC_CLEAN  (data/processed/nsdcc_clean.csv)
 
    Validated output: 47 rows x 38 cols, 0 missing values.
    """
    print("\n[STEP 3] Data Merging (NSDCC -> nsdcc_clean.csv)")
    print("─" * 50)
 
    from scripts.merge_data import run_merge as _merge
    merged = _merge(save=True)
    print(f"  Merged shape: {merged.shape}")
    return merged