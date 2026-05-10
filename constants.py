# =============================================================
# constants.py — HIV Care Gap AI
# Single source of truth for ALL paths, filenames, parameters,
# and column names used across every notebook and script.
#
# HOW TO USE:
#   from constants import *
#   OR
#   from constants import RAW_DIR, COUNTY_NAME_MAP, KMEANS_K
#
# RULE: Never hard-code any path, filename, column name,
#       weight, or parameter anywhere else in the codebase.
#       Change it here — it updates everywhere.
# =============================================================

import os

# ── ROOT DIRECTORY
# Works when constants.py is in repo root.
# For notebooks: add this to Cell 1 before importing constants:
#   import sys, os
#   sys.path.append(os.path.abspath('..'))
#   from constants import *
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── DATA PATHS
RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")

# ── RAW INPUT FILES
ADULT_ART_FILE = os.path.join(RAW_DIR, "Adult_on_ART.xlsx")
HTS_FILE       = os.path.join(RAW_DIR, "Adult_on_HTS.xlsx")
VLT_FILE       = os.path.join(RAW_DIR, "VLT.xlsx")           # Viral Load Testing
IIT_FILE       = os.path.join(RAW_DIR, "IIT.xlsx")           # Interruption in Treatment
DHS_REDUCED    = os.path.join(RAW_DIR, "individual_features.csv")  # already reduced 

# ── PROCESSED / INTERMEDIATE FILES
ART_CLEAN    = os.path.join(PROCESSED_DIR, "adult_on_art_clean.csv")
HTS_CLEAN    = os.path.join(PROCESSED_DIR, "hts_clean.csv")
VLT_CLEAN    = os.path.join(PROCESSED_DIR, "vlt_clean.csv")
IIT_CLEAN    = os.path.join(PROCESSED_DIR, "iit_clean.csv")
DHS_CLEAN    = os.path.join(PROCESSED_DIR, "individual_features_clean.csv")
NSDCC_CLEAN  = os.path.join(PROCESSED_DIR, "nsdcc_clean.csv")
COUNTY_PROF  = os.path.join(PROCESSED_DIR, "county_profiles.csv")
TIER_TS      = os.path.join(PROCESSED_DIR, "tier_timeseries.csv")

# ── MODEL ARTIFACT FILES
KMEANS_MODEL  = os.path.join(MODELS_DIR, "kmeans_county_tiers.pkl")
XGBOOST_MODEL = os.path.join(MODELS_DIR, "xgboost_dropout.pkl")
PROPHET_MODEL = os.path.join(MODELS_DIR, "prophet_bau.pkl")

# ── PROPHET FORECAST CSVs (one per tier, saved by notebook 07 / train_model3.py)
FORECAST_CRITICAL = os.path.join(PROCESSED_DIR, "forecast_critical.csv")
FORECAST_HIGH     = os.path.join(PROCESSED_DIR, "forecast_high.csv")
FORECAST_MODERATE = os.path.join(PROCESSED_DIR, "forecast_moderate.csv")
FORECAST_LOW      = os.path.join(PROCESSED_DIR, "forecast_low.csv")
FORECAST_NATIONAL = os.path.join(PROCESSED_DIR, "forecast_national.csv")

# ── CARE GAP INDEX WEIGHTS  (must sum to 1.0)
IIT_WEIGHT    = 0.4   # weight for IIT rate
VLS_WEIGHT    = 0.4   # weight for (1 - VLS rate)
HTS_WEIGHT    = 0.2   # weight for HTS positivity rate
CGI_SCALE_MIN = 0
CGI_SCALE_MAX = 100

# ── KMEANS PARAMETERS
KMEANS_K            = 4
KMEANS_RANDOM_STATE = 42
TIER_LABELS         = ["Critical", "High", "Moderate", "Low"]
KMEANS_FEATURES     = [
    "iit_rate",
    "vls_rate",
    "hts_positivity_rate",
    "art_coverage",
    "iit_yoy_change",
]

# ── TIER COLOURS (for charts)
TIER_COLORS = {
    "Critical": "#C0392B",  # dark red
    "High":     "#E67E22",  # orange
    "Moderate": "#F1C40F",  # yellow
    "Low":      "#27AE60",  # green
}

# ── XGBOOST / CLASSIFICATION PARAMETERS
TEST_SIZE            = 0.2
RANDOM_STATE         = 42
XGB_N_ESTIMATORS     = 200
XGB_MAX_DEPTH        = 5
XGB_LEARNING_RATE    = 0.05
XGB_SCALE_POS_WEIGHT = 3    # class imbalance — update after checking Cell 13 of NB02

# ── MODEL 2 FEATURES (from individual_features_clean.csv)
MODEL2_FEATURES = [
    "county",
    "age_group",
    "education_level",
    "wealth_index",
    "distance_to_facility",
    "ever_tested_hiv",
    "tested_hiv_last_12months",
    "marital_status",
    "num_sexual_partners",
    "has_health_insurance",
]
MODEL2_TARGET = "dropout"

# ── PROPHET PARAMETERS
FORECAST_YEAR_END         = 2030
IIT_REDUCTION_RATE        = 0.30   # 30% IIT reduction in Bridged Gap scenario
BRIDGED_TIERS             = ["Critical", "High"]
BRIDGED_START_YEAR        = 2026
PROPHET_CHANGEPOINT_PRIOR = 0.05   # low = smoother trend (important for short series)
PROPHET_SEASONALITY_MODE  = "additive"

# ── STREAMLIT
STREAMLIT_PORT = 8505
TAB1_TITLE     = "County Gap Map"
TAB2_TITLE     = "Dropout Risk Calculator"
TAB3_TITLE     = "2030 Forecast"

# ── IIT ALERT THRESHOLD
IIT_ALERT_FALLBACK_THRESHOLD = 0.15   # 15% — update after EDA in notebook 04


# ── NSDCC COLUMN NAME CONFIG
# The column name for county in each NSDCC raw file (all use "County").
# ART values have a " County" suffix (e.g. "Baringo County") — stripped in cleaning.
# Update these if your file export uses different column names.
NSDCC_COUNTY_COL = "County"
NSDCC_PERIOD_COL = "Period"

# ART files include " County" suffix in county values — strip this during cleaning
ART_COUNTY_SUFFIX = " County"

# =============================================================
# COUNTY NAME MAP
# Maps every raw MOH county name variant → standardised name.
# Built by Person 1 during notebook 01.
# DO NOT duplicate this dictionary anywhere else in the codebase.
# =============================================================
COUNTY_NAME_MAP = {
    # ── Standardised → standardised (identity, defensive)
    "Baringo":         "Baringo",
    "Bomet":           "Bomet",
    "Bungoma":         "Bungoma",
    "Busia":           "Busia",
    "Elgeyo Marakwet": "Elgeyo Marakwet",
    "Embu":            "Embu",
    "Garissa":         "Garissa",
    "Homa Bay":        "Homa Bay",
    "Isiolo":          "Isiolo",
    "Kajiado":         "Kajiado",
    "Kakamega":        "Kakamega",
    "Kericho":         "Kericho",
    "Kiambu":          "Kiambu",
    "Kilifi":          "Kilifi",
    "Kirinyaga":       "Kirinyaga",
    "Kisii":           "Kisii",
    "Kisumu":          "Kisumu",
    "Kitui":           "Kitui",
    "Kwale":           "Kwale",
    "Laikipia":        "Laikipia",
    "Lamu":            "Lamu",
    "Machakos":        "Machakos",
    "Makueni":         "Makueni",
    "Mandera":         "Mandera",
    "Marsabit":        "Marsabit",
    "Meru":            "Meru",
    "Migori":          "Migori",
    "Mombasa":         "Mombasa",
    "Murang'a":        "Murang'a",
    "Muranga":         "Murang'a",
    "Nairobi":         "Nairobi",
    "Nakuru":          "Nakuru",
    "Nandi":           "Nandi",
    "Narok":           "Narok",
    "Nyamira":         "Nyamira",
    "Nyandarua":       "Nyandarua",
    "Nyeri":           "Nyeri",
    "Samburu":         "Samburu",
    "Siaya":           "Siaya",
    "Taita Taveta":    "Taita Taveta",
    "Tana River":      "Tana River",
    "Tharaka Nithi":   "Tharaka Nithi",
    "Trans Nzoia":     "Trans Nzoia",
    "Turkana":         "Turkana",
    "Uasin Gishu":     "Uasin Gishu",
    "Vihiga":          "Vihiga",
    "Wajir":           "Wajir",
    "West Pokot":      "West Pokot",

    # ── Common raw MOH variants (add more as found during notebook 01)
    "Nairobi City":        "Nairobi",
    "Nairobi County":      "Nairobi",
    "Nairobi City County": "Nairobi",
    "NAIROBI":             "Nairobi",
    "NAIROBI CITY":        "Nairobi",
    "Homa-Bay":            "Homa Bay",
    "HomaBay":             "Homa Bay",
    "HOMA BAY":            "Homa Bay",
    "Taita-Taveta":        "Taita Taveta",
    "TAITA TAVETA":        "Taita Taveta",
    "Tana-River":          "Tana River",
    "TANA RIVER":          "Tana River",
    "Trans-Nzoia":         "Trans Nzoia",
    "TRANS NZOIA":         "Trans Nzoia",
    "Uasin-Gishu":         "Uasin Gishu",
    "UASIN GISHU":         "Uasin Gishu",
    "West-Pokot":          "West Pokot",
    "WEST POKOT":          "West Pokot",
    "Elgeyo-Marakwet":     "Elgeyo Marakwet",
    "ELGEYO MARAKWET":     "Elgeyo Marakwet",
    "Tharaka-Nithi":       "Tharaka Nithi",
    "THARAKA NITHI":       "Tharaka Nithi",
    "Murang'A":            "Murang'a",
    "MURANG'A":            "Murang'a",
    "MURANGA":             "Murang'a",
    "KISUMU":              "Kisumu",
    "MOMBASA":             "Mombasa",
    "NAKURU":              "Nakuru",
    "KIAMBU":              "Kiambu",
    "KAKAMEGA":            "Kakamega",
    "BUNGOMA":             "Bungoma",
    "MIGORI":              "Migori",
    "SIAYA":               "Siaya",
    "KISII":               "Kisii",
    "NYAMIRA":             "Nyamira",
    "MACHAKOS":            "Machakos",
    "MAKUENI":             "Makueni",
    "KITUI":               "Kitui",
    "GARISSA":             "Garissa",
    "MANDERA":             "Mandera",
    "WAJIR":               "Wajir",
    "TURKANA":             "Turkana",
    "MARSABIT":            "Marsabit",
    "ISIOLO":              "Isiolo",
    "MERU":                "Meru",
    "EMBU":                "Embu",
    "KIRINYAGA":           "Kirinyaga",
    "NYERI":               "Nyeri",
    "NYANDARUA":           "Nyandarua",
    "LAIKIPIA":            "Laikipia",
    "SAMBURU":             "Samburu",
    "BARINGO":             "Baringo",
    "KERICHO":             "Kericho",
    "BOMET":               "Bomet",
    "NAROK":               "Narok",
    "KAJIADO":             "Kajiado",
    "NANDI":               "Nandi",
    "VIHIGA":              "Vihiga",
    "BUSIA":               "Busia",
    "KWALE":               "Kwale",
    "KILIFI":              "Kilifi",
    "LAMU":                "Lamu",
    # ── ART file variants — county values include " County" suffix
    # These are auto-stripped in clean_data.py but kept here as a fallback
    "Baringo County":         "Baringo",
    "Bomet County":           "Bomet",
    "Bungoma County":         "Bungoma",
    "Busia County":           "Busia",
    "Elgeyo Marakwet County": "Elgeyo Marakwet",
    "Embu County":            "Embu",
    "Garissa County":         "Garissa",
    "Homa Bay County":        "Homa Bay",
    "Isiolo County":          "Isiolo",
    "Kajiado County":         "Kajiado",
    "Kakamega County":        "Kakamega",
    "Kericho County":         "Kericho",
    "Kiambu County":          "Kiambu",
    "Kilifi County":          "Kilifi",
    "Kirinyaga County":       "Kirinyaga",
    "Kisii County":           "Kisii",
    "Kisumu County":          "Kisumu",
    "Kitui County":           "Kitui",
    "Kwale County":           "Kwale",
    "Laikipia County":        "Laikipia",
    "Lamu County":            "Lamu",
    "Machakos County":        "Machakos",
    "Makueni County":         "Makueni",
    "Mandera County":         "Mandera",
    "Marsabit County":        "Marsabit",
    "Meru County":            "Meru",
    "Migori County":          "Migori",
    "Mombasa County":         "Mombasa",
    "Murang'a County":       "Murang'a",
    "Nairobi County":         "Nairobi",
    "Nakuru County":          "Nakuru",
    "Nandi County":           "Nandi",
    "Narok County":           "Narok",
    "Nyamira County":         "Nyamira",
    "Nyandarua County":       "Nyandarua",
    "Nyeri County":           "Nyeri",
    "Samburu County":         "Samburu",
    "Siaya County":           "Siaya",
    "Taita Taveta County":    "Taita Taveta",
    "Tana River County":      "Tana River",
    "Tharaka Nithi County":   "Tharaka Nithi",
    "Trans Nzoia County":     "Trans Nzoia",
    "Turkana County":         "Turkana",
    "Uasin Gishu County":     "Uasin Gishu",
    "Vihiga County":          "Vihiga",
    "Wajir County":           "Wajir",
    "West Pokot County":      "West Pokot",
}

# =============================================================
# DHS CODE MAPS
# The DHS Individual Recode uses integer codes.
# These dictionaries map codes → readable labels.
# =============================================================

# DHS COUNTY CODE MAP  (v024 integer 1–47 → standardised county name)
DHS_COUNTY_MAP = {
    1:  "Mombasa",         2:  "Kwale",           3:  "Kilifi",
    4:  "Tana River",      5:  "Lamu",            6:  "Taita Taveta",
    7:  "Garissa",         8:  "Wajir",           9:  "Mandera",
    10: "Marsabit",        11: "Isiolo",           12: "Meru",
    13: "Tharaka Nithi",   14: "Embu",            15: "Kitui",
    16: "Machakos",        17: "Makueni",         18: "Nyandarua",
    19: "Nyeri",           20: "Kirinyaga",       21: "Murang'a",
    22: "Kiambu",          23: "Turkana",         24: "West Pokot",
    25: "Samburu",         26: "Trans Nzoia",     27: "Uasin Gishu",
    28: "Elgeyo Marakwet", 29: "Nandi",           30: "Baringo",
    31: "Laikipia",        32: "Nakuru",          33: "Narok",
    34: "Kajiado",         35: "Kericho",         36: "Bomet",
    37: "Kakamega",        38: "Vihiga",          39: "Bungoma",
    40: "Busia",           41: "Siaya",           42: "Kisumu",
    43: "Homa Bay",        44: "Migori",          45: "Kisii",
    46: "Nyamira",         47: "Nairobi",
}

# DHS AGE GROUP MAP  (v013)
DHS_AGE_GROUP_MAP = {
    1: "15-19", 2: "20-24", 3: "25-29",
    4: "30-34", 5: "35-39", 6: "40-44", 7: "45-49",
}

# DHS EDUCATION MAP  (v106)
DHS_EDUCATION_MAP = {
    0: "No education",
    1: "Primary",
    2: "Secondary",
    3: "Higher",
}

# DHS WEALTH INDEX MAP  (v190)
DHS_WEALTH_MAP = {
    1: "Poorest",
    2: "Poorer",
    3: "Middle",
    4: "Richer",
    5: "Richest",
}

# DHS MARITAL STATUS MAP  (v501)
DHS_MARITAL_MAP = {
    0: "Never married",
    1: "Married",
    2: "Living together",
    3: "Widowed",
    4: "Divorced",
    5: "Separated",
}

# DHS DISTANCE TO FACILITY MAP  (v826a)
# 0=<1km  1=1-2km  2=2-5km  3=5-10km  4=10+km  998=don't know
DHS_DISTANCE_MAP = {
    0:   "<1 km",
    1:   "1-2 km",
    2:   "2-5 km",
    3:   "5-10 km",
    4:   "10+ km",
    998: "Unknown",
}

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def get_tier_color(tier: str) -> str:
    """Return hex colour for a given tier string. Grey fallback."""
    return TIER_COLORS.get(tier, "#808080")


def standardise_county(raw_name: str) -> str:
    """Map one raw MOH county name to the standardised form."""
    return COUNTY_NAME_MAP.get(raw_name, raw_name)
