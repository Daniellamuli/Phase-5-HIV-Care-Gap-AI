# =============================================================
# constants.py — HIV Care Gap AI Project
# Built from actual file inspection. Do not guess column names.
#
# REAL FILE STRUCTURES:
#   ART (47x11): Period, County, MOH cols..., Total, Total_Males, Total_Females
#   HTS (47x7) : Period, County, MOH cols..., Total, Total_Males, Total_Females
#   VLT (47x7) : County only (NO Period), Valid VL / Suppressed by age/sex
#   IIT (9x13) : Region only (NO Period), 9 regions, IIT counts + percentages
#   DHS        : county integers 1-47, already reduced
# =============================================================

import os

ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")

# ── RAW FILES (exact filenames in data/raw/)
ADULT_ART_FILE = os.path.join(RAW_DIR, "Adult_on_ART.xlsx")
HTS_FILE       = os.path.join(RAW_DIR, "Adult_on_HTS.xlsx")
VLT_FILE       = os.path.join(RAW_DIR, "VLT.xlsx")
IIT_FILE       = os.path.join(RAW_DIR, "IIT.xlsx")
DHS_REDUCED       = os.path.join(RAW_DIR, "individual_features.csv")
HTS_POSITIVE_FILE = os.path.join(RAW_DIR, "HTS_Positive.xlsx")  # HIV+ tested counts by county + period

# ── PROCESSED FILES
ART_CLEAN    = os.path.join(PROCESSED_DIR, "adult_on_art_clean.csv")
HTS_CLEAN     = os.path.join(PROCESSED_DIR, "hts_clean.csv")
HTS_POS_CLEAN = os.path.join(PROCESSED_DIR, "hts_positive_clean.csv")  # HIV+ counts cleaned
VLT_CLEAN    = os.path.join(PROCESSED_DIR, "vlt_clean.csv")
IIT_CLEAN    = os.path.join(PROCESSED_DIR, "iit_clean.csv")
DHS_CLEAN    = os.path.join(PROCESSED_DIR, "individual_features_clean.csv")
NSDCC_CLEAN  = os.path.join(PROCESSED_DIR, "nsdcc_clean.csv")
COUNTY_PROF  = os.path.join(PROCESSED_DIR, "county_profiles.csv")
TIER_TS      = os.path.join(PROCESSED_DIR, "tier_timeseries.csv")
IIT_ALERTS   = os.path.join(PROCESSED_DIR, "iit_alerts.csv")  # counties flagged above IIT threshold

# ── MODEL FILES
KMEANS_MODEL  = os.path.join(MODELS_DIR, "kmeans_county_tiers.pkl")
XGBOOST_MODEL = os.path.join(MODELS_DIR, "xgboost_dropout.pkl")
MODEL3_BUNDLE = os.path.join(MODELS_DIR, "model3_scenario.pkl")
# PROPHET_MODEL removed — Model 3 uses tier-based scenario projection, not Prophet

# ── FORECAST FILES
FORECAST_CRITICAL = os.path.join(PROCESSED_DIR, "forecast_critical.csv")
FORECAST_HIGH     = os.path.join(PROCESSED_DIR, "forecast_high.csv")
FORECAST_MODERATE = os.path.join(PROCESSED_DIR, "forecast_moderate.csv")
FORECAST_LOW      = os.path.join(PROCESSED_DIR, "forecast_low.csv")
FORECAST_NATIONAL = os.path.join(PROCESSED_DIR, "forecast_national.csv")

# ── ADDITIONAL PROCESSED FILES
COUNTY_COMPARISON = os.path.join(PROCESSED_DIR, "county_comparison.csv")  # from projection.py
PATIENTS_RETAINED = os.path.join(PROCESSED_DIR, "patients_retained.csv")  # from projection.py

# ── MODEL 2 OUTPUT FILES
LOGREG_BASELINE_JSON  = os.path.join(PROCESSED_DIR, "logreg_baseline.json")
DROPOUT_RISK_FACTORS  = os.path.join(PROCESSED_DIR, "dropout_risk_factors.csv")
ODDS_RATIOS_CI_JSON   = os.path.join(PROCESSED_DIR, "odds_ratios_with_ci.json")

# =============================================================
# COLUMN NAME CONSTANTS — confirmed from actual files
# =============================================================

# ART and HTS: have both Period and County columns
ART_COUNTY_COL  = "County"
ART_PERIOD_COL  = "Period"
ART_COUNTY_SUFFIX = " County"   # values are "Baringo County" — strip this

HTS_COUNTY_COL  = "County"
HTS_PERIOD_COL  = "Period"
HTS_COUNTY_SUFFIX = " County"   # same suffix as ART

# HTS Positive file column constants (same format as HTS)
HTS_POS_COUNTY_COL    = "County"
HTS_POS_PERIOD_COL    = "Period"
HTS_POS_COUNTY_SUFFIX = " County"

# VLT: has County only — NO Period column (single snapshot)
VLT_COUNTY_COL  = "County"
VLT_HAS_PERIOD  = False

# IIT: has Region only — NO Period column, 9 rows only
IIT_REGION_COL  = "Region"
IIT_COUNTY_COL  = "Region"   # alias — IIT uses Region not County
IIT_HAS_PERIOD  = False

# Shared period col name (ART + HTS only)
NSDCC_PERIOD_COL = "Period"
NSDCC_COUNTY_COL = "County"

# Post-cleaning column names (lowercase)
# Use these in feature_engineering.py and all notebooks after cleaning
CLEAN_COUNTY_COL = "county"
CLEAN_PERIOD_COL = "period"

# ── CGI COLUMN NAMES
# These are the standardised column names used in the CGI formula
# across notebook 04, notebook 05, and src/projection.py.
# Verah (NB04): must map iit_rate_pct → iit_rate before CGI calculation
# Naomi  (NB05): Cell 6 already maps iit_rate_pct → iit_rate  ✓
# projection.py: reads iit_rate directly from county_profiles.csv  ✓
CGI_IIT_COL = "iit_rate"        # column used in CGI formula (not iit_rate_pct)
CGI_VLS_COL = "vls_rate_adult"  # column used in CGI formula
CGI_HTS_COL = "hts_positivity_rate"  # column used in CGI formula

# =============================================================
# COLUMN RENAME MAPS — exact raw names → clean names
# Used by clean_data.py and notebook 02
# =============================================================

ART_RENAME = {
    "Total":        "adults_on_art",
    "Total_Males":  "art_total_males",
    "Total_Females":"art_total_females",
}

HTS_RENAME = {
    "Total":        "hts_tested",
    "Total_Males":  "hts_tested_males",
    "Total_Females":"hts_tested_females",
}

HTS_POSITIVE_RENAME = {
    "Total":        "hts_positive",
    "Total_Males":  "hts_positive_males",
    "Total_Females":"hts_positive_females",
}

VLT_RENAME = {
    "Valid VL <15yrs":       "vlt_valid_under15",
    "Suppressed <15yrs":     "vls_suppressed_under15",
    "Valid VL 15+Male":      "vlt_valid_male_15plus",
    "Suppressed 15+Male":    "vls_suppressed_male_15plus",
    "Valid VL 15+Female":    "vlt_valid_female_15plus",
    "Suppressed 15+Female":  "vls_suppressed_female_15plus",
}

IIT_RENAME = {
    "All Adult(Male & Female) on Treament at the begining of 2025": "adults_on_treatment",
    "All Adult(Male & Female) IIT":                                  "iit_count",
    "All Adult(Male & Female) IIT percentage(%)":                    "iit_rate_pct",
    "Children IIT":                                                  "iit_children",
    "Children IIT(%)":                                               "iit_rate_children_pct",
    "Male adult IIT":                                                "iit_male",
    "Male adult IIT(%)":                                             "iit_rate_male_pct",
    "Female adult IIT":                                              "iit_female",
    "Female adult IIT(%)":                                           "iit_rate_female_pct",
}

# =============================================================
# IIT REGION → COUNTIES  (9 MOH regions → 47 counties)
# =============================================================

IIT_REGION_MAP = {
    "Nairobi":       "Nairobi",
    "Central":       "Central",
    "Coast":         "Coast",
    "Eastern":       "Eastern",
    "North Eastern": "North Eastern",
    "Nyanza":        "Nyanza",
    "Rift Valley":   "Rift Valley",
    "Western":       "Western",
    # variants
    "NAIROBI":           "Nairobi",
    "CENTRAL":           "Central",
    "COAST":             "Coast",
    "EASTERN":           "Eastern",
    "NORTH EASTERN":     "North Eastern",
    "North-Eastern":     "North Eastern",
    "NYANZA":            "Nyanza",
    "RIFT VALLEY":       "Rift Valley",
    "Rift-Valley":       "Rift Valley",
    "WESTERN":           "Western",
    "Kenya":             None,
    "KENYA":             None,
    "National":          None,
    "Total":             None,
}

REGION_TO_COUNTIES = {
    "Nairobi":        ["Nairobi"],
    "Central":        ["Kiambu", "Murang'a", "Nyandarua", "Nyeri", "Kirinyaga"],
    "Coast":          ["Mombasa", "Kwale", "Kilifi", "Tana River", "Lamu", "Taita Taveta"],
    "Eastern":        ["Marsabit", "Isiolo", "Meru", "Tharaka Nithi", "Embu",
                       "Kitui", "Machakos", "Makueni"],
    "North Eastern":  ["Garissa", "Wajir", "Mandera"],
    "Nyanza":         ["Siaya", "Kisumu", "Homa Bay", "Migori", "Kisii", "Nyamira"],
    "Rift Valley":    ["Turkana", "West Pokot", "Samburu", "Trans Nzoia", "Uasin Gishu",
                       "Elgeyo Marakwet", "Nandi", "Baringo", "Laikipia", "Nakuru",
                       "Narok", "Kajiado", "Kericho", "Bomet"],
    "Western":        ["Kakamega", "Vihiga", "Bungoma", "Busia"],
}

_all = [c for cs in REGION_TO_COUNTIES.values() for c in cs]
assert len(_all) == 47, f"REGION_TO_COUNTIES has {len(_all)} counties, expected 47"
assert len(_all) == len(set(_all)), "Duplicate county in REGION_TO_COUNTIES"

# =============================================================
# CARE GAP INDEX
# =============================================================
IIT_WEIGHT    = 0.4
VLS_WEIGHT    = 0.4
HTS_WEIGHT    = 0.2
CGI_SCALE_MIN = 0
CGI_SCALE_MAX = 100

# =============================================================
# MODEL PARAMETERS
# =============================================================
KMEANS_K            = 4
KMEANS_RANDOM_STATE = 42
TIER_LABELS         = ["Critical", "High", "Moderate", "Low"]

# CLUSTER FEATURES — confirmed from notebook 05 validation (2025 data only):
#   - "iit_rate"      : from iit_rate_pct         — valid, used
#   - "vls_rate"      : from vls_rate_adult        — valid, used
#   - "art_coverage"  : DROPPED — adults_on_treatment is IIT regional data
#                       (not PLHIV denominator), so adults_on_art /
#                       adults_on_treatment >> 1 for all counties → clips
#                       to constant 1.0 → zero clustering information
#   - "iit_yoy_change": DROPPED — only 2025 period available; no prior
#                       year to compute year-on-year change
# Re-add art_coverage / iit_yoy_change here when multi-period data is available.
KMEANS_FEATURES     = ["iit_rate", "vls_rate"]

# k=4 is a deliberate project decision to produce 4 named tiers
# (Critical / High / Moderate / Low). Silhouette analysis favours k=3
# (score=0.6582) over k=4 (score=0.6359), but 4 tiers are required
# for the intervention framework — do not change without team sign-off.

TIER_COLORS = {
    "Critical": "#C0392B",
    "High":     "#E67E22",
    "Moderate": "#F1C40F",
    "Low":      "#27AE60",
}

TEST_SIZE            = 0.2
RANDOM_STATE         = 42
XGB_N_ESTIMATORS     = 200
XGB_MAX_DEPTH        = 5
XGB_LEARNING_RATE    = 0.05
XGB_SCALE_POS_WEIGHT = 1236  # Calculated from Lorenah's DHS cleaning output
                             # rec_weight = 32130 / 26 = 1235.77 → rounded up
                             # Dropout: 26 individuals (0.08%) vs 32130 retained (99.92%)

# DHS ACTUAL COLUMNS (confirmed from 03_dhs_cleaning.ipynb output)
# Raw columns in individual_features.csv (15 total):
#   case_id, county, age_group, education_level, wealth_index,
#   worked_last_12months, ever_tested_hiv, tested_hiv_last_12months,
#   distance_to_facility, marital_status, currently_in_union,
#   num_sexual_partners, knows_aids_death, told_hiv_positive, has_health_insurance
#
# Notes from Lorenah's cleaning:
#   - knows_aids_death     : 100% missing → DROPPED
#   - has_health_insurance : 100% missing → DROPPED (imputed to all 0s)
#   - anc_visits           : column does NOT exist in this dataset → REMOVED
#   - distance_to_facility : 57% missing but kept, imputed
#   - education_level      : one-hot encoded → edu_Higher, edu_No education,
#                            edu_Primary, edu_Secondary
#   - wealth_index         : one-hot encoded → wealth_Middle, wealth_Poorer,
#                            wealth_Poorest, wealth_Richer, wealth_Richest

# Features used for MODEL2 training (post one-hot encoding column names)
MODEL2_FEATURES = [
    "county",
    "age_group",
    "marital_status",
    "distance_to_facility",
    "ever_tested_hiv",
    "tested_hiv_last_12months",
    "num_sexual_partners",
    "worked_last_12months",
    "currently_in_union",
    # One-hot encoded education (edu_ prefix from Lorenah's DHSCleaner)
    "edu_Higher",
    "edu_No education",
    "edu_Primary",
    "edu_Secondary",
    # One-hot encoded wealth (wealth_ prefix from Lorenah's DHSCleaner)
    "wealth_Middle",
    "wealth_Poorer",
    "wealth_Poorest",
    "wealth_Richer",
    "wealth_Richest",
]
MODEL2_TARGET = "dropout"

FORECAST_YEAR_END  = 2030
IIT_REDUCTION_RATE = 0.30
BRIDGED_TIERS      = ["Critical", "High"]
BRIDGED_START_YEAR = 2026
# PROPHET_CHANGEPOINT_PRIOR + PROPHET_SEASONALITY_MODE removed — no Prophet in project

STREAMLIT_PORT = 8505
TAB1_TITLE     = "County Gap Map"
TAB2_TITLE     = "Dropout Risk Calculator"
TAB3_TITLE     = "2030 Forecast"

IIT_ALERT_FALLBACK_THRESHOLD = 0.15

# =============================================================
# COUNTY NAME MAP
# =============================================================
COUNTY_NAME_MAP = {
    "Baringo": "Baringo", "Bomet": "Bomet", "Bungoma": "Bungoma",
    "Busia": "Busia", "Elgeyo Marakwet": "Elgeyo Marakwet", "Embu": "Embu",
    "Garissa": "Garissa", "Homa Bay": "Homa Bay", "Isiolo": "Isiolo",
    "Kajiado": "Kajiado", "Kakamega": "Kakamega", "Kericho": "Kericho",
    "Kiambu": "Kiambu", "Kilifi": "Kilifi", "Kirinyaga": "Kirinyaga",
    "Kisii": "Kisii", "Kisumu": "Kisumu", "Kitui": "Kitui",
    "Kwale": "Kwale", "Laikipia": "Laikipia", "Lamu": "Lamu",
    "Machakos": "Machakos", "Makueni": "Makueni", "Mandera": "Mandera",
    "Marsabit": "Marsabit", "Meru": "Meru", "Migori": "Migori",
    "Mombasa": "Mombasa", "Murang'a": "Murang'a", "Muranga": "Murang'a",
    "Nairobi": "Nairobi", "Nakuru": "Nakuru", "Nandi": "Nandi",
    "Narok": "Narok", "Nyamira": "Nyamira", "Nyandarua": "Nyandarua",
    "Nyeri": "Nyeri", "Samburu": "Samburu", "Siaya": "Siaya",
    "Taita Taveta": "Taita Taveta", "Tana River": "Tana River",
    "Tharaka Nithi": "Tharaka Nithi", "Tharaka-Nithi": "Tharaka Nithi",
    "Trans Nzoia": "Trans Nzoia", "Turkana": "Turkana", "Uasin Gishu": "Uasin Gishu",
    "Vihiga": "Vihiga", "Wajir": "Wajir", "West Pokot": "West Pokot",
    # ART/HTS suffix variants
    "Baringo County": "Baringo", "Bomet County": "Bomet",
    "Bungoma County": "Bungoma", "Busia County": "Busia",
    "Elgeyo Marakwet County": "Elgeyo Marakwet", "Embu County": "Embu",
    "Garissa County": "Garissa", "Homa Bay County": "Homa Bay",
    "Isiolo County": "Isiolo", "Kajiado County": "Kajiado",
    "Kakamega County": "Kakamega", "Kericho County": "Kericho",
    "Kiambu County": "Kiambu", "Kilifi County": "Kilifi",
    "Kirinyaga County": "Kirinyaga", "Kisii County": "Kisii",
    "Kisumu County": "Kisumu", "Kitui County": "Kitui",
    "Kwale County": "Kwale", "Laikipia County": "Laikipia",
    "Lamu County": "Lamu", "Machakos County": "Machakos",
    "Makueni County": "Makueni", "Mandera County": "Mandera",
    "Marsabit County": "Marsabit", "Meru County": "Meru",
    "Migori County": "Migori", "Mombasa County": "Mombasa",
    "Murang'a County": "Murang'a", "Nairobi County": "Nairobi",
    "Nakuru County": "Nakuru", "Nandi County": "Nandi",
    "Narok County": "Narok", "Nyamira County": "Nyamira",
    "Nyandarua County": "Nyandarua", "Nyeri County": "Nyeri",
    "Samburu County": "Samburu", "Siaya County": "Siaya",
    "Taita Taveta County": "Taita Taveta", "Tana River County": "Tana River",
    "Tharaka Nithi County": "Tharaka Nithi", "Trans Nzoia County": "Trans Nzoia",
    "Turkana County": "Turkana", "Uasin Gishu County": "Uasin Gishu",
    "Vihiga County": "Vihiga", "Wajir County": "Wajir",
    "West Pokot County": "West Pokot",
    # uppercase variants
    "NAIROBI": "Nairobi", "MOMBASA": "Mombasa", "NAKURU": "Nakuru",
    "KISUMU": "Kisumu", "KAKAMEGA": "Kakamega", "KIAMBU": "Kiambu",
    "HOMA BAY": "Homa Bay", "Homa-Bay": "Homa Bay",
    "TAITA TAVETA": "Taita Taveta", "TANA RIVER": "Tana River",
    "TRANS NZOIA": "Trans Nzoia", "UASIN GISHU": "Uasin Gishu",
    "WEST POKOT": "West Pokot", "ELGEYO MARAKWET": "Elgeyo Marakwet",
    "THARAKA NITHI": "Tharaka Nithi", "MURANG'A": "Murang'a",
}

# =============================================================
# DHS CODE MAPS
# =============================================================
DHS_COUNTY_MAP = {
    1:"Mombasa", 2:"Kwale", 3:"Kilifi", 4:"Tana River", 5:"Lamu",
    6:"Taita Taveta", 7:"Garissa", 8:"Wajir", 9:"Mandera", 10:"Marsabit",
    11:"Isiolo", 12:"Meru", 13:"Tharaka Nithi", 14:"Embu", 15:"Kitui",
    16:"Machakos", 17:"Makueni", 18:"Nyandarua", 19:"Nyeri", 20:"Kirinyaga",
    21:"Murang'a", 22:"Kiambu", 23:"Turkana", 24:"West Pokot", 25:"Samburu",
    26:"Trans Nzoia", 27:"Uasin Gishu", 28:"Elgeyo Marakwet", 29:"Nandi",
    30:"Baringo", 31:"Laikipia", 32:"Nakuru", 33:"Narok", 34:"Kajiado",
    35:"Kericho", 36:"Bomet", 37:"Kakamega", 38:"Vihiga", 39:"Bungoma",
    40:"Busia", 41:"Siaya", 42:"Kisumu", 43:"Homa Bay", 44:"Migori",
    45:"Kisii", 46:"Nyamira", 47:"Nairobi",
}

DHS_AGE_GROUP_MAP  = {1:"15-19",2:"20-24",3:"25-29",4:"30-34",5:"35-39",6:"40-44",7:"45-49"}
DHS_EDUCATION_MAP  = {0:"No education",1:"Primary",2:"Secondary",3:"Higher"}
DHS_WEALTH_MAP     = {1:"Poorest",2:"Poorer",3:"Middle",4:"Richer",5:"Richest"}
DHS_MARITAL_MAP    = {0:"Never married",1:"Married",2:"Living together",3:"Widowed",4:"Divorced",5:"Separated"}
DHS_DISTANCE_MAP   = {0:"<1 km",1:"1-2 km",2:"2-5 km",3:"5-10 km",4:"10+ km",998:"Unknown"}

# DHS WORKED LAST 12 MONTHS MAP (v731)
# 0 = Never worked / no job
# 1 = Worked in the past year (had work at some point in last 12 months but it stopped)
# 2 = Currently working
# 3 = Have a job but not currently working (on leave, sick leave, maternity leave,
#     seasonal worker, or otherwise temporarily absent from work at time of survey)
DHS_WORKED_MAP = {
    0: "Never worked",
    1: "In the past year",
    2: "Currently working",
    3: "Have job but not currently working",
}

# DHS CURRENTLY IN UNION MAP (v502)
# 0 = Not in union, 1 = Currently in union, 2 = Formerly in union
DHS_UNION_MAP = {
    0: "Not in union",
    1: "Currently in union",
    2: "Formerly in union",
}

# =============================================================
# HELPERS
# =============================================================
def get_tier_color(tier): return TIER_COLORS.get(tier, "#808080")
def standardise_county(name): return COUNTY_NAME_MAP.get(name, name)