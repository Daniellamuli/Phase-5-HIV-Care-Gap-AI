"""
src/projection.py
═══════════════════════════════════════════════════════════════
Production version of: notebooks/07_model_3_projection.ipynb

Tasks covered:
  G1DFP5CP-105 : project_bau()          — flat projection 2025→2030
  G1DFP5CP-106 : project_bridged_gap()  — 30% IIT reduction Critical/High from 2026
  G1DFP5CP-107 : build_scenario_df()    — combined scenario dataframes per tier
  G1DFP5CP-108 : cross_sectional_compare() — county rankings + regional aggregation
  G1DFP5CP-109 : patients_retained_counter() — ART patients retained under Scenario B

Usage:
    from src.projection import (
        project_bau, project_bridged_gap,
        build_scenario_df, cross_sectional_compare,
        patients_retained_counter,
    )

    OR standalone:
        python src/projection.py
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from constants import (
    COUNTY_PROF, PROCESSED_DIR,
    FORECAST_CRITICAL, FORECAST_HIGH,
    FORECAST_MODERATE, FORECAST_LOW,
    FORECAST_NATIONAL,
    IIT_REDUCTION_RATE, BRIDGED_TIERS,
    BRIDGED_START_YEAR, FORECAST_YEAR_END,
    TIER_LABELS, TIER_COLORS,
    REGION_TO_COUNTIES,
)

# ── Constants used throughout
BASE_YEAR      = 2025
FORECAST_YEARS = list(range(BASE_YEAR, FORECAST_YEAR_END + 1))

# Tier → forecast file path mapping
TIER_FORECAST_FILES = {
    "Critical": FORECAST_CRITICAL,
    "High":     FORECAST_HIGH,
    "Moderate": FORECAST_MODERATE,
    "Low":      FORECAST_LOW,
}

# county → region lookup (built from REGION_TO_COUNTIES in constants)
COUNTY_TO_REGION = {
    county: region
    for region, counties in REGION_TO_COUNTIES.items()
    for county in counties
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_county_profiles() -> pd.DataFrame:
    """
    Load county_profiles.csv from data/processed/.
    Validates that the tier column exists (must run after notebook 05).
    """
    if not os.path.exists(COUNTY_PROF):
        raise FileNotFoundError(
            f"county_profiles.csv not found at {COUNTY_PROF}\n"
            "Run notebook 05_model_1_county_clustering.ipynb first."
        )
    df = pd.read_csv(COUNTY_PROF)
    if "tier" not in df.columns:
        raise ValueError(
            "'tier' column missing from county_profiles.csv.\n"
            "Run notebook 05_model_1_county_clustering.ipynb first."
        )
    # Normalise rates to decimal if stored as percentages
    if df["iit_rate"].max() > 1:
        df["iit_rate"] = df["iit_rate"] / 100
    if df["vls_rate_adult"].max() > 1:
        df["vls_rate_adult"] = df["vls_rate_adult"] / 100
    if "art_coverage" in df.columns:
        df["art_coverage"] = df["art_coverage"].clip(0, 1)
    return df


def _tier_mean_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean IIT rate, VLS rate, CGI per tier."""
    return (
        df.groupby("tier")[["iit_rate", "vls_rate_adult", "care_gap_index"]]
        .mean()
        .reset_index()
    )


# ─────────────────────────────────────────────────────────────
# G1DFP5CP-105
# ─────────────────────────────────────────────────────────────

def project_bau(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Scenario A — Business As Usual (flat projection).

    Assumption: since only 2025 data exists, all future years
    keep the same IIT rate, VLS rate, and CGI as 2025.

    Parameters
    ----------
    df : pd.DataFrame, optional
        county_profiles.csv. Loaded automatically if not provided.

    Returns
    -------
    pd.DataFrame
        Columns: tier, year, iit_rate, vls_rate_adult, care_gap_index, scenario
    """
    if df is None:
        df = load_county_profiles()

    tier_rates = _tier_mean_rates(df)
    rows = []

    for _, row in tier_rates.iterrows():
        for year in FORECAST_YEARS:
            rows.append({
                "tier":           row["tier"],
                "year":           year,
                "iit_rate":       row["iit_rate"],
                "vls_rate_adult": row["vls_rate_adult"],
                "care_gap_index": row["care_gap_index"],
                "scenario":       "BAU",
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# G1DFP5CP-106
# ─────────────────────────────────────────────────────────────

def project_bridged_gap(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Scenario B — Bridged Gap.

    Applies a 30% IIT reduction (IIT_REDUCTION_RATE) to
    Critical and High tier counties (BRIDGED_TIERS) starting
    from BRIDGED_START_YEAR (2026). Moderate and Low tiers
    remain unchanged.

    VLS improves proportionally: ΔVLS = −0.5 × ΔIIT
    (conservative estimate from PEPFAR Kenya programme data).

    Parameters
    ----------
    df : pd.DataFrame, optional
        county_profiles.csv. Loaded automatically if not provided.

    Returns
    -------
    pd.DataFrame
        Columns: tier, year, iit_rate, vls_rate_adult,
                 care_gap_index, scenario
    """
    if df is None:
        df = load_county_profiles()

    tier_rates = _tier_mean_rates(df)
    rows = []

    for _, row in tier_rates.iterrows():
        tier        = row["tier"]
        iit_base    = row["iit_rate"]
        vls_base    = row["vls_rate_adult"]
        cgi_base    = row["care_gap_index"]

        for year in FORECAST_YEARS:
            if tier in BRIDGED_TIERS and year >= BRIDGED_START_YEAR:
                iit_rate = iit_base * (1 - IIT_REDUCTION_RATE)
            else:
                iit_rate = iit_base

            # VLS improves as IIT falls
            delta_iit = iit_rate - iit_base
            vls_rate  = min(1.0, max(0.0, vls_base + (-0.5 * delta_iit)))

            rows.append({
                "tier":           tier,
                "year":           year,
                "iit_rate":       round(iit_rate, 6),
                "vls_rate_adult": round(vls_rate, 6),
                "care_gap_index": cgi_base,
                "scenario":       "Bridged",
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# G1DFP5CP-107
# ─────────────────────────────────────────────────────────────

def build_scenario_df(
    df: pd.DataFrame = None,
    save: bool = True,
) -> dict:
    """
    Build combined scenario DataFrames for each tier and national level.

    Runs both project_bau() and project_bridged_gap() and
    combines them into one DataFrame per tier + one national DataFrame.

    Parameters
    ----------
    df : pd.DataFrame, optional
        county_profiles.csv. Loaded automatically if not provided.
    save : bool
        If True, saves CSVs to data/processed/.

    Returns
    -------
    dict with keys:
        'bau'      : BAU projection DataFrame (all tiers)
        'bridged'  : Bridged Gap projection DataFrame (all tiers)
        'combined' : Both scenarios combined
        'national_bau'     : National aggregate BAU
        'national_bridged' : National aggregate Bridged
    """
    if df is None:
        df = load_county_profiles()

    bau     = project_bau(df)
    bridged = project_bridged_gap(df)
    combined = pd.concat([bau, bridged], ignore_index=True)

    # ── Per-tier CSVs (BAU only — matches notebook 07 Cell 7)
    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        for tier in bau["tier"].unique():
            tier_df  = bau[bau["tier"] == tier].copy()
            filename = os.path.join(
                PROCESSED_DIR,
                f"prediction_{tier.lower()}.csv"
            )
            tier_df.to_csv(filename, index=False)

        # ── Tier-level forecast files (BAU + Bridged combined per tier)
        for tier, path in TIER_FORECAST_FILES.items():
            tier_df = combined[combined["tier"] == tier].copy()
            tier_df.to_csv(path, index=False)

    # ── National aggregates
    national_bau = (
        bau.groupby("year")
        .agg(iit_rate=("iit_rate","mean"),
             vls_rate_adult=("vls_rate_adult","mean"))
        .reset_index()
        .assign(scenario="BAU")
    )
    national_bridged = (
        bridged.groupby("year")
        .agg(iit_rate=("iit_rate","mean"),
             vls_rate_adult=("vls_rate_adult","mean"))
        .reset_index()
        .assign(scenario="Bridged")
    )
    national = pd.concat([national_bau, national_bridged], ignore_index=True)

    if save:
        national.to_csv(FORECAST_NATIONAL, index=False)
        print(f"  ✓ Saved national forecast   → {FORECAST_NATIONAL}")
        for tier, path in TIER_FORECAST_FILES.items():
            print(f"  ✓ Saved {tier:<10} forecast → {path}")

    return {
        "bau":              bau,
        "bridged":          bridged,
        "combined":         combined,
        "national_bau":     national_bau,
        "national_bridged": national_bridged,
        "national":         national,
    }


# ─────────────────────────────────────────────────────────────
# G1DFP5CP-108
# ─────────────────────────────────────────────────────────────

def cross_sectional_compare(
    df: pd.DataFrame = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    County rankings and regional aggregation from 2025 baseline data.

    Produces:
    - County ranked by IIT rate (worst first)
    - County ranked by VLS rate (lowest first)
    - County ranked by Care Gap Index (worst first)
    - Regional average IIT rate and VLS rate
    - county_comparison.csv saved to data/processed/

    Parameters
    ----------
    df : pd.DataFrame, optional
        county_profiles.csv. Loaded automatically if not provided.
    save : bool
        If True, saves county_comparison.csv.

    Returns
    -------
    dict with keys:
        'iit_ranking'  : counties ranked by IIT rate descending
        'vls_ranking'  : counties ranked by VLS rate ascending
        'cgi_ranking'  : counties ranked by CGI descending
        'regional'     : regional aggregate IIT + VLS rates
        'comparison_df': full county comparison DataFrame
    """
    if df is None:
        df = load_county_profiles()

    # Add region column using COUNTY_TO_REGION from constants
    df = df.copy()
    df["region"] = df["county"].map(COUNTY_TO_REGION)

    # Rankings
    iit_ranking = (
        df[["county", "tier", "iit_rate"]]
        .sort_values("iit_rate", ascending=False)
        .reset_index(drop=True)
    )
    iit_ranking.index += 1   # rank starts at 1

    vls_ranking = (
        df[["county", "tier", "vls_rate_adult"]]
        .sort_values("vls_rate_adult", ascending=True)
        .reset_index(drop=True)
    )
    vls_ranking.index += 1

    cgi_ranking = (
        df[["county", "tier", "care_gap_index"]]
        .sort_values("care_gap_index", ascending=False)
        .reset_index(drop=True)
    )
    cgi_ranking.index += 1

    # Regional aggregation
    regional = (
        df.groupby("region")[["iit_rate", "vls_rate_adult", "care_gap_index"]]
        .mean()
        .round(4)
        .reset_index()
        .sort_values("iit_rate", ascending=False)
    )

    # Full comparison DataFrame — matches notebook Cell 18
    comparison_df = df[[
        "county", "region", "tier",
        "iit_rate", "vls_rate_adult", "care_gap_index"
    ]].copy()

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        path = os.path.join(PROCESSED_DIR, "county_comparison.csv")
        comparison_df.to_csv(path, index=False)
        print(f"  ✓ Saved county_comparison.csv → {path}")

    return {
        "iit_ranking":   iit_ranking,
        "vls_ranking":   vls_ranking,
        "cgi_ranking":   cgi_ranking,
        "regional":      regional,
        "comparison_df": comparison_df,
    }


# ─────────────────────────────────────────────────────────────
# G1DFP5CP-109
# ─────────────────────────────────────────────────────────────

def patients_retained_counter(
    df: pd.DataFrame = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Calculate patients retained on ART under Scenario B (Bridged Gap)
    compared to Scenario A (BAU) for each county, each year 2025–2030.

    Formula:
        retention_rate      = 1 − iit_rate
        art_retained(year)  = adults_on_art × retention_rate^(year − 2025)
        patients_saved      = art_retained_bridged − art_retained_bau

    Parameters
    ----------
    df : pd.DataFrame, optional
        county_profiles.csv. Loaded automatically if not provided.
    save : bool
        If True, saves patients_retained.csv to data/processed/.

    Returns
    -------
    pd.DataFrame
        Columns: county, tier, year,
                 art_retained_bau, art_retained_bridged, patients_saved
    """
    if df is None:
        df = load_county_profiles()

    rows = []

    for _, row in df.iterrows():
        county   = row["county"]
        tier     = row["tier"]
        iit_base = row["iit_rate"]
        art_base = pd.to_numeric(row.get("adults_on_art", 0), errors="coerce")
        if pd.isna(art_base):
            art_base = 0

        for year in FORECAST_YEARS:
            years_elapsed = year - BASE_YEAR

            # BAU — IIT unchanged
            iit_bau           = iit_base
            retention_bau     = 1 - iit_bau
            art_retained_bau  = art_base * (retention_bau ** years_elapsed)

            # Bridged — 30% IIT reduction for Critical/High from 2026
            if tier in BRIDGED_TIERS and year >= BRIDGED_START_YEAR:
                iit_bridged = iit_base * (1 - IIT_REDUCTION_RATE)
            else:
                iit_bridged = iit_base

            retention_bridged    = 1 - iit_bridged
            art_retained_bridged = art_base * (retention_bridged ** years_elapsed)
            patients_saved       = art_retained_bridged - art_retained_bau

            rows.append({
                "county":               county,
                "tier":                 tier,
                "year":                 year,
                "iit_bau":              round(iit_bau, 6),
                "iit_bridged":          round(iit_bridged, 6),
                "art_retained_bau":     round(art_retained_bau, 0),
                "art_retained_bridged": round(art_retained_bridged, 0),
                "patients_saved":       round(patients_saved, 0),
            })

    result = pd.DataFrame(rows)

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        path = os.path.join(PROCESSED_DIR, "patients_retained.csv")
        result.to_csv(path, index=False)
        print(f"  ✓ Saved patients_retained.csv → {path}")

    # Print summary
    summary_2030 = (
        result[result["year"] == 2030]
        .groupby("tier")
        .agg(
            total_patients_saved=("patients_saved", "sum"),
            n_counties=("county", "count"),
        )
        .reset_index()
    )
    print("\n  Patients additionally retained by 2030 (Bridged vs BAU):")
    print(f"  {'Tier':<12} {'Counties':>8} {'Patients Saved':>16}")
    print(f"  {'─'*38}")
    for _, r in summary_2030.iterrows():
        print(f"  {r['tier']:<12} {int(r['n_counties']):>8} {int(r['total_patients_saved']):>16,}")
    total = int(summary_2030["total_patients_saved"].sum())
    print(f"  {'─'*38}")
    print(f"  {'TOTAL':<12} {'47':>8} {total:>16,}")

    return result


# ─────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────

def run_projection_pipeline(save: bool = True) -> dict:
    """
    Run all five projection functions in order.

    Returns dict with all outputs.
    """
    print("=" * 58)
    print("  src/projection.py — Full Projection Pipeline")
    print("=" * 58)

    df = load_county_profiles()
    print(f"\n  Loaded county_profiles.csv: {df.shape[0]} counties")
    print(f"  Tier distribution: {df['tier'].value_counts().to_dict()}")

    print("\n  Running Scenario A — BAU...")
    bau = project_bau(df)

    print("  Running Scenario B — Bridged Gap...")
    bridged = project_bridged_gap(df)

    print("\n  Building scenario DataFrames and saving CSVs...")
    scenarios = build_scenario_df(df, save=save)

    print("\n  Running cross-sectional county comparison...")
    comparison = cross_sectional_compare(df, save=save)

    print("\n  Counting patients retained under Scenario B...")
    retained = patients_retained_counter(df, save=save)

    print(f"\n{'=' * 58}")
    print("  ✓ Projection pipeline complete")
    print(f"  Outputs saved to: {PROCESSED_DIR}")
    print(f"  Files: prediction_critical/high/moderate/low.csv,")
    print(f"         forecast_*.csv, forecast_national.csv,")
    print(f"         county_comparison.csv, patients_retained.csv")
    print("=" * 58)

    return {
        "df":         df,
        "bau":        bau,
        "bridged":    bridged,
        "scenarios":  scenarios,
        "comparison": comparison,
        "retained":   retained,
    }


# ─────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_projection_pipeline(save=True)