"""
scripts/train_model3.py
HIV Care Gap AI — Model 3: Scenario Projection + Cross-Sectional Comparison Wrapper
=====================================================================================
Production wrapper for Model 3.
Mirrors the logic in notebooks/07_model_3_projection.ipynb.

What this script does
---------------------
Component A — Scenario-Based Projection (2025 → 2030):
    1. Loads county_profiles.csv (Verah's feature engineering + Naomi's tier labels — must have tier column)
    2. Runs Scenario A (BAU): current 2025 IIT + VLS rates held flat to 2030
    3. Runs Scenario B (Bridged Gap): 30% IIT reduction in Critical + High tiers
       from 2026, VLS improves proportionally (ΔVLS = −0.5 × ΔIIT)
    4. Saves per-tier forecast CSVs (forecast_critical/high/moderate/low.csv)
    5. Saves national forecast CSV (forecast_national.csv)
    6. Runs patients_retained_counter(): ART patients additionally retained
       under Scenario B vs Scenario A per county per year

Component B — Cross-Sectional Comparison (2025 snapshot):
    7. Ranks all 47 counties by IIT rate, VLS rate, and Care Gap Index
    8. Computes regional average IIT + VLS rates
    9. Saves county_comparison.csv

Final:
    10. Saves full model bundle to models/model3_scenario.pkl

Usage
-----
    # From project root:
    python scripts/train_model3.py

    # From a notebook or main.py:
    import sys, os
    sys.path.append(os.path.abspath('..'))
    from scripts.train_model3 import run_model3
    results = run_model3()

Author : Eve Michelle
Project: HIV Care Gap AI — Phase 5 Capstone
"""

import sys
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# ── Root imports
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

# ── Use Naomi's projection utilities from src/projection.py
from src.projection import (
    load_county_profiles,
    project_bau,
    project_bridged_gap,
    build_scenario_df,
    cross_sectional_compare,
    patients_retained_counter,
)

from constants import (
    COUNTY_PROF,
    PROCESSED_DIR,
    MODELS_DIR,
    MODEL3_BUNDLE,
    FORECAST_CRITICAL,
    FORECAST_HIGH,
    FORECAST_MODERATE,
    FORECAST_LOW,
    FORECAST_NATIONAL,
    TIER_LABELS,
    TIER_COLORS,
    IIT_REDUCTION_RATE,
    BRIDGED_TIERS,
    BRIDGED_START_YEAR,
    FORECAST_YEAR_END,
)


# ============================================================
# HELPERS
# ============================================================

def validate_county_profiles(df):
    """
    Confirm county_profiles.csv has the tier column and
    the required rate columns before running projections.

    Raises ValueError with a clear message if validation fails.
    """
    required_cols = ['county', 'tier', 'iit_rate', 'vls_rate_adult',
                     'care_gap_index', 'adults_on_art']

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f'Missing required columns in county_profiles.csv: {missing_cols}\n'
            f'Run 04_feature_engineering.ipynb + 05_model_1_county_clustering.ipynb first.'
        )

    if 'tier' not in df.columns or df['tier'].isnull().all():
        raise ValueError(
            "'tier' column is missing or empty in county_profiles.csv.\n"
            'Run 05_model_1_county_clustering.ipynb first.'
        )

    n_counties  = df['county'].nunique()
    n_tiers     = df['tier'].nunique()
    missing_iit = df['iit_rate'].isnull().sum()
    missing_vls = df['vls_rate_adult'].isnull().sum()

    ok = n_counties == 47 and n_tiers == 4 and missing_iit == 0 and missing_vls == 0

    print(f'  {"OK" if ok else "WARNING"} county_profiles:')
    print(f'    Counties  : {n_counties} (expected 47)')
    print(f'    Tiers     : {n_tiers} (expected 4 — {sorted(df["tier"].unique())})')
    print(f'    IIT nulls : {missing_iit}')
    print(f'    VLS nulls : {missing_vls}')

    if not ok:
        raise ValueError(
            'county_profiles.csv failed validation. '
            'Check notebooks 04 and 05 before running Model 3.'
        )


def print_tier_baseline(df):
    """
    Print 2025 baseline IIT + VLS rates per tier.
    Used as a quick sense-check before running projections.
    """
    tier_rates = (
        df.groupby('tier')[['iit_rate', 'vls_rate_adult', 'care_gap_index']]
        .mean()
        .round(4)
    )
    print('\n  2025 baseline rates per tier:')
    print(f'  {"Tier":<12} {"IIT Rate":>10} {"VLS Rate":>10} {"CGI":>8}')
    print(f'  {"─" * 42}')
    for tier in TIER_LABELS:
        if tier in tier_rates.index:
            row = tier_rates.loc[tier]
            color = TIER_COLORS.get(tier, '')
            print(
                f'  {tier:<12} {row["iit_rate"]:>10.4f} '
                f'{row["vls_rate_adult"]:>10.4f} '
                f'{row["care_gap_index"]:>8.2f}'
            )


# ============================================================
# COMPONENT A — SCENARIO PROJECTION
# ============================================================

def run_component_a(df, save=True):
    """
    Run Scenario A (BAU) and Scenario B (Bridged Gap) projections.

    Parameters
    ----------
    df : pd.DataFrame — county_profiles.csv with tier column
    save : bool — if True, saves all forecast CSVs

    Returns
    -------
    dict with keys: bau, bridged, combined, national_projection
    """
    print('\n[Component A] Scenario-Based Projection (2025 → 2030)')
    print(f'  IIT reduction rate  : {IIT_REDUCTION_RATE * 100:.0f}%')
    print(f'  Bridged tiers       : {BRIDGED_TIERS}')
    print(f'  Intervention start  : {BRIDGED_START_YEAR}')
    print(f'  Forecast end        : {FORECAST_YEAR_END}')

    # ── Scenario A — BAU
    print('\n  Running Scenario A (BAU flat projection)...')
    scenario_a = project_bau(df)
    print(f'    Shape: {scenario_a.shape}')

    # ── Scenario B — Bridged Gap
    print('\n  Running Scenario B (30% IIT reduction Critical + High)...')
    scenario_b = project_bridged_gap(df)
    print(f'    Shape: {scenario_b.shape}')

    # ── Validate VLS cap
    vls_max = scenario_b['vls_rate_adult'].max()
    vls_ok  = vls_max <= 1.0
    print(f'\n  VLS sanity check (must be ≤ 1.0): {vls_max:.6f} {"✓" if vls_ok else "✗ EXCEEDS 1.0"}')
    if not vls_ok:
        raise ValueError('Scenario B VLS rate exceeds 1.0 — check IIT_REDUCTION_RATE in constants.py')

    # ── Combined
    combined = pd.concat([scenario_a, scenario_b], ignore_index=True)
    print(f'\n  Combined scenarios shape: {combined.shape}')
    print(f'  Scenarios present: {sorted(combined["scenario"].unique())}')

    # ── National projection (mean across tiers, weighted equally)
    national_projection = (
        combined.groupby(['year', 'scenario'])[['iit_rate', 'vls_rate_adult']]
        .mean()
        .round(6)
        .reset_index()
    )
    print(f'  National projection shape: {national_projection.shape}')

    # ── Save per-tier CSVs
    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)

        tier_file_map = {
            'Critical': FORECAST_CRITICAL,
            'High':     FORECAST_HIGH,
            'Moderate': FORECAST_MODERATE,
            'Low':      FORECAST_LOW,
        }

        print('\n  Saving per-tier forecast CSVs...')
        for tier, path in tier_file_map.items():
            tier_df = combined[combined['tier'] == tier].copy()
            tier_df.to_csv(path, index=False)
            size_kb = os.path.getsize(path) / 1024
            print(f'    {tier:<10} → {path}  ({size_kb:.1f} KB, {len(tier_df)} rows)')

        # Save national
        national_projection.to_csv(FORECAST_NATIONAL, index=False)
        size_kb = os.path.getsize(FORECAST_NATIONAL) / 1024
        print(f'    {"National":<10} → {FORECAST_NATIONAL}  ({size_kb:.1f} KB)')

    # ── Scenario B improvement summary
    print('\n  Scenario B vs Scenario A improvement at 2030:')
    a_2030 = scenario_a[scenario_a['year'] == FORECAST_YEAR_END][['tier', 'iit_rate', 'vls_rate_adult']]
    b_2030 = scenario_b[scenario_b['year'] == FORECAST_YEAR_END][['tier', 'iit_rate', 'vls_rate_adult']]
    print(f'  {"Tier":<12} {"IIT A":>8} {"IIT B":>8} {"IIT Δ":>8} {"VLS A":>8} {"VLS B":>8}')
    print(f'  {"─" * 56}')
    for tier in TIER_LABELS:
        a_row = a_2030[a_2030['tier'] == tier]
        b_row = b_2030[b_2030['tier'] == tier]
        if not a_row.empty and not b_row.empty:
            iit_a = a_row['iit_rate'].values[0]
            iit_b = b_row['iit_rate'].values[0]
            vls_a = a_row['vls_rate_adult'].values[0]
            vls_b = b_row['vls_rate_adult'].values[0]
            delta = iit_b - iit_a
            print(
                f'  {tier:<12} {iit_a:>8.4f} {iit_b:>8.4f} '
                f'{delta:>+8.4f} {vls_a:>8.4f} {vls_b:>8.4f}'
            )

    return {
        'bau':                 scenario_a,
        'bridged':             scenario_b,
        'combined':            combined,
        'national_projection': national_projection,
    }


# ============================================================
# COMPONENT B — CROSS-SECTIONAL COMPARISON
# ============================================================

def run_component_b(df, save=True):
    """
    Run cross-sectional comparison of all 47 counties
    using 2025 snapshot data.

    Parameters
    ----------
    df : pd.DataFrame — county_profiles.csv with tier column
    save : bool — if True, saves county_comparison.csv

    Returns
    -------
    dict with keys: iit_ranking, vls_ranking, cgi_ranking,
                    regional, comparison_df
    """
    print('\n[Component B] Cross-Sectional Comparison (2025 snapshot)')

    results = cross_sectional_compare(df, save=save)

    # Print top 5 worst counties by IIT
    print('\n  Top 5 counties by IIT rate (worst → best care retention):')
    print(f'  {"Rank":<6} {"County":<20} {"Tier":<12} {"IIT Rate":>10}')
    print(f'  {"─" * 52}')
    for rank, row in results['iit_ranking'].head(5).iterrows():
        print(f'  {rank:<6} {row["county"]:<20} {row["tier"]:<12} {row["iit_rate"]:>10.4f}')

    # Print top 5 worst counties by VLS
    print('\n  Top 5 counties by VLS rate (lowest suppression):')
    print(f'  {"Rank":<6} {"County":<20} {"Tier":<12} {"VLS Rate":>10}')
    print(f'  {"─" * 52}')
    for rank, row in results['vls_ranking'].head(5).iterrows():
        print(f'  {rank:<6} {row["county"]:<20} {row["tier"]:<12} {row["vls_rate_adult"]:>10.4f}')

    # Print regional breakdown
    print('\n  Regional IIT + VLS averages:')
    print(f'  {"Region":<18} {"IIT Rate":>10} {"VLS Rate":>10} {"CGI":>8}')
    print(f'  {"─" * 50}')
    for _, row in results['regional'].iterrows():
        print(
            f'  {str(row["region"]):<18} '
            f'{row["iit_rate"]:>10.4f} '
            f'{row["vls_rate_adult"]:>10.4f} '
            f'{row["care_gap_index"]:>8.2f}'
        )

    if save:
        path = os.path.join(PROCESSED_DIR, 'county_comparison.csv')
        size_kb = os.path.getsize(path) / 1024
        print(f'\n  Saved county_comparison.csv → {path}  ({size_kb:.1f} KB)')

    return results


# ============================================================
# PATIENTS RETAINED COUNTER
# ============================================================

def run_patients_retained(df, save=True):
    """
    Calculate additional patients retained on ART under Scenario B
    vs Scenario A for each county from 2025 to 2030.

    Parameters
    ----------
    df : pd.DataFrame — county_profiles.csv with tier column
    save : bool — if True, saves patients_retained.csv

    Returns
    -------
    pd.DataFrame — county × year × patients_saved
    """
    print('\n[Patients Retained] Counting ART patients saved under Scenario B...')
    patients_df = patients_retained_counter(df, save=save)
    print(f'  patients_retained.csv shape: {patients_df.shape}')
    return patients_df


# ============================================================
# SAVE MODEL BUNDLE
# ============================================================

def save_model3_bundle(component_a, component_b, patients_df, save=True):
    """
    Save the full Model 3 bundle to models/model3_scenario.pkl.

    Bundle contents:
        scenario_a         : BAU projection DataFrame
        scenario_b         : Bridged Gap projection DataFrame
        combined           : Both scenarios combined
        national_projection: National-level projections
        comparison_df      : Full county comparison table
        iit_ranking        : Counties ranked by IIT rate
        vls_ranking        : Counties ranked by VLS rate
        cgi_ranking        : Counties ranked by CGI
        regional           : Regional aggregations
        patients_retained  : Patients saved per county per year
        metadata           : Run parameters from constants.py

    Parameters
    ----------
    component_a : dict — output of run_component_a()
    component_b : dict — output of run_component_b()
    patients_df : DataFrame — output of run_patients_retained()
    save : bool
    """
    bundle = {
        # Scenario projections
        'scenario_a':          component_a['bau'],
        'scenario_b':          component_a['bridged'],
        'combined':            component_a['combined'],
        'national_projection': component_a['national_projection'],

        # Cross-sectional comparison
        'comparison_df':       component_b['comparison_df'],
        'iit_ranking':         component_b['iit_ranking'],
        'vls_ranking':         component_b['vls_ranking'],
        'cgi_ranking':         component_b['cgi_ranking'],
        'regional':            component_b['regional'],

        # Patients retained counter
        'patients_retained':   patients_df,

        # Run metadata
        'metadata': {
            'iit_reduction_rate': IIT_REDUCTION_RATE,
            'bridged_tiers':      BRIDGED_TIERS,
            'bridged_start_year': BRIDGED_START_YEAR,
            'forecast_year_end':  FORECAST_YEAR_END,
            'tier_labels':        TIER_LABELS,
        }
    }

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(bundle, MODEL3_BUNDLE)
        size_kb = os.path.getsize(MODEL3_BUNDLE) / 1024
        print(f'\n  Model bundle saved → {MODEL3_BUNDLE}  ({size_kb:.1f} KB)')

    return bundle


# ============================================================
# CORE PIPELINE
# ============================================================

def run_model3(save=True):
    """
    Run the full Model 3 pipeline.

    Steps
    -----
    1. Load + validate county_profiles.csv (must have tier column from notebook 05)
    2. Print 2025 baseline rates per tier
    3. Component A: Run Scenario A (BAU) + Scenario B (Bridged Gap) projections
       → Save per-tier forecast CSVs + national forecast CSV
    4. Component B: Cross-sectional comparison of all 47 counties
       → Save county_comparison.csv
    5. Run patients_retained_counter()
       → Save patients_retained.csv
    6. Save full model bundle to models/model3_scenario.pkl
    7. Print final validation summary

    Parameters
    ----------
    save : bool
        If True, saves all CSVs and the model bundle.

    Returns
    -------
    dict with keys:
        df          : county_profiles DataFrame
        component_a : Scenario projection outputs
        component_b : Cross-sectional comparison outputs
        patients    : Patients retained DataFrame
        bundle      : Full model bundle
    """

    print('=' * 58)
    print('Model 3 — Scenario Projection + Cross-Sectional Comparison')
    print('=' * 58)

    # ── Step 1: Load + validate county_profiles.csv
    print('\n[1] Loading county_profiles.csv...')
    df = load_county_profiles()
    print(f'    Shape: {df.shape}')

    print('\n[2] Validating county_profiles...')
    validate_county_profiles(df)

    # ── Step 2: Baseline rates
    print('\n[3] 2025 Baseline Summary')
    print_tier_baseline(df)

    # ── Step 3: Component A — Scenario Projection
    print('\n[4] Running Component A...')
    component_a = run_component_a(df, save=save)

    # ── Step 4: Component B — Cross-Sectional Comparison
    print('\n[5] Running Component B...')
    component_b = run_component_b(df, save=save)

    # ── Step 5: Patients Retained Counter
    print('\n[6] Running Patients Retained Counter...')
    patients_df = run_patients_retained(df, save=save)

    # ── Step 6: Save model bundle
    print('\n[7] Saving model bundle...')
    bundle = save_model3_bundle(component_a, component_b, patients_df, save=save)

    # ── Step 7: Final validation
    print('\n[8] Final Validation')
    print(f'  {"Output":<30} {"Shape":>12} {"Status"}')
    print(f'  {"─" * 55}')

    checks = [
        ('Scenario A (BAU)',          component_a['bau'].shape,                 component_a['bau'].shape[0] == 24),
        ('Scenario B (Bridged Gap)',   component_a['bridged'].shape,             component_a['bridged'].shape[0] == 24),
        ('Combined scenarios',         component_a['combined'].shape,            component_a['combined'].shape[0] == 48),
        ('National projection',        component_a['national_projection'].shape, component_a['national_projection'].shape[0] == 12),
        ('County comparison',          component_b['comparison_df'].shape,       component_b['comparison_df'].shape[0] == 47),
        ('Patients retained',          patients_df.shape,                        patients_df.shape[0] == 282),
    ]

    all_ok = True
    for label, shape, ok in checks:
        status = '✓' if ok else '✗ UNEXPECTED'
        all_ok = all_ok and ok
        print(f'  {label:<30} {str(shape):>12}  {status}')

    # Scenarios present
    scenarios = sorted(component_a['combined']['scenario'].unique())
    print(f'\n  Scenarios in combined_df: {scenarios}')

    # VLS cap check
    vls_max = component_a['bridged']['vls_rate_adult'].max()
    print(f'  Scenario B VLS max     : {vls_max:.6f}  {"✓" if vls_max <= 1.0 else "✗ EXCEEDS 1.0"}')

    print(f'\n{"=" * 58}')
    print(f'  {"All done ✓" if all_ok else "Completed with warnings ⚠"}')
    print('=' * 58)

    return {
        'df':          df,
        'component_a': component_a,
        'component_b': component_b,
        'patients':    patients_df,
        'bundle':      bundle,
    }


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == '__main__':
    results = run_model3(save=True)

    print()
    print('Files written:')
    for path in [
        FORECAST_CRITICAL, FORECAST_HIGH,
        FORECAST_MODERATE, FORECAST_LOW,
        FORECAST_NATIONAL, MODEL3_BUNDLE,
        os.path.join(PROCESSED_DIR, 'county_comparison.csv'),
        os.path.join(PROCESSED_DIR, 'patients_retained.csv'),
    ]:
        exists = os.path.exists(path)
        size   = f'{os.path.getsize(path) / 1024:.1f} KB' if exists else 'MISSING'
        print(f'  {"✓" if exists else "✗"} {path}  ({size})')