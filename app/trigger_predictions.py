"""
app/trigger_predictions.py
═══════════════════════════════════════════════════════════════
Tasks:
  G1DFP5CP-187 : argparse --new-data-year flag
  G1DFP5CP-188 : Retraining sequence:
                   clean_nsdcc → merge → feature_engineering
                   → train_model1 → train_model2 → train_model3
  G1DFP5CP-189 : Log tier changes year-on-year

Usage:
    python app/trigger_predictions.py --new-data-year 2026

What it does when new NSDCC data arrives:
    1. Re-clean all 4 NSDCC files (IIT, VLT, HTS, ART)
    2. Re-merge into nsdcc_clean.csv
    3. Re-run feature engineering → county_profiles.csv
    4. Retrain Model 1 (KMeans) → new tier assignments
    5. Retrain Model 2 (Logistic Regression) → updated odds ratios
    6. Re-run Model 3 projections → updated forecast CSVs
    7. Log which counties changed tier (G1DFP5CP-189)
    8. Print instructions to restart Streamlit
"""

import argparse
import os
import sys
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import pandas as pd

# ── Add project root to path
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import constants as c


# ─────────────────────────────────────────────────────────────
# G1DFP5CP-189 — TIER CHANGE LOGGER
# ─────────────────────────────────────────────────────────────

def log_tier_changes(new_profiles: pd.DataFrame, year: int) -> dict:
    """
    Compare new county tier assignments vs the previous year's log.
    Save a running tier_change_log.json in data/processed/.

    Parameters
    ----------
    new_profiles : pd.DataFrame  — county_profiles.csv with new tiers
    year         : int           — the new data year just processed

    Returns
    -------
    dict  — summary of changes { year, changes, n_changes }
    """
    TIER_LOG = os.path.join(c.PROCESSED_DIR, "tier_change_log.json")

    # Tier order (lower index = worse)
    TIER_ORDER = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3}

    # Load existing log
    if os.path.exists(TIER_LOG):
        with open(TIER_LOG, "r") as f:
            log = json.load(f)
        prev_entries = log.get("entries", [])
        prev_tiers   = {
            e["county"]: e["tier"]
            for e in (prev_entries[-1].get("county_tiers", []) if prev_entries else [])
        }
    else:
        log        = {"entries": []}
        prev_tiers = {}

    # Current assignments
    current_tiers = dict(zip(new_profiles["county"], new_profiles["tier"]))

    # Find counties that changed tier
    changes = []
    for county, new_tier in current_tiers.items():
        old_tier = prev_tiers.get(county)
        if old_tier and old_tier != new_tier:
            old_rank = TIER_ORDER.get(old_tier, 99)
            new_rank = TIER_ORDER.get(new_tier, 99)
            changes.append({
                "county":    county,
                "from_tier": old_tier,
                "to_tier":   new_tier,
                "direction": "IMPROVED" if new_rank > old_rank else "WORSENED",
            })

    # Build log entry
    entry = {
        "year":         year,
        "timestamp":    datetime.now().isoformat(),
        "county_tiers": [{"county": k, "tier": v} for k, v in current_tiers.items()],
        "tier_changes": changes,
        "n_changes":    len(changes),
        "tier_counts":  new_profiles["tier"].value_counts().to_dict(),
    }
    log["entries"].append(entry)

    with open(TIER_LOG, "w") as f:
        json.dump(log, f, indent=2)

    # Print summary
    print(f"\n  ── Tier Change Log ({year}) ──")
    print(f"  Counties that changed tier: {len(changes)}")
    if changes:
        for ch in sorted(changes, key=lambda x: x["direction"]):
            icon = "📈" if ch["direction"] == "IMPROVED" else "📉"
            print(f"  {icon}  {ch['county']:<22} {ch['from_tier']:>10} → {ch['to_tier']}")
    else:
        print("  No tier changes — all counties maintained same tier")

    print(f"\n  2025 tier distribution:")
    for tier in ["Critical", "High", "Moderate", "Low"]:
        n = new_profiles["tier"].value_counts().get(tier, 0)
        print(f"    {tier:<12} {n} counties")

    print(f"  Log saved → {TIER_LOG}")
    return {"year": year, "changes": changes, "n_changes": len(changes)}


# ─────────────────────────────────────────────────────────────
# G1DFP5CP-188 — RETRAINING SEQUENCE
# ─────────────────────────────────────────────────────────────

def run_retraining_pipeline(year: int):
    """
    Full retraining sequence:
    clean_nsdcc → merge → feature_engineering
    → train_model1 → train_model2 → train_model3 → log_tier_changes
    """
    print("=" * 60)
    print(f"  TRIGGER PREDICTIONS — New Data Year: {year}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    passed = []
    failed = []

    # ── Step 1: Clean NSDCC
    print("\n  [1/7] Cleaning NSDCC files...")
    try:
        from scripts.clean_data import clean_nsdcc
        clean_nsdcc()
        passed.append("clean_nsdcc")
        print("  ✓ Step 1 done")
    except Exception as e:
        failed.append(("clean_nsdcc", str(e)))
        print(f"  ✗ Step 1 FAILED: {e}")
        print("  Cannot continue — fix cleaning step first.")
        _print_summary(passed, failed)
        return

    # ── Step 2: Merge NSDCC
    print("\n  [2/7] Merging NSDCC files → nsdcc_clean.csv...")
    try:
        from scripts.merge_data import merge_nsdcc_files
        merge_nsdcc_files()
        passed.append("merge_nsdcc_files")
        print("  ✓ Step 2 done")
    except Exception as e:
        failed.append(("merge_nsdcc_files", str(e)))
        print(f"  ✗ Step 2 FAILED: {e}")
        _print_summary(passed, failed)
        return

    # ── Step 3: Feature engineering → county_profiles.csv
    print("\n  [3/7] Running feature engineering → county_profiles.csv...")
    try:
        nsdcc_df = pd.read_csv(c.NSDCC_CLEAN)
        from src.feature_engineering import run_feature_engineering
        run_feature_engineering(nsdcc_df, save=True)
        passed.append("feature_engineering")
        print("  ✓ Step 3 done")
    except Exception as e:
        failed.append(("feature_engineering", str(e)))
        print(f"  ✗ Step 3 FAILED: {e}")
        _print_summary(passed, failed)
        return

    # ── Step 4: Train Model 1 (KMeans → new tier assignments)
    print("\n  [4/7] Retraining Model 1 — KMeans clustering...")
    try:
        from scripts.train_model1 import train_kmeans
        train_kmeans()
        passed.append("train_model1")
        print("  ✓ Step 4 done")
    except Exception as e:
        failed.append(("train_model1", str(e)))
        print(f"  ✗ Step 4 FAILED: {e}")
        _print_summary(passed, failed)
        return

    # ── Step 5: Train Model 2 (Logistic Regression)
    print("\n  [5/7] Retraining Model 2 — Logistic Regression + odds ratios...")
    try:
        from scripts.train_model2 import train_logreg
        train_logreg()
        passed.append("train_model2")
        print("  ✓ Step 5 done")
    except Exception as e:
        # Model 2 failure is non-fatal — projections can still run
        failed.append(("train_model2", str(e)))
        print(f"  ⚠ Step 5 FAILED (non-fatal): {e}")
        print("  Continuing to Model 3...")

    # ── Step 6: Train Model 3 (scenario projections)
    print("\n  [6/7] Rerunning Model 3 — scenario projections...")
    try:
        from src.projection import run_projection_pipeline
        run_projection_pipeline(save=True)
        passed.append("train_model3")
        print("  ✓ Step 6 done")
    except Exception as e:
        failed.append(("train_model3", str(e)))
        print(f"  ✗ Step 6 FAILED: {e}")

    # ── Step 7: Log tier changes (G1DFP5CP-189)
    print("\n  [7/7] Logging tier changes year-on-year...")
    try:
        new_profiles = pd.read_csv(c.COUNTY_PROF)
        log_tier_changes(new_profiles, year)
        passed.append("tier_logging")
        print("  ✓ Step 7 done")
    except Exception as e:
        failed.append(("tier_logging", str(e)))
        print(f"  ✗ Step 7 FAILED: {e}")

    _print_summary(passed, failed)


def _print_summary(passed: list, failed: list):
    """Print pipeline completion summary."""
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Steps passed : {len(passed)}/7  {passed}")
    if failed:
        print(f"  Steps failed : {len(failed)}")
        for step, err in failed:
            print(f"    ✗ {step}: {err}")
    print()
    print("  Restart Streamlit to load updated data:")
    print(f"  streamlit run app/streamlit_app.py --server.port {c.STREAMLIT_PORT}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# G1DFP5CP-187 — ARGPARSE ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "HIV Care Gap AI — Retrain all models when new NSDCC data arrives.\n\n"
            "Steps: clean_nsdcc → merge → feature_engineering\n"
            "       → train_model1 → train_model2 → train_model3\n"
            "       → log_tier_changes"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--new-data-year",
        type=int,
        required=True,
        metavar="YEAR",
        help="Year of the new NSDCC data (e.g. 2026). Used to label the tier change log.",
    )
    args = parser.parse_args()
    run_retraining_pipeline(year=args.new_data_year)