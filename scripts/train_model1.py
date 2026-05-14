"""
scripts/train_model1.py
HIV Care Gap AI — Model 1: KMeans County Clustering Wrapper
=============================================================
Production wrapper for the KMeans county risk tier model.
Mirrors the logic in notebooks/05_model_1_county_clustering.ipynb.

What this script does
---------------------
1. Loads county_profiles.csv (Verah's feature engineering output)
2. Extracts KMEANS_FEATURES from constants.py
3. Scales features using StandardScaler
4. Trains KMeans (k=4) using Dennis's train_kmeans() from src/model_training.py
5. Maps cluster IDs to tier labels (Critical, High, Moderate, Low)
6. Validates with Silhouette Score via src/evaluation.py
7. Saves trained model to models/kmeans_county_tiers.pkl
8. Updates county_profiles.csv with cluster + tier columns

Usage
-----
    # From project root:
    python scripts/train_model1.py

    # From a notebook or main.py:
    import sys, os
    sys.path.append(os.path.abspath('..'))
    from scripts.train_model1 import run_kmeans
    profiles = run_kmeans()

Author : Eve Michelle
Project: HIV Care Gap AI — Phase 5 Capstone
"""

import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle

# ── Root imports
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from sklearn.preprocessing import StandardScaler

# ── Use Dennis's model_training utility
from src.model_training import train_kmeans

# ── Use Dennis's evaluation utility
from src.evaluation import calculate_silhouette_score

from constants import (
    COUNTY_PROF,
    KMEANS_MODEL,
    MODELS_DIR,
    PROCESSED_DIR,
    KMEANS_K,
    KMEANS_RANDOM_STATE,
    KMEANS_FEATURES,
    TIER_LABELS,
    TIER_COLORS,
)


# ============================================================
# HELPERS
# ============================================================

def load_county_profiles():
    """
    Load county_profiles.csv from constants.py path.
    Raises FileNotFoundError if Verah's feature engineering
    has not been run yet.
    """
    if not os.path.exists(COUNTY_PROF):
        raise FileNotFoundError(
            f'county_profiles.csv not found at {COUNTY_PROF}.\n'
            'Run 04_feature_engineering.ipynb first.'
        )
    df = pd.read_csv(COUNTY_PROF)
    print(f'  Loaded county_profiles: {df.shape}')
    return df


def validate_features(df):
    """
    Confirm all KMEANS_FEATURES are present in the DataFrame
    and contain no missing values.

    KMeans crashes on NaNs — raising here gives a clear error
    message rather than a cryptic sklearn failure.
    """
    # Check columns exist
    missing_cols = [f for f in KMEANS_FEATURES if f not in df.columns]
    if missing_cols:
        raise ValueError(
            f'Missing KMEANS_FEATURES in county_profiles: {missing_cols}\n'
            f'Available columns: {df.columns.tolist()}'
        )

    # Check no NaN values in feature columns
    n_missing = df[KMEANS_FEATURES].isnull().sum().sum()
    if n_missing > 0:
        raise ValueError(
            f'Missing values found in KMEANS_FEATURES: '
            f'{df[KMEANS_FEATURES].isnull().sum().to_dict()}\n'
            'Impute before clustering.'
        )

    print(f'  KMEANS_FEATURES confirmed (no NaNs): {KMEANS_FEATURES}')


def assign_tier_labels(df):
    """
    Map KMeans cluster IDs (0, 1, 2, 3) to tier labels
    (Critical, High, Moderate, Low) based on mean CGI per cluster.

    Clusters are ranked by mean care_gap_index:
        Highest CGI -> Critical
        Next        -> High
        Next        -> Moderate
        Lowest CGI  -> Low

    Parameters
    ----------
    df : DataFrame with cluster column added

    Returns
    -------
    df              : DataFrame with tier column added
    cluster_to_tier : dict mapping cluster_id -> tier label
    """
    # Rank clusters by mean care_gap_index (descending = worst first)
    cluster_cgi = (
        df.groupby('cluster')['care_gap_index']
        .mean()
        .sort_values(ascending=False)
    )

    # Map: cluster_id -> tier label
    cluster_to_tier = {
        cluster_id: TIER_LABELS[rank]
        for rank, cluster_id in enumerate(cluster_cgi.index)
    }

    df['tier'] = df['cluster'].map(cluster_to_tier)

    print('  Cluster -> Tier mapping:')
    for cluster_id, tier in cluster_to_tier.items():
        mean_cgi = cluster_cgi[cluster_id]
        n        = (df['cluster'] == cluster_id).sum()
        color    = TIER_COLORS.get(tier, '#808080')
        print(f'    Cluster {cluster_id} -> {tier:<10} | '
              f'mean CGI: {mean_cgi:.2f} | counties: {n} | color: {color}')

    return df, cluster_to_tier


# ============================================================
# CORE TRAINING
# ============================================================

def run_kmeans(save=True):
    """
    Run the full KMeans training pipeline.

    Steps
    -----
    1. Load county_profiles.csv
    2. Validate KMEANS_FEATURES (present + no NaNs)
    3. Scale features with StandardScaler
    4. Fit KMeans via Dennis's train_kmeans() from src/model_training.py
    5. Assign tier labels ranked by mean CGI
    6. Validate with calculate_silhouette_score() from src/evaluation.py
    7. Save model bundle to KMEANS_MODEL path (constants.py)
    8. Save updated county_profiles.csv with cluster + tier

    Parameters
    ----------
    save : bool
        If True, saves model + updated county_profiles.csv

    Returns
    -------
    df : DataFrame — county_profiles with cluster + tier columns
    """

    print('=' * 55)
    print('Model 1 — KMeans County Clustering')
    print('=' * 55)

    # ── Step 1: Load
    print('\n[1] Loading county_profiles.csv...')
    df = load_county_profiles()

    # ── Step 2: Validate features
    print('\n[2] Validating KMEANS_FEATURES...')
    validate_features(df)

    # ── Step 3: Extract and scale
    print('\n[3] Scaling features...')
    X = df[KMEANS_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f'  Feature matrix shape: {X_scaled.shape}')
    print(f'  Features used: {KMEANS_FEATURES}')

    # ── Step 4: Fit KMeans via Dennis's train_kmeans()
    print(f'\n[4] Fitting KMeans (k={KMEANS_K}, '
          f'random_state={KMEANS_RANDOM_STATE})...')
    kmeans, labels = train_kmeans(
        X_scaled,
        n_clusters=KMEANS_K,
        random_state=KMEANS_RANDOM_STATE,
    )
    df['cluster'] = labels
    print(f'  KMeans fitted | inertia: {kmeans.inertia_:.2f}')
    print(f'  Cluster distribution:')
    for c, n in pd.Series(labels).value_counts().sort_index().items():
        print(f'    Cluster {c}: {n} counties')

    # ── Step 5: Assign tier labels
    print('\n[5] Assigning tier labels...')
    df, cluster_to_tier = assign_tier_labels(df)

    print('\n  Tier distribution:')
    for tier in TIER_LABELS:
        n        = (df['tier'] == tier).sum()
        counties = df[df['tier'] == tier]['county'].tolist()
        print(f'    {tier:<10}: {n} counties')
        print(f'               {counties}')

    # ── Step 6: Silhouette Score via Dennis's evaluation utility
    print('\n[6] Validating with Silhouette Score...')
    sil_score = calculate_silhouette_score(X_scaled, labels)

    # ── Step 7: Save model bundle
    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Save KMeans + scaler + metadata together in one pkl
        model_bundle = {
            'kmeans'          : kmeans,
            'scaler'          : scaler,
            'features'        : KMEANS_FEATURES,
            'cluster_to_tier' : cluster_to_tier,
            'silhouette_score': sil_score,
            'n_clusters'      : KMEANS_K,
        }
        with open(KMEANS_MODEL, 'wb') as f:
            pickle.dump(model_bundle, f)

        size_kb = os.path.getsize(KMEANS_MODEL) / 1024
        print(f'\n[7] Model saved -> {KMEANS_MODEL}  ({size_kb:.1f} KB)')

        # ── Step 8: Update county_profiles.csv
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_csv(COUNTY_PROF, index=False)
        print(f'[8] Updated -> {COUNTY_PROF}')

    print('\nAll done ✓')
    print('=' * 55)

    return df


# ============================================================
# PREDICT TIER FOR A NEW COUNTY PROFILE  (used by dashboard)
# ============================================================

def predict_tier(feature_values: dict) -> str:
    """
    Predict tier for a new county profile using the saved model.

    Parameters
    ----------
    feature_values : dict
        Keys must match KMEANS_FEATURES from constants.py.
        e.g. {'iit_rate': 0.15, 'vls_rate': 0.93}

    Returns
    -------
    str — predicted tier label (Critical / High / Moderate / Low)
    """
    if not os.path.exists(KMEANS_MODEL):
        raise FileNotFoundError(
            f'Model not found at {KMEANS_MODEL}. '
            'Run train_model1.py first.'
        )

    with open(KMEANS_MODEL, 'rb') as f:
        bundle = pickle.load(f)

    kmeans          = bundle['kmeans']
    scaler          = bundle['scaler']
    features        = bundle['features']
    cluster_to_tier = bundle['cluster_to_tier']

    X        = np.array([[feature_values[f] for f in features]])
    X_scaled = scaler.transform(X)
    cluster  = kmeans.predict(X_scaled)[0]

    return cluster_to_tier[cluster]


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == '__main__':
    profiles = run_kmeans(save=True)
    print()
    print('Final tier assignments:')
    print(
        profiles[['county', 'tier', 'care_gap_index',
                   'iit_rate', 'vls_rate']]
        .sort_values('care_gap_index', ascending=False)
        .to_string(index=False)
    )