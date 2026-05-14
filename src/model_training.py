import os
import joblib
import pandas as pd
import numpy as np
from typing import Any, Optional, Tuple, Dict
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Importing project-wide constants
import src.constants as c

def train_kmeans(
    X: pd.DataFrame, 
    n_clusters: int = c.KMEANS_K, 
    random_state: int = c.KMEANS_RANDOM_STATE
) -> Tuple[KMeans, np.ndarray]:
    """
    Fits a KMeans model for county care-gap tiering.

    Args:
        X: Scaled feature set (iit_rate, vls_rate, etc.).
        n_clusters: Number of tiers (default from constants).
        random_state: Seed for reproducibility.

    Returns:
        Tuple containing the fitted KMeans model and generated cluster labels.
    """
    print(f"Training Model 1: KMeans with k={n_clusters}...")
    
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = model.fit_predict(X)
    
    print("✓ Model 1 training complete.")
    return model, labels

def train_xgboost(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    params: Optional[Dict[str, Any]] = None
) -> XGBClassifier:
    """
    Fits the XGBoost classifier for individual dropout prediction.
    Automatically applies scale_pos_weight to handle the 0.08% dropout rate.

    Args:
        X_train: Training features (MODEL2_FEATURES).
        y_train: Binary target (dropout).
        params: Optional dictionary to override default constants.

    Returns:
        Fitted XGBClassifier model.
    """
    print("Training Model 2: XGBoost Classifier...")
    
    # Default parameters from constants.py
    default_params = {
        'n_estimators': c.XGB_N_ESTIMATORS,
        'max_depth': c.XGB_MAX_DEPTH,
        'learning_rate': c.XGB_LEARNING_RATE,
        'scale_pos_weight': c.XGB_SCALE_POS_WEIGHT,
        'random_state': c.RANDOM_STATE,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    
    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    
    print("✓ Model 2 (XGBoost) training complete.")
    return model

def train_logistic(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> LogisticRegression:
    """
    Fits a Logistic Regression baseline model using balanced class weights.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Fitted LogisticRegression model.
    """
    print("Training Model 2 Baseline: Logistic Regression...")
    
    model = LogisticRegression(
        class_weight='balanced',
        random_state=c.RANDOM_STATE,
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    
    print("✓ Baseline Model 2 (Logistic) training complete.")
    return model

def save_model(model: Any, path: str) -> None:
    """
    Utility to serialize a trained model to a file.

    Args:
        model: The model object to save.
        path: Destination path (e.g., models/kmeans_model.pkl).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        joblib.dump(model, path)
        print(f"✓ Model successfully saved to: {path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")

def load_model(path: str) -> Optional[Any]:
    """
    Utility to load a serialized model from a file.

    Args:
        path: Path to the .pkl file.

    Returns:
        The loaded model object if found, else None.
    """
    if not os.path.exists(path):
        print(f"⚠ Warning: Model file not found at {path}")
        return None
    
    try:
        model = joblib.load(path)
        print(f"✓ Model loaded from: {path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
