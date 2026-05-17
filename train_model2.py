"""
scripts/train_model2.py
═══════════════════════════════════════════════════════════════
Production version of: notebooks/06_model_2_dropout_prediction.ipynb
Step 5 in main.py pipeline.

What it does:
  - Loads individual_features_clean.csv
  - Trains Logistic Regression (balanced class weights, liblinear solver)
  - Calculates odds ratios with 95% confidence intervals via bootstrap
  - Saves model bundle to models/xgboost_dropout.pkl (kept for Streamlit compatibility)
  - Saves odds_ratios_with_ci.json, logreg_baseline.json, dropout_risk_factors.csv

Note: Model 2 uses Logistic Regression + odds ratios (not XGBoost).
The extreme class imbalance (26 dropouts out of 32,156) makes
logistic regression + odds ratio analysis more appropriate and
more interpretable for MOH policymakers.

Input  : DHS_CLEAN (individual_features_clean.csv)
Outputs: XGBOOST_MODEL (.pkl) — model bundle
         ODDS_RATIOS_CI_JSON   — odds ratios with 95% CI
         LOGREG_BASELINE_JSON  — performance metrics
         DROPOUT_RISK_FACTORS  — full risk factor table (.csv)
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, confusion_matrix,
)

import constants as c


# ─────────────────────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────────────────────

def train_logreg() -> dict:
    """
    Train Logistic Regression on DHS individual features.
    Compute odds ratios + 95% CI via bootstrap.
    Save all outputs.

    Returns dict with model, metrics, and odds ratio DataFrames.
    """
    print("=" * 58)
    print("  STEP 5 — Model 2: Dropout Risk Factor Analysis")
    print("  Logistic Regression + Odds Ratios")
    print("=" * 58)

    # ── Load data
    if not os.path.exists(c.DHS_CLEAN):
        raise FileNotFoundError(
            f"DHS clean file not found: {c.DHS_CLEAN}\n"
            "Run notebooks/03_dhs_cleaning.ipynb first."
        )

    df = pd.read_csv(c.DHS_CLEAN)
    print(f"\n  Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  Target distribution:")
    print(f"    dropout=1 : {df[c.MODEL2_TARGET].sum():,}  ({df[c.MODEL2_TARGET].mean():.4%})")
    print(f"    dropout=0 : {(df[c.MODEL2_TARGET]==0).sum():,}  ({(1-df[c.MODEL2_TARGET].mean()):.4%})")

    # ── Features and target
    available = [col for col in c.MODEL2_FEATURES if col in df.columns]
    missing   = [col for col in c.MODEL2_FEATURES if col not in df.columns]
    if missing:
        print(f"\n  ⚠  Features not in data (skipped): {missing}")

    X = df[available].copy()
    y = df[c.MODEL2_TARGET].copy()
    print(f"\n  Features: {len(available)}  |  Target: {c.MODEL2_TARGET}")

    # ── Encode categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    print(f"  Encoded {len(categorical_cols)} categorical columns: {categorical_cols}")

    # ── Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=c.TEST_SIZE,
        random_state=c.RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Train Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=c.RANDOM_STATE,
        class_weight="balanced",
        solver="liblinear",
    )
    lr_model.fit(X_train, y_train)
    print(f"\n  ✓ Logistic Regression trained ({lr_model.n_iter_[0]} iterations)")

    # ── Evaluate
    y_pred      = lr_model.predict(X_test)
    y_pred_prob = lr_model.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, y_pred_prob)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Performance metrics:")
    print(f"    AUC-ROC   : {auc_score:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")

    # ── Odds ratios
    coefficients = lr_model.coef_[0]
    odds_ratios  = np.exp(coefficients)

    risk_df = pd.DataFrame({
        "Feature":    X.columns,
        "Coefficient":coefficients,
        "Odds_Ratio": odds_ratios.round(3),
    }).sort_values("Odds_Ratio", ascending=False)

    risk_df["Risk_Direction"] = [
        "Higher" if or_ > 1 else "Lower" if or_ < 1 else "Neutral"
        for or_ in risk_df["Odds_Ratio"]
    ]

    high_risk  = risk_df[risk_df["Odds_Ratio"] > 1.5]
    protective = risk_df[risk_df["Odds_Ratio"] < 0.7]

    print(f"\n  Top risk factors (OR > 1.5): {len(high_risk)}")
    for _, row in high_risk.head(5).iterrows():
        print(f"    {row['Feature']:<35} OR={row['Odds_Ratio']:.2f}")

    print(f"\n  Protective factors (OR < 0.7): {len(protective)}")
    for _, row in protective.head(5).iterrows():
        print(f"    {row['Feature']:<35} OR={row['Odds_Ratio']:.2f}")

    # ── Bootstrap 95% CI
    print(f"\n  Computing 95% CI via bootstrap (500 samples)...")
    n_bootstrap  = 500
    n_samples    = len(X_train)
    bootstrap_ors = []
    np.random.seed(42)

    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/500")
        idx    = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train.iloc[idx]
        y_boot = y_train.iloc[idx]
        lr_b   = LogisticRegression(
            max_iter=1000, random_state=c.RANDOM_STATE,
            class_weight="balanced", solver="liblinear"
        )
        lr_b.fit(X_boot, y_boot)
        bootstrap_ors.append(np.exp(lr_b.coef_[0]))

    bootstrap_ors = np.array(bootstrap_ors)
    ci_lower = np.percentile(bootstrap_ors, 2.5,  axis=0)
    ci_upper = np.percentile(bootstrap_ors, 97.5, axis=0)
    print("  ✓ Bootstrap complete")

    odds_ci_df = pd.DataFrame({
        "Feature":    X.columns,
        "Coefficient":coefficients,
        "Odds_Ratio": odds_ratios.round(3),
        "CI_Lower":   ci_lower.round(3),
        "CI_Upper":   ci_upper.round(3),
        "CI_Range":   (ci_upper - ci_lower).round(3),
    }).sort_values("Odds_Ratio", ascending=False)

    # ── Save outputs
    os.makedirs(c.MODELS_DIR, exist_ok=True)
    os.makedirs(c.PROCESSED_DIR, exist_ok=True)

    # Model bundle
    bundle = {
        "model":         lr_model,
        "encoders":      encoders,
        "features":      list(X.columns),
        "feature_names": available,
        "metrics": {
            "auc_roc": round(auc_score, 4),
            "recall":  round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        },
        "risk_factors":  risk_df.to_dict(orient="records"),
        "odds_ci":       odds_ci_df.to_dict(orient="records"),
    }
    with open(c.XGBOOST_MODEL, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n  ✓ Model bundle saved → {c.XGBOOST_MODEL}")

    # Odds ratios with CI JSON
    odds_ci_json = {
        "model":            "LogisticRegression",
        "confidence_level": 0.95,
        "bootstrap_samples":n_bootstrap,
        "features":         odds_ci_df.to_dict(orient="records"),
        "top_risk_factors": odds_ci_df.head(10)[
            ["Feature","Odds_Ratio","CI_Lower","CI_Upper"]
        ].to_dict(orient="records"),
        "protective_factors": odds_ci_df.tail(10)[
            ["Feature","Odds_Ratio","CI_Lower","CI_Upper"]
        ].to_dict(orient="records"),
    }
    with open(c.ODDS_RATIOS_CI_JSON, "w") as f:
        json.dump(odds_ci_json, f, indent=2)
    print(f"  ✓ Odds ratios CI saved → {c.ODDS_RATIOS_CI_JSON}")

    # Baseline metrics JSON
    baseline = {
        "model":           "LogisticRegression",
        "auc_roc":         round(auc_score, 4),
        "recall":          round(recall, 4),
        "precision":       round(precision, 4),
        "f1":              round(f1, 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "features_encoded": categorical_cols,
        "total_features":  X.shape[1],
        "top_risk_factors": high_risk.head(10)["Feature"].tolist(),
        "top_risk_ors":     high_risk.head(10)["Odds_Ratio"].tolist(),
    }
    with open(c.LOGREG_BASELINE_JSON, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"  ✓ Baseline metrics saved → {c.LOGREG_BASELINE_JSON}")

    # Risk factor CSV
    risk_df.to_csv(c.DROPOUT_RISK_FACTORS, index=False)
    print(f"  ✓ Risk factors saved → {c.DROPOUT_RISK_FACTORS}")

    print(f"\n{'=' * 58}")
    print(f"  ✓ Model 2 complete")
    print(f"  Next: scripts/train_model3.py")
    print()

    return {
        "model":      lr_model,
        "bundle":     bundle,
        "risk_df":    risk_df,
        "odds_ci_df": odds_ci_df,
        "metrics":    baseline,
    }


# ─────────────────────────────────────────────────────────────
# STANDALONE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_logreg()