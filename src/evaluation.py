## silhouette_score() for Model 1 validation.

from sklearn.metrics import silhouette_score
import pandas as pd
from src.constants import KMEANS_RANDOM_STATE

def calculate_silhouette_score(data, labels):
    """
    Validates Model 1 (County Clustering).
    
    Args:
        data (DataFrame/Array): The scaled features used for clustering 
                               (e.g., iit_rate, vls_rate, art_coverage).
        labels (Series/Array): The cluster labels (0, 1, 2, 3) assigned by KMeans.
        
    Returns:
        float: The mean silhouette coefficient of all samples.
    """
    # Calculate the score
    score = silhouette_score(data, labels, random_state=KMEANS_RANDOM_STATE)
    
    print(f"--- Model 1 Evaluation ---")
    print(f"Silhouette Score: {score:.4f}")
    
    if score > 0.5:
        print("Interpretation: Strong structure. Counties are well-assigned to tiers.")
    elif score > 0.25:
        print("Interpretation: Fair structure. Some overlap between adjacent tiers.")
    else:
        print("Interpretation: Weak structure. Tiers may be too similar; consider re-scaling features.")
        
    return score
    
    
## classification_metrics() returning AUC-ROC, Recall, Precision, F1.
from sklearn.metrics import (
    roc_auc_score, 
    recall_score, 
    precision_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import pandas as pd
from src.constants import MODEL2_TARGET

def classification_metrics(y_true, y_pred, y_prob):
    """
    Evaluates Model 2 (XGBoost Dropout Prediction).
    
    Args:
        y_true: Actual 'dropout' labels (0 or 1).
        y_pred: Binary predictions (0 or 1) from the model.
        y_prob: Predicted probabilities for the positive class (used for AUC-ROC).
        
    Returns:
        dict: A dictionary containing all four requested metrics.
    """
    # Calculate Core Metrics
    auc = roc_auc_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n--- Model 2 Evaluation: {MODEL2_TARGET.upper()} ---")
    print(f"AUC-ROC Score: {auc:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    
    # Detailed Clinical Context
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nClinical Impact Summary:")
    print(f"- Correctly identified {tp} high-risk patients (True Positives)")
    print(f"- MISSED {fn} patients who subsequently dropped out (False Negatives)")
    print(f"- Flagged {fp} stable patients for unnecessary follow-up (False Positives)")
    
    return {
        "auc_roc": auc,
        "recall": recall,
        "precision": precision,
        "f1_score": f1
    }

## forecast_errors() returning MAE and RMSE

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def forecast_errors(y_true, y_pred):
    """
    Evaluates Model 3 (2030 Time-Series Forecasting).
    
    Args:
        y_true: Historical actual values (e.g., actual iit_count).
        y_pred: Model predictions for the same historical period (yhat).
        
    Returns:
        dict: A dictionary containing MAE and RMSE.
    """
    # Calculate Core Errors
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE for business context (Mean Absolute Percentage Error)
    # Using a small epsilon to avoid division by zero if data is sparse
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    print(f"\n--- Model 3 Evaluation: Forecasting Accuracy ---")
    print(f"Mean Absolute Error (MAE):      {mae:.2f}")
    print(f"Root Mean Square Error (RMSE):  {rmse:.2f}")
    print(f"Mean Absolute % Error (MAPE):   {mape:.2f}%")
    
    if mape < 10:
        print("Interpretation: Highly accurate forecast.")
    elif mape < 25:
        print("Interpretation: Good forecast; reliable for strategic planning.")
    else:
        print("Interpretation: High variance; consider adding holidays or changepoints.")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }
