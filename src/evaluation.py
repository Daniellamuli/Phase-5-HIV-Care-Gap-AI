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
