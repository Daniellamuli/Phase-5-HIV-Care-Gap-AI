import os
import pandas as pd
from typing import Optional
from pathlib import Path

# These constants should ideally live in src/constants.py, 
# but are imported here to support the utility functions.
try:
    from constants import COUNTY_NAME_MAP, TIER_COLORS
except ImportError:
    # Fallback/Placeholder if constants.py isn't available yet
    COUNTY_NAME_MAP = {}
    TIER_COLORS = {
        "High": "#D7191C",
        "Medium": "#FDAE61",
        "Low": "#ABDDA4",
        "Minimal": "#2B83BA"
    }

def ensure_dir(file_path: str) -> None:
    """
    Ensures that the directory for a given file path exists.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def standardise_county(name: str) -> str:
    """
    Standardises county names using the COUNTY_NAME_MAP.
    Removes ' County' suffix and strips whitespace before mapping.
    """
    if not isinstance(name, str):
        return name
    
    # Clean suffix seen in Adult_on_ART.xlsx and Adult_on_HTS.xlsx
    clean_name = name.replace(" County", "").strip()
    
    # Return mapped name or the cleaned name if not in map
    return COUNTY_NAME_MAP.get(clean_name, clean_name)

def get_tier_color(tier_name: str) -> str:
    """
    Returns the hex color code for a specific priority tier.
    Defaults to grey (#808080) if tier is not found.
    """
    return TIER_COLORS.get(tier_name, "#808080")

def save_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    """
    Saves a DataFrame to CSV with automatic directory creation.
    """
    try:
        ensure_dir(path)
        df.to_csv(path, index=index)
        print(f"Successfully saved: {path}")
    except Exception as e:
        print(f"Error saving file to {path}: {e}")

def load_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Loads a CSV file and provides helpful error messaging if the file is missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        print(f"FAILED TO LOAD: The file '{path}' does not exist.")
        print("Please ensure you have run the preceding notebooks (01 and 02) to generate data.")
        return None
    
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"An unexpected error occurred while loading {path}: {e}")
        return None