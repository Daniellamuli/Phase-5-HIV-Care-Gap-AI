# src/dhs_cleaner.py

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Optional


class DHSCleaner:
    """
    DHS Data Cleaner for HIV Care Gap Analysis
    """

    def __init__(self, constants_module):
        """
        Initialize with constants module containing mappings

        Parameters:
        -----------
        constants_module : module
            Imported constants.py with DHS mappings
        """
        self.constants = constants_module
        self.raw_df = None
        self.clean_df = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load DHS individual features CSV

        Parameters:
        -----------
        filepath : str
            Path to individual_features.csv

        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        self.raw_df = pd.read_csv(filepath)
        # Strip whitespace from column names
        self.raw_df.columns = self.raw_df.columns.str.strip()
        self.clean_df = self.raw_df.copy()
        print(f"Loaded {len(self.raw_df)} rows with {len(self.raw_df.columns)} columns")
        print(f"Columns: {self.raw_df.columns.tolist()}")
        return self.raw_df

    def decode_county(self, county_col: str = "county") -> None:
        """
        Map county codes (1-47) to names using DHS_COUNTY_MAP
        """
        if county_col in self.clean_df.columns:
            self.clean_df["county_name"] = self.clean_df[county_col].map(
                self.constants.DHS_COUNTY_MAP
            )
            print(f"Decoded {self.clean_df['county_name'].nunique()} counties")
        else:
            print(f"Warning: {county_col} not found in dataframe")

    # In src/dhs_cleaner.py, replace decode_demographics method with:

    def decode_demographics(self) -> None:
        """
        Decode age, education, wealth, marital status, distance, work, and union using maps from constants
        """
        # Columns that need mapping (numeric codes → labels)
        numeric_mappings = {
            "age_group": self.constants.DHS_AGE_GROUP_MAP,
            "education_level": self.constants.DHS_EDUCATION_MAP,
            "wealth_index": self.constants.DHS_WEALTH_MAP,
            "distance_to_facility": self.constants.DHS_DISTANCE_MAP,
        }

        # Apply mappings for numeric columns
        for col, mapping in numeric_mappings.items():
            if col in self.clean_df.columns:
                # Only apply if column is numeric
                if self.clean_df[col].dtype in ["int64", "float64"]:
                    self.clean_df[col] = self.clean_df[col].map(mapping)
                    print(f"Decoded {col} (numeric)")
                else:
                    print(f"Note: {col} is already decoded, skipping")

        # Columns that are already strings (no mapping needed)
        string_cols = ["age_group", "marital_status"]
        for col in string_cols:
            if col in self.clean_df.columns:
                print(f"Note: {col} already has string values, no mapping needed")
                print(f"  Sample: {self.clean_df[col].iloc[0]}")

        # Handle work status
        if "worked_last_12months" in self.clean_df.columns:
            if self.clean_df["worked_last_12months"].isnull().all():
                self.clean_df["worked_last_12months"] = "No"
                print("Filled worked_last_12months with 'No' (100% missing)")
            elif self.clean_df["worked_last_12months"].dtype in ["int64", "float64"]:
                self.clean_df["worked_last_12months"] = self.clean_df[
                    "worked_last_12months"
                ].map(self.constants.DHS_WORKED_MAP)
                print("Decoded worked_last_12months")

        # Handle union status
        if "currently_in_union" in self.clean_df.columns:
            if self.clean_df["currently_in_union"].isnull().all():
                self.clean_df["currently_in_union"] = "Not in union"
                print("Filled currently_in_union with 'Not in union' (100% missing)")
            elif self.clean_df["currently_in_union"].dtype in ["int64", "float64"]:
                self.clean_df["currently_in_union"] = self.clean_df[
                    "currently_in_union"
                ].map(self.constants.DHS_UNION_MAP)
                print("Decoded currently_in_union")

    def impute_binary_flags(self) -> None:
        """
        Impute binary flags (ever tested, tested last 12m, told positive, has insurance) with 0
        """
        binary_cols = [
            "ever_tested_hiv",
            "tested_hiv_last_12months",
            "told_hiv_positive",
            "has_health_insurance",
        ]

        imputed_cols = []
        for col in binary_cols:
            if col in self.clean_df.columns:
                self.clean_df[col] = self.clean_df[col].fillna(0)
                imputed_cols.append(col)

        print(f"Imputed {len(imputed_cols)} binary columns with 0")

    def impute_numeric_columns(self, numeric_cols: List[str] = None) -> None:
        """
        Impute numeric columns with median

        Parameters:
        -----------
        numeric_cols : List[str]
            List of column names to impute (default: ['num_sexual_partners', 'anc_visits'])
        """
        if numeric_cols is None:
            numeric_cols = ["num_sexual_partners", "anc_visits"]

        for col in numeric_cols:
            if col in self.clean_df.columns:
                median_val = self.clean_df[col].median()
                self.clean_df[col] = self.clean_df[col].fillna(median_val)
                print(f"Imputed {col} with median: {median_val}")
            else:
                print(f"Warning: {col} not found - skipping")

    def engineer_dropout_target(self) -> None:
        """
        Engineer dropout target variable:
        1 if told_hiv_positive == 1 AND tested_hiv_last_12months == 0
        """
        if (
            "told_hiv_positive" in self.clean_df.columns
            and "tested_hiv_last_12months" in self.clean_df.columns
        ):

            self.clean_df["dropout"] = (
                (self.clean_df["told_hiv_positive"] == 1)
                & (self.clean_df["tested_hiv_last_12months"] == 0)
            ).astype(int)

            print("Engineered dropout target variable")
            print("Class distribution:")
            print(self.clean_df["dropout"].value_counts(normalize=True))
        else:
            print("Error: Required columns for dropout target not found")

    def one_hot_encode(self, columns: List[str]) -> None:
        """
        One-hot encode specified columns

        Parameters:
        -----------
        columns : List[str]
            Columns to one-hot encode (typically ['education_level', 'wealth_index'])
        """
        available_cols = [col for col in columns if col in self.clean_df.columns]

        if available_cols:
            self.clean_df = pd.get_dummies(
                self.clean_df, columns=available_cols, prefix=["edu", "wealth"]
            )
            print(f"One-hot encoded columns: {available_cols}")
        else:
            print("No columns available for one-hot encoding")

    def save_clean_data(self, output_path: str) -> None:
        """
        Save cleaned dataframe to CSV

        Parameters:
        -----------
        output_path : str
            Path to save cleaned data (e.g., data/processed/individual_features_clean.csv)
        """
        self.clean_df.to_csv(output_path, index=False)
        print(f"Saved clean data to {output_path}")

    def get_class_ratio(self) -> dict:
        """
        Return class ratio for dropout target

        Returns:
        --------
        dict
            Dictionary with dropout class ratios
        """
        if "dropout" in self.clean_df.columns:
            return self.clean_df["dropout"].value_counts(normalize=True).to_dict()
        else:
            return {}
