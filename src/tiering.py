"""
src/tiering.py
Model 1: County Tiering using KMeans clustering
Creates county profiles, CGI scores, and tiers
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants as c


class CountyTiering:
    """
    County tiering model using KMeans clustering
    Creates tiers: Critical, High, Moderate, Low
    """
    
    def __init__(self, constants_module):
        self.constants = constants_module
        self.df = None
        self.scaler = None
        self.kmeans = None
        
    def load_nsdcc_data(self):
        """
        Load cleaned NSDCC data from processed directory
        """
        print("=" * 50)
        print("LOADING NSDCC DATA")
        print("=" * 50)
        
        # Load all NSDCC cleaned files
        self.art_df = pd.read_csv(self.constants.ART_CLEAN)
        self.hts_df = pd.read_csv(self.constants.HTS_CLEAN)
        self.iit_df = pd.read_csv(self.constants.IIT_CLEAN)
        self.vlt_df = pd.read_csv(self.constants.VLT_CLEAN)
        
        print(f"✅ ART data: {self.art_df.shape}")
        print(f"✅ HTS data: {self.hts_df.shape}")
        print(f"✅ IIT data: {self.iit_df.shape}")
        print(f"✅ VLT data: {self.vlt_df.shape}")
        
        return self
    
    def clean_county_names(self):
        """
        Standardize county names across all datasets
        """
        def clean_name(name):
            return str(name).replace(" County", "").strip()
        
        for df in [self.art_df, self.hts_df, self.iit_df, self.vlt_df]:
            if 'County' in df.columns:
                df['county'] = df['County'].apply(clean_name)
        
        print("✅ County names standardized")
        return self
    
    def merge_county_data(self):
        """
        Merge all NSDCC data into single county dataframe
        """
        print("\n" + "=" * 50)
        print("MERGING COUNTY DATA")
        print("=" * 50)
        
        # Start with ART data
        self.df = self.art_df[['county', 'adults_on_art', 'art_total_males', 'art_total_females']].copy()
        
        # Add HTS data
        if 'hts_tested' in self.hts_df.columns:
            self.df = self.df.merge(
                self.hts_df[['county', 'hts_tested', 'hts_tested_males', 'hts_tested_females']],
                on='county', how='left'
            )
        
        # Add IIT data
        if 'iit_rate_pct' in self.iit_df.columns:
            self.df = self.df.merge(
                self.iit_df[['county', 'iit_rate_pct', 'iit_count']],
                on='county', how='left'
            )
        
        # Add VLT data (viral load suppression)
        vlt_cols = ['county']
        for col in ['vls_suppressed_female_15plus', 'vls_suppressed_male_15plus', 
                    'vlt_valid_female_15plus', 'vlt_valid_male_15plus']:
            if col in self.vlt_df.columns:
                vlt_cols.append(col)
        
        self.df = self.df.merge(self.vlt_df[vlt_cols], on='county', how='left')
        
        print(f"✅ Merged {len(self.df)} counties")
        print(f"📊 Columns: {self.df.columns.tolist()}")
        
        return self
    
    def calculate_cgi_score(self):
        """
        Calculate Care Gap Index (CGI)
        Lower score = worse care gap
        """
        print("\n" + "=" * 50)
        print("CALCULATING CGI SCORES")
        print("=" * 50)
        
        # Normalize IIT rate (higher IIT = worse gap)
        if 'iit_rate_pct' in self.df.columns:
            self.df['iit_normalized'] = self.df['iit_rate_pct'] / self.df['iit_rate_pct'].max()
            print(f"✅ IIT normalized: range 0-{self.df['iit_normalized'].max():.2f}")
        else:
            self.df['iit_normalized'] = 0.5
            print("⚠️ IIT rate not found, using default 0.5")
        
        # Calculate viral suppression rate
        if 'vls_suppressed_female_15plus' in self.df.columns:
            self.df['suppression_rate'] = (
                self.df['vls_suppressed_female_15plus'] + self.df['vls_suppressed_male_15plus']
            ) / (self.df['vlt_valid_female_15plus'] + self.df['vlt_valid_male_15plus'])
            self.df['suppression_normalized'] = 1 - self.df['suppression_rate']
            print(f"✅ Suppression normalized: range 0-{self.df['suppression_normalized'].max():.2f}")
        else:
            self.df['suppression_normalized'] = 0.5
            print("⚠️ VL suppression not found, using default 0.5")
        
        # CGI = weighted average (lower = worse gap)
        self.df['cgi_score'] = (
            self.df['iit_normalized'] * 0.6 + 
            self.df['suppression_normalized'] * 0.4
        ) * 100
        
        print(f"✅ CGI scores calculated")
        print(f"   Range: {self.df['cgi_score'].min():.1f} - {self.df['cgi_score'].max():.1f}")
        
        return self
    
    def create_tiers_kmeans(self, n_clusters=4):
        """
        Create tiers using KMeans clustering on CGI scores
        """
        print("\n" + "=" * 50)
        print(f"CREATING TIERS (KMeans, k={n_clusters})")
        print("=" * 50)
        
        # Prepare features
        features = self.df[['cgi_score']].copy()
        
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply KMeans
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.constants.KMEANS_RANDOM_STATE,
            n_init=10
        )
        self.df['cluster'] = self.kmeans.fit_predict(features_scaled)
        
        # Map clusters to tier labels (order by mean CGI score)
        cluster_order = self.df.groupby('cluster')['cgi_score'].mean().sort_values().index
        tier_mapping = {
            cluster_order[0]: 'Critical',
            cluster_order[1]: 'High',
            cluster_order[2]: 'Moderate',
            cluster_order[3]: 'Low'
        }
        self.df['tier'] = self.df['cluster'].map(tier_mapping)
        
        print(f"✅ Tiers created:")
        for tier in ['Critical', 'High', 'Moderate', 'Low']:
            count = len(self.df[self.df['tier'] == tier])
            print(f"   {tier}: {count} counties")
        
        return self
    
    def create_tiers_manual(self):
        """
        Alternative: Create tiers using manual thresholds on CGI scores
        (Use if KMeans fails)
        """
        print("\n" + "=" * 50)
        print("CREATING TIERS (Manual thresholds)")
        print("=" * 50)
        
        def assign_tier(score):
            if score <= 40:
                return 'Critical'
            elif score <= 60:
                return 'High'
            elif score <= 75:
                return 'Moderate'
            else:
                return 'Low'
        
        self.df['tier'] = self.df['cgi_score'].apply(assign_tier)
        
        print(f"✅ Tiers created (manual):")
        for tier in ['Critical', 'High', 'Moderate', 'Low']:
            count = len(self.df[self.df['tier'] == tier])
            print(f"   {tier}: {count} counties")
        
        return self
    
    def save_county_profiles(self):
        """
        Save county profiles to CSV
        """
        print("\n" + "=" * 50)
        print("SAVING COUNTY PROFILES")
        print("=" * 50)
        
        # Select columns for output
        output_cols = ['county', 'tier', 'cgi_score', 'iit_rate_pct', 'adults_on_art']
        available_cols = [col for col in output_cols if col in self.df.columns]
        
        output_df = self.df[available_cols].copy()
        
        # Sort by CGI (worst first)
        output_df = output_df.sort_values('cgi_score', ascending=True)
        
        # Save to file
        output_path = self.constants.COUNTY_PROF
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        print(f"✅ Saved to: {output_path}")
        print(f"📊 Shape: {output_df.shape}")
        print(f"\n📋 Preview:")
        print(output_df.head(10))
        
        return self
    
    def get_summary(self):
        """
        Print summary statistics
        """
        print("\n" + "=" * 50)
        print("COUNTY PROFILES SUMMARY")
        print("=" * 50)
        
        summary = self.df.groupby('tier').agg({
            'cgi_score': ['count', 'mean', 'min', 'max'],
            'iit_rate_pct': 'mean'
        }).round(1)
        
        print(summary)
        return summary
    
    def run_pipeline(self, use_kmeans=True):
        """
        Run complete county tiering pipeline
        """
        print("\n" + "=" * 60)
        print("COUNTY TIERING PIPELINE")
        print("=" * 60)
        
        self.load_nsdcc_data()
        self.clean_county_names()
        self.merge_county_data()
        self.calculate_cgi_score()
        
        if use_kmeans:
            self.create_tiers_kmeans(n_clusters=self.constants.KMEANS_K)
        else:
            self.create_tiers_manual()
        
        self.save_county_profiles()
        self.get_summary()
        
        print("\n✅ County tiering complete!")
        return self.df


# Run if executed directly
if __name__ == "__main__":
    import constants as c
    tiering = CountyTiering(c)
    df = tiering.run_pipeline(use_kmeans=False)  # Use manual thresholds first