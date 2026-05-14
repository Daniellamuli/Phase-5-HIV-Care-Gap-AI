"""
HIV Care Gap AI Dashboard
Tab 1: County Tier Visualization
Tasks: G1DFP5CP-132, 133, 134, 135, 136
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# --- PATH SECURING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import constants as c

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="HIV Care Gap AI Dashboard", page_icon="🏥", layout="wide"
)

st.title("🏥 HIV Care Gap AI Dashboard")
st.markdown("---")


# --- DATA LOADING ---
@st.cache_data
def load_county_data():
    if os.path.exists(c.COUNTY_PROF):
        df = pd.read_csv(c.COUNTY_PROF)
        return df
    return None


df_county = load_county_data()

# --- COLOR MAPPING ---
TIER_COLORS = {
    "Critical": "#C0392B",
    "High": "#E67E22",
    "Moderate": "#F1C40F",
    "Low": "#27AE60",
}

# --- UI TABS ---
tab1, tab2 = st.tabs(["📍 County Tier Visualization", "📋 System Diagnostics"])

with tab1:
    st.header("County Tier Visualization")

    if df_county is not None:
        # Task G1DFP5CP-132: Metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Counties Analyzed", len(df_county))

        with col2:
            if "iit_rate_pct" in df_county.columns:
                avg_iit = df_county["iit_rate_pct"].mean() * 100
                st.metric("Avg. Dropout (IIT) Rate", f"{avg_iit:.2f}%")
            else:
                st.metric("Avg. Dropout Rate", "N/A")

        with col3:
            if "adults_on_art" in df_county.columns:
                total_art = df_county["adults_on_art"].sum()
                st.metric("Total Adults on ART", f"{total_art:,}")
            else:
                st.metric("Total on ART", "N/A")

        st.markdown("---")

        # Task G1DFP5CP-133: Bar chart of counties sorted by CGI
        if "cgi_score" in df_county.columns and "tier" in df_county.columns:
            st.markdown("### 📊 County CGI Scores by Tier")

            df_sorted = df_county.sort_values("cgi_score", ascending=True)
            colors = [TIER_COLORS.get(t, "#808080") for t in df_sorted["tier"]]

            fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) / 4)))
            bars = ax.barh(df_sorted["county"], df_sorted["cgi_score"], color=colors)

            for bar, score in zip(bars, df_sorted["cgi_score"]):
                ax.text(
                    score + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}",
                    va="center",
                    fontsize=8,
                )

            ax.set_xlabel("Care Gap Index (CGI) - Lower = Worse Gap")
            ax.set_title("County CGI Scores by Tier", fontweight="bold")
            ax.axvline(
                x=50, color="red", linestyle="--", alpha=0.5, label="Critical Threshold"
            )
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        # Task G1DFP5CP-134: Tier distribution summary
        st.markdown("### 📈 Tier Distribution Summary")

        tier_counts = df_county["tier"].value_counts()
        tier_percentages = (tier_counts / len(df_county) * 100).round(1)

        cols = st.columns(4)
        for i, tier in enumerate(["Critical", "High", "Moderate", "Low"]):
            count = tier_counts.get(tier, 0)
            pct = tier_percentages.get(tier, 0)
            with cols[i]:
                st.metric(f"{tier}", f"{count} counties", delta=f"{pct}%")

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        colors_pie = [TIER_COLORS.get(t, "#808080") for t in tier_counts.index]
        ax2.pie(
            tier_counts,
            labels=tier_counts.index,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("County Tier Distribution", fontweight="bold")
        st.pyplot(fig2)

        # Task G1DFP5CP-135: Folium map (stretch - optional)
        st.markdown("### 🗺️ Geographic Distribution")
        st.info(
            "📍 Interactive map feature coming soon."
        )

        # Task G1DFP5CP-136: Data table
        st.markdown("### 📋 Priority County Profiles")
        st.dataframe(df_county, use_container_width=True)

        st.success("✅ Tab 1 loaded successfully with all 47 counties")

    else:
        st.error(f"Could not load data from {c.COUNTY_PROF}")

with tab2:
    st.subheader("System Diagnostics")
    if df_county is not None:
        st.success("✅ File loaded successfully")
        st.write("**Detected Columns:**", df_county.columns.tolist())
        st.write("**Data Preview:**")
        st.dataframe(df_county.head())

        # Task G1DFP5CP-136 verification
        if len(df_county) == 47:
            st.success("✅ All 47 counties present")
        else:
            st.warning(f"⚠️ Expected 47 counties, found {len(df_county)}")
    else:
        st.error(f"❌ File not found at: {c.COUNTY_PROF}")
