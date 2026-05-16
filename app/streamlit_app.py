"""
HIV Care Gap AI Dashboard
Integrates team's county_profiles.csv and forecast files
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

# Page config
st.set_page_config(
    page_title="HIV Care Gap AI Dashboard", page_icon="🏥", layout="wide"
)

st.title("🏥 HIV Care Gap AI Dashboard")
st.markdown("---")


# Load team's data
@st.cache_data
def load_county_data():
    if os.path.exists(c.COUNTY_PROF):
        return pd.read_csv(c.COUNTY_PROF)
    return None


@st.cache_data
def load_forecast(tier):
    file_map = {
        "Critical": getattr(c, "FORECAST_CRITICAL", ""),
        "High": getattr(c, "FORECAST_HIGH", ""),
        "Moderate": getattr(c, "FORECAST_MODERATE", ""),
        "Low": getattr(c, "FORECAST_LOW", ""),
    }
    path = file_map.get(tier, "")
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


df_county = load_county_data()

# Color mapping
TIER_COLORS = {
    "Critical": "#C0392B",
    "High": "#E67E22",
    "Moderate": "#F1C40F",
    "Low": "#27AE60",
}

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["📍 County Tier Map", "📊 Risk Factors", "📈 Scenario Projections"]
)

# ============================================================
# TAB 1: COUNTY TIER VISUALIZATION
# ============================================================
with tab1:
    st.header("County Tier Visualization")

    if df_county is not None:
        # Dynamic Column Safe-Checks
        cgi_col = (
            "cgi_score"
            if "cgi_score" in df_county.columns
            else ("care_gap_index" if "care_gap_index" in df_county.columns else None)
        )
        iit_col = (
            "iit_rate_pct"
            if "iit_rate_pct" in df_county.columns
            else ("iit_rate" if "iit_rate" in df_county.columns else None)
        )

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        tier_counts = (
            df_county["tier"].value_counts()
            if "tier" in df_county.columns
            else pd.Series()
        )

        with col1:
            st.metric(
                "Critical Counties",
                tier_counts.get("Critical", 0),
                delta="Highest Priority",
            )
        with col2:
            st.metric("High Counties", tier_counts.get("High", 0))
        with col3:
            st.metric("Moderate Counties", tier_counts.get("Moderate", 0))
        with col4:
            st.metric("Low Counties", tier_counts.get("Low", 0))

        # Bar chart
        if cgi_col:
            st.subheader("CGI Scores by County")
            df_sorted = df_county.sort_values(cgi_col, ascending=True)
            colors = (
                [TIER_COLORS.get(t, "#808080") for t in df_sorted["tier"]]
                if "tier" in df_sorted.columns
                else ["#3498db"] * len(df_sorted)
            )

            fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) / 4)))
            bars = ax.barh(df_sorted["county"], df_sorted[cgi_col], color=colors)
            ax.set_xlabel("Care Gap Index (CGI)")
            ax.set_title("County Care Gap Index by Tier")
            ax.axvline(
                x=df_sorted[cgi_col].mean(),
                color="red",
                linestyle="--",
                label="National Average",
            )
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("CGI Column details not found for plotting.")

        # Task G1DFP5CP-135: Folium choropleth map
        st.subheader("🗺️ Geographic Distribution of County Tiers")

        try:
            import folium
            from streamlit_folium import st_folium

            # Create base map centered on Kenya
            kenya_center = [0.5, 38.0]
            m = folium.Map(
                location=kenya_center, zoom_start=6, tiles="CartoDB positron"
            )

            # County coordinates (approximate centroids)
            county_coords = {
                "Baringo": [0.4667, 35.9667],
                "Bomet": [-0.7833, 35.3333],
                "Bungoma": [0.5667, 34.5667],
                "Busia": [0.4667, 34.1167],
                "Elgeyo Marakwet": [0.8333, 35.5833],
                "Embu": [-0.5333, 37.4500],
                "Garissa": [-0.4500, 39.6500],
                "Homa Bay": [-0.5167, 34.4500],
                "Isiolo": [0.3500, 37.5833],
                "Kajiado": [-1.8500, 36.7833],
                "Kakamega": [0.2833, 34.7500],
                "Kericho": [-0.3667, 35.2833],
                "Kiambu": [-1.1667, 36.8333],
                "Kilifi": [-3.6333, 39.8500],
                "Kirinyaga": [-0.5000, 37.2833],
                "Kisii": [-0.6833, 34.7667],
                "Kisumu": [-0.1000, 34.7500],
                "Kitui": [-1.3667, 38.0167],
                "Kwale": [-4.1667, 39.4500],
                "Laikipia": [0.1833, 36.9500],
                "Lamu": [-2.2667, 40.9000],
                "Machakos": [-1.5167, 37.2667],
                "Makueni": [-2.2833, 37.8333],
                "Mandera": [3.9333, 41.8667],
                "Marsabit": [2.3333, 37.9833],
                "Meru": [0.0500, 37.6500],
                "Migori": [-1.0667, 34.4667],
                "Mombasa": [-4.0500, 39.6667],
                "Murang'a": [-0.7167, 37.1500],
                "Nairobi": [-1.2833, 36.8167],
                "Nakuru": [-0.3000, 36.0667],
                "Nandi": [0.1667, 35.1167],
                "Narok": [-1.0833, 35.8667],
                "Nyamira": [-0.5667, 34.9333],
                "Nyandarua": [-0.1667, 36.6333],
                "Nyeri": [-0.4167, 36.9500],
                "Samburu": [1.1667, 36.6667],
                "Siaya": [0.0667, 34.2833],
                "Taita Taveta": [-3.4000, 38.3667],
                "Tana River": [-1.7333, 39.6667],
                "Tharaka Nithi": [-0.3000, 37.9500],
                "Trans Nzoia": [1.0167, 34.9667],
                "Turkana": [3.3167, 35.5667],
                "Uasin Gishu": [0.5167, 35.2833],
                "Vihiga": [0.0833, 34.7167],
                "Wajir": [1.7500, 40.0667],
                "West Pokot": [1.1667, 35.1167],
            }

            # Add circle markers for each county
            for _, row in df_county.iterrows():
                county = row["county"]
                tier = row["tier"] if "tier" in row else "Unknown"
                cgi = row[cgi_col] if cgi_col else 0.0

                if county in county_coords:
                    coords = county_coords[county]
                    color = TIER_COLORS.get(tier, "#808080").lstrip("#")

                    folium.CircleMarker(
                        location=coords,
                        radius=10,
                        popup=f"<b>{county}</b><br>Tier: {tier}<br>CGI: {cgi:.1f}",
                        color=f"#{color}",
                        fill=True,
                        fill_color=f"#{color}",
                        fill_opacity=0.7,
                        weight=2,
                    ).add_to(m)

            # Add legend
            legend_html = """
            <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; 
                        background-color: white; padding: 10px; border-radius: 5px; 
                        border: 1px solid grey;">
                <b>Tier Legend</b><br>
                <i class="fa fa-circle" style="color:#C0392B"></i> Critical<br>
                <i class="fa fa-circle" style="color:#E67E22"></i> High<br>
                <i class="fa fa-circle" style="color:#F1C40F"></i> Moderate<br>
                <i class="fa fa-circle" style="color:#27AE60"></i> Low
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            # Display map
            st_map = st_folium(m, width=800, height=600, returned_objects=[])
            st.caption(
                "🗺️ County tier distribution map (circle size indicates relative CGI)"
            )

        except ImportError:
            st.info(
                "📦 Install folium and streamlit-folium: pip install folium streamlit-folium"
            )
        except Exception as e:
            st.warning(f"⚠️ Map could not be loaded: {e}")

        # Data table
        st.subheader("County Data Summary Profiles")
        st.dataframe(df_county, use_container_width=True)
    else:
        st.error(
            f"❌ Missing file structural data at: {getattr(c, 'COUNTY_PROF', 'Path undefined')}"
        )

# ============================================================
# TAB 2: RISK FACTORS (From Model 2)
# ============================================================
with tab2:
    st.header("Dropout Risk Factors")

    # Load odds ratios from Model 2
    risk_file = "data/processed/dropout_risk_factors.csv"
    if os.path.exists(risk_file):
        risk_df = pd.read_csv(risk_file)
        st.subheader("Top Risk Factors (Odds Ratios)")
        st.dataframe(risk_df.head(10), use_container_width=True)

        # Forest plot
        fig, ax = plt.subplots(figsize=(10, 6))
        top_risk = risk_df.head(10)
        colors_risk = [
            "#E74C3C" if x > 1 else "#27AE60" for x in top_risk["Odds_Ratio"]
        ]
        ax.barh(top_risk["Feature"], top_risk["Odds_Ratio"], color=colors_risk)
        ax.axvline(x=1, color="black", linestyle="--")
        ax.set_xlabel("Odds Ratio")
        ax.set_title("Top 10 Risk Factors for Dropout")
        st.pyplot(fig)
    else:
        st.info(
            "Run 06_model_2_dropout_prediction.ipynb or output `dropout_risk_factors.csv` to look at factors."
        )

# ============================================================
# TAB 3: SCENARIO PROJECTIONS
# ============================================================
with tab3:
    st.header("📈 Scenario Projections (2025-2030)")

    # Load all forecast files
    @st.cache_data
    def load_all_forecasts():
        forecasts = {}
        for tier in ["Critical", "High", "Moderate", "Low"]:
            file_map = {
                "Critical": getattr(c, "FORECAST_CRITICAL", ""),
                "High": getattr(c, "FORECAST_HIGH", ""),
                "Moderate": getattr(c, "FORECAST_MODERATE", ""),
                "Low": getattr(c, "FORECAST_LOW", ""),
            }
            path = file_map.get(tier, "")
            if path and os.path.exists(path):
                try:
                    forecasts[tier] = pd.read_csv(path)
                except Exception as e:
                    forecasts[tier] = pd.DataFrame()
            else:
                forecasts[tier] = pd.DataFrame()
        return forecasts

    forecasts = load_all_forecasts()

    # Load national projection
    @st.cache_data
    def load_national_forecast():
        path = getattr(c, "FORECAST_NATIONAL", "")
        if path and os.path.exists(path):
            try:
                return pd.read_csv(path)
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    national_df = load_national_forecast()

    # Task: 4 dual-scenario line charts
    st.subheader("📊 IIT Rate Projections by Tier")

    for tier in ["Critical", "High", "Moderate", "Low"]:
        df_tier = forecasts.get(tier, pd.DataFrame())
        if df_tier is not None and not df_tier.empty:
            st.markdown(f"### {tier} Tier")

            fig, ax = plt.subplots(figsize=(10, 4))

            # Scenario A
            df_a = df_tier[df_tier["scenario"] == "A"]
            if not df_a.empty:
                ax.plot(
                    df_a["year"],
                    df_a["iit_rate"],
                    "o--",
                    color="blue",
                    linewidth=2,
                    label="Scenario A (Business as Usual)",
                )

            # Scenario B
            df_b = df_tier[df_tier["scenario"] == "B"]
            if not df_b.empty:
                ax.plot(
                    df_b["year"],
                    df_b["iit_rate"],
                    "s-",
                    color="red",
                    linewidth=2,
                    label="Scenario B (30% Reduction)",
                )

            ax.set_xlabel("Year")
            ax.set_ylabel("IIT Rate")
            ax.set_title(f"{tier} Tier - IIT Rate Projections", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # Task: National headline chart
    st.subheader("🇰🇪 National Projections")

    if not national_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 4))

        df_nat_a = national_df[national_df["scenario"] == "A"]
        df_nat_b = national_df[national_df["scenario"] == "B"]

        if not df_nat_a.empty:
            ax2.plot(
                df_nat_a["year"],
                df_nat_a["iit_rate"],
                "o--",
                color="blue",
                linewidth=2,
                label="Scenario A (Business as Usual)",
            )

        if not df_nat_b.empty:
            ax2.plot(
                df_nat_b["year"],
                df_nat_b["iit_rate"],
                "s-",
                color="red",
                linewidth=2,
                label="Scenario B (30% Reduction)",
            )

        ax2.set_xlabel("Year")
        ax2.set_ylabel("IIT Rate")
        ax2.set_title("Kenya National IIT Rate Projections", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        if not df_nat_b.empty and not df_nat_a.empty:
            final_b = df_nat_b[df_nat_b["year"] == 2030]["iit_rate"].values
            final_a = df_nat_a[df_nat_a["year"] == 2030]["iit_rate"].values
            if len(final_b) > 0 and len(final_a) > 0:
                reduction = (final_a[0] - final_b[0]) / final_a[0] * 100
                st.success(
                    f"📉 Scenario B reduces national IIT rate by {reduction:.0f}% by 2030"
                )

    # Cross-sectional comparison table
    if df_county is not None:
        st.subheader("📋 County Comparison (2025 Baseline)")
        col_worst, col_best = st.columns(2)

        cgi_col = (
            "cgi_score"
            if "cgi_score" in df_county.columns
            else ("care_gap_index" if "care_gap_index" in df_county.columns else None)
        )

        if cgi_col:
            # Reusable sub-columns list checking
            sub_cols = ["county", "tier", cgi_col]
            if "iit_rate_pct" in df_county.columns:
                sub_cols.append("iit_rate_pct")
            elif "iit_rate" in df_county.columns:
                sub_cols.append("iit_rate")

            with col_worst:
                st.markdown("**🔴 Highest Risk Counties (Worst/Lowest CGI)**")
                top_worst = df_county.nsmallest(10, cgi_col)[sub_cols]
                st.dataframe(top_worst, use_container_width=True)

            with col_best:
                st.markdown("**🟢 Best Performing Counties (Highest CGI)**")
                top_best = df_county.nlargest(10, cgi_col)[sub_cols]
                st.dataframe(top_best, use_container_width=True)

    # CSV download button
    st.subheader("📥 Export Projection Data")
    all_projections = []
    for tier, df_tier in forecasts.items():
        if df_tier is not None and not df_tier.empty:
            df_tier_copy = df_tier.copy()
            df_tier_copy["tier"] = tier
            all_projections.append(df_tier_copy)

    if all_projections:
        export_df = pd.concat(all_projections, ignore_index=True)
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download All Projections (CSV)",
            data=csv,
            file_name="hiv_care_gap_projections.csv",
            mime="text/csv",
        )

    # Tooltips / Info
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **Data Sources:**
        - County profiles: NSDCC 2025 data integration layer
        - Projections: Scenario-based forecasting engine
        """)
