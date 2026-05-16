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
# TAB 2: RISK CALCULATOR AND FACTORS (From Model 2)
# ============================================================
with tab2:
    st.header(c.TAB2_TITLE)
    st.markdown(
        "Enter a patient profile below to calculate their HIV care dropout "
        "risk probability and see the top risk factors driving that score."
    )
    st.markdown("---")
 
    # ── Load odds ratios with CI from constants path
    @st.cache_data
    def load_odds_ratios():
        if not os.path.exists(c.ODDS_RATIOS_CI_JSON):
            return None
        with open(c.ODDS_RATIOS_CI_JSON, "r") as f:
            import json
            return json.load(f)
 
    # ── Load XGBoost model — fall back to logistic regression proxy
    @st.cache_resource
    def load_dropout_model():
        import joblib
        if os.path.exists(c.XGBOOST_MODEL):
            model = joblib.load(c.XGBOOST_MODEL)
            return model, "XGBoost"
        return None, "unavailable"
 
    odds_data  = load_odds_ratios()
    model, model_name = load_dropout_model()
 
    # ── County list from constants (set() removes duplicates)
    county_list = sorted(set(c.COUNTY_NAME_MAP.values()))
 
    # ── Age group options from constants
    age_options      = list(c.DHS_AGE_GROUP_MAP.values())
    wealth_options   = list(c.DHS_WEALTH_MAP.values())
    edu_options      = list(c.DHS_EDUCATION_MAP.values())
    distance_options = [v for k, v in c.DHS_DISTANCE_MAP.items() if k != 998]
    marital_options  = list(c.DHS_MARITAL_MAP.values())
 
    # ============================================================
    # SECTION 1 — RISK CALCULATOR
    # ============================================================
    st.subheader("🧮 Patient Risk Calculator")
 
    if model_name == "unavailable":
        st.warning(
            "⚠️ XGBoost model (`xgboost_dropout.pkl`) not found. "
            "Run `scripts/train_model2.py` to generate it. "
            "Risk factor analysis below is still fully available."
        )
    else:
        st.caption(f"Model: {model_name}")
 
    with st.form("risk_calculator_form"):
        col1, col2, col3 = st.columns(3)
 
        with col1:
            county       = st.selectbox("County", county_list)
            age_group    = st.selectbox("Age Group", age_options)
 
        with col2:
            wealth_index     = st.selectbox("Wealth Index", wealth_options)
            education_level  = st.selectbox("Education Level", edu_options)
 
        with col3:
            distance_to_facility = st.selectbox(
                "Distance to Facility", distance_options
            )
            marital_status = st.selectbox("Marital Status", marital_options)
 
        submitted = st.form_submit_button(
            "Calculate Dropout Risk", use_container_width=True
        )
 
    if submitted:
        if model_name != "unavailable" and model is not None:
 
            # ── Build feature vector matching MODEL2_FEATURES
            # Map inputs back to numeric codes
            age_code      = [k for k, v in c.DHS_AGE_GROUP_MAP.items()
                             if v == age_group][0]
            marital_code  = [k for k, v in c.DHS_MARITAL_MAP.items()
                             if v == marital_status][0]
            dist_code     = [k for k, v in c.DHS_DISTANCE_MAP.items()
                             if v == distance_to_facility][0]
            county_code   = [k for k, v in c.DHS_COUNTY_MAP.items()
                             if v == county][0] if hasattr(c, "DHS_COUNTY_MAP") else 1
 
            # One-hot encode education
            edu_higher     = 1 if education_level == "Higher"       else 0
            edu_no_edu     = 1 if education_level == "No education"  else 0
            edu_primary    = 1 if education_level == "Primary"       else 0
            edu_secondary  = 1 if education_level == "Secondary"     else 0
 
            # One-hot encode wealth
            wealth_middle  = 1 if wealth_index == "Middle"   else 0
            wealth_poorer  = 1 if wealth_index == "Poorer"   else 0
            wealth_poorest = 1 if wealth_index == "Poorest"  else 0
            wealth_richer  = 1 if wealth_index == "Richer"   else 0
            wealth_richest = 1 if wealth_index == "Richest"  else 0
 
            import numpy as np
            feature_vector = np.array([[
                county_code,    # county
                age_code,       # age_group
                marital_code,   # marital_status
                dist_code,      # distance_to_facility
                1,              # ever_tested_hiv (patient is in HIV care workflow)
                1,              # tested_hiv_last_12months (patient is active in system)
                0,              # num_sexual_partners (median default)
                0,              # worked_last_12months
                0,              # currently_in_union
                edu_higher,
                edu_no_edu,
                edu_primary,
                edu_secondary,
                wealth_middle,
                wealth_poorer,
                wealth_poorest,
                wealth_richer,
                wealth_richest,
            ]])
 
            try:
                prob = model.predict_proba(feature_vector)[0][1]
                pct  = prob * 100
 
                st.markdown("### 🎯 Dropout Risk Score")
 
                # Color-coded risk display
                if pct >= 60:
                    st.error(
                        f"🔴 **HIGH RISK: {pct:.1f}%** — "
                        f"Immediate follow-up recommended"
                    )
                    risk_color = "#C0392B"
                elif pct >= 30:
                    st.warning(
                        f"🟠 **MODERATE RISK: {pct:.1f}%** — "
                        f"Schedule follow-up within 30 days"
                    )
                    risk_color = "#E67E22"
                else:
                    st.success(
                        f"🟢 **LOW RISK: {pct:.1f}%** — "
                        f"Routine monitoring"
                    )
                    risk_color = "#27AE60"
 
                # Risk gauge bar
                st.progress(int(min(pct, 100)))
                st.caption(
                    f"Risk probability: {pct:.2f}% | "
                    f"Model: {model_name} | "
                    f"Threshold: 30% = moderate, 60% = high"
                )
 
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.info(
                "Risk calculator requires `xgboost_dropout.pkl`. "
                "Risk factor analysis below is still available."
            )
 
    st.markdown("---")
 
    # ============================================================
    # SECTION 2 — TOP 5 RISK FACTORS WITH CIs
    # ============================================================
    st.subheader("📊 Top 5 Risk Factors for HIV Care Dropout")
    st.caption(
        "Odds ratios with 95% bootstrap confidence intervals — "
        "from Logistic Regression (Kenya DHS 2022, n=32,156)"
    )
 
    if odds_data is None:
        st.warning(
            f"⚠️ `odds_ratios_with_ci.json` not found. "
            f"Run `06_model_2_dropout_prediction.ipynb` to generate it."
        )
    else:
        # ── Pull top 5 risk factors (OR > 1, sorted descending)
        all_features = odds_data.get("features", [])
        top5 = sorted(
            [f for f in all_features if f["Odds_Ratio"] > 1],
            key=lambda x: x["Odds_Ratio"],
            reverse=True
        )[:5]
 
        if top5:
            # ── Forest plot with CI error bars
            features   = [f["Feature"]    for f in top5]
            ors        = [f["Odds_Ratio"] for f in top5]
            ci_lower   = [f["CI_Lower"]   for f in top5]
            ci_upper   = [f["CI_Upper"]   for f in top5]
 
            # Cap x-axis — ever_tested_hiv CI_Upper is 6500
            # Display capped at 40 for readability; value shown in table
            X_CAP = 40
            ors_plot      = [min(o, X_CAP)       for o in ors]
            ci_upper_plot = [min(u, X_CAP)       for u in ci_upper]
            ci_lower_plot = [max(l, 0)            for l in ci_lower]
 
            xerr_lower = [o - l for o, l in zip(ors_plot, ci_lower_plot)]
            xerr_upper = [u - o for u, o in zip(ci_upper_plot, ors_plot)]
 
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#0F1117")
            ax.set_facecolor("#1E2130")
 
            colors_risk = ["#C0392B" if o >= 2 else "#E67E22" for o in ors]
 
            ax.barh(
                features, ors_plot,
                xerr=[xerr_lower, xerr_upper],
                color=colors_risk,
                error_kw=dict(ecolor="white", capsize=5, linewidth=1.5),
                height=0.5,
            )
            ax.axvline(x=1, color="white", linestyle="--",
                       linewidth=1, label="No effect (OR=1)")
            ax.set_xlabel("Odds Ratio (95% CI)", color="white", fontsize=11)
            ax.set_title(
                "Top 5 Risk Factors for HIV Care Dropout",
                color="white", fontsize=13, fontweight="bold", pad=12
            )
            ax.tick_params(colors="white")
            ax.set_xlim(0, X_CAP)
            for sp in ax.spines.values():
                sp.set_edgecolor("#2C3E50")
            ax.legend(
                facecolor="#1E2130", edgecolor="#2C3E50",
                labelcolor="white", fontsize=9
            )
 
            # Annotate OR values on bars
            for i, (feature, or_val, cap_val) in enumerate(
                zip(features, ors, ors_plot)
            ):
                label = (
                    f"OR={or_val:.2f} ⚠ capped at {X_CAP}"
                    if or_val > X_CAP
                    else f"OR={or_val:.2f}"
                )
                ax.text(
                    cap_val + 0.3, i, label,
                    va="center", color="white", fontsize=8.5
                )
 
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
 
            # ── Risk factors table
            st.markdown("#### Risk Factor Summary Table")
            table_data = []
            for f in top5:
                ci_note = (
                    f"[{f['CI_Lower']:.3f}, {f['CI_Upper']:.1f}]"
                    if f["CI_Upper"] > X_CAP
                    else f"[{f['CI_Lower']:.3f}, {f['CI_Upper']:.3f}]"
                )
                table_data.append({
                    "Risk Factor":        f["Feature"],
                    "Odds Ratio":         f"{f['Odds_Ratio']:.3f}",
                    "95% CI":             ci_note,
                    "Interpretation":     (
                        "🔴 Strong risk factor"
                        if f["Odds_Ratio"] >= 5
                        else "🟠 Moderate risk factor"
                        if f["Odds_Ratio"] >= 2
                        else "🟡 Mild risk factor"
                    ),
                })
            st.dataframe(
                pd.DataFrame(table_data),
                use_container_width=True,
                hide_index=True,
            )
 
        # ── Protective factors section
        st.markdown("---")
        st.subheader("🛡️ Protective Factors (OR < 1)")
        st.caption("Features associated with lower dropout risk")
 
        protective = sorted(
            [f for f in all_features if f["Odds_Ratio"] < 1
             and f["Feature"] != "tested_hiv_last_12months"],
            key=lambda x: x["Odds_Ratio"]
        )[:5]
 
        if protective:
            prot_data = []
            for f in protective:
                prot_data.append({
                    "Protective Factor": f["Feature"],
                    "Odds Ratio":        f"{f['Odds_Ratio']:.3f}",
                    "95% CI":            f"[{f['CI_Lower']:.3f}, {f['CI_Upper']:.3f}]",
                    "Risk Reduction":    f"{(1 - f['Odds_Ratio']) * 100:.0f}% lower risk",
                })
            st.dataframe(
                pd.DataFrame(prot_data),
                use_container_width=True,
                hide_index=True,
            )
 
    st.markdown("---")
 
    # ── Model performance summary
    st.subheader("📈 Model Performance")
 
    @st.cache_data
    def load_logreg_baseline():
        import json
        if not os.path.exists(c.LOGREG_BASELINE_JSON):
            return None
        with open(c.LOGREG_BASELINE_JSON, "r") as f:
            return json.load(f)
 
    baseline = load_logreg_baseline()
    if baseline:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("AUC-ROC",   f"{baseline.get('auc_roc', 0):.4f}")
        with m2:
            st.metric("Recall",    f"{baseline.get('recall', 0):.4f}")
        with m3:
            st.metric("Precision", f"{baseline.get('precision', 0):.4f}")
        with m4:
            st.metric("F1 Score",  f"{baseline.get('f1', 0):.4f}")
 
        with st.expander("ℹ️ Model Notes"):
            st.markdown(f"""
            **Model:** {baseline.get('model', 'Logistic Regression')}
            **Features used:** {baseline.get('total_features', 18)}
            **Training data:** Kenya DHS 2022 Individual Recode (n=32,156)
            **Dropout cases:** Only 26 confirmed dropouts (0.08% prevalence)
            **Key limitation:** Extremely low dropout prevalence means
            precision is low — recall is the primary metric for this
            public health use case (missing a high-risk patient is worse
            than a false alarm).
            """)
    else:
        st.info("Run `06_model_2_dropout_prediction.ipynb` to generate model metrics.")

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
