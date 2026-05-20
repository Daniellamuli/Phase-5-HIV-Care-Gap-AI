"""
HIV Care Gap AI Dashboard
Kenya HIV Care Gap AI - County Risk Mapping · Dropout Prediction · 2030 Scenario Forecasting
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import json
import sys
import os
import numpy as np

# ── Path setup
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import constants as c

# ── Page config
st.set_page_config(
    page_title="HIV Care Gap AI Dashboard",

    layout="wide"
)

# ── Logo + Title
logo_path = os.path.join(PROJECT_ROOT, "figures", "logo.png")
if not os.path.exists(logo_path):
    logo_path = os.path.join(PROJECT_ROOT, "figures", "logo.jpg")
col_logo, col_title = st.columns([1, 8])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
with col_title:
    st.title("🏥 HIV Care Gap AI Dashboard")
    st.caption(
        "Kenya HIV Care Gap AI · County Risk Mapping · Dropout Prediction · 2030 Scenario Forecasting"
    )
st.markdown("---")

# ── Tier colours
TIER_COLORS = {
    "Critical": "#C0392B",
    "High":     "#E67E22",
    "Moderate": "#F1C40F",
    "Low":      "#27AE60",
}

# ── Readable column names for Tab 1 summary table
# art_coverage excluded - placeholder value (0.5) for all counties
READABLE_COLS = {
    "county":              "County",
    "period":              "Year",
    "tier":                "Risk Tier",
    "care_gap_index":      "Care Gap Index",
    "iit_rate":            "Treatment Interruption Rate",
    "vls_rate_adult":      "Viral Load Suppression Rate",
    "hts_positivity_rate": "HTS Positivity Rate",
    "adults_on_art":       "Adults on ART",
}

# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_county_data():
    if os.path.exists(c.COUNTY_PROF):
        df = pd.read_csv(c.COUNTY_PROF)
        if "period" in df.columns:
            df["period"] = pd.to_numeric(df["period"], errors="coerce").astype("Int64")
        return df
    return None

@st.cache_data
def load_all_forecasts():
    forecasts = {}
    for tier in ["Critical", "High", "Moderate", "Low"]:
        file_map = {
            "Critical": getattr(c, "FORECAST_CRITICAL", ""),
            "High":     getattr(c, "FORECAST_HIGH", ""),
            "Moderate": getattr(c, "FORECAST_MODERATE", ""),
            "Low":      getattr(c, "FORECAST_LOW", ""),
        }
        path = file_map.get(tier, "")
        if path and os.path.exists(path):
            try:
                forecasts[tier] = pd.read_csv(path)
            except Exception:
                forecasts[tier] = pd.DataFrame()
        else:
            forecasts[tier] = pd.DataFrame()
    return forecasts

@st.cache_data
def load_national_forecast():
    path = getattr(c, "FORECAST_NATIONAL", "")
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data
def load_odds_ratios():
    if not os.path.exists(c.ODDS_RATIOS_CI_JSON):
        return None
    with open(c.ODDS_RATIOS_CI_JSON, "r") as f:
        return json.load(f)

@st.cache_resource
def load_dropout_model():
    if os.path.exists(c.XGBOOST_MODEL):
        with open(c.XGBOOST_MODEL, "rb") as f:
            model = pickle.load(f)
        return model, "Logistic Regression"
    return None, "unavailable"

@st.cache_data
def load_logreg_baseline():
    if not os.path.exists(c.LOGREG_BASELINE_JSON):
        return None
    with open(c.LOGREG_BASELINE_JSON, "r") as f:
        return json.load(f)

# ── Load all data
df_county   = load_county_data()
forecasts   = load_all_forecasts()
national_df = load_national_forecast()
odds_data   = load_odds_ratios()
model, model_name = load_dropout_model()
baseline    = load_logreg_baseline()

cgi_col = (
    "cgi_score"       if df_county is not None and "cgi_score"       in df_county.columns
    else "care_gap_index" if df_county is not None and "care_gap_index" in df_county.columns
    else None
)

# ── Tabs
tab1, tab2, tab3 = st.tabs(
    ["📍 County Tier Map", "📊 Risk Factors", "📈 Scenario Projections"]
)

# ============================================================
# TAB 1: COUNTY TIER MAP
# ============================================================
with tab1:
    st.header(getattr(c, "TAB1_TITLE", "HIV Care Gap County Map"))

    st.info(
        "This tab shows Kenya's 47 counties ranked by their "
        "HIV Care Gap Index (CGI). Counties are grouped into 4 tiers based on "
        "how urgently they need HIV care interventions. "
        "🔴 **Critical** = most urgent · 🟠 **High** = urgent · "
        "🟡 **Moderate** = moderate · 🟢 **Low** = performing well."
    )

    if df_county is not None:
        tier_counts = df_county["tier"].value_counts() if "tier" in df_county.columns else pd.Series()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔴 Critical Counties", tier_counts.get("Critical", 0), delta="Highest Priority")
        with col2:
            st.metric("🟠 High Counties", tier_counts.get("High", 0))
        with col3:
            st.metric("🟡 Moderate Counties", tier_counts.get("Moderate", 0))
        with col4:
            st.metric("🟢 Low Counties", tier_counts.get("Low", 0))

        if cgi_col:
            st.subheader("County Risk Stratification - Tier Distribution")
            st.caption(
                "Each bar represents one county. Longer bar = larger care gap = higher priority. "
                "The red dashed line is the national average CGI."
            )
            df_sorted = df_county.sort_values(cgi_col, ascending=True)
            colors = [TIER_COLORS.get(t, "#808080") for t in df_sorted["tier"]] if "tier" in df_sorted.columns else ["#3498db"] * len(df_sorted)
            fig, ax = plt.subplots(figsize=(12, max(6, len(df_sorted) / 4)))
            ax.barh(df_sorted["county"], df_sorted[cgi_col], color=colors)
            ax.set_xlabel("Care Gap Index (CGI)")
            ax.set_title("County Care Gap Index by Tier")
            ax.axvline(x=df_sorted[cgi_col].mean(), color="red", linestyle="--", label="National Average")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("CGI column not found.")

        # ── Folium map
        st.subheader("🗺️ Geographic Distribution of County Tiers")
        st.caption("Click any circle on the map to see the county name, tier, and key indicators.")

        try:
            import folium
            from streamlit_folium import st_folium

            m = folium.Map(location=[0.5, 38.0], zoom_start=6, tiles="CartoDB positron")


            # CHOROPLETH LAYER
            geojson_path = os.path.join(PROJECT_ROOT, "data", "kenya_counties.geojson")
            if os.path.exists(geojson_path) and df_county is not None:
                import json as _json
                with open(geojson_path) as _f:
                    kenya_geojson = _json.load(_f)
                COUNTY_NAME_FIXES = {
                    "THARAKA-NITHI": "Tharaka Nithi", "ELGEYO-MARAKWET": "Elgeyo Marakwet",
                    "HOMA BAY": "Homa Bay", "TAITA TAVETA": "Taita Taveta",
                    "TANA RIVER": "Tana River", "WEST POKOT": "West Pokot",
                    "UASIN GISHU": "Uasin Gishu", "MURANG'A": "Murang'a",
                    "TRANS-NZOIA": "Trans Nzoia", "TRANS NZOIA": "Trans Nzoia",
                }
                tier_color_lookup = {}
                if "tier" in df_county.columns:
                    for _, _r in df_county.iterrows():
                        tier_color_lookup[_r["county"]] = TIER_COLORS.get(_r["tier"], "#808080")
                def _style_choropleth(feature):
                    raw    = feature["properties"].get("COUNTY_NAM") or ""
                    county = COUNTY_NAME_FIXES.get(raw.upper(), raw.title())
                    fill   = tier_color_lookup.get(county, "#CCCCCC")
                    return {"fillColor": fill, "color": "#555555", "weight": 1.0, "fillOpacity": 0.85}
                try:
                    folium.GeoJson(
                        kenya_geojson, style_function=_style_choropleth, name="County Risk Tiers",
                        tooltip=folium.GeoJsonTooltip(fields=["COUNTY_NAM"], aliases=["County:"], style="font-size:12px;"),
                    ).add_to(m)
                except Exception as _err:
                    st.caption(f"Choropleth skipped: {_err}")

            county_coords = {
                "Baringo": [0.4667, 35.9667], "Bomet": [-0.7833, 35.3333],
                "Bungoma": [0.5667, 34.5667], "Busia": [0.4667, 34.1167],
                "Elgeyo Marakwet": [0.8333, 35.5833], "Embu": [-0.5333, 37.4500],
                "Garissa": [-0.4500, 39.6500], "Homa Bay": [-0.5167, 34.4500],
                "Isiolo": [0.3500, 37.5833], "Kajiado": [-1.8500, 36.7833],
                "Kakamega": [0.2833, 34.7500], "Kericho": [-0.3667, 35.2833],
                "Kiambu": [-1.1667, 36.8333], "Kilifi": [-3.6333, 39.8500],
                "Kirinyaga": [-0.5000, 37.2833], "Kisii": [-0.6833, 34.7667],
                "Kisumu": [-0.1000, 34.7500], "Kitui": [-1.3667, 38.0167],
                "Kwale": [-4.1667, 39.4500], "Laikipia": [0.1833, 36.9500],
                "Lamu": [-2.2667, 40.9000], "Machakos": [-1.5167, 37.2667],
                "Makueni": [-2.2833, 37.8333], "Mandera": [3.9333, 41.8667],
                "Marsabit": [2.3333, 37.9833], "Meru": [0.0500, 37.6500],
                "Migori": [-1.0667, 34.4667], "Mombasa": [-4.0500, 39.6667],
                "Murang'a": [-0.7167, 37.1500], "Nairobi": [-1.2833, 36.8167],
                "Nakuru": [-0.3000, 36.0667], "Nandi": [0.1667, 35.1167],
                "Narok": [-1.0833, 35.8667], "Nyamira": [-0.5667, 34.9333],
                "Nyandarua": [-0.1667, 36.6333], "Nyeri": [-0.4167, 36.9500],
                "Samburu": [1.1667, 36.6667], "Siaya": [0.0667, 34.2833],
                "Taita Taveta": [-3.4000, 38.3667], "Tana River": [-1.7333, 39.6667],
                "Tharaka Nithi": [-0.3000, 37.9500], "Trans Nzoia": [1.0167, 34.9667],
                "Turkana": [3.3167, 35.5667], "Uasin Gishu": [0.5167, 35.2833],
                "Vihiga": [0.0833, 34.7167], "Wajir": [1.7500, 40.0667],
                "West Pokot": [1.1667, 35.1167],
            }

            for _, row in df_county.iterrows():
                county = row["county"]
                tier   = row.get("tier", "Unknown")
                cgi    = row[cgi_col] if cgi_col else 0.0
                iit    = row.get("iit_rate", 0.0)
                vls    = row.get("vls_rate_adult", 0.0)
                if county in county_coords:
                    color = TIER_COLORS.get(tier, "#808080").lstrip("#")
                    folium.CircleMarker(
                        location=county_coords[county],
                        radius=10,
                        popup=folium.Popup(
                            f"<b>{county}</b><br>Tier: {tier}<br>"
                            f"Care Gap Index: {cgi:.2f}<br>"
                            f"Treatment Interruption: {iit:.1%}<br>"
                            f"Viral Suppression: {vls:.1%}",
                            max_width=200
                        ),
                        color=f"#{color}", fill=True,
                        fill_color=f"#{color}", fill_opacity=0.7, weight=2,
                    ).add_to(m)

            legend_html = """
            <div style="position:fixed;bottom:50px;right:50px;z-index:1000;
                        background:white;padding:10px;border-radius:5px;
                        border:1px solid grey;font-size:13px;">
                <b>Risk Tier</b><br>
                <span style="color:#C0392B">●</span> Critical<br>
                <span style="color:#E67E22">●</span> High<br>
                <span style="color:#F1C40F">●</span> Moderate<br>
                <span style="color:#27AE60">●</span> Low
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            st_folium(m, width="100%", height=450, returned_objects=[])

        except ImportError:
            st.info("📦 To enable the interactive map, run: `pip install folium streamlit-folium`")
        except Exception as e:
            st.warning(f"⚠️ Map could not be loaded: {e}")

        # ── Readable county summary table
        st.subheader("County Data Summary")
        st.caption(
            "Key HIV programme indicators for each county. "
            "Treatment Interruption Rate = % of patients who missed their ART visit by 28+ days. "
            "Viral Load Suppression Rate = % of patients whose HIV is undetectable (target: 95%)."
        )

        display_cols = [col for col in READABLE_COLS.keys() if col in df_county.columns]
        df_display   = df_county[display_cols].copy()
        df_display.rename(columns=READABLE_COLS, inplace=True)

        if "Year" in df_display.columns:
            df_display["Year"] = df_display["Year"].apply(
                lambda x: str(int(x)) if pd.notna(x) else ""
            )
        for col in ["Treatment Interruption Rate", "Viral Load Suppression Rate",
                    "HTS Positivity Rate", "ART Coverage"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.1%}" if pd.notna(x) else ""
                )
        if "Care Gap Index" in df_display.columns:
            df_display["Care Gap Index"] = df_display["Care Gap Index"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )

        st.dataframe(df_display, use_container_width=True)

    else:
        st.error("❌ County data not found. Run the pipeline first.")

# ============================================================
# TAB 2: DROPOUT RISK CALCULATOR
# ============================================================
with tab2:
    st.header(c.TAB2_TITLE)

    st.info(
        "This tab identifies which demographic factors are most strongly associated with "
        "HIV care dropout in Kenya, using Logistic Regression on Kenya DHS 2022 (32,156 individuals).\n\n"
        "⚠️ This is **NOT an individual clinical prediction tool.** With only 26 confirmed dropout cases "
        "(0.08% prevalence), no model can reliably predict individual outcomes. "
        "The Odds Ratio forest plot and tables below are the primary output and are fully reliable. "
        "The demographic profile explorer is a supplementary population-risk interpretation tool."
    )
    st.markdown("---")

    county_list      = sorted(set(c.COUNTY_NAME_MAP.values()))
    age_options      = list(c.DHS_AGE_GROUP_MAP.values())
    wealth_options   = list(c.DHS_WEALTH_MAP.values())
    edu_options      = list(c.DHS_EDUCATION_MAP.values())
    distance_options = [v for k, v in c.DHS_DISTANCE_MAP.items() if k != 998]
    marital_options  = list(c.DHS_MARITAL_MAP.values())

    st.subheader("🧮 Demographic Profile Explorer")
    st.caption(
        "Select a demographic profile and click 'Explore Risk Profile'. "
        "Results show which associated-risk category this profile belongs to, "
        "based on odds ratios from Logistic Regression on Kenya DHS 2022."
    )

    if model_name == "unavailable":
        st.warning(
            "⚠️ Logistic Regression model not found. "
            "Run `scripts/train_model2.py` to generate it. "
            "Risk factor analysis below is still fully available."
        )
    else:
        st.caption(f"Model: {model_name}")

    with st.form("risk_calculator_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            county      = st.selectbox("County", county_list)
            age_group   = st.selectbox("Age Group", age_options)
        with col2:
            wealth_index    = st.selectbox("Wealth Index", wealth_options)
            education_level = st.selectbox("Education Level", edu_options)
        with col3:
            distance_to_facility = st.selectbox("Distance to Facility", distance_options)
            marital_status       = st.selectbox("Marital Status", marital_options)
        submitted = st.form_submit_button("🔍 Explore Risk Profile", use_container_width=True)

    if submitted:
        if odds_data is not None:
            edu_no_edu     = 1 if education_level == "No education" else 0
            edu_higher     = 1 if education_level == "Higher"       else 0
            edu_primary    = 1 if education_level == "Primary"      else 0
            wealth_richest = 1 if wealth_index == "Richest"         else 0
            marital_code   = [k for k, v in c.DHS_MARITAL_MAP.items() if v == marital_status][0]
            dist_code      = [k for k, v in c.DHS_DISTANCE_MAP.items() if v == distance_to_facility][0]

            strong, moderate, protective = [], [], []
            if edu_no_edu:
                strong.append("No formal education (OR=6.26 - strongest risk factor)")
            if wealth_richest:
                strong.append("Richest wealth quintile (OR=2.59)")
            if marital_code in [2, 3, 4]:
                moderate.append("Marital status associated with moderate risk (OR=1.46)")
            if dist_code == 1:
                protective.append("Close to health facility (OR=0.63 - protective)")
            if edu_higher:
                protective.append("Higher education (OR=0.47 - strongly protective)")
            if edu_primary:
                protective.append("Primary education (OR=0.85 - mildly protective)")

            st.markdown("### 🎯 Demographic Risk Profile Result")
            if len(strong) >= 1:
                st.error("🔴 **Strong Associated-Risk Profile**\n\nThis profile matches characteristics strongly associated with HIV care dropout. Priority follow-up recommended for patient groups with this profile.")
            elif len(moderate) >= 1 and len(protective) == 0:
                st.warning("🟠 **Moderate Associated-Risk Profile**\n\nThis profile has characteristics moderately associated with dropout risk. Proactive check-ins recommended.")
            else:
                st.success("🟢 **Lower Associated-Risk Profile**\n\nThis profile does not strongly match high-risk characteristics. Routine monitoring is appropriate. Individual clinical assessment always takes precedence.")

            if strong:
                st.markdown("**High-OR factors present:**")
                for r in strong: st.markdown(f"  - {r}")
            if moderate:
                st.markdown("**Moderate-OR factors present:**")
                for r in moderate: st.markdown(f"  - {r}")
            if protective:
                st.markdown("**Protective factors present:**")
                for r in protective: st.markdown(f"  - {r}")

            st.caption("Population-level indicator only - not an individual clinical prediction. Based on odds ratios from Logistic Regression (Kenya DHS 2022, n=32,156, 26 dropout cases).")
        else:
            st.info("Run `scripts/train_model2.py` to generate odds ratio data.")
    st.markdown("---")

    st.subheader("📊 Top 5 Risk Factors for HIV Care Dropout")
    st.caption("Odds ratios with 95% bootstrap confidence intervals - Logistic Regression (Kenya DHS 2022, n=32,156)")
    st.info(
        "📖 **How to read this chart:** An Odds Ratio (OR) greater than 1 means that factor "
        "is associated with a HIGHER risk of dropping out of HIV care. "
        "OR=2.65 for 'No education' means patients with no education are 2.65× more likely "
        "to interrupt treatment. The error bars show the 95% confidence interval (uncertainty range)."
    )

    if odds_data is None:
        st.warning("⚠️ `odds_ratios_with_ci.json` not found. Run `06_model_2_dropout_prediction.ipynb`.")
    else:
        all_features = odds_data.get("features", [])
        top5 = sorted([f for f in all_features if f["Odds_Ratio"] > 1],
                      key=lambda x: x["Odds_Ratio"], reverse=True)[:5]

        if top5:
            features      = [f["Feature"]    for f in top5]
            ors           = [f["Odds_Ratio"] for f in top5]
            ci_lower      = [f["CI_Lower"]   for f in top5]
            ci_upper      = [f["CI_Upper"]   for f in top5]
            X_CAP         = 40
            ors_plot      = [min(o, X_CAP) for o in ors]
            ci_upper_plot = [min(u, X_CAP) for u in ci_upper]
            ci_lower_plot = [max(l, 0)     for l in ci_lower]
            xerr_lower    = [o - l for o, l in zip(ors_plot, ci_lower_plot)]
            xerr_upper    = [u - o for u, o in zip(ci_upper_plot, ors_plot)]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#0F1117")
            ax.set_facecolor("#1E2130")
            colors_risk = ["#C0392B" if o >= 2 else "#E67E22" for o in ors]
            ax.barh(features, ors_plot, xerr=[xerr_lower, xerr_upper], color=colors_risk,
                    error_kw=dict(ecolor="white", capsize=5, linewidth=1.5), height=0.5)
            ax.axvline(x=1, color="white", linestyle="--", linewidth=1, label="No effect (OR=1)")
            ax.set_xlabel("Odds Ratio (95% CI)", color="white", fontsize=11)
            ax.set_title("Top 5 Risk Factors for HIV Care Dropout", color="white",
                         fontsize=13, fontweight="bold", pad=12)
            ax.tick_params(colors="white")
            ax.set_xlim(0, X_CAP)
            for sp in ax.spines.values():
                sp.set_edgecolor("#2C3E50")
            ax.legend(facecolor="#1E2130", edgecolor="#2C3E50", labelcolor="white", fontsize=9)
            for i, (feature, or_val, cap_val) in enumerate(zip(features, ors, ors_plot)):
                label = f"OR={or_val:.2f} ⚠ capped" if or_val > X_CAP else f"OR={or_val:.2f}"
                ax.text(cap_val + 0.3, i, label, va="center", color="white", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### Risk Factor Summary Table")
            table_data = []
            for f in top5:
                ci_note = (f"[{f['CI_Lower']:.3f}, >40]" if f["CI_Upper"] > X_CAP
                           else f"[{f['CI_Lower']:.3f}, {f['CI_Upper']:.3f}]")
                table_data.append({
                    "Risk Factor":    f["Feature"],
                    "Odds Ratio":     f"{f['Odds_Ratio']:.3f}",
                    "95% CI":         ci_note,
                    "Interpretation": (
                        "🔴 Strong risk factor"       if f["Odds_Ratio"] >= 5
                        else "🟠 Moderate risk factor" if f["Odds_Ratio"] >= 2
                        else "🟡 Mild risk factor"
                    ),
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("🛡️ Protective Factors (OR < 1)")
        st.caption("Features associated with lower dropout risk")
        st.info(
            "📖 **How to read this:** OR less than 1 = that factor is associated with LOWER risk. "
            "OR=0.361 for 'Primary education' means those patients are 64% less likely to "
            "interrupt treatment compared to the baseline group."
        )
        protective = sorted([f for f in all_features if f["Odds_Ratio"] < 1
                             and f["Feature"] != "tested_hiv_last_12months"],
                            key=lambda x: x["Odds_Ratio"])[:5]
        if protective:
            prot_data = [{
                "Protective Factor": f["Feature"],
                "Odds Ratio":        f"{f['Odds_Ratio']:.3f}",
                "95% CI":            f"[{f['CI_Lower']:.3f}, {f['CI_Upper']:.3f}]",
                "Risk Reduction":    f"{(1 - f['Odds_Ratio']) * 100:.0f}% lower risk",
            } for f in protective]
            st.dataframe(pd.DataFrame(prot_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📈 Model Performance")
    st.info(
        "📖 **AUC-ROC:** How well the model separates high-risk from low-risk patients "
        "(0.5 = random guess, 1.0 = perfect). "
        "**Recall:** The most important metric here - how many truly at-risk patients "
        "the model correctly flags. Missing a high-risk patient is worse than a false alarm."
    )
    if baseline:
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("AUC-ROC",   f"{baseline.get('auc_roc', 0):.4f}")
        with m2: st.metric("Recall",    f"{baseline.get('recall', 0):.4f}")
        with m3: st.metric("Precision", f"{baseline.get('precision', 0):.4f}")
        with m4: st.metric("F1 Score",  f"{baseline.get('f1', 0):.4f}")
        with st.expander("ℹ️ Model Notes"):
            st.markdown("""
**Model:** Logistic Regression (balanced class weights)
**Training data:** Kenya DHS 2022 Individual Recode (n=32,156)
**Dropout cases:** Only 26 confirmed dropouts (0.08% prevalence)
**Key limitation:** Very low dropout prevalence means precision is low.
Recall is the primary metric in this public health context.
**Note on `ever_tested_hiv` OR:** Wide CI [0.038, >40] is a known limitation -
nearly all 26 dropouts share the same value, making bootstrap unstable.
            """)
    else:
        st.info("Run `06_model_2_dropout_prediction.ipynb` to generate model metrics.")

# ============================================================
# TAB 3: SCENARIO PROJECTIONS
# ============================================================
with tab3:
    st.header(getattr(c, "TAB3_TITLE", "2030 Forecast"))
    st.caption("Scenario-based projection of IIT and VLS rates per county tier, 2025–2030")

    st.info(
        "**What happens to Kenya's HIV programme by 2030?**\n\n"
        "🔵 **Scenario A (Business as Usual):** Current IIT rates stay flat through 2030. "
        "This is what happens if no new action is taken.\n\n"
        "🔴 **Scenario B (Bridged Gap):** A 30% reduction in IIT rates is applied to Critical and High tier "
        "counties from 2026 through targeted CHW retention programmes. "
        "The gap between the two lines = the quantified impact of that intervention.\n\n"
        "**Why does Scenario B show a one-time drop in 2026 then stay flat?** "
        "This is by design. The model applies a sustained 30% reduction from 2026 onwards, "
        "not a continuing annual decline. With only one year of data (2025), there is no evidence base "
        "to model year-on-year improvement. The 30% reduction is grounded in PEPFAR Kenya retention "
        "programme evidence. Moderate and Low tiers show identical lines as they do not receive the intervention."
    )

    st.subheader("📊 Treatment Interruption Rate Projections by Tier")
    st.caption(
        "IIT = Interruption in Treatment - when a patient misses their ART visit by 28+ days. "
        "Lower rate = more patients staying on treatment = better outcomes."
    )

    for tier in ["Critical", "High", "Moderate", "Low"]:
        df_tier = forecasts.get(tier, pd.DataFrame())
        if df_tier is not None and not df_tier.empty:
            st.markdown(f"### {tier} Tier")
            fig, ax = plt.subplots(figsize=(10, 4))
            df_a = df_tier[df_tier["scenario"] == "A"]
            df_b = df_tier[df_tier["scenario"] == "B"]
            if not df_a.empty:
                ax.plot(df_a["year"], df_a["iit_rate"], "o--", color="blue",
                        linewidth=2, label="Scenario A (Business as Usual)")
            if not df_b.empty:
                ax.plot(df_b["year"], df_b["iit_rate"], "s-", color="red",
                        linewidth=2, label="Scenario B (30% IIT Reduction)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Treatment Interruption Rate")
            ax.set_title(f"{tier} Tier - Treatment Interruption Rate 2025–2030", fontweight="bold")
            ax.set_xticks([2025, 2026, 2027, 2028, 2029, 2030])
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("🇰🇪 National Treatment Interruption Rate")
    st.caption("Kenya-wide average treatment interruption rate under both scenarios.")

    if not national_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        df_nat_a = national_df[national_df["scenario"] == "A"]
        df_nat_b = national_df[national_df["scenario"] == "B"]
        if not df_nat_a.empty:
            ax2.plot(df_nat_a["year"], df_nat_a["iit_rate"], "o--", color="blue",
                     linewidth=2, label="Scenario A (Business as Usual)")
        if not df_nat_b.empty:
            ax2.plot(df_nat_b["year"], df_nat_b["iit_rate"], "s-", color="red",
                     linewidth=2, label="Scenario B (30% Reduction)")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Treatment Interruption Rate")
        ax2.set_title("Kenya National Treatment Interruption Rate 2025–2030", fontweight="bold")
        ax2.set_xticks([2025, 2026, 2027, 2028, 2029, 2030])
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

        if not df_nat_b.empty and not df_nat_a.empty:
            final_b = df_nat_b[df_nat_b["year"] == 2030]["iit_rate"].values
            final_a = df_nat_a[df_nat_a["year"] == 2030]["iit_rate"].values
            if len(final_b) > 0 and len(final_a) > 0:
                reduction = (final_a[0] - final_b[0]) / final_a[0] * 100
                st.success(f"📉 Scenario B reduces national treatment interruption rate by {reduction:.0f}% by 2030")

    st.subheader("🇰🇪 National Viral Load Suppression Rate")
    st.caption(
        "VLS = Viral Load Suppression - when ART reduces HIV to undetectable levels. "
        "Higher is better. The UNAIDS 95% target means 95% of ART patients should be "
        "virally suppressed by 2030 (part of the global 95-95-95 targets)."
    )

    if not national_df.empty and "vls_rate_adult" in national_df.columns:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        df_nat_a = national_df[national_df["scenario"] == "A"]
        df_nat_b = national_df[national_df["scenario"] == "B"]
        if not df_nat_a.empty:
            ax3.plot(df_nat_a["year"], df_nat_a["vls_rate_adult"], "o--", color="blue",
                     linewidth=2, label="Scenario A (Business as Usual)")
        if not df_nat_b.empty:
            ax3.plot(df_nat_b["year"], df_nat_b["vls_rate_adult"], "s-", color="green",
                     linewidth=2, label="Scenario B (Bridged Gap)")
        ax3.axhline(y=0.95, color="orange", linestyle=":", linewidth=2,
                    label="UNAIDS 95% VLS Target")
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Viral Load Suppression Rate")
        ax3.set_title("Kenya National Viral Load Suppression Rate 2025–2030", fontweight="bold")
        ax3.set_xticks([2025, 2026, 2027, 2028, 2029, 2030])
        ax3.set_ylim(0.85, 1.0)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)

    st.subheader("🏥 Patients Additionally Retained Under Scenario B")
    st.info(
        "📖 **What does this mean?** This shows how many additional patients would REMAIN "
        "on HIV treatment by 2030 if Kenya implements Scenario B interventions, "
        "compared to doing nothing (Scenario A). "
        "These are real people who would otherwise miss their ART visits and risk viral rebound, "
        "onward transmission, and disease progression to AIDS."
    )

    if os.path.exists(c.PATIENTS_RETAINED):
        pr_df = pd.read_csv(c.PATIENTS_RETAINED)
        total_saved = pr_df["patients_saved"].sum() if "patients_saved" in pr_df.columns else 0
        st.metric(
            "Total Additional Patients Retained on ART by 2030",
            f"{int(total_saved):,}",
            delta="Scenario B vs Business as Usual"
        )
        if "tier" in pr_df.columns and "patients_saved" in pr_df.columns:
            tier_saved = (
                pr_df.groupby("tier")["patients_saved"]
                .sum().reset_index()
                .rename(columns={"patients_saved": "Additional Patients Retained by 2030"})
                .sort_values("Additional Patients Retained by 2030", ascending=False)
            )
            st.dataframe(tier_saved, use_container_width=True, hide_index=True)
    else:
        st.info("Run `scripts/train_model3.py` to generate patients_retained.csv")

    st.subheader("📋 County Comparison (2025 Baseline)")
    st.info(
        "📖 This table compares all 47 counties using 2025 data. "
        "**Highest Risk Counties** have the largest Care Gap Index - meaning more patients "
        "are interrupting treatment and fewer are virally suppressed. These need help first. "
        "**Best Performing Counties** have the smallest gaps and are closest to the UNAIDS targets. "
        "Note: A county's tier (Critical/High/Moderate/Low) is assigned by the AI clustering model "
        "and may differ from a simple ranking by CGI score alone."
    )

    if df_county is not None and cgi_col:
        col_worst, col_best = st.columns(2)
        sub_cols = ["county", "tier", cgi_col]
        if "iit_rate" in df_county.columns:
            sub_cols.append("iit_rate")

        with col_worst:
            st.markdown("**🔴 Highest Risk Counties (Largest Care Gap)**")
            top_worst = df_county.nlargest(10, cgi_col)[sub_cols].copy()
            top_worst.rename(columns={cgi_col: "Care Gap Index", "iit_rate": "Interruption Rate"}, inplace=True)
            st.dataframe(top_worst, use_container_width=True, hide_index=True)

        with col_best:
            st.markdown("**🟢 Best Performing Counties (Smallest Care Gap)**")
            top_best = df_county.nsmallest(10, cgi_col)[sub_cols].copy()
            top_best.rename(columns={cgi_col: "Care Gap Index", "iit_rate": "Interruption Rate"}, inplace=True)
            st.dataframe(top_best, use_container_width=True, hide_index=True)

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

    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
**Data Sources:**
- County profiles: NSDCC 2025 raw programme data (47 counties)
- Individual risk factors: Kenya DHS 2022 Individual Recode (32,156 records)
- Projections: Scenario-based linear projection (2025–2030)

**Key Terms:**
- **IIT (Interruption in Treatment):** Patient misses ART visit by 28+ days
- **VLS (Viral Load Suppression):** HIV reduced to undetectable levels on ART
- **CGI (Care Gap Index):** IIT rate (40%) + inverse VLS rate (40%) + HTS positivity (20%)
- **ART:** Antiretroviral Therapy - medicine that treats HIV

**Scenarios:**
- **Scenario A (BAU):** Current treatment interruption rates remain constant through 2030
- **Scenario B (Bridged Gap):** 30% reduction in interruption rates for Critical/High tier counties from 2026

**UNAIDS 95% Target:** 95% of patients on ART should be virally suppressed by 2030.

**Model retraining:** When new NSDCC data is available, run:
```
python scripts/train_model3.py
python app/trigger_alerts.py
```
        """)

# ── Footer
st.markdown("---")
st.caption("Built using Kenya DHS 2022 + NSDCC 2025 programme data")
