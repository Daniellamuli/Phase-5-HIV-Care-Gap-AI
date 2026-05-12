<div align="center">
  <img src="figures/logo.jpg" alt="HIV Care Gap AI Logo" width="200"/>
  
  ## Team Members
  **Daniella Muli • Eve Michelle • Naomi Opiyo • Pheonverah Achieng' • Lorenah Mbogo • Dennis Kamuri**
  
  # HIV Care Gap AI: Kenya
  
  **County Risk Mapping · Individual Dropout Prediction · 2030 Scenario Forecasting**
  
  **Using Machine Learning to Predict Who Kenya is Leaving Behind**
</div>

---


# Phase 5 HIV Care Gap AI 🩺

## Project Overview

An end-to-end machine learning system that identifies HIV care gaps across Kenya's 47 counties, predicts individual treatment dropout risk, and forecasts 2030 outcomes under different intervention scenarios — giving the Ministry of Health precision tools to close the gap.

---

## Business Understanding

**Problem:** Kenya's HIV response is losing ground. New infections rose 19% in 2024, reversing years of decline. Just 10 counties account for 60% of all new infections, yet no AI tool currently ranks all 47 counties by care gap urgency using raw programme data.

**Solution:** Three integrated models that give MOH actionable intelligence:

| Model | Question | Output |
|-------|----------|--------|
| **Model 1 — County Gap Map** | Where is the health system losing patients right now? | 47 counties ranked by Care Gap Index, tiered 🔴🟠🟡🟢 |
| **Model 2 — Dropout Risk Calculator** | Who is about to interrupt treatment? | Individual risk score (0–1) + top 3 dropout drivers |
| **Model 3 — 2030 Forecast** | What happens if we act — or don't? | BAU vs Bridged Gap projections for IIT and VLS to 2030 |

---

## Stakeholders

| Stakeholder | How They Use This |
|-------------|------------------|
| Ministry of Health (MOH) Kenya | National strategic planning and resource allocation |
| County Directors of Health | Subnational intervention prioritisation |
| NSDCC / NASCOP | National HIV programme management |
| Community Health Workers | Patient-level follow-up prioritisation |
| UNAIDS / World Bank | Evidence-based investment decisions |

---

## Data Sources

### Dataset 1 — NSDCC Raw Programme Data
Source: [analytics.nsdcc.go.ke](https://analytics.nsdcc.go.ke) → HIV Estimates 2025

> **Download all raw files here:** [Google Drive](https://drive.google.com/drive/folders/1vJ6NUUUjVKgZPfy37ODNRvDQW81kTDQK?usp=sharing)
> After downloading, place all files in `data/raw/`

| File | Contents |
|------|----------|
| `IIT.xlsx` | Interruption in Treatment counts by county and period |
| `VLT.xlsx` | Viral Load Testing: tested and suppressed counts |
| `Adult_on_HTS.xlsx` | HIV Testing Services: tested and positive counts |
| `Adult_on_ART.xlsx` | Adults on ART by county and period |

### Dataset 2 — DHS Kenya 2022 Individual Recode
Source: [dhsprogram.com](https://dhsprogram.com) — approved and downloaded.
Processed file: `data/processed/individual_features_clean.csv` (32,156 individuals, 15 features)

---

## Project Structure
hiv-care-gap-ai/
│
├── data/
│   ├── raw/                         
│   │   ├── Adult_on_ART.xlsx
│   │   ├── Adult_on_HTS.xlsx
│   │   ├── IIT.xlsx
│   │   ├── VLT.xlsx
│   │   └── individual_features.csv
│   │
│   └── processed/                  
│   │   ├── adult_on_art_clean.csv 
│   │   ├── hts_clean.csv                   
│   │   ├── vlt_clean.csv                   
│   │   ├── iit_clean.csv                   
│   │   ├── individual_features_clean.csv   
│   │   ├── nsdcc_merged.csv                
│   │   ├── county_profiles.csv             
│   │   ├── tier_timeseries.csv             
│   │   ├── forecast_critical.csv           
│   │   ├── forecast_high.csv              
│   │   ├── forecast_moderate.csv          
│   │   ├── forecast_low.csv               
│   │   └── forecast_national.csv 
│
├── notebooks/                      
│   ├── 01_data_extraction.ipynb              
│   ├── 02_nsdcc_cleaning.ipynb
│   ├── 03_dhs_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_1_county_clustering.ipynb
│   ├── 06_model_2_dropout_prediction.ipynb
│   ├── 07_model_3_forecasting.ipynb
│   ├── 08_model_evaluation.ipynb
│   ├── 09_deployment.ipynb
│   ├── 10_monitoring.ipynb
│   └── final_notebook.ipynb
│
├── src/                            ← Python modules 
│   ├── __init__.py
│   ├── utils.py                              ← Shared helpers
│   ├── nsdcc_cleaner.py                      ← Load + clean nsdcc functions
│   ├── dhs_cleaner.py                        ← Load + clean dhs functions
│   ├── feature_engineering.py                ← CGI + tier aggregation
│   ├── model_training.py                     ← KMeans + XGBoost + Prophet
│   ├── evaluation.py                         ← Metrics functions
│   └── forecasting.py                        ← Prophet scenario functions
│
├── scripts/                        ← Production pipeline scripts
│   ├── extract_data.py                       ← Wrapper for data loading
│   ├── merge_data.py                         ← Merge NSDCC files
│   ├── prepare_data.py                       ← Feature engineering wrapper
│   ├── train_model1.py                       ← KMeans wrapper
│   ├── train_model2.py                       ← XGBoost wrapper
│   └── train_model3.py                       ← Prophet wrapper
│
├── app/
│   ├── streamlit_app.py
│   ├── trigger_alerts.py
│   └── trigger_predictions.py
│
├── models/                         ← Will hold .pkl files
├── figures/                        ← logo.jpg + exported charts
├── presentation/                   ← Slides
├── tableau/                        ← Tableau dashboard files
│
├── constants.py                    ← On main
├── main.py                         ← To be created
├── requirements.txt
├── .gitignore
└── PROJECT_PLAN.md                 ← This file

---

## Quick Start

### Prerequisites
- Python 3.8+
- Conda or virtualenv
- 4GB RAM minimum

### Setup

```bash
# Clone the repo
git clone https://github.com/Daniellamuli/Phase-5-HIV-Care-Gap-AI.git
cd Phase-5-HIV-Care-Gap-AI

# Create and activate environment
conda create -n learn-env python=3.10
conda activate learn-env

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python main.py
```

### Run Notebooks Manually (CRISP-DM order)

```bash
jupyter notebook notebooks/01_data_extraction.ipynb
# Continue through 02 → 11 sequentially
```

### Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Models

### Model 1 — County Care Gap Map
- **Algorithm:** Care Gap Index (engineered) + KMeans Clustering (k=4)
- **Data:** NSDCC IIT, VLT, HTS, ART
- **Output:** 47 counties tiered 🔴 Critical → 🟢 Low

### Model 2 — Individual Dropout Risk
- **Algorithm:** Logistic Regression (baseline) + XGBoost (primary)
- **Data:** DHS Kenya 2022 (32,156 individuals, 15 features)
- **Output:** Risk probability (0–1) + top 3 demographic/behavioural drivers

### Model 3 — 2030 Dual Scenario Forecast
- **Algorithm:** Facebook Prophet (per county tier, per scenario)
- **Data:** NSDCC IIT and VLS time series 2020–2025
- **Output:** Scenario A (BAU) vs Scenario B (30% IIT reduction in Critical/High counties by 2026)

---

## Key Metrics Defined

| Metric | Definition |
|--------|-----------|
| **IIT Rate** | % of ART patients who interrupt treatment (miss visit by 28+ days) |
| **VLS Rate** | % of patients on ART with viral load below suppression threshold |
| **HTS Positivity Rate** | % of individuals tested who receive a positive HIV result |
| **ART Coverage** | % of PLHIV currently receiving ART |
| **Care Gap Index** | Composite score combining IIT, VLS, HTS, and ART — higher = worse |

---

## Recommendations

| Model | Recommendation |
|-------|---------------|
| Model 1 | Immediately prioritise retention in Critical tier counties (IIT above national average, VLS below 90%) |
| Model 2 | Deploy differentiated follow-up for wealth quintiles 1–2, aged 15–34, living 10km+ from a facility |
| Model 3 | Closing the IIT gap in Critical/High counties by 2026 is projected to substantially improve national VLS by 2030 |