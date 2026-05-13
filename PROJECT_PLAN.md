# HIV Care Gap AI: Kenya — Project Plan 
**County Risk Mapping · Individual Dropout Prediction · 2030 Scenario Forecasting**

> **Team:** Daniella (Lead) · Eve · Verah · Naomi · Lorenah · Dennis  
> **Timeline:** 10 working days · **Methodology:** CRISP-DM  

---

## ⚠️ Model 3 Methodology Update

**Original plan:** Facebook Prophet time-series forecasting

**Change made:** Switched to **scenario-based linear projection + cross-sectional comparison**

**Reason:** NSDCC portal only provides 2025 raw data. Prophet requires a minimum of 3-4 years of historical data to detect trends. With a single year, Prophet would produce unreliable forecasts.

**Impact on deliverables:** None — all original deliverables are preserved.

**New Model 3 has two components:**

### Component A — Scenario-Based Projection (2025 → 2030)
- **Scenario A (BAU):** Current 2025 IIT and VLS rates held flat to 2030
- **Scenario B (Bridged Gap):** 30% IIT reduction in Critical + High tier
  counties from 2026 (`IIT_REDUCTION_RATE = 0.30` in constants.py)
- Output: Four tier line charts + national headline chart + 5 projection CSVs

### Component B — Cross-Sectional Comparison (NEW)
Uses 2025 snapshot to compare all 47 counties against each other:

| Analysis | What it shows |
|----------|---------------|
| Best vs worst counties | Which counties need immediate attention |
| Regional patterns | Which regions have systemic issues |
| Urban vs rural | Access disparities |

---

## Data Sources

We have **6 raw files** from two sources:

### NSDCC (National Syndemic Diseases Control Council) — 5 files
| File | Description |
|------|-------------|
| `Adult_on_ART.xlsx` | Adults currently on antiretroviral therapy, by county and period |
| `Adult_on_HTS.xlsx` | HIV testing services (tested counts) |
| `HTS_Positive.xlsx` | HIV testing services (positive results) |
| `IIT.xlsx` | Interruption in treatment (patients who missed ART by 28+ days) |
| `VLT.xlsx` | Viral load testing (tested counts + suppressed counts) |

### DHS (Demographic and Health Survey) — 1 file
| File | Description |
|------|-------------|
| `individual_features.csv` | Individual-level survey data (32,156 records, 15 features) |

---

## The Golden Rules

1. **One person, one file per day.** No two people edit the same file on the same day.
2. **`constants.py` is the single source of truth.** Never hard-code anything.
3. **Notebooks for exploration. `src/` for reusable logic.**
4. **If you write a notebook, you write/update the matching `src/` module.**
5. **Branch → Work → Pull Request → Merge → Pull.** Never commit directly to `main`.
6. **Data never goes to GitHub.** `data/` folder is in `.gitignore`.

---

# 📅 DAY-BY-DAY PLAN (10 Days)

## Day 1 — Data Cleaning: NSDCC + DHS (Parallel)

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Complete `02_nsdcc_cleaning.ipynb`: Load 5 NSDCC files, strip county suffix, standardise names, rename MOH columns, convert to numeric, impute missing values (county mean → column median), create before/after missing-value heatmaps, validate 47 counties and 0 duplicates, save 5 clean CSVs to `data/processed/`. THEN create `src/nsdcc_cleaner.py` with reusable functions. | 5 clean CSVs, `nsdcc_cleaner.py` |
| **Lorenah** | Complete `03_dhs_cleaning.ipynb`: Load `individual_features.csv`, map county codes (1-47) to names, decode age/education/wealth/marital/distance columns using constants, impute binary flags with 0 and numeric with median, engineer dropout target (`told_hiv_positive=1` AND `tested_hiv_last_12months=0`), create before/after missing-value heatmap, one-hot encode education and wealth, save to `individual_features_clean.csv`. THEN create `src/dhs_cleaner.py`. | `individual_features_clean.csv`, `dhs_cleaner.py` |
| **Verah** | Create `src/feature_engineering.py`: Add functions for CGI calculation and tier aggregation. | `feature_engineering.py` |
| **Naomi** | Review DHS class balance from Lorenah's output. Update `XGB_SCALE_POS_WEIGHT` in `constants.py`. | `constants.py` updated |
| **Dennis** | Create `src/utils.py`: Add `standardise_county()`, `get_tier_color()`, `save_csv()`, `load_csv()` helpers. | `utils.py` |
| **Daniella** | Review PRs, merge Day 1 work. Create `main.py` skeleton. | `main.py` |

---

## Day 2 — Data Merging + Feature Engineering Start

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create `scripts/merge_data.py`: Merge 5 clean NSDCC CSVs → `nsdcc_clean.csv`. Validate merge (47 counties, no lost rows). | `merge_data.py`, `nsdcc_clean.csv` |
| **Lorenah** | Begin `06_model_2_dropout_prediction.ipynb`: Load `individual_features_clean.csv`, split data, train Logistic Regression baseline. Document AUC-ROC, Recall. **Extract odds ratios with 95% confidence intervals. Save to `data/processed/odds_ratios.json`.** | LR baseline metrics, `odds_ratios.json` |
| **Verah** | Complete `04_feature_engineering.ipynb` AND add reusable functions to `src/feature_engineering.py`. Compute Care Gap Index and build `county_profiles.csv`. Note: YOY change columns will be 0 (single year) — include them as placeholders for future multi-year data. | `county_profiles.csv`, `feature_engineering.py`, `feature_engineering.ipynb` |
| **Naomi** | Complete `05_model_1_county_clustering.ipynb`: KMeans clustering, silhouette score, assign tier labels, save model. | `kmeans_county_tiers.pkl` |
| **Dennis** | Update `src/evaluation.py`: Add `silhouette_score()`, `classification_metrics()`, `forecast_errors()` functions. | `evaluation.py` |
| **Daniella** | Wire `merge_data` and `feature_engineering` into `main.py`. Review PRs. | `main.py` Days 1-2 running |

---

## Day 3 — Model 1 + Model 2 + Model 3 Start (Parallel)

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create `scripts/train_model1.py` wrapper for KMeans. | `train_model1.py` |
| **Lorenah** | Complete `06_model_2_dropout_prediction.ipynb`: Train XGBoost, evaluate, plot feature importance. | XGBoost trained, metrics |
| **Verah** | Begin `07_model_3_forecasting.ipynb`: **Component A** — Run Scenario A (BAU flat projection) per tier from 2025 to 2030. **Component B** — Cross-sectional comparison: rank all 47 counties by IIT rate, VLS rate, HTS positivity. Identify best vs worst counties, regional patterns, urban vs rural disparities. Save `county_comparison.csv`. | BAU projection plotted, `county_comparison.csv` |
| **Naomi** | Create `src/projection.py`: Add `project_bau()`, `project_bridged_gap()`, `build_scenario_df()`, `cross_sectional_compare()`, `patients_retained_counter()` functions. **Note: No Prophet — scenario-based projection only.** | `projection.py` |
| **Dennis** | Create `src/model_training.py`: Add KMeans wrapper, XGBoost wrapper functions. | `model_training.py` |
| **Daniella** | Update `main.py` with Model 1 + Model 2 training steps. Review PRs. | `main.py` Day 3 running |

---

## Day 4 — Dashboard Foundation + Forecasting + Choropleth Map (Stretch)

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create `scripts/train_model3.py`: Wrapper that runs both Component A (scenario projection) and Component B (cross-sectional comparison) and saves all forecast CSVs + `county_comparison.csv`. | `train_model3.py` |
| **Lorenah** | Build `app/streamlit_app.py` Tab 1 (County Gap Map) — full implementation. Stretch: Add Folium choropleth map. | Tab 1 working, choropleth map (if time) |
| **Verah** | Complete `07_model_3_forecasting.ipynb`: Add Scenario B (30% IIT reduction in Critical + High tiers from 2026). Add patients-retained counter. Save all 5 forecast CSVs. | Forecast charts, all CSVs saved |
| **Naomi** | Create `scripts/train_model2.py` wrapper for XGBoost + Set up Streamlit Cloud account. | `train_model2.py`, Cloud ready |
| **Dennis** | Test Tab 1 (verify county colors, sorting, data loading). Report issues. | Tab 1 validated |
| **Daniella** | Update `main.py` with Model 3 + Review PRs. | `main.py` Day 4 running |

---

## Day 5 — Dashboard Completion + Triggers

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Build `app/streamlit_app.py` Tab 2 (Risk Factor Identification) — full implementation. Load `odds_ratios.json` and display top risk factors with odds ratios and confidence intervals. | Tab 2 working |
| **Lorenah** | Build `app/streamlit_app.py` Tab 3 (2030 Projection Charts) — full implementation. Include both Scenario A and Scenario B lines + cross-sectional comparison table. | Tab 3 working |
| **Verah** | Create `app/trigger_alerts.py` — IIT alert system. | `trigger_alerts.py` |
| **Naomi** | Integrate all 3 tabs, handle cross-tab dependencies + Create `app/trigger_predictions.py`. | Full dashboard, `trigger_predictions.py` |
| **Dennis** | Test Tabs 2-3 with sample data (risk scores, charts display). Report issues. | Tabs 2-3 validated |
| **Daniella** | Begin `notebooks/final_notebook.ipynb` + Prepare deployment checklist. | Final notebook started |

---

## Day 6 — Evaluation + Test Deployment

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Run `08_model_evaluation.ipynb` for Model 1 metrics. | Model 1 eval |
| **Lorenah** | Run `08_model_evaluation.ipynb` for Model 2 metrics + **Extract odds ratios from Logistic Regression**. Create **forest plot** of top 5-7 risk factors (wealth, distance, age, education, marital status) with 95% confidence intervals. Add odds ratios table to dashboard footer. | Model 2 eval, odds ratios table, forest plot |
| **Verah** | Run `08_model_evaluation.ipynb` for Model 3 metrics. For scenario projection: document assumptions, validate Scenario B vs Scenario A gap, validate cross-sectional rankings. | Model 3 eval |
| **Naomi** | Deploy dashboard to Streamlit Cloud (test deployment) + Compile all evaluation metrics. | Test URL, evaluation compiled |
| **Dennis** | Help Verah with Model 3 evaluation (verify projection CSVs, cross-sectional comparison). | Support |
| **Daniella** | Test deployed dashboard on 3 browsers + Create `presentation/slides.md` outline. | Test report, slides outline |

---

## Day 7 — Final Notebook + Final Deployment + Triggers

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Test `trigger_predictions.py` with dry run. | Trigger tested |
| **Lorenah** | Test `trigger_alerts.py` with current data. | Alerts tested |
| **Verah** | Update `README.md` with setup and run instructions. **Include Model 3 methodology update section explaining switch from Prophet to scenario projection.** | README complete |
| **Naomi** | Final deployment to Streamlit Cloud (production) + Complete `final_notebook.ipynb`. | Live dashboard URL, final notebook |
| **Dennis** | Complete `10_monitoring.ipynb` (documentation for annual updates). | Monitoring notebook |
| **Daniella** | Review PRs, coordinate final checks. | All PRs merged |

---

## Day 8 — Final Dashboard Polish + Documentation

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Run full pipeline on clean environment + Document any issues. | Pipeline verified |
| **Lorenah** | Polish dashboard (error handling, loading states, tooltips) + Final browser check. | Dashboard final |
| **Verah** | Review README, ensure all commands work + Final `trigger_alerts.py` test. | README verified, alerts ready |
| **Naomi** | Final dashboard check on all browsers + mobile + Fix any issues found. | Dashboard verified |
| **Dennis** | Audit `constants.py` for hard-coded values + Complete `10_monitoring.ipynb`. | `constants.py` final, monitoring done |
| **Daniella** | Create complete presentation deck (10-12 slides): Title, Problem, Data, Model 1, Model 2, Model 3 (scenario projection + cross-sectional), Dashboard, Recommendations, Limitations, Conclusion. | Slides complete |

---

## Day 9 — Rehearsal + Final Fixes

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Test `trigger_predictions.py` with dry run + Practice Model 1 presentation. | Trigger tested, ready to present |
| **Lorenah** | Test trigger scripts with sample new data + Practice Model 2 presentation. | Triggers verified, ready to present |
| **Verah** | Verify README commands work + Practice Model 3 presentation. | README verified, ready to present |
| **Naomi** | Final dashboard check + Practice Data + Methodology presentation. | Dashboard verified, ready to present |
| **Dennis** | Final deployment check + Practice Dashboard + Deployment presentation. | Dashboard live, ready to present |
| **Daniella** | Lead 1-hour full presentation rehearsal + Polish slides based on feedback + Time each section. | Team ready, slides final, timing notes |

---

## Day 10 — Final Submission + Presentation

| Person | Task | Deliverable |
|--------|------|-------------|
| **All** | Final presentation (15 min + 5 min Q&A) | Live presentation |
| **All** | Submit GitHub repo link | Submission |
| **All** | Submit live dashboard URL | Submission |
| **All** | Submit final notebook as PDF | Submission |

---

## Success Checklist (End of Day 10)

- [ ] `main.py` runs end-to-end without errors on clean environment
- [ ] All 3 model `.pkl` files saved in `models/`
- [ ] All forecast CSVs saved in `data/processed/`
- [ ] `county_comparison.csv` saved in `data/processed/`
- [ ] Dashboard runs locally — all 3 tabs functional
- [ ] Dashboard deployed to Streamlit Cloud — public URL works
- [ ] `trigger_alerts.py` correctly flags counties above IIT threshold
- [ ] `trigger_predictions.py` runs without errors
- [ ] `final_notebook.ipynb` imports from src and shows all outputs
- [ ] All code uses `constants.py` — zero hard-coded values
- [ ] All 10 notebooks run in sequence without errors
- [ ] Presentation slides complete (15 min timed)
- [ ] README has data download link + step-by-step instructions + Model 3 methodology note
- [ ] GitHub repo is public/accessible
- [ ] Team has rehearsed at least once

---

## Repository Structure

```text
hiv-care-gap-ai/
│
├── data/
│   ├── raw/                         ← All 6 files present
│   │   ├── Adult_on_ART.xlsx
│   │   ├── Adult_on_HTS.xlsx
│   │   ├── HTS_Positive.xlsx
│   │   ├── IIT.xlsx
│   │   ├── VLT.xlsx
│   │   └── individual_features.csv
│   │
│   └── processed/                  ← Empty — will be populated
│       ├── adult_on_art_clean.csv
│       ├── hts_clean.csv
│       ├── hts_positive_clean.csv
│       ├── vlt_clean.csv
│       ├── iit_clean.csv
│       ├── individual_features_clean.csv
│       ├── nsdcc_clean.csv
│       ├── county_profiles.csv
│       ├── tier_timeseries.csv
│       ├── county_comparison.csv         ←cross-sectional comparison
│       ├── projection_critical.csv
│       ├── projection_high.csv
│       ├── projection_moderate.csv
│       ├── projection_low.csv
│       └── projection_national.csv
│
├── notebooks/                      ← 11 notebooks total
│   ├── 01_data_extraction.ipynb              COMPLETE (Naomi)
│   ├── 02_nsdcc_cleaning.ipynb
│   ├── 03_dhs_cleaning.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_1_county_clustering.ipynb
│   ├── 06_model_2_dropout_prediction.ipynb
│   ├── 07_model_3_projection.ipynb
│   ├── 08_model_evaluation.ipynb
│   ├── 09_deployment.ipynb
│   ├── 10_monitoring.ipynb
│   └── final_notebook.ipynb
│
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── nsdcc_cleaner.py
│   ├── dhs_cleaner.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── projection.py                 
│
├── scripts/
│   ├── extract_data.py
│   ├── merge_data.py
│   ├── prepare_data.py
│   ├── train_model1.py
│   ├── train_model2.py
│   └── train_model3.py              
│
├── app/
│   ├── streamlit_app.py
│   ├── trigger_alerts.py
│   └── trigger_predictions.py
│
├── models/
├── figures/
├── presentation/
├── tableau/
│
├── constants.py
├── main.py
├── requirements.txt
├── .gitignore
└── PROJECT_PLAN.md
```
---
