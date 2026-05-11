# HIV Care Gap AI: Kenya — Project Plan 
**County Risk Mapping · Individual Dropout Prediction · 2030 Scenario Forecasting**

> **Team:** Daniella (Lead) · Eve · Verah · Naomi · Lorenah · Dennis  
> **Timeline:** 10 working days · **Methodology:** CRISP-DM  
> **Status:** Day 1 starts with `01_data_extraction.ipynb`  

---

## Current State (As of Day 1 Morning)

| Item | Status | Owner |
|------|--------|-------|
| `01_data_extraction.ipynb` | COMPLETE | Naomi |
| Raw data files in `data/raw/` | All 5 files present | Daniella |
| `constants.py` | On main | Naomi |
| Folder structure | Created | Daniella |
| `src/` utilities | Needs population | Team |

---
## Data Sources

We have **5 raw files** from two sources:

### NSDCC (National Syndemic Diseases Control Council) — 4 files
| File | Description |
|------|-------------|
| `Adult_on_ART.xlsx` | Adults currently on antiretroviral therapy, by county and period |
| `Adult_on_HTS.xlsx` | HIV testing services (tested counts + positive results) |
| `IIT.xlsx` | Interruption in treatment (patients who missed ART by 28+ days) |
| `VLT.xlsx` | Viral load testing (tested counts + suppressed counts) |

### DHS (Demographic and Health Survey) — 1 file
| File | Description |
|------|-------------|
| `individual_features.csv` | Individual-level survey data (32,156 records, 15 features including age, wealth, education, HIV testing history) |

---

## The Golden Rules

1. **One person, one file per day.** No two people edit the same file on the same day.
2. **`constants.py` is the single source of truth.** Never hard-code anything.
3. **Notebooks for exploration. `src/` for reusable logic.**
4. **If you write a notebook, you write/update the matching `src/` module.**
5. **Branch → Work → Pull Request → Merge → Pull.** Never commit directly to `main`.
6. **Data never goes to GitHub.** `data/` folder is in `.gitignore`.

---

## Repository Structure 

```text
hiv-care-gap-ai/
│
├── data/
│   ├── raw/                         ← All 5 files present
│   │   ├── Adult_on_ART.xlsx
│   │   ├── Adult_on_HTS.xlsx
│   │   ├── IIT.xlsx
│   │   ├── VLT.xlsx
│   │   └── individual_features.csv
│   │
│   └── processed/                  ← Empty — will be populated
│
├── notebooks/                      ← 11 notebooks total
│   ├── 01_data_extraction.ipynb              COMPLETE (Naomi)
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
├── src/                            ← Python modules (NOT scripts/)
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
├── models/                         ← Empty — will hold .pkl files
├── figures/                        ← logo.jpg + exported charts
├── presentation/                   ← Slides
├── tableau/                        ← Tableau dashboard files
│
├── constants.py                    ← On main
├── main.py                         ← To be created
├── requirements.txt
├── .gitignore
└── PROJECT_PLAN.md                 ← This file
```
---

# 📅 DAY-BY-DAY PLAN (10 Days)

## Day 1 — Data Cleaning: NSDCC + DHS (Parallel)

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Complete `02_nsdcc_cleaning.ipynb`: Load 4 NSDCC files, strip county suffix, standardise names, rename MOH columns, convert to numeric, impute missing values (county mean → column median), create before/after missing-value heatmaps, validate 47 counties and 0 duplicates, save 4 clean CSVs to `data/processed/`. THEN create `src/nsdcc_cleaner.py` with reusable functions. | 4 clean CSVs, `nsdcc_cleaner.py` |
| **Lorenah** | Complete `03_dhs_cleaning.ipynb`: Load `individual_features.csv`, map county codes (1-47) to names, decode age/education/wealth/marital/distance columns using constants, impute binary flags with 0 and numeric with median, engineer dropout target (`told_hiv_positive=1` AND `tested_hiv_last_12months=0`), create before/after missing-value heatmap, one-hot encode education and wealth, save to `individual_features_clean.csv`. THEN create `src/dhs_cleaner.py` with reusable functions. | `individual_features_clean.csv`, `dhs_cleaner.py` |
| **Verah** | Create `src/feature_engineering.py`: Add functions for CGI calculation and tier aggregation. | `feature_engineering.py` |
| **Naomi** | Review DHS class balance from Lorenah's output (Cell 17). Update `XGB_SCALE_POS_WEIGHT` in `constants.py`. | `constants.py` updated |
| **Dennis** | Create `src/utils.py`: Add `standardise_county()`, `get_tier_color()`, `save_csv()`, `load_csv()` helpers. | `utils.py` |
| **Daniella** | Review PRs, merge Day 1 work. Create `main.py` skeleton. | `main.py` |

---

## Day 2 — Data Merging + Feature Engineering Start

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create `scripts/merge_data.py`: Merge 4 clean NSDCC CSVs on county + period → `nsdcc_merged.csv`. Validate merge (47 counties, no lost rows). | `merge_data.py`, `nsdcc_merged.csv` |
| **Lorenah** | Begin `06_model_2_dropout_prediction.ipynb`: Load `individual_features_clean.csv`, split data, train Logistic Regression baseline. Document AUC-ROC, Recall. | LR baseline metrics |
| **Verah** | Complete `04_feature_engineering.ipynb`: Compute IIT yoy change, VLS yoy change, Engineer Care Gap Index, build county_profiles.csv and tier_timeseries.csv. | `county_profiles.csv`, `tier_timeseries.csv` |
| **Naomi** | Update `src/feature_engineering.py`: Add `calculate_yoy_changes()`, `calculate_cgi()`, `build_tier_timeseries()` functions. | `feature_engineering.py` |
| **Dennis** | Update `src/evaluation.py`: Add `silhouette_score()`, `classification_metrics()`, `forecast_errors()` functions. | `evaluation.py` |
| **Daniella** | Wire `merge_data` and `feature_engineering` into `main.py`. Review PRs. | `main.py` Days 1-2 running |

---

## Day 3 — Model 1 (KMeans) + Model 2 (XGBoost) Parallel

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create `scripts/train_model1.py` wrapper for KMeans. | `train_model1.py` |
| **Lorenah** | Complete `06_model_2_dropout_prediction.ipynb`: Train XGBoost, evaluate, plot feature importance. | XGBoost trained, metrics |
| **Verah** | Begin `07_model_3_forecasting.ipynb`: Load tier_timeseries.csv, run Prophet BAU forecast per tier. | BAU forecasts plotted |
| **Naomi** | Complete `05_model_1_county_clustering.ipynb`: KMeans clustering, silhouette score, assign tier labels, save model. | `kmeans_county_tiers.pkl` |
| **Dennis** | Create `src/model_training.py`: Add KMeans wrapper, XGBoost wrapper functions. | `model_training.py` |
| **Daniella** | Update `main.py` with Model 1 + Model 2 training steps. Review PRs. | `main.py` Day 3 running |

---

## Day 4 — Model 3 (Prophet) + Dashboard Tab 1 + Testing

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create `scripts/train_model3.py` wrapper for Prophet. | `train_model3.py` |
| **Lorenah** | Create `scripts/train_model2.py` wrapper for XGBoost. | `train_model2.py` |
| **Verah** | Complete `07_model_3_forecasting.ipynb`: Add Scenario B, backtest, save forecasts. | Forecast charts, CSVs |
| **Naomi** | Update `src/forecasting.py` AND test dashboard Tab 1 (verify county colors, sorting). | `forecasting.py`, Tab 1 validated |
| **Dennis** | Begin `app/streamlit_app.py` Tab 1 (County Gap Map). | Tab 1 working locally |
| **Daniella** | Update `main.py` with Model 3 + Set up Streamlit Cloud account. | `main.py` Day 4 running, Cloud ready |

---

## Day 5 — Dashboard Tabs 2 & 3 + Triggers + Testing

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Build `app/streamlit_app.py` Tab 2 (Dropout Risk Calculator). | Tab 2 working |
| **Lorenah** | Build `app/streamlit_app.py` Tab 3 (2030 Forecast Charts). | Tab 3 working |
| **Verah** | Create `app/trigger_alerts.py` — IIT alert system. | `trigger_alerts.py` |
| **Naomi** | Create `app/trigger_predictions.py` AND test Tabs 2-3 with sample data. | `trigger_predictions.py`, Tabs 2-3 validated |
| **Dennis** | Integrate Tabs 2-3 into dashboard, handle cross-tab dependencies. | Dashboard full |
| **Daniella** | Begin `final_notebook.ipynb` + Prepare deployment checklist. | Final notebook started, checklist ready |

---

## Day 6 — Evaluation + Test Deployment

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Run `08_model_evaluation.ipynb` for Model 1 metrics. | Model 1 eval |
| **Lorenah** | Run `08_model_evaluation.ipynb` for Model 2 metrics. | Model 2 eval |
| **Verah** | Run `08_model_evaluation.ipynb` for Model 3 metrics. | Model 3 eval |
| **Naomi** | Complete `08_model_evaluation.ipynb` — compile all metrics. | Evaluation notebook |
| **Dennis** | Deploy dashboard to Streamlit Cloud (test deployment). | Test URL |
| **Daniella** | Test deployed dashboard on 3 browsers + Document issues + Create `presentation/slides.md` outline. | Test report, slides outline |

---

## Day 7 — Final Notebook + Final Deployment + Triggers

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Test `trigger_predictions.py` with dry run. | Trigger tested |
| **Lorenah** | Test `trigger_alerts.py` with current data. | Alerts tested |
| **Verah** | Update `README.md` with setup and run instructions. | README complete |
| **Naomi** | Complete `10_monitoring.ipynb` + Test final deployment URL. | Monitoring notebook, URL validated |
| **Dennis** | Final deployment to Streamlit Cloud (production). | Live dashboard URL |
| **Daniella** | Complete `final_notebook.ipynb` with all 3 models and recommendations. | Final notebook ready |

---

## Day 8 — Presentation Slides + Dashboard Polish

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create slide content for Model 1 (2-3 slides). | Slides section |
| **Lorenah** | Create slide content for Model 2 (2-3 slides). | Slides section |
| **Verah** | Create slide content for Model 3 (2-3 slides). | Slides section |
| **Naomi** | Create slide content for Data + Methodology + Testing (2-3 slides). | Slides section |
| **Dennis** | Polish dashboard + Create slide for Dashboard + Deployment. | Dashboard final, slides section |
| **Daniella** | Assemble all slides into final deck + Create Intro/Conclusion/Recommendation slides. | Slides complete |

---

## Day 9 — Rehearsal + Final Fixes

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Run full pipeline on clean environment. Fix any bugs. | Pipeline verified |
| **Lorenah** | Test trigger scripts with sample new data. | Triggers verified |
| **Verah** | Review README, ensure all commands work. | README verified |
| **Naomi** | Review constants.py for hard-coded values + Time presentation sections. | constants.py final, timing notes |
| **Dennis** | Final dashboard check on all browsers + mobile. | Dashboard verified |
| **Daniella** | Lead 1-hour full presentation rehearsal + Note fixes needed. | Team ready, fix list |

---

## Day 10 — Final Submission + Presentation

| Person | Task | Deliverable |
|--------|------|-------------|
| **All** | Final presentation (15 min + 5 min Q&A) | Live presentation |
| **All** | Submit GitHub repo link | Submission |
| **All** | Submit live dashboard URL | Submission |
| **All** | Submit final notebook as PDF | Submission |

---
## Parallel Flow Summary

### Days 1-2: Maximum Parallelism (Everyone Works)

| Person | Day 1 | Day 2 |
|--------|-------|-------|
| Eve | `02_nsdcc_cleaning.ipynb` | `scripts/merge_data.py` |
| Lorenah | `03_dhs_cleaning.ipynb` | `06_model_2_dropout_prediction.ipynb` (Logistic Regression) |
| Verah | `src/data_preprocessing.py` | `04_feature_engineering.ipynb` |
| Naomi | Update `constants.py` | `src/feature_engineering.py` |
| Dennis | `src/utils.py` | `src/evaluation.py` |
| Daniella | `main.py` (skeleton) | `main.py` (wire Days 1-2) |

** ALL 6 WORKING IN PARALLEL — NO ONE WAITING**

### Days 3-4: Three Parallel Streams

| Stream | Day 3 | Day 4 |
|--------|-------|-------|
| **A (Model 1)** | Eve: `scripts/train1.py` | Eve: `scripts/train3.py` |
| | Naomi: `05_model_1_county_clustering.ipynb`-KMeans | Naomi: `src/forecasting.py` |
| **B (Model 2)** | Lorenah: `06_model_2_dropout_prediction.ipynb` -XGBoost | Lorenah: `scripts/train2.py` |
| | Verah: `07_model_3_forecasting.ipynb`-BAU | Verah: `07_model_3_forecasting.ipynb`-Scenario B |
| **C (Core)** | Dennis: `src/model_training.py` | Dennis: app Tab1 |
| | Daniella: `main.py` | Daniella: `main.py` |

**Stream A: County Clustering (Model 1)**
- Eve writes `scripts/train_model1.py`
- Naomi completes `05_model_1_county_clustering.ipynb`
- *Internal dependency:* Naomi needs Verah's `county_profiles.csv` (from Day 2)

**Stream B: Dropout Prediction (Model 2) + Forecasting Start (Model 3)**
- Lorenah completes `06_model_2_dropout_prediction.ipynb` (XGBoost)
- Verah starts `07_model_3_forecasting.ipynb` (BAU forecast)

**Stream C: Core Infrastructure**
- Dennis writes `src/model_training.py`
- Daniella wires Models 1-2 into `main.py`

** All three streams run at the same time. No stream waits for another.**
** ALL STREAMS RUN IN PARALLEL**

### Days 5-10: Fully Parallel (No Blocks)

Each person has their own distinct task each day. No dependencies between people.

| Day | Eve | Lorenah | Verah | Naomi | Dennis | Daniella |
|-----|-----|---------|-------|-------|--------|----------|
| 5 | Support dashboard | Support dashboard | `app/trigger_alerts.py` | `app/trigger_predictions.py` | Complete dashboard Tabs 2-3 | Start `final_notebook.ipynb` |
| 6 | Run `08_model_evaluation.ipynb` (Model 1) | Run `08_model_evaluation.ipynb` (Model 2) | Run `08_model_evaluation.ipynb` (Model 3) | Compile all metrics | Add metrics footer to dashboard | Create `presentation/slides.md` outline |
| 7 | Test `trigger_predictions.py` | Test `trigger_alerts.py` | Update `README.md` | Complete `10_monitoring.ipynb` | Deploy dashboard (test) | Complete `final_notebook.ipynb` |
| 8 | Create Model 1 slides | Create Model 2 slides | Create Model 3 slides | Create Data + Methodology slides | Polish dashboard | Assemble all slides |
| 9 | Run full pipeline on clean environment | Test triggers with sample data | Verify README commands | Audit `constants.py` | Final deployment | Lead rehearsal |
| 10 | Present | Present | Present | Present | Present | Submit all |

** COMPLETELY PARALLEL — NO DEPENDENCIES**

---

### Who Blocks Whom (The Only Dependencies)

| If you are... | You need this from... | By when | If delayed, do this instead |
|---------------|----------------------|---------|----------------------------|
| Dennis (Day 2 merge) | Eve's 4 clean NSDCC CSVs | End of Day 2 | Work on `src/evaluation.py` (doesn't need NSDCC data) |
| Naomi (Day 3 clustering) | Verah's `county_profiles.csv` | End of Day 2 | Start silhouette score exploration with dummy data |
| Verah (Day 4 Scenario B) | Naomi's tier labels | End of Day 3 | Complete BAU forecast first, add Scenario B later |
| Dennis (Day 5 dashboard Tabs 2-3) | All 3 `.pkl` files | End of Day 4 | Build Tab 1 first (needs only CSV, no models) |

---

# Success Checklist (End of Day 10)

- [ ] `main.py` runs end-to-end without errors on clean environment
- [ ] All 3 model `.pkl` files saved in `models/`
- [ ] All forecast CSVs saved in `data/processed/`
- [ ] Dashboard runs locally — all 3 tabs functional
- [ ] Dashboard deployed to Streamlit Cloud — public URL works
- [ ] `trigger_alerts.py` correctly flags counties above IIT threshold
- [ ] `trigger_predictions.py` runs without errors
- [ ] `final_notebook.ipynb` imports from src and shows all outputs
- [ ] All code uses `constants.py` — zero hard-coded values
- [ ] All 10 notebooks run in sequence without errors
- [ ] Presentation slides complete (15 min timed)
- [ ] README has data download link + step-by-step instructions
- [ ] GitHub repo is public/accessible
- [ ] Team has rehearsed at least once

---

# Quick Reference: File Ownership by Day

| Day | Eve | Lorenah | Verah | Naomi | Dennis | Daniella |
|-----|-----|---------|-------|-------|--------|----------|
| 1 | `02_nsdcc_cleaning.ipynb` + `src/nsdcc_cleaner.py` | `03_dhs_cleaning.ipynb` + `src/dhs_cleaner.py` | `src/feature_engineering.py` | `constants.py` | `src/utils.py` | `main.py` |
| 2 | `scripts/merge_data.py` | `06_model_2_dropout_prediction.ipynb` (Logistic Regression) | `04_feature_engineering.ipynb` | `src/feature_engineering.py` | `src/evaluation.py` | `main.py` |
| 3 | `scripts/train_model1.py` | `06_model_2_dropout_prediction.ipynb` (XGBoost) | `07_model_3_forecasting.ipynb` (BAU) | `05_model_1_county_clustering.ipynb` | `src/model_training.py` | `main.py` |
| 4 | `scripts/train_model3.py` | `scripts/train_model2.py` | `07_model_3_forecasting.ipynb` (Scenario B) | `src/forecasting.py` | `app/streamlit_app.py` (Tab 1) | `main.py` |
| 5 | Support dashboard | Support dashboard Tab 2 | `app/trigger_alerts.py` | `app/trigger_predictions.py` | `app/streamlit_app.py` (Tabs 2-3) | `notebooks/final_notebook.ipynb` |
| 6 | `08_model_evaluation.ipynb` (Model 1) | `08_model_evaluation.ipynb` (Model 2) | `08_model_evaluation.ipynb` (Model 3) | `08_model_evaluation.ipynb` (compile) | Dashboard metrics footer | `presentation/slides.md` |
| 7 | Test `trigger_predictions.py` | Test `trigger_alerts.py` | `README.md` | `10_monitoring.ipynb` | Deploy dashboard (test) | `notebooks/final_notebook.ipynb` |
| 8 | Slides (Model 1) | Slides (Model 2) | Slides (Model 3) | Slides (Data + Methodology) | Dashboard polish | Assemble all slides |
| 9 | Full pipeline test | Triggers test with sample data | Verify README | Audit `constants.py` | Final deployment | Lead rehearsal |
| 10 | Present | Present | Present | Present | Present | Submit all |
