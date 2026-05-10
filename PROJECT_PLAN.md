# HIV Care Gap AI: Kenya — Project Plan 
**County Risk Mapping · Individual Dropout Prediction · 2030 Scenario Forecasting**

> **Team:** Daniella (Lead) · Eve · Verah · Naomi · Lorenah · Dennis  
> **Timeline:** 10 working days · **Methodology:** CRISP-DM  
> **Status:** Day 1 starts with `01_data_extraction.ipynb` ✅ COMPLETE

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

## The Golden Rules

1. **One person, one file per day.** No two people edit the same file on the same day.
2. **`constants.py` is the single source of truth.** Never hard-code anything.
3. **Notebooks for exploration. `src/` for reusable logic.**
4. **If you write a notebook, you write/update the matching `src/` module.**
5. **Branch → Work → Pull Request → Merge → Pull.** Never commit directly to `main`.
6. **Data never goes to GitHub.** `data/` folder is in `.gitignore`.

---

## Repository Structure (Your Actual Layout)
hiv-care-gap-ai/
│
├── data/
│ ├── raw/ ← All 5 files present 
│ │ ├── Adult_on_ART.xlsx
│ │ ├── Adult_on_HTS.xlsx
│ │ ├── IIT.xlsx
│ │ ├── VLT.xlsx
│ │ └── individual_features.csv
│ └── processed/ ← Empty — will be populated
│
├── notebooks/ ← 10 notebooks total
│ ├── 01_data_extraction.ipynb  COMPLETE (Naomi)
│ ├── 02_nsdcc_cleaning.ipynb  Needs NSDCC + DHS cells
│ ├── 03_dhs_cleaning.ipynb  Needs completion
│ ├── 04_feature_engineering.ipynb ⚠️ Needs completion
│ ├── 05_model_1_county_clustering.ipynb
│ ├── 06_model_2_dropout_prediction.ipynb
│ ├── 07_model_3_forecasting.ipynb
│ ├── 08_model_evaluation.ipynb
│ ├── 09_deployment.ipynb
│ └── 10_monitoring.ipynb
│
├── src/ ← Python modules (NOT scripts/)
│ ├── init.py
│ ├── utils.py ← Shared helpers
│ ├── data_preprocessing.py ← Load + clean functions
│ ├── feature_engineering.py ← CGI + tier aggregation
│ ├── model_training.py ← KMeans + XGBoost + Prophet
│ ├── evaluation.py ← Metrics functions
│ └── forecasting.py ← Prophet scenario functions
│
├── scripts/ ← Production pipeline scripts
│ ├── extract_data.py ← Wrapper for data loading
│ ├── merge_data.py ← Merge NSDCC files
│ ├── prepare_data.py ← Feature engineering wrapper
│ ├── train_model1.py ← KMeans wrapper
│ ├── train_model2.py ← XGBoost wrapper
│ └── train_model3.py ← Prophet wrapper
│
├── app/
│ ├── streamlit_app.py
│ ├── trigger_alerts.py
│ └── trigger_predictions.py
│
├── models/ ← Empty — will hold .pkl files
├── figures/ ← logo.jpg + exported charts
├── presentation/ ← Slides
├── tableau/ ← Tableau dashboard files
├── constants.py ←  On main
├── main.py ←  To be created
├── requirements.txt
├── .gitignore
└── PROJECT_PLAN.md ← This file
---

# 📅 DAY-BY-DAY PLAN (10 Days)

## Day 1 — Data Cleaning: NSDCC + DHS (Parallel)

**Starting point:** `01_data_extraction.ipynb` is complete. Raw data is in place.

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Complete `02_nsdcc_cleaning.ipynb` (NSDCC cells 1-8): Load 4 NSDCC files, apply `COUNTY_NAME_MAP`, parse MOH codes, reshape wide→long, impute missing values, calculate IIT/VLS/HTS rates. | 4 clean CSVs in `data/processed/` |
| **Lorenah** | Complete `03_dhs_cleaning.ipynb`: Load `individual_features.csv`, apply DHS code maps from `constants.py`, engineer dropout target, one-hot encode categoricals. | `individual_features_clean.csv` |
| **Verah** | Update `src/data_preprocessing.py`: Add `load_nsdcc()`, `load_dhs()`, `clean_nsdcc()`, `clean_dhs()` functions. | `data_preprocessing.py` importable |
| **Naomi** | Review & document: Confirm DHS column names match `constants.py`. Update `XGB_SCALE_POS_WEIGHT` in `constants.py` based on Lorenah's class balance. | `constants.py` updated |
| **Dennis** | Create `src/utils.py`: Add `standardise_county()`, `get_tier_color()`, `save_csv()`, `load_csv()` helpers. | `utils.py` complete |
| **Daniella** | Standup, review PRs, merge Day 1 work. Create `main.py` skeleton with function stubs. | `main.py` started |

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

## Day 4 — Model 3 (Prophet) + Feature Engineering Completion

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create `scripts/train_model3.py` wrapper for Prophet. | `train_model3.py` |
| **Lorenah** | Create `scripts/train_model2.py` wrapper for XGBoost. | `train_model2.py` |
| **Verah** | Complete `07_model_3_forecasting.ipynb`: Add Scenario B (Bridged Gap), backtest, save forecasts. | Forecast charts, forecast CSVs |
| **Naomi** | Update `src/forecasting.py`: Add Prophet wrapper, scenario builder functions. | `forecasting.py` |
| **Dennis** | Begin `app/streamlit_app.py` Tab 1 (County Gap Map). | Tab 1 working locally |
| **Daniella** | Update `main.py` with Model 3 training. Review PRs. | `main.py` Day 4 running |

---

## Day 5 — Dashboard Tabs 2 & 3 + Trigger Scripts Start

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Review & help Dennis with dashboard data connections. | Support |
| **Lorenah** | Help Dennis with Tab 2 (risk calculator) integration. | Support |
| **Verah** | Create `app/trigger_alerts.py` — IIT alert system. | `trigger_alerts.py` |
| **Naomi** | Create `app/trigger_predictions.py` — annual retraining trigger. | `trigger_predictions.py` |
| **Dennis** | Complete Tab 2 and Tab 3 in dashboard. | Dashboard full |
| **Daniella** | Begin `notebooks/final_notebook.ipynb`. Review PRs. | Final notebook started |

---

## Day 6 — Evaluation Notebook + Slides Start

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Run `08_model_evaluation.ipynb` for Model 1 metrics. | Model 1 eval |
| **Lorenah** | Run `08_model_evaluation.ipynb` for Model 2 metrics. | Model 2 eval |
| **Verah** | Run `08_model_evaluation.ipynb` for Model 3 metrics. | Model 3 eval |
| **Naomi** | Complete `08_model_evaluation.ipynb` — compile all metrics into comparison table. | Evaluation notebook |
| **Dennis** | Add evaluation metrics to dashboard (optional footer). | Dashboard update |
| **Daniella** | Create `presentation/slides.md` outline. Assign slide owners. | Slides started |

---

## Day 7 — Final Notebook + Trigger Scripts Testing

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Test `trigger_predictions.py` with dry run. | Trigger tested |
| **Lorenah** | Test `trigger_alerts.py` with current data. | Alerts tested |
| **Verah** | Update `README.md` with setup and run instructions. | README complete |
| **Naomi** | Complete `10_monitoring.ipynb` — documentation for annual updates. | Monitoring notebook |
| **Dennis** | Deploy dashboard to Streamlit Cloud (test deployment). | Test URL |
| **Daniella** | Complete `final_notebook.ipynb` with all 3 models and recommendations. | Final notebook ready |

---

## Day 8 — Presentation Slides + Dashboard Polish

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Create slide content for Model 1 (2-3 slides). | Slides section |
| **Lorenah** | Create slide content for Model 2 (2-3 slides). | Slides section |
| **Verah** | Create slide content for Model 3 (2-3 slides). | Slides section |
| **Naomi** | Create slide content for Data + Methodology (2 slides). | Slides section |
| **Dennis** | Polish dashboard: add error handling, loading states, tooltips. | Dashboard final |
| **Daniella** | Assemble all slides into final deck. Review with team. | Slides complete |

---

## Day 9 — Rehearsal + Final Fixes

| Person | Task | Deliverable |
|--------|------|-------------|
| **Eve** | Run full pipeline on clean environment. Fix any bugs. | Pipeline verified |
| **Lorenah** | Test trigger scripts with sample new data. | Triggers verified |
| **Verah** | Review README, ensure all commands work. | README verified |
| **Naomi** | Review constants.py for any remaining hard-coded values. | constants.py final |
| **Dennis** | Final deployment to Streamlit Cloud (production URL). | Live dashboard |
| **Daniella** | Lead 1-hour full presentation rehearsal. | Team ready |

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

## Dependency Graph
DAY 1-2 (FULLY PARALLEL)
─────────────────────────────────────────────────────────────────
Eve ──────────┐
Lorenah ──────┼────► All produce independent outputs
Verah ────────┤         No one waits on anyone
Naomi ────────┤
Dennis ───────┤
Daniella ─────┘
│
▼
DAY 3-4 (THREE STREAMS)
─────────────────────────────────────────────────────────────────
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ STREAM A: Model 1 │ │ STREAM B: Model 2 │ │ STREAM C: Core │
│ Eve ──► scripts/ │ │ Lorenah ──► NB06 │ │ Dennis ──► src/ │
│ Naomi ──► NB05 │ │ Verah ──► NB07 │ │ Daniella ──► main │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
│
▼
DAY 5-10 (FULLY PARALLEL)
─────────────────────────────────────────────────────────────────
Eve ──────────┐
Lorenah ──────┤
Verah ────────┼────► No blocks — everyone independent
Naomi ────────┤
Dennis ───────┤
Daniella ─────┘

---

### Who Blocks Whom (The Only Dependencies)

| If you are... | You need this from... | By when | If delayed, do this instead |
|---------------|----------------------|---------|----------------------------|
| Dennis (Day 2 merge) | Eve's 4 clean NSDCC CSVs | End of Day 2 | Work on `src/evaluation.py` (doesn't need NSDCC data) |
| Naomi (Day 3 clustering) | Verah's `county_profiles.csv` | End of Day 2 | Start silhouette score exploration with dummy data |
| Verah (Day 4 Scenario B) | Naomi's tier labels | End of Day 3 | Complete BAU forecast first, add Scenario B later |
| Dennis (Day 5 dashboard Tabs 2-3) | All 3 `.pkl` files | End of Day 4 | Build Tab 1 first (needs only CSV, no models) |

**Net result:** Maximum wait time = 1 day. Most waits are under 4 hours.

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
| 1 | `02_nsdcc_cleaning.ipynb` | `03_dhs_cleaning.ipynb` | `src/data_preprocessing.py` | `constants.py` | `src/utils.py` | `main.py` |
| 2 | `scripts/merge_data.py` | `06_model_2_dropout_prediction.ipynb` (Logistic Regression) | `04_feature_engineering.ipynb` | `src/feature_engineering.py` | `src/evaluation.py` | `main.py` |
| 3 | `scripts/train_model1.py` | `06_model_2_dropout_prediction.ipynb` (XGBoost) | `07_model_3_forecasting.ipynb` (BAU) | `05_model_1_county_clustering.ipynb` | `src/model_training.py` | `main.py` |
| 4 | `scripts/train_model3.py` | `scripts/train_model2.py` | `07_model_3_forecasting.ipynb` (Scenario B) | `src/forecasting.py` | `app/streamlit_app.py` (Tab 1) | `main.py` |
| 5 | Support dashboard | Support dashboard Tab 2 | `app/trigger_alerts.py` | `app/trigger_predictions.py` | `app/streamlit_app.py` (Tabs 2-3) | `notebooks/final_notebook.ipynb` |
| 6 | `08_model_evaluation.ipynb` (Model 1) | `08_model_evaluation.ipynb` (Model 2) | `08_model_evaluation.ipynb` (Model 3) | `08_model_evaluation.ipynb` (compile) | Dashboard metrics footer | `presentation/slides.md` |
| 7 | Test `trigger_predictions.py` | Test `trigger_alerts.py` | `README.md` | `10_monitoring.ipynb` | Deploy dashboard (test) | `notebooks/final_notebook.ipynb` |
| 8 | Slides (Model 1) | Slides (Model 2) | Slides (Model 3) | Slides (Data + Methodology) | Dashboard polish | Assemble all slides |
| 9 | Full pipeline test | Triggers test with sample data | Verify README | Audit `constants.py` | Final deployment | Lead rehearsal |
| 10 | Present | Present | Present | Present | Present | Submit all |