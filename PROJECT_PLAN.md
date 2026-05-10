# HIV Care Gap AI: Kenya - Project Plan  

> **Team:** Daniella · Eve · Verah · Naomi · Lorenah · Dennis  
> **Timeline:** 10 working days  
> **Methodology:** CRISP-DM  
> **Status:** Day 1 starts with `01_data_extraction.ipynb`

---

# Current State (As of Day 1 Morning)

| Item | Status | Owner |
|------|--------|-------|
| `01_data_extraction.ipynb` | COMPLETE | Naomi |
| Raw data files in `data/raw/` | All 5 files present | Daniella |
| `constants.py` | On main | Naomi |
| Folder structure | Created | Daniella |
| `src/` utilities | Needs population | Team |

---

# The Golden Rules

1. **One person, one file per day.** No two people edit the same file on the same day.
2. **`constants.py` is the single source of truth.** Never hard-code anything.
3. **Notebooks for exploration. `src/` for reusable logic.**
4. **If you write a notebook, you write/update the matching `src/` module.**
5. **Branch → Work → Pull Request → Merge → Pull.** Never commit directly to `main`.
6. **Data never goes to GitHub.** `data/` folder is in `.gitignore`.

---

# Repository Structure

## Project Structure

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
├── notebooks/                      ← 10 notebooks total
│   ├── 01_data_extraction.ipynb               COMPLETE (Naomi)
│   ├── 02_nsdcc_cleaning.ipynb                Needs NSDCC + DHS cells
│   ├── 03_dhs_cleaning.ipynb                  Needs completion
│   ├── 04_feature_engineering.ipynb           Needs completion
│   ├── 05_model_1_county_clustering.ipynb
│   ├── 06_model_2_dropout_prediction.ipynb
│   ├── 07_model_3_forecasting.ipynb
│   ├── 08_model_evaluation.ipynb
│   ├── 09_deployment.ipynb
│   └── 10_monitoring.ipynb
│
├── src/                            ← Python modules (NOT scripts/)
│   ├── __init__.py
│   ├── utils.py                              ← Shared helpers
│   ├── data_preprocessing.py                 ← Load + clean functions
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
├── figures/                        ← Logo.jpg + exported charts
├── presentation/                   ← Slides
├── tableau/                        ← Tableau dashboard files
│
├── constants.py                    ← On main
├── main.py                         ← To be created
├── requirements.txt
├── .gitignore
└── PROJECT_PLAN.md
```

---

# 📅 DAY-BY-DAY PLAN (10 Days)

---

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

# Parallel Flow Summary

## Days 1-2: Maximum Parallelism (Everyone Works)

| Person | Day 1 | Day 2 |
|--------|-------|-------|
| Eve | `02_nsdcc_cleaning.ipynb` | `scripts/merge_data.py` |
| Lorenah | `03_dhs_cleaning.ipynb` | `06_model_2_dropout_prediction.ipynb` (Logistic Regression) |
| Verah | `src/data_preprocessing.py` | `04_feature_engineering.ipynb` |
| Naomi | Update `constants.py` | `src/feature_engineering.py` |
| Dennis | `src/utils.py` | `src/evaluation.py` |
| Daniella | `main.py` (skeleton) | `main.py` (wire Days 1-2) |

> **ALL 6 WORKING IN PARALLEL — NO ONE WAITING**

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
