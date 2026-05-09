**HIV Care Gap AI — Kenya**

County Risk Mapping · Individual Dropout Prediction · 2030 Scenario Forecasting

Using Machine Learning to Predict Who Kenya is Leaving Behind


## **1. Business Understanding:**

Over the years, Kenya's HIV response has made significant national progress yet critical gaps are widening.
New HIV infections rose 19% in 2024 (from 16,752 to 19,991), reversing years of decline.
Just 10 counties account for 60% of all new infections, yet no AI tool currently ranks all
47 counties by care gap urgency using raw programme data.

This project gives the Ministry of Health three tools it has never had before:
- Model 1 — County Gap Map:Where is the health system losing patients right now?
- Model 2 — Dropout Risk Calculator: Who is about to interrupt treatment?
- Model 3 — 2030 Forecast: What happens if we act — or if we don't?

#### Why This Matters: 

Treatment Interruption (IIT) — missing an ART(Antiretrovirals) visit by 28+ days is one of the strongest predictors of viral rebound and onward transmission. However, it is not currently modelled predictively at county level in Kenya.

Therefore, no existing system shows what Kenya's IIT and VLS(Viral Load Suppression) trends will look like in 2030 under different intervention scenarios

#### **Stakeholders:How they Utilise the Project**

- **Ministry of Health (MOH) Kenya:** National strategic planning and resource allocation
- **County Directors of Health:** Subnational intervention prioritisation
- **NSDCC / NASCOP:** National HIV programme managers
- **Community Health Workers:** Patient-level follow-up prioritisation
- **UNAIDS / World Bank:** Evidence-based investment prioritisation

#### **Three Models: Three Outputs**

Model 1: County Care Gap Map

Algorithm: Care Gap Index (engineered score) + KMeans Clustering (k=4)
Data: NSDCC raw programme data — IIT, VLT, Adult on HTS, Adult on ART
Output: All 47 counties ranked by Care Gap Index and assigned a tier:

🔴 Critical — highest IIT rate, lowest VLS rate: immediate intervention needed
🟠 High — above-average care gaps: priority follow-up
🟡 Moderate — programme gaps present but manageable
🟢 Low — relatively strong programme performance

Who uses it: MOH planners and County Directors of Health


Model 2: Individual Dropout Risk Score — Who?

Algorithm: Logistic Regression (baseline) + XGBoost Classifier (primary)
Data: Kenya DHS 2022 Individual Recode — 32,156 individual records, 15 features
Output: Probability score (0–1) that a person will interrupt HIV treatment,
plus the top 3 demographic/behavioural drivers of that risk
Who uses it: Community health workers and clinic staff — to prioritise which
patients need follow-up before they miss their next ART visit


Model 3: 2030 Dual Scenario Forecast — What next?

Algorithm: Facebook Prophet — run per county tier, per scenario
Data: NSDCC IIT and VLS rates per county, 2020–2025 time series
Output:

Scenario A (BAU): IIT and VLS trends to 2030 if nothing changes
Scenario B (Bridged Gap): Projected improvement if IIT is reduced 30%
in Critical and High tier counties by 2026


Who uses it: MOH policy makers and development partners — to quantify the
programme impact of investing in retention


### **Data-Driven Recommendations:**

Three actionable recommendations will be produced for MOH, one per model:

- Recommendation 1 (Model 1): Immediately prioritise retention interventions in
Critical tier counties — specifically those with IIT rates above the national average
and VLS rates below 90%. These counties represent the highest return-on-investment
targets for reducing new infections.

- Recommendation 2 (Model 2): Deploy differentiated follow-up protocols targeting
the demographic profiles flagged as highest dropout risk — specifically individuals
in wealth quintiles 1–2, aged 15–34, living more than 10km from a health facility.

- Recommendation 3 (Model 3): Closing the IIT gap in Critical and High tier counties
by 2026 is projected to substantially improve VLS rates nationally by 2030 — providing
a quantified evidence base for donor investment in county-level retention programmes.


# Existing Work This Builds On

| Reference | Contribution of This Project |
|-----------|------------------------------|
| Johns Hopkins / CDC HIV dropout models for Uganda | Applies dropout-risk modelling to Kenya for the first time using raw county-level programme data |

| UNAIDS Spectrum & Naomi national estimates | Adds a county-level machine learning intelligence layer on top of national estimates |

| South Africa THEMBISA county model | Kenya currently has no equivalent county-level predictive HIV care-gap system |

| UNAIDS Prevention 2025 Roadmap | Operationalises precision-prevention AI for priority geographic targeting |

**What is novel:** No existing Kenya HIV tool combines raw NSDCC programme data processing, individual dropout prediction, and cluster-based 2030 scenario forecasting in one integrated system.


# **2. DATA UNDERSTANDING:**

#### **Datasets:**

Two approved, confirmed datasets are used. No additional data collection is required.

#### **Dataset 1 — NSDCC Raw Programme Data (4 files)**

Source: analytics.nsdcc.go.ke/estimates

HIV Estimates 2025 Raw Data section. No login required.

| Filename | What it Contains | Size |

| IIT.xlsx | Interruption in Treatment counts by county and period | ~47 counties × 6 periods |
| VLT.xlsx | Viral Load Testing: tested and suppressed counts by county | ~47 counties × 6 periods |
| Adult_on_HTS.xlsx | HIV Testing Services: tested and positive counts by county, age, sex | ~47 counties × 6 periods |
| Adult_on_ART.xlsx | Adults on Antiretroviral Therapy by county and period | ~47 counties × 6 periods |

# Dataset 2 — DHS Kenya 2022 Individual Recode

**Source:** dhsprogram.com  
**Status:** ✅ Approved, downloaded, and reduced.

Processed dataset:

```text
data/processed/individual_features.csv
```

- **Rows:** 32,156 individuals
- **Columns:** 15 renamed features


## DHS Feature Dictionary

| Feature Column | DHS Code | Description |

| `county` | `v024` | County identifier (1–47 mapped to county names) |
| `age_group` | `v013` | Age group (5-year bands, 15–49) |
| `education_level` | `v106` | Highest education level |
| `wealth_index` | `v190` | Wealth quintile (1 = Poorest, 5 = Richest) |
| `distance_to_facility` | `v826a` | Distance to nearest health facility (km) |
| `ever_tested_hiv` | `v781` | Ever tested for HIV (binary) |
| `tested_hiv_last_12months` | `v828` | Tested for HIV in the last 12 months |
| `told_hiv_positive` | `v763a` | Told HIV positive |
| `marital_status` | `v501` | Marital status |
| `num_sexual_partners` | `v766b` | Lifetime number of sexual partners |
| `has_health_insurance` | `v481` | Has health insurance |
| `urban_rural` | `v025` | Urban or rural residence |
| `anc_visits` | `m14_1` | Antenatal care visits |
| `sex` | `v012` | Sex |
| `hiv_status` | `v781` | HIV status |



## Engineered Target Variable

```python
dropout = 1 if told_hiv_positive == 1 and tested_hiv_last_12months == 0
dropout = 0 otherwise
```


# Data Quality Challenges

| Challenge | Where It Appears | Mitigation Strategy |

| MOH indicator codes as column headers | All 4 NSDCC files | Parsed and renamed in notebook `02_preprocessing.ipynb` |

| County name inconsistencies | All 4 NSDCC files | Standardised using `COUNTY_NAME_MAP` in `constants.py` |

| Missing programme data | IIT and VLT datasets | Imputed using county-period averages |
| DHS numeric recode mappings | DHS datasets | Mapped using `DHS_COUNTY_MAP`, `DHS_AGE_GROUP_MAP`, etc. |

| Class imbalance in dropout target | DHS datasets | Addressed using `scale_pos_weight` in XGBoost |

| Wide-format county-period exports | All NSDCC datasets | Reshaped using `pandas.melt()` |



# Dataset Sizes

| Dataset | Records | Columns |

| `IIT.xlsx` | ~282 county-period rows | MOH indicator columns |
| `VLT.xlsx` | ~282 county-period rows | MOH indicator columns |
| `Adult_on_HTS.xlsx` | ~282 county-period rows | MOH indicator columns |
| `Adult_on_ART.xlsx` | ~282 county-period rows | MOH indicator columns |
| DHS Kenya 2022 (raw) | 32,156 individuals | 5,925 columns |
| `individual_features.csv` | 32,156 individuals | 15 columns |



# Planned Visualisations

| Model | Visualisation |

| Model 1 | Ranked horizontal bar chart of all 47 counties by Care Gap Index |
| Model 1 | Stacked HIV programme bar chart: IIT rate, VLS rate, ART coverage |
| Model 1 (Stretch) | Folium choropleth map of Kenya counties coloured Critical → Low |
| Model 2 | XGBoost feature importance chart (top dropout predictors) |
| Model 2 | Risk score distribution histogram by county tier |
| Model 3 | Four dual-scenario line charts: BAU vs Bridged Gap (2025–2030) |
| Model 3 | National headline projection chart for IIT and VLS trends |