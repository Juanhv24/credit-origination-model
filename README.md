# Credit Origination Model & External Vendor Benchmarking
### Technical assessment — Data Scientist Jr. role at a financial institution

> **Note on data:** the datasets used are proprietary to the institution that issued the assessment and are not included in this repository. The notebook documents the full methodology and reproduces every result from those inputs.

> **Objective:** Build a credit scoring model for a new product with limited credit history, evaluate it on an out-of-time holdout, audit it with SHAP, and benchmark two external scoring vendors.

---

## Project Overview

This project covers the full credit risk modeling pipeline for a consumer lending product — from raw data exploration through model deployment decisions. It includes an ethical AI analysis comparing a full-feature model against a gender-blind version, and a rigorous external vendor comparison using bootstrap confidence intervals.

| Component | Detail |
|-----------|--------|
| **Model** | LightGBM (gradient boosting) |
| **Hyperparameter optimization** | Optuna — Bayesian optimization, 30 trials |
| **Validation strategy** | Out-of-Time (OOT) — 2018 holdout cohort |
| **Target variable** | Mora60 (60-day delinquency) |
| **Class imbalance** | 7.6:1 (good:bad) — handled via `scale_pos_weight` |
| **Explainability** | SHAP beeswarm + waterfall analysis |

---

## Results

### Model Performance (OOT Test — 2018 Cohort)

| Model | AUC-ROC | GINI |
|-------|---------|------|
| V1 — Full features (incl. Gender) | 0.6267 | 25.34% |
| V2 — Ethical (Gender excluded) | 0.617 | 23.4% |

> A **1.94 pp GINI drop** from removing Gender — a modest performance cost for a significantly fairer credit decision system.

### Cutoff Analysis

| Criterion | Threshold | Precision | Recall | F1 | Approval Rate |
|-----------|-----------|-----------|--------|----|---------------|
| Max F1-Score | 0.1306 | 11.7% | 71.4% | 0.201 | 44% |
| Youden J (risk-focused) | 0.1296 | 11.5% | 71.4% | 0.197 | 42.9% |

### External Vendor Benchmarking

| Vendor | GINI | KS Statistic | Coverage |
|--------|------|-------------|----------|
| Proveedor AB | −0.26% | 0.0052 | 100% |
| Proveedor XY | +0.35% | 0.0051 | 88.78% |

> Neither vendor shows statistically significant discriminatory power (GINI ≈ 0%), suggesting the internal model provides more reliable scoring for this portfolio.

---

## Pipeline Architecture

```
Raw Data
   │
   ├── Temporal Stability Analysis (monthly default rate)
   ├── Leakage Detection → removed MoraMax_UltimoSemestre (r=0.79) & Mora30 (r=0.64)
   │
   ├── Train / Validation / OOT Split (temporal)
   │
   └── sklearn Pipeline
         ├── Numeric: median imputation + RobustScaler
         ├── Categorical: mode imputation + OneHotEncoding
         ├── Engineered features: Ratio_CargaFija, Ingreso_PerCapita, Velocidad_Endeudamiento
         └── LightGBM (Optuna-tuned)
```

---

## Key Technical Decisions

**Why LightGBM?**  
Handles mixed numeric/categorical features natively, efficient on moderately-sized datasets, and supports `scale_pos_weight` for class imbalance without resampling artifacts.

**Why OOT validation?**  
Random splits inflate performance on credit data due to temporal correlation. Holding out the 2018 cohort strictly simulates real deployment conditions.

**Why remove Mora30 and MoraMax_UltimoSemestre?**  
Both are proxies of the target computed over periods that overlap with the label window — classic data leakage that would produce inflated but non-generalizable metrics.

**Why build V2 without Gender?**  
Regulatory trends in Latin American fintech increasingly restrict the use of protected attributes in credit decisions. V2 demonstrates the operational feasibility of a compliant model with minimal performance loss.

---

## Files

| File | Description |
|------|-------------|
| `Notebooks/ProductoNuevo.ipynb` | Full pipeline: EDA → feature engineering → training → OOT evaluation → SHAP → vendor benchmarking |
| `requirements.txt` | Pinned dependencies |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-EC6F1A?style=flat)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Optimization-2dd4bf?style=flat)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-4f8ef7?style=flat)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Bootstrap%20CI-8CAAE6?style=flat&logo=scipy&logoColor=white)
