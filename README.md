# Credit Origination Model & External Vendor Benchmarking
### Technical assessment — Data Scientist Jr. role at a financial institution

> **Note on data:** the datasets used are proprietary to the institution that issued the assessment and are not included in this repository. The notebook documents the full methodology and reproduces every result from those inputs.

> **Objective:** Build a credit scoring model for a new product with limited credit history, evaluate it on an out-of-time holdout, audit it with SHAP, and benchmark two external scoring vendors.

---

## Interactive cutoff calculator

A model returns a score; a credit policy returns a decision. The threshold is what connects them.
This panel walks every threshold over the out-of-time cohort and shows the operating consequences
of each one — approval rate, observed delinquency inside the approved population, recall and
precision — for both V1 and V2.

**→ [Open the calculator](https://juanhv24.github.io/credit-origination-model/)**

It also makes two findings visible that the metrics tables hide: V1's scores collapse into a
two-hundredths band, and a threshold computed on one model cannot be carried over to the other.

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
| V2 — Ethical (Gender excluded) | 0.6169 | 23.38% |

> A **1.96 pp GINI drop** from removing Gender — a modest performance cost for a significantly fairer credit decision system.

### Cutoff Analysis — Model V1

| Criterion | Threshold | Precision | Recall | F1 | Approval Rate |
|-----------|-----------|-----------|--------|----|---------------|
| Max F1-Score | 0.1306 | 11.7% | 71.4% | 0.201 | 44% |
| Youden J (risk-focused) | 0.1296 | 11.5% | 71.4% | 0.197 | 42.9% |

### Cutoff Analysis — Model V2

A threshold belongs to the score distribution it was estimated on. V1's cutoff (0.1296) falls below
V2's score range and rejects the entire population, so it was recalculated on V2's own validation
set. Both criteria converge on 0.4532 — which is a well-computed statistic and a poor policy:

| Threshold | Approval Rate | Recall | Delinquency among approved |
|-----------|---------------|--------|----------------------------|
| 0.4532 (statistical optimum) | 77.8% | 35.7% | 7.56% |
| 0.3350 (risk-focused) | 42.0% | 73.8% | 5.70% |

Baseline delinquency with no model is 9.15%. At its own optimum V2 lets two thirds of the
delinquent accounts through for a 1.6-point improvement. Youden's J implicitly prices a false
positive and a false negative equally, which origination does not: approving a defaulter costs
capital, rejecting a good applicant costs margin. The threshold is a risk-appetite decision, not a
statistical one — which is what the calculator above is for.

### Known limitations

- **V1 is effectively a single tree.** Early stopping fixed `best_iteration=1`: the selected
  hyperparameters (`num_leaves=86`, `max_depth=9`) overfit 3,084 records immediately, so boosting
  contributed nothing. V1's probabilities collapse into a two-hundredths band, which makes its
  cutoff extremely sensitive. Constraining the search space is the natural fix.
- **Small out-of-time cohort.** The test set holds 459 records with roughly 42 delinquent accounts,
  so one- or two-point differences in recall are sampling noise.
- **Structural signal ceiling.** After removing leakage, no individual feature correlates above 0.10
  with the target. Improving beyond this range requires richer data, not a better algorithm.
- **Proxy risk after removing gender.** SHAP places academic level fourth in importance; in the
  Colombian context it correlates with socioeconomic status and may act as a proxy. Quarterly
  monitoring of approval rate by education level is recommended.

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
