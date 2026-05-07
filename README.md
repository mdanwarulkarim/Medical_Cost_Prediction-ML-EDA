
# 🏥 Medical Cost Prediction
<img width="1024" height="1536" alt="Medical_Cost_Prediction" src="https://github.com/user-attachments/assets/114e84ee-7d08-4647-a807-74f3f68aa38a" />

> Predicting individual medical insurance charges using patient demographics, health indicators, and engineered interaction features — with full feature importance analysis across 7 regression models.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Key Findings](#-key-findings)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Feature Importance](#-feature-importance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Visualizations](#-visualizations)

---

## 🔍 Project Overview

This project performs end-to-end machine learning on the [Kaggle Insurance Dataset](https://www.kaggle.com/mirichoi0218/insurance) to predict individual medical costs billed by health insurance.

**Improvements over baseline:**

| Area | What Was Added |
|------|---------------|
| Feature Engineering | 6 new interaction & derived features |
| Feature Importance | RF importance + XGBoost gain + Permutation importance |
| Model Coverage | 7 models (Linear, Ridge, Lasso, DT, RF, GBM, XGBoost) |
| Evaluation | 5-fold cross-validation + residual diagnostics |
| Visualizations | 12 publication-quality plots with proper formatting |
| Code Quality | All deprecated `sns.distplot` calls replaced |

---

## 📊 Dataset

**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/mirichoi0218/insurance)

| Variable | Type | Description |
|----------|------|-------------|
| `age` | int | Age of primary beneficiary (18–64) |
| `sex` | str | Gender: male / female |
| `bmi` | float | Body Mass Index (kg/m²) |
| `children` | int | Number of children/dependents covered |
| `smoker` | str | Smoking status: yes / no |
| `region` | str | US residential region: NE / NW / SE / SW |
| `charges` | **float** | **Target** — Individual medical costs billed ($) |

- **Rows:** 1,338
- **Missing values:** None
- **Target range:** $1,122 – $63,770 (mean: $13,270)

---

## 📁 Project Structure

```
medical-cost-prediction/
│
├── Medical_Cost_Prediction_Improved.ipynb   # Main analysis notebook
├── insurance.csv                            # Dataset
├── README.md                                # This file
└── requirements.txt                         # Dependencies
```

---

## 💡 Key Findings

1. **Smoking is the #1 cost driver** — smokers pay **3–4× more** on average than non-smokers
2. **Smoker + Obese (BMI ≥ 30)** is the single strongest predictor when treated as a compound feature
3. **Age increases costs monotonically**, but the effect is amplified for smokers (accumulative damage)
4. **BMI ≥ 30** creates a distinct second cost tier even among non-smokers
5. **Linear models surprisingly competitive** — the relationship is largely additive after feature engineering
6. **Feature engineering improved R² by ~1–2%** over raw features alone

---

## ⚙️ Feature Engineering

Six new features were created based on domain knowledge and EDA insights:

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `bmi_category` | `cut(bmi, [0, 18.5, 25, 30, ∞])` | Captures non-linear BMI risk tiers |
| `age_group` | `cut(age, [0, 30, 45, ∞])` | Young / Middle / Senior risk segments |
| `smoker_obese` | `(smoker==yes) AND (bmi≥30)` | Compound high-risk flag |
| `smoker_age` | `smoker_binary × age` | Cumulative smoking damage proxy |
| `bmi_age` | `bmi × age` | Health deterioration interaction |
| `has_children` | `children > 0` | Binary coverage indicator |

---

## 📈 Models & Results

All models evaluated on 20% held-out test set + 5-fold cross-validation:

| Model | R² | RMSE ($) | MAE ($) | CV R² |
|-------|----|---------|---------|-------|
| Linear Regression | **0.8727** | 4,445 | 2,482 | ~0.86 |
| Ridge Regression | **0.8730** | 4,441 | 2,489 | ~0.86 |
| Lasso Regression | 0.8727 | 4,446 | 2,482 | ~0.86 |
| Gradient Boosting | 0.8677 | 4,532 | 2,483 | ~0.85 |
| Random Forest | 0.8638 | 4,598 | 2,374 | ~0.84 |
| XGBoost | 0.8561 | 4,726 | 2,617 | ~0.84 |
| Decision Tree | 0.8181 | 5,315 | 2,775 | ~0.80 |

> **Winner:** Ridge Regression (R² = 0.8730) — strong linear signal after feature engineering means linear models are competitive

---

## 🔑 Feature Importance

Three complementary importance methods were used to rank features:

| Rank | Feature | Consensus Importance |
|------|---------|---------------------|
| 1 | `smoker_obese` | ⭐⭐⭐⭐⭐ — Dominant predictor |
| 2 | `smoker` | ⭐⭐⭐⭐⭐ — 3× charge multiplier |
| 3 | `age` / `smoker_age` | ⭐⭐⭐⭐ — Aging + smoking compounding |
| 4 | `bmi` / `bmi_age` | ⭐⭐⭐ — Obesity amplifies with age |
| 5 | `children` | ⭐⭐ — More dependents, higher costs |
| 6 | `region` | ⭐ — Regional pricing variance |
| 7 | `sex` | ⭐ — Marginal after controlling for smoking |

**Methods Used:**
- 🌲 Random Forest built-in impurity importance
- 🚀 XGBoost gain-based feature importance
- 🔀 Permutation Importance (model-agnostic, 15 repeats)

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/medical-cost-prediction.git
cd medical-cost-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Medical_Cost_Prediction_Improved.ipynb
```

### Requirements

```
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
xgboost>=1.7
jupyter>=1.0
```

---

## 🚀 Usage

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load and engineer features
df = pd.read_csv('insurance.csv')
df['smoker_obese'] = ((df['smoker']=='yes') & (df['bmi']>=30)).astype(int)
df['smoker_age']   = (df['smoker']=='yes').astype(int) * df['age']
# ... (see notebook for full pipeline)

# Train
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
```

---

## 📸 Visualizations

The notebook generates 12 plots including:

- Distribution plots for all features
- Charges by smoking status, BMI category, age group
- Scatter: Age/BMI vs Charges coloured by smoker status
- Correlation heatmap (lower triangle)
- Actual vs Predicted for all 7 models
- Residual diagnostics (scatter, histogram, KDE)
- Feature importance bars (RF, XGBoost, Permutation)
- Consensus feature ranking table

---

## 🔮 Next Steps

- [ ] Hyperparameter tuning with `Optuna` / `GridSearchCV`
- [ ] SHAP value analysis for instance-level explanations
- [ ] Stacking ensemble (LR + RF + GBM)
- [ ] Log-transform `charges` target to handle right-skewed residuals
- [ ] Deploy as a web app using Streamlit or FastAPI

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements
 
- Dataset: [Miri Choi on Kaggle](https://www.kaggle.com/mirichoi0218/insurance)
- Original notebook inspiration from public Kaggle kernels
### Analysis By 
*Md Anwarul karim* <br>
*Masters in DataScience*
