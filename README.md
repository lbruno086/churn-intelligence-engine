# Churn Intelligence Engine

> Production-grade customer attrition prediction system using gradient boosting, deployed across multiple geographies. Includes model monitoring, explainability, and automated retraining triggers.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-red)](https://shap.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-tracked-blue)](https://mlflow.org)

---

## Business Problem

Customer churn is one of the highest-cost problems in financial services and telecom. Acquiring a new customer costs 5–7x more than retaining an existing one. The challenge: identify at-risk customers **before** they churn, with enough lead time to act — and do it with a model that generalizes across markets with different behavioral profiles.

**Context:** Built and deployed for multi-country financial services operations, including markets in Spain, Colombia, Argentina, and Paraguay.

---

## Key Results

| Geography | Metric | Result |
|-----------|--------|--------|
| Spain | Model performance lift | **+30%** vs. previous model |
| Multi-market | First-to-market KS score | **KS = 52** |
| Colombia | Survival Score | **+20%** performance improvement |
| LATAM | Deployment footprint | **8 countries** |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                            │
│  CRM · Transaction history · Product usage · Support logs  │
│  Bureau data · Digital behavior · Payment patterns          │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                   FEATURE ENGINEERING                        │
│                                                              │
│  Recency-Frequency-Monetary (RFM) features                  │
│  Product penetration & cross-sell depth                     │
│  Engagement decay (exponential smoothing)                   │
│  Delinquency trajectory (0-30-60-90 day flags)              │
│  Seasonality-adjusted tenure features                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────▼──┐    ┌────────▼───┐   ┌────────▼──────┐
│  Churn     │    │  Survival  │   │  Segment-level│
│  Score     │    │  Score     │   │  Propensity   │
│ LightGBM   │    │  Cox PH /  │   │  XGBoost      │
│            │    │  XGBoost   │   │               │
└─────────┬──┘    └────────┬───┘   └────────┬──────┘
          └────────────────┼────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│              SCORING ENGINE & ACTION LAYER                   │
│  Risk tiers · Retention offer assignment · CRM integration  │
│  Champion/Challenger · Drift monitoring (PSI/CSI)           │
└──────────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology |
|-------|-----------|
| Modeling | LightGBM, XGBoost, Cox Proportional Hazard |
| Explainability | SHAP (TreeExplainer) |
| Experiment tracking | MLflow |
| Feature engineering | pandas, feature-engine, tsfresh |
| Monitoring | PSI, CSI, performance drift alerts |
| Big Data | Hadoop (Hive, Impala), PySpark |
| Serving | Batch scoring via AWS Glue + S3 |

---

## Repository Structure

```
churn-intelligence-engine/
├── data/
│   └── sample/                     # Telco Customer Churn (public)
├── notebooks/
│   ├── 01_EDA_churn_patterns.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_churn_model_lightgbm.ipynb
│   ├── 04_survival_analysis.ipynb
│   ├── 05_shap_explainability.ipynb
│   └── 06_model_monitoring.ipynb
├── src/
│   ├── features/
│   │   ├── rfm_features.py
│   │   ├── engagement_decay.py
│   │   └── delinquency_trajectory.py
│   ├── models/
│   │   ├── churn_classifier.py
│   │   ├── survival_model.py
│   │   └── segment_propensity.py
│   ├── monitoring/
│   │   ├── psi_monitor.py
│   │   ├── performance_tracker.py
│   │   └── retraining_trigger.py
│   └── explainability/
│       └── shap_report.py
├── configs/
│   ├── model_config.yaml
│   └── geography_configs/
│       ├── spain.yaml
│       ├── colombia.yaml
│       └── argentina.yaml
├── tests/
├── docs/
│   ├── model_card.md
│   └── multi_geography_adaptation.md
├── requirements.txt
└── README.md
```

---

## Feature Engineering Deep Dive

### Engagement Decay Features

```python
def compute_engagement_decay(df: pd.DataFrame, halflife_days: int = 30) -> pd.DataFrame:
    """
    Exponentially weighted activity score — recent interactions matter more.
    Captures disengagement trajectory before formal churn event.
    """
    alpha = 1 - np.exp(-np.log(2) / halflife_days)
    df["engagement_score"] = (
        df.groupby("customer_id")["activity_flag"]
        .transform(lambda x: x.ewm(alpha=alpha).mean())
    )
    return df
```

### Delinquency Trajectory

```python
# Rolling delinquency flags capture worsening patterns
for days in [30, 60, 90]:
    df[f"ever_delinquent_{days}d"] = (
        df.groupby("customer_id")[f"days_past_due"]
        .transform(lambda x: (x >= days).rolling(6).max())
    )
```

---

## Multi-Geography Adaptation

A key challenge was making the same model work across markets with structurally different customer behaviors (Spain vs. Colombia vs. Paraguay). The solution:

1. **Base model** trained on pooled data with market indicators
2. **Market-specific** calibration layer (isotonic regression per country)
3. **Local feature** augmentation: bureau integration varies by country
4. **Threshold tuning** per market based on local retention economics

```yaml
# configs/geography_configs/spain.yaml
churn_threshold: 0.35
retention_offer_min_score: 0.40
psi_alert_threshold: 0.15
feature_set: ["rfm", "digital", "payment_behavior"]
bureau_source: "ASNEF"
```

---

## Model Validation

```python
# KS statistic — key metric for credit/churn scoring
def compute_ks(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score", ascending=False)
    df["cumulative_bad"]  = df["y"].cumsum() / df["y"].sum()
    df["cumulative_good"] = (1 - df["y"]).cumsum() / (1 - df["y"]).sum()
    return (df["cumulative_bad"] - df["cumulative_good"]).abs().max()

# Target: KS > 40 for production deployment
```

---

## SHAP Explainability Output

```
Top features driving churn probability (SHAP):
┌────────────────────────────────────┬──────────────┐
│ Feature                            │ Mean |SHAP|  │
├────────────────────────────────────┼──────────────┤
│ days_since_last_transaction        │    0.142     │
│ engagement_score_30d               │    0.118     │
│ product_penetration_depth          │    0.097     │
│ delinquency_ever_60d               │    0.089     │
│ payment_consistency_score          │    0.076     │
└────────────────────────────────────┴──────────────┘
```

---

## Data

Public reference: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)  
Financial churn: [Bank Customer Churn](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

---

## How to Run

```bash
git clone https://github.com/lbruno086/churn-intelligence-engine
cd churn-intelligence-engine
pip install -r requirements.txt

# Run full pipeline
python src/models/churn_classifier.py --config configs/model_config.yaml

# Generate SHAP report
python src/explainability/shap_report.py --model_path models/churn_lgbm.pkl

# Monitor drift
python src/monitoring/psi_monitor.py --reference data/train_scores.parquet --current data/recent_scores.parquet
```

---

## Interview Talking Points

**"Walk me through a churn model you built at scale."**

> "At ERGO, I was responsible for deploying ML models for Prosegur across 8 countries. The churn model for Spain was underperforming, so I rebuilt it from the ground up — re-engineering features around engagement decay, payment trajectory, and product utilization patterns. I moved from a logistic regression to LightGBM with proper calibration, and performance improved by 30%. For a separate client, I built the first market-grade churn score from scratch, achieving a KS of 52 — which became the benchmark for the market. The key was making the model multi-geography-ready: same architecture, country-specific calibration and thresholds."

---

## LinkedIn Description

> Designed and deployed multi-country churn prediction systems for financial services clients across 8 countries. Achieved +30% performance improvement on Spain's churn model via gradient boosting (LightGBM, XGBoost) and feature re-engineering. Developed first-to-market churn score with KS=52. Implemented survival analysis for long-horizon attrition risk. Full ownership from feature design to regulatory documentation and business stakeholder presentation.
