# Customer Churn Risk & Retention Intelligence System

A full-stack, enterprise-grade capstone analytics and machine learning solution that transforms raw invoice-level e-commerce transactions into a predictive churn-risk radar and prescriptive customer retention dashboard.

---

## 📌 Business Problem & Overview

In non-contractual e-commerce, customer churn is silent. Unlike subscription models with explicit cancellation events, retail customers simply stop placing orders. Traditional commercial teams often recognize churn only after months of inactivity, when win-back efforts are expensive and largely ineffective.

The **Customer Churn Risk & Retention Intelligence System** addresses this challenge by:
1. Transforming raw, multi-item transaction logs into clean, customer-level behavioral metrics (RFM+).
2. Implementing a **leakage-safe temporal snapshot** to forecast customer churn without lookahead bias.
3. Training a calibrated **Logistic Regression classifier** that prioritizes statistical **Recall (75.4%)** to intercept at-risk revenue early.
4. Providing business and CRM managers with an interactive **three-page Streamlit portal** to inspect portfolio health, explore customer risk profiles, and simulate prescriptive retention ROI.

---

## 📂 Dataset & Data Hygiene Rules

The system is trained on the UCI Online Retail dataset (`data.csv`), covering transactions from **December 01, 2010 to December 09, 2011** for a UK-based gifts and homeware retailer.

### Data Cleaning & Audit Pipeline
- **Missing CustomerID**: 135,080 records lacked customer identifiers and were filtered out.
- **Cancellations & Returns**: Invoices starting with `'C'` (9,288 records) were excluded.
- **Sanity Auditing**: Quantities $\le 0$ and Unit Prices $\le 0$ were purged.
- **Revenue Calculation**: Added `Revenue = Quantity * UnitPrice`.
- **Deduplication**: Redundant transaction records were removed.

| Quality Audit Metric | Recorded Value | Significance |
| :--- | :--- | :--- |
| **Original Rows** | 541,909 | Raw transaction line items |
| **Cleaned Rows** | 392,692 | Verified valid purchases |
| **Excluded Rows** | 149,217 (27.5%) | Filtered invalid / cancelled rows |
| **Verified Gross Revenue** | \$8,887,208.89 | Total verified spend across orders |
| **Unique Customers** | 4,338 | Identified retail accounts |
| **Date Span** | 2010-12-01 to 2011-12-09 | 374 days of transactional history |

---

## ⏱️ Leakage-Safe Churn Definition & Feature Engineering

To guarantee zero data leakage between feature extraction and target evaluation:
- **Maximum Date**: `2011-12-09 12:50:00`
- **Temporal Snapshot Date**: `2011-09-10 12:50:00` (90 days prior to dataset conclusion)
- **Historical Feature Window**: Transactions occurring strictly **before** `2011-09-10`.
- **Future Observation Window**: Transactions occurring **on or after** `2011-09-10` (90 calendar days).
- **Target Label (Churn)**:
  - `Churn = 1`: Customer made **zero** purchases during the 90-day observation window.
  - `Churn = 0`: Customer made **at least one** purchase during the 90-day observation window.
- **Cohort Size**: 3,370 active historical customers with a baseline observed churn rate of **43.0%**.

### Customer Features Engineered:
- **Recency ($R$)**: Days between the customer's most recent order and the snapshot date.
- **Frequency ($F$)**: Total count of unique completed invoices.
- **Monetary ($M$)**: Total gross customer spend prior to the snapshot date.
- **Average Order Value (AOV)**: Monetary spend divided by frequency.
- **Average Items per Order**: Total units divided by frequency.
- **Customer Tenure**: Days between customer's first purchase and the snapshot date.
- **Geographic Group**: Country categorization (UK, Germany, France, EIRE, Spain, Netherlands, Other).

---

## 🤖 Machine Learning Modeling & Evaluation

A Scikit-Learn `Pipeline` combines continuous variable standardization (`StandardScaler`), one-hot encoding for categorical country features (`OneHotEncoder`), and a `LogisticRegression` classifier configured with **balanced class weighting**.

### Primary Metric: Recall
In retention economics, **False Negatives** (failing to detect a churning customer) are significantly more costly than **False Positives** (sending an automated email or offer to an active customer). The model is tuned to maximize **Recall**.

| Metric | Score | Commercial Meaning |
| :--- | :--- | :--- |
| **Recall (Primary)** | **75.41%** | Detects ~3 out of 4 churning customers |
| **ROC-AUC** | **0.7320** | Strong separation across discrimination thresholds |
| **Overall Accuracy** | **65.72%** | Balanced multi-class classification rate |
| **Precision** | **57.72%** | Positive predictive accuracy on flagged accounts |
| **F1-Score** | **0.6539** | Harmonic balance between precision and recall |

### Top Predictive Drivers (Coefficients & Odds Ratios):
1. **Recency ($\beta = +1.11$, Odds Ratio $= 3.03$)**: Strongly increases churn risk. Inactivity is the #1 driver of customer loss.
2. **Tenure ($\beta = -0.56$, Odds Ratio $= 0.57$)**: Longer-standing customers exhibit strong loyalty resilience.
3. **Frequency & Items per Order ($\beta < 0$)**: Higher repeat order volume acts as a strong protective factor against churn.

> **Disclaimer**: Churn probabilities are predictive estimates based on historical statistical correlations, not deterministic guarantees.

---

## 🎯 Risk Segmentation & Prescriptive Playbooks

| Risk Tier | Churn Probability | Cohort Size | Recommended Strategic Playbook | Primary Channel |
| :--- | :---: | :---: | :--- | :--- |
| **High Risk** | $\ge 70\%$ | 558 accounts | Immediate 25% win-back voucher, dedicated VIP outreach, cart re-engagement | Direct Phone / VIP Concierge / SMS |
| **Medium Risk** | $40\% \text{ to } < 70\%$ | 1,756 accounts | Double loyalty points, automated category recommendations, re-engagement drip | Automated Email Drip / Push Notification |
| **Low Risk** | $< 40\%$ | 1,056 accounts | Ongoing retention nurture, cross-sell adjacent categories, VIP rewards tier | Product Newsletter / In-App Reward Banner |

---

## 🖥️ Streamlit Multi-Page Application Architecture

The application (`app.py`) delivers three distinct operational portals:

### 1. Executive Overview
- **Executive KPI Cards**: Total active customers (4,338), gross revenue ($8.89M), completed orders (18,532), observed baseline churn rate (43.0%), high-risk customer count (558), and average order value ($479.56).
- **Interactive Trajectory Charts**: Monthly revenue and order volume trends with Plotly tooltips.
- **Geographic & Risk Breakdown**: Top customer markets and donut charts for churn risk tiers.
- **Automated Insights**: Dynamic revenue-at-risk analysis and primary churn drivers.

### 2. Customer Risk Analytics
- **Searchable & Filterable Table**: Filter by country, risk tier, or search by `CustomerID`. Sorted by high risk first.
- **Analytical Charts**: Churn probability histogram with 40% and 70% threshold cutoffs, and RFM comparative box plots.
- **Customer 360 Drill-Down**: Instant historical inspection, RFM metrics, and tailored retention recommendation for any selected account.
- **CSV Cohort Export**: One-click download for CRM and marketing automation platforms.

### 3. Prescriptive Action Portal
- **Individual Account Diagnostician**: Root-cause behavioral breakdown (inactivity gap, frequency decay, basket size decline).
- **Tailored Win-Back Blueprints**: Ready-to-use message templates, discount suggestions, and contact SLAs.
- **Campaign Planning Matrix**: High-, medium-, and low-risk multi-channel engagement blueprint.
- **Interactive Revenue-at-Risk Simulator**: Real-time slider to model win-back conversion rates, gross revenue saved, and net marketing ROI.

---

## 🚀 Installation & Quick Start

### Prerequisites
- Python 3.9+ (Python 3.10 to 3.14 fully supported).

### Setup Commands
```bash
# 1. Clone or navigate to the project directory
cd customer_churn_intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```

*Note: If running in a virtual environment, activate the virtual environment or run:*
```bash
D:\venv\Scripts\streamlit.exe run app.py
```

The application will launch locally at `http://localhost:8501`.

---

## 📑 Project Deliverables

- `app.py`: Complete multi-page Streamlit application.
- `ml_pipeline.py`: Modular data ingestion, feature engineering, and model training engine.
- `requirements.txt`: Pinned production dependencies.
- `Project_Report.docx`: Professional Word capstone report with executive typography and tables.
- `Project_Report.pdf`: Publication-ready PDF report.
- `README.md`: Project documentation and architecture guide.

---

## ⚠️ Academic & Decision-Support Disclaimer

This system is developed for academic, demonstration, and executive decision-support purposes. Predicted churn probabilities represent statistical likelihoods derived from historical transaction patterns. They should be utilized alongside human managerial judgment and customer success domain expertise.
