# Credit Risk Analysis — End-to-End PD Modeling

This project presents an end-to-end **credit risk analysis pipeline**, from data preparation
and exploratory analysis to **Probability of Default (PD)** modeling and bank-style decision rules.

The objective is to demonstrate how credit decisions are made in realistic lending environments.

---

## Project Highlights

- Realistic **Probability of Default (PD)** modeling
- Bank-style decisions: **APPROVE / REVIEW / REJECT**
- Robust preprocessing and feature engineering
- Statistical validation (Spearman, Kruskal–Wallis, ANOVA)
- Loan application **simulation (example-based)**
- Simple, explainable decision logic

---

## Methodology Overview

### 1. Data Preparation
- Final loan outcomes only (Fully Paid vs Default / Charged Off)
- No aggressive outlier removal (to preserve real credit behavior)

### 2. Exploratory Data Analysis
- One-page EDA dashboard
- Risk segmentation by grade, DTI, utilization, and inquiries

### 3. Modeling
- Logistic Regression (baseline)
- HistGradientBoosting (non-linear benchmark)
- Evaluation using ROC-AUC and Average Precision

### 4. Decision Policy
- Cost-sensitive thresholds
- Multi-tier decisions: **Approve / Review / Reject**

### 5. Simulation
- Example-based loan application simulation
- End-to-end inference using the trained pipeline
- PD translated into a bank-style decision

---

## Key Design Choices

- Outliers are retained because extreme values are informative in credit risk
- Tree-based models are used for robustness to non-normal distributions
- A **review bucket** is included to reflect real-world underwriting uncertainty

---

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Credit_Risk_Project.ipynb


