# VITA Tri-Board Layout Plan

## Center Panel (Main Story)
### Title
- **VITA: Lifestyle Risk Optimization Under Uncertainty**

### Big Visuals
- Dashboard screenshot (risk fan chart + health fan chart)
- Optimization tradeoff plot (risk vs time cost)

### Core Message (short text)
- Predict baseline risk from real data.
- Simulate many possible futures using Monte Carlo.
- Optimize habit plans under realistic time constraints.
- Adapt weights using weekly logs.

## Left Panel (Background + Methods)
### Problem & Motivation
- Lifestyle choices affect health risk over time.
- Real life is uncertain, so a single prediction is not enough.

### Dataset + Baseline ML (Layer 1)
- UCI Heart Disease dataset (or equivalent)
- Logistic regression + standardization
- Metrics: accuracy, ROC-AUC, confusion matrix

### Stochastic Model (Layer 2)
- Latent health state `H_t` in `[0,100]`
- Weekly update = drift + noise
- Noise increases with stress and sleep variability

## Right Panel (Results + Demo + Impact)
### Optimization
- Candidate plans generated under constraints
- Top 3 plans ranked by risk/time/adherence objective

### Personalization
- Weekly logs update sensitivity weights
- “Your model adapted because...” explanation

### Real-World Impact / Limitations
- Educational decision-support tool
- Not medical advice/diagnosis
- Future work: better datasets, wearables, longer calibration

## Table Demo Setup
- Laptop running Streamlit app
- Printed quick-start demo script (30-60 seconds)
- Backup screenshots and test checklist

