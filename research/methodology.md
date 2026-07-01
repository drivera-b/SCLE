# SLCE Research Methodology

## Research question

How can a transparent decision-support system combine a population risk prior, personal lifestyle observations, stochastic uncertainty, and feasibility constraints to compare lifestyle scenarios?

## Data architecture

SLCE deliberately separates two public datasets instead of combining incompatible records:

1. UCI Cleveland Heart Disease supplies the supervised binary target for the baseline logistic-regression model.
2. NHANES 2017-2018 supplies population-reference distributions for sleep, activity, blood pressure, BMI, cholesterol, HbA1c, and fasting glucose.

The NHANES build pipeline downloads official CDC XPT modules, merges them on `SEQN`, derives comparable lifestyle fields, retains survey-design columns, and saves a reproducible CSV extract.

In-app biomarker percentiles use the NHANES examination weights. Strata and PSU values remain available for future design-based variance estimation.

## Feature provenance

Every baseline prediction distinguishes among:

- Observed inputs: directly supplied demographics or measured biomarkers.
- Proxy-derived inputs: estimated from available lifestyle inputs.
- Median-imputed inputs: filled from the training dataset.

The Input Evidence Score summarizes data coverage and population applicability. It is not a probability, confidence interval, or medical certainty score.

## Baseline model

The baseline is a standardized logistic regression with median imputation. Evaluation uses a stratified train/test split and reports accuracy, ROC-AUC, and confusion matrix.

Measured systolic blood pressure, total cholesterol, and fasting glucose map to existing UCI features. Supplemental biomarkers such as HbA1c and BMI are not inserted into the classifier unless a future model is explicitly trained with those features.

The optional one-row lab CSV uses explicit standard units and rejects unsupported units, nonnumeric values, out-of-range values, and multiple-person files. For transparency, the dashboard recomputes the same profile with measured labs disabled and displays the probability difference caused by replacing proxy/imputed inputs.

## Evaluation design

Aggregate and subgroup results use shared five-fold stratified out-of-fold predictions. Sex and age slices are evaluated after prediction so every row remains out-of-fold. Bootstrap intervals quantify sampling variability in ROC-AUC; subgroup results remain descriptive because the dataset is small.

Missing-feature ablations remove selected feature groups before each fold is fitted. The resulting metric changes measure predictive dependence within UCI Cleveland, not causal importance.

## Stochastic layer

The latent health state is updated weekly:

`H[t+1] = clamp(H[t] + drift + reversion_to_start + noise, 0, 100)`

Noise increases with stress and sleep variability. Monte Carlo paths produce percentile bands, threshold-crossing probabilities, and final-state distributions.

Lifestyle drift uses a conservative `0.10` scale factor so repeated weekly assumptions do not overwhelm the clinical baseline over multi-year horizons. These coefficients encode directional scenarios for decision analysis; they are not estimated causal treatment effects.

Week-zero simulated risk is calibrated to equal the Layer 1 baseline probability. Later risk is mapped from health-state movement relative to that individual starting point, avoiding an unexplained intercept shift between the two layers.

## Optimization

Candidate plans are constrained by time, exercise frequency, and weekly change limits. Plans are evaluated with common random numbers (the same simulation seed) to reduce plan-to-plan Monte Carlo noise.

`objective = expected_mean_risk + lambda_time * time_cost - lambda_adherence * adherence`

The app reports objective-ranked plans and a risk/time Pareto frontier.

## Personalization

Weekly logs update bounded lifestyle-sensitivity weights. This is an adaptive heuristic, not causal inference. The app labels observed relationships as associations and retains the educational-use disclaimer.
