# SLCE Baseline Model Card

## Model details
- **Model:** median imputation, standardization, logistic regression
- **Version:** generated with `python -m scripts.generate_research_figures`
- **Training data:** UCI Cleveland Heart Disease, 303 rows
- **Output:** binary heart-disease baseline probability used to initialize SLCE scenarios

## Intended use
Educational demonstration of transparent baseline modeling, uncertainty propagation, and constrained decision support. The probability is an input to a research simulation, not a diagnosis or treatment recommendation.

## Out-of-scope use
- Clinical diagnosis, screening, triage, or treatment selection
- Predictions for populations not represented by the training data
- Causal claims about lifestyle or laboratory changes

## Aggregate evaluation
Five-fold stratified out-of-fold results:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.908 |
| Accuracy | 0.832 |
| Brier score | 0.119 |
| ROC-AUC bootstrap 95% interval | 0.875-0.939 |

## Subgroup evaluation
These slices reuse shared out-of-fold predictions. Small samples make differences descriptive, not proof of fairness.

| Slice | Rows | ROC-AUC | Brier score |
|---|---:|---:|---:|
| Sex: Female | 97 | 0.909 | 0.087 |
| Sex: Male | 206 | 0.889 | 0.134 |
| Age: 29-49 | 87 | 0.933 | 0.093 |
| Age: 50-59 | 125 | 0.901 | 0.131 |
| Age: 60-77 | 91 | 0.882 | 0.129 |

## Inputs and provenance
The app records whether each model feature is observed, proxy-derived, or median-imputed. Measured systolic blood pressure, total cholesterol, and fasting glucose can replace corresponding proxy/imputed fields. HbA1c and BMI remain context-only because this classifier was not trained with them.

## Limitations and risks
- Training ages span 29-77; teen outputs are out-of-distribution.
- The dataset is small, historical, single-center data and lacks broad demographic metadata.
- Many clinical fields are not available from the consumer form and require imputation.
- Calibration and subgroup estimates have substantial sampling uncertainty.
- The stochastic layer contains transparent assumptions, not learned treatment effects.

## Monitoring and reproducibility
- CI runs unit and Streamlit smoke tests.
- `reports/research_metrics.json` stores machine-readable evaluation output.
- `reports/figures/subgroup_performance.png` and `reports/figures/missing_feature_ablation.png` expose slice and missingness behavior.
