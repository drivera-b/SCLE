# Assumptions and Limitations

## Population alignment

The UCI Cleveland dataset is small and its observed age range is 29-77. Predictions for younger users are out-of-distribution and receive a lower Input Evidence Score. The NHANES reference extract includes participants aged 16-80, but it is not used to relabel or silently retrain the UCI classifier.

## No causal claims

Lifestyle drift coefficients are modeling assumptions. They demonstrate stochastic control and scenario analysis; they do not prove that a habit change causes a specific medical-risk reduction.

## Dataset compatibility

UCI and NHANES have different participants, labels, schemas, and sampling designs. Their rows are not merged. NHANES is used for reference distributions and future model research.

## Survey design

The in-app percentile context uses NHANES examination weights. The extract also retains strata and PSU fields, but the app does not currently estimate design-based standard errors or confidence intervals.

## Missing and proxy features

The UCI classifier has clinical features that are not available from the lifestyle form. SLCE exposes whether each input was measured, proxy-derived, or median-imputed rather than hiding that limitation.

The measured-versus-proxy comparison changes only input provenance. It is not a before/after health comparison, and its probability difference must not be interpreted as a treatment effect.

## Subgroup evaluation

Age- and sex-sliced metrics are based on small samples and a binary sex field from the source dataset. They identify possible performance variation but do not establish fairness, generalization, or performance for identities and populations absent from the data.

## Simulation interpretation

Monte Carlo intervals quantify uncertainty under the specified model and parameter assumptions. They are not clinical prediction intervals and do not include every source of real-world uncertainty.

## Safety boundary

SLCE is an educational research prototype. It does not diagnose disease, prescribe treatment, or replace a qualified clinician.
