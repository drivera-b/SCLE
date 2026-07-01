# SLCE Data Card

## Dataset roles

### UCI Cleveland Heart Disease
- **Role:** supervised binary baseline model
- **Rows:** 303
- **License:** CC BY 4.0
- **Local artifact:** `data/heart.csv`
- **DOI:** `10.24432/C52P4X`

### CDC NHANES 2017-2018
- **Role:** survey-weighted population context for lifestyle and laboratory measurements
- **Processed rows:** 6,161
- **Local artifact:** `data/nhanes_lifestyle_biomarkers.csv`
- **Source:** [CDC/NCHS NHANES 2017-2018](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017)

The datasets are not row-concatenated because they contain different participants, schemas, sampling designs, and outcomes.

## NHANES measurement coverage
| Field | Non-missing coverage |
|---|---:|
| sleep_mean_hours | 99.3% |
| exercise_days_per_week | 100.0% |
| resting_hr | 90.1% |
| systolic_bp | 89.8% |
| bmi | 93.0% |
| total_cholesterol | 88.4% |
| hdl_cholesterol | 88.4% |
| hba1c | 89.9% |
| fasting_glucose | 43.2% |

Fasting glucose is collected in a subsample, so its lower coverage is expected. SLCE retains survey weights, strata, and PSU columns; in-app percentiles use positive examination weights, while full design-based variance estimation remains future work.

## Processing
- Official CDC XPT files are merged on `SEQN`.
- Physiologically impossible values are set missing using explicit bounds.
- Adult activity days use the maximum of vigorous/moderate recreation days to avoid double-counting unknown overlap.
- The processed extract can be rebuilt with `python -m src.nhanes_dataset --build`.

## Missingness and quality risks
- Laboratory eligibility, fasting subsampling, and nonresponse vary by measurement and participant characteristics.
- UCI missing model fields use fold-local medians during evaluation and training medians at inference.
- User-uploaded lab CSVs are validated for one-row structure, units, numeric type, and bounded ranges.
- No names or personal identifiers are required or retained by the import format.

## Responsible use
NHANES percentiles describe position in an age/sex reference sample and are not clinical thresholds. See the [CDC laboratory overview](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/overviewlab.aspx?BeginYear=2017) for collection, quality-control, subsampling, and analytic guidance.
