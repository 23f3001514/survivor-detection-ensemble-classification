# Survivor Detection Challenge — Maritime Classification

Predicting passenger survival outcomes in a Titanic-style maritime disaster dataset using an ensemble of Random Forest, Histogram Gradient Boosting, and Logistic Regression models.

**Challenge:** Survivor Detection Challenge — Maritime Classification (Modified)
**Cross-Validated Accuracy:** 0.8226 ± 0.0239 (Repeated 5-Fold CV, 3 repeats)

---

## Overview

This project builds a supervised binary classification pipeline to predict `Outcome` (survived / did not survive) for passengers in a maritime disaster dataset. The dataset is a Titanic-style challenge deliberately modified with injected Gaussian noise across numeric fields, decoy columns, and realistic missing-data patterns — reflecting the kind of messy, adversarial data conditions common in real-world classification tasks.

The pipeline covers exploratory data analysis of missingness, category-wise survival rates, and correlations; denoising and feature engineering to recover clean signal from corrupted columns; and a soft-voting ensemble evaluated via repeated cross-validation.

## Dataset

- **Train:** 712 rows | **Test:** 179 rows | **Features:** 16 columns (train)
- Derived from a Titanic-style maritime disaster dataset, with several numeric columns perturbed by injected noise and three columns (`PassengerId`, `CLass`, `FamilySize`) outside the official data dictionary

| Column | Description |
|---|---|
| `PassengerName` | Full passenger name, including title |
| `TicketTier` | Passenger class proxy — continuous/noisy, rounds to 1–3 |
| `Gender` | Passenger sex |
| `Age` | Passenger age — 27.7% missing in train |
| `RelativesAboard` | Siblings/spouses aboard — noisy, rounds to a non-negative integer |
| `ParentsChildren` | Parents/children aboard — noisy, rounds to a non-negative integer |
| `TicketCost` | Fare paid — noisy, contains negative values |
| `Berth` | Cabin identifier — 77.7% missing in train |
| `BoardingPort` | Port of embarkation |
| `Singleton` | Traveling alone flag |
| `FarePerPerson` | Fare divided by family size |
| `Title` | Extracted/provided honorific |
| `Outcome` | **Target** — 0 = did not survive, 1 = survived |

Known data quality issues: `PassengerId` and `TicketCost` contain implausible negative values, and `TicketTier` is a continuous float rather than a clean 1/2/3 category — all consistent with deliberately injected Gaussian noise. A `CLass` column (558 unique ticket-number-style strings) sits outside the data dictionary and was treated as a decoy.

## Approach

1. **EDA** — missingness audit, target balance, survival rate by Gender/Tier/Port/Title/Singleton, Age and Fare distributions, correlation with target
2. **Denoising & feature engineering** — Title extracted from name, Gender reconciled via Title, Tier rounded/clipped to valid range, family counts rounded/clipped to non-negative integers, plus engineered `FamilySize`, `Alone`, `IsChild`, `Mother`, log-fare features, `TicketGroup`, `SurnameGroup`, `HasBerth`, and one-hot Title/Port
3. **Modeling** — Random Forest, Histogram Gradient Boosting, and Logistic Regression compared individually and as a soft-voting ensemble
4. **Evaluation** — Repeated Stratified 5-Fold Cross-Validation (3 repeats), accuracy as the scoring metric
5. **Submission** — Ensemble predictions merged into the sample submission format and validated with sanity checks

## Results

| Model | Accuracy | Std. Dev |
|---|---|---|
| Random Forest | 0.8211 | ± 0.0292 |
| Histogram Gradient Boosting | 0.8188 | ± 0.0291 |
| Logistic Regression | 0.8239 | ± 0.0309 |
| **Ensemble (Soft Voting)** | **0.8226** | **± 0.0239** |

Logistic Regression scores marginally highest on average, but the ensemble has the **lowest variance across folds**, making it the more stable choice for final submission. Key EDA signals: passengers with a recorded `Berth` value survived at more than double the rate of those without (64.8% vs. 29.8%), and missing `Age` correlated with lower survival (29.4% vs. 40.8%) — both consistent with `Berth`/`Age` availability acting as a proxy for socioeconomic status.

**Final sanity check:** predicted survival rate (39.7%) closely tracks the training survival rate (37.6%), indicating no significant prediction bias.

## Repository Structure

```
survivor-detection-ensemble-classification/
├── README.md
├── requirements.txt
├── Notebook
     └── Survivor_Detection_ensemble.ipynb
├── report/
│   └── Survivor_Detection_Challenge_Report.pdf
└── Datasets/
    ├── maritime_train.csv
   
```

## Tech Stack

- **pandas / numpy** — data loading, cleaning, feature computation
- **matplotlib** — exploratory and diagnostic visualizations
- **scikit-learn** — imputation, scaling, pipeline construction, cross-validation, ensemble modeling, metrics

## Future Work

- Hyperparameter tuning via grid/Bayesian search across all three base models
- Stacking (meta-learner) in place of soft voting to potentially improve on the ensemble's accuracy ceiling
- Deeper feature interactions (e.g., Title × Tier, FamilySize × Fare)
- Model-based imputation for `Age` in place of simple median imputation
- Threshold tuning if the competition metric rewards precision/recall trade-offs differently than raw accuracy


A full write-up with methodology, EDA findings, and detailed results is available in [`report/Survivor_Detection_Challenge_Report.pdf`](./report/Survivor_Detection_Challenge_Report.pdf).
