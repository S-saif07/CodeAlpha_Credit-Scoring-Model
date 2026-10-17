# Project Report: Credit Scoring Using Machine Learning

*Educational project. All numbers below were produced by running `credit_scoring.ipynb` (seed 42, scikit-learn 1.8.0).*

## 1. Introduction
Lenders must decide whether an applicant is likely to repay a loan. Machine-learning classifiers can learn patterns from
historical applicants to estimate this risk. This project builds and compares three such classifiers on the public German Credit
dataset, with emphasis on *correct evaluation* and *responsible interpretation* rather than on chasing a high score.

## 2. Objectives
1. Build a reproducible pipeline that predicts good vs. bad credit.
2. Compare Logistic Regression, Decision Tree and Random Forest using appropriate metrics.
3. Choose a model using credit-risk objectives (cost of errors), not accuracy alone.
4. Provide an educational prediction demo, explainability and a responsible-AI discussion.

## 3. Methodology
**Data.** Statlog German Credit: 1,000 applicants, 20 features (7 numeric, 13 categorical), no missing values, no duplicates.
Target recoded as `default_risk` (1 = bad credit = **positive class**, 0 = good). Class distribution: 700 good (70%), 300 bad (30%).

**Feature engineering.** `monthly_payment = credit_amount / duration`; `payment_burden_index = monthly_payment × installment_rate`;
`has_multiple_credits`; `history_delay_or_critical`; `has_other_installment_plans`. Debt-to-income, loan-to-income, credit
utilisation and number of previous defaults were **not created** because the dataset has no income, debt-balance, credit-limit
or default-count columns. `payment_burden_index` is a relative index, not a true debt-to-income ratio.

**Preprocessing.** Stratified 80/20 split (800 train / 200 test) before any fitting. A `ColumnTransformer` inside each `Pipeline`:
numeric → median imputation + standardisation; categorical → most-frequent imputation + one-hot encoding. Preprocessing is fitted
on training folds only. `personal_status_sex`, `age` and `foreign_worker` were excluded from the model inputs and used only for
a fairness audit.

**Models and tuning.** `GridSearchCV` with 5-fold stratified cross-validation optimising ROC-AUC; `class_weight="balanced"`.
Best settings: Logistic Regression `C=0.1`; Decision Tree `max_depth=7, min_samples_leaf=20`; Random Forest
`n_estimators=400, max_depth=None, min_samples_leaf=3`.

**Evaluation.** Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix and classification report, all with `pos_label=1`
(bad credit); ROC-AUC uses the predicted probability of class 1.

## 4. Results

### 4.1 Exploratory findings
* Bad-credit rate by checking-account status: `<0 DM` 49%, `0–200 DM` 39%, `≥200 DM` 22%, **no checking account 12%**.
* Savings: `<100 DM` 36% bad vs `≥1000 DM` 12.5%.
* Longer loans and larger amounts are riskier (correlation with target: duration 0.22, credit amount 0.16).
* **Credit history is counter-intuitive:** "critical/other credits" has the *lowest* bad rate (17%), while "no credits / all paid" (62%, n=40)
  and "all paid at this bank" (57%, n=49) have the highest – probably selection effects. Consequently the `history_delay_or_critical` flag is a weak feature.
* Engineered payment features are only weakly related to risk (correlations 0.02–0.05).

### 4.2 Model comparison – default threshold 0.5 (test set, n = 200; 60 bad, 140 good)
| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC | FN rate | FP rate |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.774 | 0.740 | 0.547 | 0.783 | 0.644 | **0.805** | 0.217 | 0.279 |
| Decision Tree | 0.698 | 0.615 | 0.407 | 0.617 | 0.490 | 0.677 | 0.383 | 0.386 |
| Random Forest | **0.797** | **0.775** | **0.636** | 0.583 | 0.609 | 0.791 | 0.417 | **0.143** |
| Baseline: always "good" | – | 0.700 | 0.000 | 0.000 | 0.000 | 0.500 | 1.000 | 0.000 |

The baseline shows why accuracy misleads: 70% accuracy with zero risky applicants detected. The tuned Decision Tree's accuracy is even below the baseline.

### 4.3 Cost-based threshold selection
Using an illustrative cost of **5 per false negative and 1 per false positive** (suggested in the dataset's documentation), the threshold
for each model was chosen on **out-of-fold training predictions**, then applied once to the test set.

| Model @ threshold | Precision | Recall | FN | FP | Test cost / applicant |
|---|---|---|---|---|---|
| Logistic Regression @ 0.34 | 0.400 | 0.867 | 8 | 78 | 0.590 |
| Decision Tree @ 0.08 | 0.342 | 0.900 | 6 | 104 | 0.670 |
| **Random Forest @ 0.27 (selected)** | 0.393 | 0.883 | 7 | 82 | **0.585** |
| Random Forest @ 0.50 (for reference) | 0.636 | 0.583 | 25 | 20 | 0.725 |

Reference points: approving everyone costs 1.50 per applicant; rejecting everyone costs 0.70. The selected Random Forest's cross-validated
cost (0.501) was marginally below Logistic Regression's (0.506); on the test set the order is the same but the gap (0.585 vs 0.590) is negligible.
**The two models should be considered tied.** The Random Forest test ROC-AUC is 0.791 with a 95% bootstrap interval of **0.715–0.864**.

### 4.4 Trade-offs
Moving the threshold from 0.5 to 0.27 lifts recall from 58% to 88% (misses fall from 25 to 7 risky applicants) but precision drops from 64% to 39%:
the model now flags about two-thirds of all applicants (False-positive rate ≈ 59%). Under a 5:1 cost this is "optimal" but only modestly better than rejecting everyone,
and a business that must also grow its customer base might accept a different cost ratio and threshold.

### 4.5 Explainability
Permutation importance (Random Forest, test AUC drop): **checking-account status 0.124** (by far the largest), then credit amount 0.015, savings 0.012,
purpose 0.010 and other debtors 0.010. Remaining features contribute very little; with only 200 test rows several importances are within noise.
Importance describes what the model *uses*, not what *causes* default.

### 4.6 Fairness audit (protected attributes not used as inputs; test set, selected model)
| Group | n | Actual bad rate | Flagged rate | Recall | False-positive rate |
|---|---|---|---|---|---|
| Female | 61 | 32.8% | 70.5% | 0.85 | 63.4% |
| Male | 139 | 28.8% | 66.2% | 0.90 | 56.6% |
| Age < 25 | 39 | 48.7% | 76.9% | 0.895 | 65.0% |
| Age 25–39 | 90 | 27.8% | 65.6% | 0.88 | 56.9% |
| Age 40+ | 71 | 22.5% | 64.8% | 0.875 | 58.2% |

Younger and female applicants are flagged somewhat more often and have higher false-positive rates (good applicants wrongly flagged). Group sizes are small
(and the foreign-worker "no" group has only 6 people, too few to interpret), so these gaps may be noise – but they show why a proper fairness review is required.
The model is unaware of age and sex, yet other features can act as proxies.

### 4.7 Prediction demonstration
`predict_applicant()` returns a class and the estimated probability of bad credit. In the notebook's two illustrative profiles, a short, small, well-saved loan
received an estimated 5.9% probability (not flagged) and a long, large loan with a negative checking balance 70.0% (flagged), at threshold 0.27.
These are labelled educational and are not lending decisions.

## 5. Limitations
* **Data:** 1,000 rows from one bank, 1990s Germany; amounts in Deutsche Marks; no income, debt, credit-limit or time information; rejected applicants are absent (selection bias).
* **Evaluation:** one random split with 200 test rows – wide uncertainty; model differences of a few hundredths are not reliable.
* **Assumptions:** the 5:1 cost ratio is illustrative; probabilities were not calibrated; no out-of-time or out-of-population validation.
* **Features:** several finance-style features could not be built; the engineered ones add little.
* **Causality:** results show association only.
* **Fairness/privacy:** proxies for protected attributes may remain; real credit data would raise consent, security and legal obligations (e.g. GDPR, anti-discrimination law).

## 6. Conclusion
On this dataset, Logistic Regression and Random Forest perform similarly (test ROC-AUC ≈ 0.79–0.81) and both clearly beat a Decision Tree and the
"always good" baseline, which looks accurate (70%) while detecting no risky applicants. The checking-account status is the dominant signal. Because
missing a bad applicant is assumed to be five times as costly as rejecting a good one, a low threshold (≈ 0.27) was selected, yielding high recall (≈ 88%)
at the price of low precision (≈ 39%) – a trade-off a real lender would have to weigh against business goals and fairness. The project demonstrates a
leakage-free, reproducible workflow, but **it does not show that the model is suitable for real lending**. Real deployment would require validation on current,
representative data, calibration, fairness and regulatory review, explanation for affected applicants, human oversight and continuous monitoring.
