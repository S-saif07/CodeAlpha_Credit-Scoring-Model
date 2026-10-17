# Credit Scoring with Machine Learning

A beginner-friendly classification project that predicts whether a loan applicant is a **good** or **bad** credit
risk from the **German Credit (Statlog)** dataset, comparing **Logistic Regression, Decision Tree and Random Forest**.

> ⚠️ **Educational project.** Not a real lending tool. Results come from a small, old dataset and must not be used
> to make financial decisions about real people.

## What's inside
| File | Purpose |
|---|---|
| `credit_scoring.ipynb` | Main deliverable: Colab-compatible notebook (already executed, outputs included) |
| `credit_scoring.py` | The same code as a plain Python script |
| `REPORT.md` | Project report (introduction → conclusion) |
| `requirements.txt` | Python dependencies |
| `outputs/` | Generated tables, figures, saved model (`credit_model.joblib`) |
| `build_notebook.py` | (Optional) script that regenerates the notebook |

## Dataset
* **Source:** UCI Machine Learning Repository – *Statlog (German Credit Data)*, 1,000 applicants, 20 features.
  <https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data>
* **Download:** the code downloads the file automatically (public GitHub mirror of the original UCI data) into `data/german_credit_raw.csv`.
  If you are offline, download it manually and save it under that name (comma-separated, no header, 21 columns).
* **Target:** originally `1 = good`, `2 = bad`. Recoded as `default_risk`: **`1 = bad credit (positive class)`**, `0 = good credit`.
  Class balance: 70% good / 30% bad.

## Run in Google Colab
1. Upload `credit_scoring.ipynb` to <https://colab.research.google.com> (File → Upload notebook).
2. *Runtime → Run all*. All libraries are pre-installed; the dataset downloads itself.

## Run locally (VS Code)
```bash
git clone <your-repo-url> && cd credit_scoring
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option A – notebook: open credit_scoring.ipynb in VS Code, select the .venv kernel, Run All
# Option B – script:
python credit_scoring.py           # add MPLBACKEND=Agg to skip chart pop-ups; figures are saved either way
```
Runtime: about 1–2 minutes. Everything is seeded (`SEED = 42`), so results are reproducible for the same library versions
(developed with scikit-learn 1.8.0, pandas 3.0.2).

## Method in brief
1. Load, decode and inspect the data (no missing values or duplicates found).
2. Engineer 5 features the data can support (monthly payment, payment-burden index, etc.). Debt-to-income, loan-to-income,
   credit utilisation and default counts are **not possible** with this dataset and were deliberately not invented.
3. EDA with Matplotlib/Seaborn.
4. Stratified 80/20 split **before** any preprocessing; `Pipeline` + `ColumnTransformer` (impute → scale / one-hot) to avoid leakage.
5. `GridSearchCV` (5-fold, ROC-AUC) for each model; `class_weight="balanced"`.
6. Evaluate with Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, classification report (positive class = bad credit).
7. Choose a decision threshold using a 5:1 false-negative:false-positive cost (illustrative, from the dataset's documentation) on
   out-of-fold training predictions, then evaluate once on the untouched test set.
8. Permutation importance, logistic-regression coefficients and a fairness audit on age, sex and foreign-worker status
   (protected attributes are **not** model inputs).
9. Educational prediction demo `predict_applicant()`.

## Headline results (test set, 200 applicants, produced by running this code)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always "good") | 0.700 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.740 | 0.547 | 0.783 | 0.644 | 0.805 |
| Decision Tree | 0.615 | 0.407 | 0.617 | 0.490 | 0.677 |
| Random Forest | 0.775 | 0.636 | 0.583 | 0.609 | 0.791 |

*Threshold 0.5.* Random Forest and Logistic Regression are statistically hard to tell apart (95% bootstrap interval for the
Random Forest AUC: 0.715–0.864). Full tables, including the cost-based threshold analysis, are in `REPORT.md` and `outputs/`.

## Limitations (please read)
Small 1990s German dataset · no income/debt data · only 200 test rows (wide uncertainty) · possible bias and proxies for protected
attributes · predictions show association, not causation · real use would require validation on current data, calibration,
fairness/legal review and monitoring.

## Dataset citation
Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository.
