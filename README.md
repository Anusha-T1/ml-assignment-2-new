# Cellar Check — spotting a premium wine from its lab chemistry

Machine Learning · Assignment 2 · M.Tech (AIML/DSE), BITS Pilani WILP

---

## a. Problem statement

A wine's commercial grade is decided by human tasters — a slow, expensive and
notoriously inconsistent process. This project asks whether the eleven
physicochemical measurements a lab can produce in minutes (acidity, sulphates,
alcohol, and so on), plus the colour of the wine, carry enough signal to flag a
**premium** bottle before anyone opens it.

Formally: a **binary classification** task. Each bottle is labelled

* `is_premium = 1` when the median sensory score awarded by the tasting panel is **≥ 7**
* `is_premium = 0` otherwise

Only about **19%** of bottles clear that bar, so the problem is deliberately
imbalanced. That makes plain accuracy a misleading headline — a model that calls
everything "everyday" already scores 81%. The comparison below therefore leans on
**MCC** and **AUC**, which do not reward that shortcut.

## b. Dataset description

**Source:** Wine Quality Data Set — Cortez, Cerdeira, Almeida, Matos & Reis (2009),
UCI Machine Learning Repository
([archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)).
Two CSVs — `winequality-red.csv` and `winequality-white.csv` — are merged here into
a single table, with a `wine_type` indicator added so that colour becomes a usable
predictor rather than a hidden confounder.

| Property | Value |
|---|---|
| Instances after merge and de-duplication | **5,320** (from 6,497 raw rows; 1,177 exact duplicates dropped) |
| Features | **12** (11 continuous physicochemical + 1 binary colour flag) |
| Target | `is_premium` — binary, derived from the 0–10 `quality` score |
| Class balance | 1,009 premium (18.97%) vs 4,311 everyday (81.03%) |
| Missing values | None |
| Train / test split | 75 / 25, stratified, `random_state=2026` → 3,990 train / **1,330 test** |

**Features**

| # | Feature | Unit / meaning |
|---|---|---|
| 1 | `fixed_acidity` | g(tartaric acid)/dm³ |
| 2 | `volatile_acidity` | g(acetic acid)/dm³ — the vinegar note |
| 3 | `citric_acid` | g/dm³ |
| 4 | `residual_sugar` | g/dm³ |
| 5 | `chlorides` | g(sodium chloride)/dm³ |
| 6 | `free_sulfur_dioxide` | mg/dm³ |
| 7 | `total_sulfur_dioxide` | mg/dm³ |
| 8 | `density` | g/cm³ |
| 9 | `pH` | acidity scale |
| 10 | `sulphates` | g(potassium sulphate)/dm³ |
| 11 | `alcohol` | % volume |
| 12 | `wine_type` | 1 = red, 0 = white |

**Preprocessing.** Exact duplicate rows are dropped (the raw files contain many —
the same wine sampled twice). All 12 features are standardised with a
`StandardScaler` fitted on the training split only, then persisted alongside the
models so the Streamlit app applies exactly the same transform at inference time.
`test_data.csv` in this repo *is* the held-out 25% slice — the models have never
seen it.

## c. GitHub repository link

`https://github.com/Anusha-T1/ml-assignment-2-new`

**Live Streamlit app:** `https://YOUR-APP.streamlit.app`

```
ml-assignment-2-new/
│-- app.py                     Streamlit front end
│-- requirements.txt
│-- README.md
│-- test_data.csv              held-out 25% slice (1,330 bottles)
│-- data/
│   │-- winequality-red.csv    raw UCI inputs
│   │-- winequality-white.csv
│-- model/
    │-- train_models.py        merge → split → scale → fit → persist
    │-- feature_scaler.joblib
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- random_forest_ensemble.joblib
    │-- support_vector_machine.joblib
    │-- metrics_summary.csv
```

Reproduce with `pip install -r requirements.txt`, then
`python model/train_models.py`, then `streamlit run app.py`.

## d. Models used

Six classifiers, all fitted on the same standardised training split and scored on
the same 1,330-bottle held-out set. Hyperparameters were hand-tuned; the
class-imbalance handling differs by model and is noted in the observations.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7308 | 0.8209 | 0.3936 | 0.7778 | 0.5227 | 0.4029 |
| Decision Tree | 0.7353 | 0.7779 | 0.3932 | 0.7302 | 0.5111 | 0.3830 |
| kNN | 0.8368 | 0.8281 | 0.6190 | 0.3611 | 0.4561 | 0.3864 |
| Naive Bayes | 0.7008 | 0.7345 | 0.3453 | 0.6468 | 0.4503 | 0.2950 |
| **Random Forest (Ensemble)** | **0.8474** | **0.8541** | **0.6738** | 0.3770 | 0.4835 | **0.4256** |
| Support Vector Machine (RBF) | 0.7436 | 0.8453 | 0.4086 | **0.7897** | **0.5386** | 0.4250 |

*Precision, recall and F1 are reported for the positive (premium) class.
AUC uses predicted probabilities. Configuration: LogReg `C=0.8, class_weight=balanced`;
Decision Tree `entropy, max_depth=9, min_samples_leaf=8, class_weight=balanced`;
kNN `k=17, distance-weighted, Manhattan`; GaussianNB `var_smoothing=1e-8`;
Random Forest `450 trees, max_features=sqrt, unweighted`; SVC `RBF, C=3.0, class_weight=balanced`.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Surprisingly competitive for a linear model — AUC 0.821 says the premium/everyday boundary is *mostly* linear in these 12 standardised features, with alcohol and volatile acidity doing most of the work. Class balancing pushes it into a high-recall, low-precision regime: it catches 78% of premium bottles but only 39% of what it flags is actually premium. Its accuracy (0.731) is *below* the 0.810 no-skill baseline, which is exactly the trade the balanced weighting makes. Cheap, stable and fully interpretable — a sensible floor for the others to beat. |
| **Decision Tree** | The weakest ranker in the set (AUC 0.778). A single tree emits coarse, piecewise-constant probabilities, so it cannot produce the smooth confidence ordering AUC rewards, even though its hard-label scores (accuracy 0.735, F1 0.511) sit right alongside logistic regression. Depth 9 with a leaf floor of 8 was needed to stop it memorising the training split; left unpruned it overfitted badly. Useful mainly as the diagnostic that shows what one tree gives you and what 450 of them add. |
| **kNN** | The clearest illustration of why accuracy alone misleads here. At 0.837 it posts the second-highest accuracy in the table, but it gets there by being conservative: recall of just 0.361 means it misses nearly two-thirds of premium bottles. With k=17 and distance weighting, a genuinely premium wine sitting in a neighbourhood of ordinary ones is simply outvoted — and in an 81/19 split most neighbourhoods are ordinary. Precision 0.619 is decent, so when it does call premium it is usually right. |
| **Naive Bayes** | Last on every single metric (MCC 0.295, AUC 0.735), and the reason is structural rather than tuning. GaussianNB assumes the 12 features are conditionally independent given the class, which this dataset flatly violates — density is nearly a deterministic function of alcohol and residual sugar, and the two sulfur dioxide columns are strongly correlated. Those redundant signals get counted several times over, producing overconfident and badly calibrated posteriors. Fast to train, but the wrong inductive bias for correlated chemistry. |
| **Random Forest (Ensemble)** | The best overall model: top accuracy (0.847), top AUC (0.854), top precision (0.674) and top MCC (0.426). Bagging plus feature subsampling fixes exactly what broke the single tree — variance — while the ensemble average restores the fine-grained probability ordering that lifts AUC by 8 points over one tree. Notably it scored *better* left unweighted: forcing `class_weight='balanced_subsample'` cost roughly 3 MCC points, because re-weighting pulled its precision down faster than it lifted recall. Its remaining weakness is recall (0.377) at the default 0.5 threshold — this is a threshold choice, not a modelling limit, and moving the cutoff would trade precision back for recall. |
| **Support Vector Machine (RBF)** | Effectively tied with the forest on MCC (0.425 vs 0.426) while occupying the opposite corner of the trade-off: best recall in the whole table (0.790) and best F1 (0.539), at the cost of accuracy (0.744). The RBF kernel plus balanced class weights carves out a curved, deliberately generous premium region. Slowest to fit of the six, and the least interpretable, but if the business cost is *missing* a good bottle rather than over-flagging one, this is the model to ship. |
| **Overall winner for this dataset** | **Random Forest (Ensemble)** — it leads on MCC (0.4256), AUC (0.8541), accuracy (0.8474) and precision (0.6738), so it is the strongest single answer on the metric that matters most under a 19% positive rate. The honest caveat: its MCC edge over the RBF SVM is 0.0006, i.e. noise, and the SVM more than doubles its recall. The tree ensemble wins on ranking quality and on being right when it commits; the SVM wins on coverage. For a screening use case where a missed premium bottle costs more than a false alarm, the SVM is the better deployment — and the app lets you switch between them and see that trade-off directly. A broader point: no model clears MCC 0.43, which suggests lab chemistry alone genuinely caps out somewhere short of replacing the tasting panel. |

## Streamlit app features

* **CSV upload** — drop in any test file with the 12 feature columns plus `is_premium`; falls back to the bundled `test_data.csv`.
* **Model dropdown** — switch between all six trained classifiers.
* **Evaluation metrics** — all six metrics rendered live for the selected model.
* **Confusion matrix, ROC curve and full classification report**, plus a "compare all six" table with a per-column heatmap and a grouped bar chart.

## Notes

* Random seed fixed at 2026 throughout, so every number above reproduces exactly.
* Models are trained offline by `model/train_models.py` and loaded from `.joblib`
  files at runtime — the deployed app does no training, which keeps it inside the
  Streamlit Community Cloud resource budget.
