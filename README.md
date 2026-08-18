# Wine Quality Classification — Comparison of Six Supervised Models

Machine Learning · Assignment 2 · M.Tech (AIML/DSE), BITS Pilani WILP

---

## a. Problem statement

Wine quality is conventionally assessed by trained human tasters, a process that
is slow, costly and subject to inter-rater variability. This project evaluates
whether the eleven physicochemical properties measurable in a laboratory
(acidity, sulphates, alcohol content and related variables), together with the
wine type, are sufficient to predict whether a wine will be rated as premium.

The task is formulated as **binary classification**. Each instance is labelled:

* `is_premium = 1` when the median sensory score awarded by the tasting panel is **≥ 7**
* `is_premium = 0` otherwise

Approximately **19%** of instances belong to the positive class, so the dataset
is imbalanced. Accuracy alone is therefore an inadequate measure of performance:
a trivial classifier predicting the majority class for every instance achieves
81% accuracy. The comparison below consequently emphasises **MCC** and **AUC**,
neither of which rewards that degenerate strategy.

## b. Dataset description

**Source:** Wine Quality Data Set — Cortez, Cerdeira, Almeida, Matos & Reis (2009),
UCI Machine Learning Repository
([archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)).
The two source files, `winequality-red.csv` and `winequality-white.csv`, are
merged into a single table with an added `wine_type` indicator, so that wine
colour is modelled explicitly as a predictor rather than left as an unobserved
confounding variable.

| Property | Value |
|---|---|
| Instances after merging and de-duplication | **5,320** (from 6,497 raw rows; 1,177 exact duplicates removed) |
| Features | **12** (11 continuous physicochemical + 1 binary colour flag) |
| Target | `is_premium` — binary, derived from the 0–10 `quality` score |
| Class balance | 1,009 premium (18.97%) vs 4,311 everyday (81.03%) |
| Missing values | None |
| Train / test split | 75 / 25, stratified, `random_state=2026` → 3,990 train / **1,330 test** |

**Features**

| # | Feature | Unit / meaning |
|---|---|---|
| 1 | `fixed_acidity` | g(tartaric acid)/dm³ |
| 2 | `volatile_acidity` | g(acetic acid)/dm³ |
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

**Preprocessing.** Exact duplicate rows are removed, as the raw files contain a
substantial number of repeated records. All 12 features are standardised using a
`StandardScaler` fitted on the training split only, and the fitted scaler is
persisted alongside the models so that the Streamlit application applies an
identical transformation at inference time. The `test_data.csv` file in this
repository is the held-out 25% partition, which was not used during training.

## c. GitHub repository link

`https://github.com/Anusha-T1/ml-assignment-2-new`

**Live Streamlit app:** `https://YOUR-APP.streamlit.app`

```
ml-assignment-2-new/
│-- app.py                     Streamlit front end
│-- requirements.txt
│-- README.md
│-- test_data.csv              held-out 25% partition (1,330 instances)
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

Six classifiers were fitted on the same standardised training split and
evaluated on the same 1,330-instance held-out set. Hyperparameters were tuned
manually; the treatment of class imbalance varies by model and is described in
the observations.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7308 | 0.8209 | 0.3936 | 0.7778 | 0.5227 | 0.4029 |
| Decision Tree | 0.7353 | 0.7779 | 0.3932 | 0.7302 | 0.5111 | 0.3830 |
| kNN | 0.8368 | 0.8281 | 0.6190 | 0.3611 | 0.4561 | 0.3864 |
| Naive Bayes | 0.7008 | 0.7345 | 0.3453 | 0.6468 | 0.4503 | 0.2950 |
| **Random Forest (Ensemble)** | **0.8474** | **0.8541** | **0.6738** | 0.3770 | 0.4835 | **0.4256** |
| Support Vector Machine (RBF) | 0.7436 | 0.8453 | 0.4086 | **0.7897** | **0.5386** | 0.4250 |

*Precision, recall and F1 are reported for the positive (premium) class.
AUC is computed from predicted class probabilities. Configuration: LogReg `C=0.8, class_weight=balanced`;
Decision Tree `entropy, max_depth=9, min_samples_leaf=8, class_weight=balanced`;
kNN `k=17, distance-weighted, Manhattan`; GaussianNB `var_smoothing=1e-8`;
Random Forest `450 trees, max_features=sqrt, unweighted`; SVC `RBF, C=3.0, class_weight=balanced`.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Performs competitively for a linear model. An AUC of 0.821 indicates that the decision boundary between the two classes is largely linear in the 12 standardised features, with alcohol and volatile acidity contributing most of the discriminative power. Balanced class weighting shifts the model into a high-recall, low-precision regime: it identifies 78% of premium instances, but only 39% of its positive predictions are correct. Its accuracy (0.731) falls below the 0.810 majority-class baseline, which is the expected consequence of the balanced weighting. The model is computationally inexpensive, stable and interpretable, and serves as a reasonable baseline. |
| **Decision Tree** | The weakest ranking model in the set (AUC 0.778). A single tree produces coarse, piecewise-constant probability estimates and therefore cannot generate the fine-grained confidence ordering that AUC rewards, even though its hard-label metrics (accuracy 0.735, F1 0.511) are comparable to logistic regression. A maximum depth of 9 and a minimum leaf size of 8 were required to control overfitting; the unpruned tree memorised the training split. Its main value in this comparison is as a baseline against which the ensemble gain can be measured. |
| **kNN** | Illustrates why accuracy is an inadequate metric for this dataset. Its accuracy of 0.837 is the second highest in the table, but this is achieved through conservative prediction: a recall of 0.361 means the model fails to identify almost two-thirds of premium instances. With k=17 and distance weighting, a premium instance located in a neighbourhood dominated by non-premium instances is outvoted, and under an 81/19 class distribution most neighbourhoods are so dominated. Precision of 0.619 indicates that its positive predictions are nonetheless reasonably reliable. |
| **Naive Bayes** | Ranks last on every metric (MCC 0.295, AUC 0.735). The cause is structural rather than a matter of tuning. GaussianNB assumes conditional independence of features given the class, an assumption clearly violated by this dataset: density is close to a deterministic function of alcohol and residual sugar, and the two sulfur dioxide variables are strongly correlated. Redundant evidence is therefore counted multiple times, yielding overconfident and poorly calibrated posterior probabilities. The model trains rapidly but carries an inappropriate inductive bias for correlated physicochemical measurements. |
| **Random Forest (Ensemble)** | The strongest model overall: highest accuracy (0.847), AUC (0.854), precision (0.674) and MCC (0.426). Bootstrap aggregation with random feature subsampling addresses the variance problem that limits the single tree, while averaging across 450 trees restores the graded probability estimates that raise AUC by approximately 0.08 relative to one tree. The model performed better without class weighting: enabling `class_weight='balanced_subsample'` reduced MCC by roughly 0.03, as the resulting gain in recall did not compensate for the loss in precision. Its principal limitation is recall (0.377) at the default 0.5 decision threshold, which is a threshold selection issue rather than a limitation of the model itself. |
| **Support Vector Machine (RBF)** | Essentially tied with the random forest on MCC (0.425 against 0.426), but occupying the opposite position in the precision-recall trade-off: it attains the highest recall (0.790) and the highest F1 (0.539) in the table, at a cost in accuracy (0.744). The RBF kernel combined with balanced class weights produces a non-linear and comparatively inclusive decision region for the positive class. It is the slowest of the six to train and the least interpretable, but is preferable where the cost of a false negative exceeds that of a false positive. |
| **Overall winner for this dataset** | **Random Forest (Ensemble)**, which leads on MCC (0.4256), AUC (0.8541), accuracy (0.8474) and precision (0.6738), and is therefore the strongest model on the metric most appropriate to a 19% positive class rate. Two qualifications should be noted. First, its MCC advantage over the RBF SVM is 0.0006, which is within noise. Second, the SVM more than doubles its recall. The ensemble is preferable in terms of ranking quality and precision; the SVM is preferable in terms of coverage, and would be the better choice for a screening application in which false negatives are more costly than false positives. More generally, no model exceeds an MCC of 0.43, which suggests that physicochemical measurements alone have limited capacity to reproduce expert sensory judgements. |

## Streamlit app features

* **CSV upload** — accepts any test file containing the 12 feature columns and `is_premium`; defaults to the bundled `test_data.csv`.
* **Model selection dropdown** — selects among the six trained classifiers.
* **Evaluation metrics** — all six metrics computed and displayed for the selected model.
* **Confusion matrix, ROC curve and classification report**, together with a comparison table across all six models and a grouped bar chart.

## Notes

* The random seed is fixed at 2026 throughout, so all reported figures are exactly reproducible.
* Models are trained offline by `model/train_models.py` and loaded from `.joblib`
  files at runtime. The deployed application performs no training, which keeps it
  within the Streamlit Community Cloud resource limits.
