"""
Wine Quality Classification -- training pipeline for six classifiers.

Reads the two raw UCI CSVs (red + white), merges them into a single frame with a
`wine_type` indicator, binarises the 0-10 sensory score into a `is_premium`
flag, then fits and persists six classifiers plus the fitted scaler.

Run:  python model/train_models.py
Out:  model/*.joblib, test_data.csv, model/metrics_summary.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 2026
PREMIUM_CUTOFF = 7          # sensory score at/above which a wine is premium
HOLDOUT_FRACTION = 0.25

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data"
ARTEFACTS = ROOT / "model"

FEATURE_ORDER = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "wine_type",          # 0 = white, 1 = red  -> 12th feature
]
TARGET = "is_premium"


def tidy(col: str) -> str:
    return col.strip().lower().replace(" ", "_")


def assemble_frame() -> pd.DataFrame:
    """Merge the red and white wine tables into a single labelled frame."""
    reds = pd.read_csv(RAW / "winequality-red.csv", sep=";")
    whites = pd.read_csv(RAW / "winequality-white.csv", sep=";")

    reds.columns = [tidy(c) for c in reds.columns]
    whites.columns = [tidy(c) for c in whites.columns]
    reds["wine_type"] = 1
    whites["wine_type"] = 0

    wines = pd.concat([reds, whites], ignore_index=True)
    wines = wines.rename(columns={"ph": "pH"})   # keep the conventional casing
    wines = wines.drop_duplicates().reset_index(drop=True)
    wines[TARGET] = (wines["quality"] >= PREMIUM_CUTOFF).astype(int)
    wines = wines.drop(columns=["quality"])
    return wines[FEATURE_ORDER + [TARGET]]


def build_classifiers() -> dict:
    """The six estimators under comparison, keyed by their display label."""
    return {
        "Logistic Regression": LogisticRegression(
            C=0.8, max_iter=3000, class_weight="balanced", random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=9, min_samples_leaf=8, criterion="entropy",
            class_weight="balanced", random_state=SEED
        ),
        "kNN": KNeighborsClassifier(n_neighbors=17, weights="distance", p=1),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "Random Forest (Ensemble)": RandomForestClassifier(
            # left unweighted deliberately: re-weighting the bootstrap samples
            # reduced MCC by roughly 0.03 during tuning
            n_estimators=450, max_depth=None, min_samples_leaf=1,
            max_features="sqrt", n_jobs=-1, random_state=SEED
        ),
        "Support Vector Machine": SVC(
            C=3.0, kernel="rbf", gamma="scale", probability=True,
            class_weight="balanced", random_state=SEED
        ),
    }


def score_model(truth, predicted, scores) -> dict:
    return {
        "Accuracy": accuracy_score(truth, predicted),
        "AUC": roc_auc_score(truth, scores),
        "Precision": precision_score(truth, predicted, zero_division=0),
        "Recall": recall_score(truth, predicted, zero_division=0),
        "F1": f1_score(truth, predicted, zero_division=0),
        "MCC": matthews_corrcoef(truth, predicted),
    }


def main() -> None:
    ARTEFACTS.mkdir(exist_ok=True)
    wines = assemble_frame()
    print(f"dataset assembled: {wines.shape[0]} instances x "
          f"{len(FEATURE_ORDER)} features")
    print(f"positive class rate: {wines[TARGET].mean():.3%}")

    X = wines[FEATURE_ORDER]
    y = wines[TARGET]
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=HOLDOUT_FRACTION, stratify=y, random_state=SEED
    )

    scaler = StandardScaler().fit(X_fit)
    X_fit_s = scaler.transform(X_fit)
    X_hold_s = scaler.transform(X_hold)
    joblib.dump(scaler, ARTEFACTS / "feature_scaler.joblib", compress=3)

    # the held-out split is also written out as the app's default test set
    holdout_csv = X_hold.copy()
    holdout_csv[TARGET] = y_hold.values
    holdout_csv.to_csv(ROOT / "test_data.csv", index=False)
    print(f"wrote test_data.csv ({len(holdout_csv)} rows)")

    rows = []
    for label, estimator in build_classifiers().items():
        estimator.fit(X_fit_s, y_fit)
        predicted = estimator.predict(X_hold_s)
        scores = estimator.predict_proba(X_hold_s)[:, 1]
        result = score_model(y_hold, predicted, scores)
        result["Model"] = label
        rows.append(result)

        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        # compressed: the 450-tree forest is ~31 MB uncompressed
        joblib.dump(estimator, ARTEFACTS / f"{slug}.joblib", compress=3)
        print(f"  {label:28s} acc={result['Accuracy']:.4f} "
              f"auc={result['AUC']:.4f} mcc={result['MCC']:.4f}")

    summary = pd.DataFrame(rows)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    summary.to_csv(ARTEFACTS / "metrics_summary.csv", index=False)
    print("\n" + summary.round(4).to_markdown(index=False))


if __name__ == "__main__":
    main()
