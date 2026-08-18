"""
Wine Quality Classification -- Streamlit interface for ML Assignment 2.

Loads six pre-trained classifiers and evaluates them on a user-supplied or
bundled test set, reporting accuracy, AUC, precision, recall, F1 and MCC.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

HERE = Path(__file__).resolve().parent
ARTEFACTS = HERE / "model"
TARGET = "is_premium"

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
    "wine_type",
]

CLASS_LABELS = ["Not premium", "Premium"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="ML Assignment 2 - Wine Quality Classification",
    layout="wide",
)


@st.cache_resource
def load_scaler():
    return joblib.load(ARTEFACTS / "feature_scaler.joblib")


@st.cache_resource
def load_estimator(filename: str):
    return joblib.load(ARTEFACTS / filename)


@st.cache_data
def load_bundled_test_set() -> pd.DataFrame:
    return pd.read_csv(HERE / "test_data.csv")


def metric_bundle(truth, predicted, scores) -> dict:
    return {
        "Accuracy": accuracy_score(truth, predicted),
        "AUC": roc_auc_score(truth, scores),
        "Precision": precision_score(truth, predicted, zero_division=0),
        "Recall": recall_score(truth, predicted, zero_division=0),
        "F1": f1_score(truth, predicted, zero_division=0),
        "MCC": matthews_corrcoef(truth, predicted),
    }


def evaluate(frame: pd.DataFrame, label: str) -> dict:
    scaler = load_scaler()
    estimator = load_estimator(MODEL_FILES[label])
    X = scaler.transform(frame[FEATURE_ORDER])
    y = frame[TARGET]
    predicted = estimator.predict(X)
    scores = estimator.predict_proba(X)[:, 1]
    return {
        "metrics": metric_bundle(y, predicted, scores),
        "y": y,
        "pred": predicted,
        "prob": scores,
    }


# ----------------------------------------------------------------- sidebar --
st.sidebar.title("Wine Quality Classification")
st.sidebar.caption(
    "Binary classification: is a wine premium (sensory score >= 7), "
    "given its physicochemical properties?"
)

uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type="csv",
    help="Must contain the 12 feature columns plus an `is_premium` column. "
         "Leave empty to use the bundled test set.",
)

chosen = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()), index=4)
show_all = st.sidebar.checkbox("Compare all six models", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Cortez et al. (2009), Wine Quality Data Set, UCI ML Repository."
)

# -------------------------------------------------------------- data intake --
if uploaded is not None:
    test_df = pd.read_csv(uploaded)
    source_note = f"uploaded file **{uploaded.name}**"
else:
    test_df = load_bundled_test_set()
    source_note = "bundled test set **test_data.csv**"

missing = [c for c in FEATURE_ORDER + [TARGET] if c not in test_df.columns]
if missing:
    st.error(f"These required columns are missing from your CSV: {missing}")
    st.stop()

st.title("Wine Quality Classification - Model Comparison")
st.write(
    f"Evaluating on **{len(test_df):,}** instances from the {source_note}. "
    f"The positive class (premium) accounts for {test_df[TARGET].sum():,} "
    f"instances ({test_df[TARGET].mean():.1%}). Because the target is "
    "imbalanced, MCC and AUC are more informative than accuracy."
)

with st.expander("Preview input data"):
    st.dataframe(test_df.head(15), width="stretch")

# ------------------------------------------------------------- single model --
outcome = evaluate(test_df, chosen)
st.subheader(f"{chosen} - evaluation metrics")

cols = st.columns(6)
for col, (name, value) in zip(cols, outcome["metrics"].items()):
    col.metric(name, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(outcome["y"], outcome["pred"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cbar=False, cmap="Blues",
        xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, width="stretch")

with right:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(outcome["y"], outcome["prob"])
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.4))
    ax2.plot(fpr, tpr, lw=2,
             label=f"AUC = {outcome['metrics']['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2, width="stretch")

st.markdown("**Classification report**")
st.code(
    classification_report(
        outcome["y"], outcome["pred"],
        target_names=CLASS_LABELS, digits=4,
    )
)

# ---------------------------------------------------------- all-model table --
if show_all:
    st.subheader("Comparison of all six models on this test set")
    results = pd.DataFrame(
        [{"Model": name, **evaluate(test_df, name)["metrics"]}
         for name in MODEL_FILES]
    ).set_index("Model")

    st.dataframe(
        results.style
        .format("{:.4f}")
        .background_gradient(cmap="Blues", axis=0),
        width="stretch",
    )

    best_model = results["MCC"].idxmax()
    st.success(
        f"Highest Matthews correlation coefficient on this test set: "
        f"**{best_model}** (MCC {results.loc[best_model, 'MCC']:.4f}, "
        f"AUC {results.loc[best_model, 'AUC']:.4f})."
    )

    fig3, ax3 = plt.subplots(figsize=(9, 3.6))
    results[["Accuracy", "F1", "MCC", "AUC"]].plot.bar(ax=ax3, width=0.8)
    ax3.set_ylabel("Score")
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    ax3.legend(ncol=4, fontsize=8)
    st.pyplot(fig3, width="stretch")
