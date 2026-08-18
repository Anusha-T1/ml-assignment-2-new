"""
Cellar Check -- an interactive comparison of six classifiers that try to spot a
premium Vinho Verde bottle from its lab chemistry alone.

Streamlit front end for ML Assignment 2.
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

CATALOGUE = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Cellar Check", page_icon="🍷", layout="wide")


@st.cache_resource
def load_scaler():
    return joblib.load(ARTEFACTS / "feature_scaler.joblib")


@st.cache_resource
def load_estimator(filename: str):
    return joblib.load(ARTEFACTS / filename)


@st.cache_data
def load_bundled_holdout() -> pd.DataFrame:
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
    estimator = load_estimator(CATALOGUE[label])
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
st.sidebar.title("🍷 Cellar Check")
st.sidebar.caption(
    "Six classifiers, one question: is this bottle a **premium** wine "
    "(sensory score ≥ 7)?"
)

uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type="csv",
    help="Needs the 12 feature columns plus an `is_premium` column. "
         "Leave empty to use the bundled hold-out slice.",
)

chosen = st.sidebar.selectbox("Model", list(CATALOGUE.keys()), index=4)
show_all = st.sidebar.checkbox("Compare all six models", value=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Cortez et al. (2009), Vinho Verde wine quality, UCI ML Repository."
)

# -------------------------------------------------------------- data intake --
if uploaded is not None:
    tasting_sheet = pd.read_csv(uploaded)
    source_note = f"uploaded file — **{uploaded.name}**"
else:
    tasting_sheet = load_bundled_holdout()
    source_note = "bundled hold-out slice — **test_data.csv**"

missing = [c for c in FEATURE_ORDER + [TARGET] if c not in tasting_sheet.columns]
if missing:
    st.error(f"These required columns are missing from your CSV: {missing}")
    st.stop()

st.title("Can chemistry alone pick out a good bottle?")
st.write(
    f"Scoring **{len(tasting_sheet):,}** bottles from the {source_note}. "
    f"{tasting_sheet[TARGET].sum():,} of them are actually premium "
    f"({tasting_sheet[TARGET].mean():.1%}) — a deliberately lopsided target, "
    "which is exactly why MCC is more honest here than accuracy."
)

with st.expander("Peek at the data"):
    st.dataframe(tasting_sheet.head(15), width="stretch")

# ------------------------------------------------------------- single model --
outcome = evaluate(tasting_sheet, chosen)
st.subheader(f"{chosen} — evaluation metrics")

cols = st.columns(6)
for col, (name, value) in zip(cols, outcome["metrics"].items()):
    col.metric(name, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(outcome["y"], outcome["pred"])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cbar=False, cmap="rocket_r",
        xticklabels=["everyday", "premium"],
        yticklabels=["everyday", "premium"], ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    st.pyplot(fig, width="stretch")

with right:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(outcome["y"], outcome["prob"])
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.4))
    ax2.plot(fpr, tpr, lw=2,
             label=f"AUC = {outcome['metrics']['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2, width="stretch")

st.markdown("**Classification report**")
st.code(
    classification_report(
        outcome["y"], outcome["pred"],
        target_names=["everyday", "premium"], digits=4,
    )
)

# ---------------------------------------------------------- all-model table --
if show_all:
    st.subheader("All six models on this test set")
    board = pd.DataFrame(
        [{"Model": name, **evaluate(tasting_sheet, name)["metrics"]}
         for name in CATALOGUE]
    ).set_index("Model")

    st.dataframe(
        board.style
        .format("{:.4f}")
        .background_gradient(cmap="Greens", axis=0),
        width="stretch",
    )

    champion = board["MCC"].idxmax()
    st.success(
        f"Best Matthews correlation on this test set: **{champion}** "
        f"(MCC {board.loc[champion, 'MCC']:.4f}, "
        f"AUC {board.loc[champion, 'AUC']:.4f})."
    )

    fig3, ax3 = plt.subplots(figsize=(9, 3.6))
    board[["Accuracy", "F1", "MCC", "AUC"]].plot.bar(ax=ax3, width=0.8)
    ax3.set_ylabel("score")
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    ax3.legend(ncol=4, fontsize=8)
    st.pyplot(fig3, width="stretch")
