"""Streamlit front-end for the handwritten-digit classifier comparison.

Run locally:   streamlit run app.py
The app loads the five saved pipelines from model/*.joblib. If a file cannot be
loaded (for example scikit-learn on the server is a different version from the
one used for training) the model is silently re-fitted from data/train_data.csv
with the hyper-parameters stored in model/best_params.json.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from model.evaluation import (
    METRIC_ORDER, confusion_frame, report_frame, score_predictions,
)
from model.model_definitions import FILE_STEMS, MODEL_NAMES, make_pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
TARGET = "target"

st.set_page_config(page_title="Digit Classifier Lab", page_icon="🔢", layout="wide")


# ------------------------------------------------------------------ loaders
@st.cache_data
def load_feature_columns():
    return json.loads((ROOT / "model" / "feature_columns.json").read_text())


@st.cache_data
def load_bundled_test_data():
    return pd.read_csv(ROOT / "test_data.csv")


@st.cache_resource
def load_models():
    """Return ({name: fitted pipeline}, [names that had to be re-fitted])."""
    feats = load_feature_columns()
    train_df = pd.read_csv(ROOT / "data" / "train_data.csv")
    best = json.loads((ROOT / "model" / "best_params.json").read_text())
    models, refitted = {}, []
    for name in MODEL_NAMES:
        try:
            m = joblib.load(ROOT / "model" / f"{FILE_STEMS[name]}.joblib")
            m.predict(train_df[feats].head(5))  # sanity check
        except Exception:
            m = make_pipeline(name, **best[name])
            m.fit(train_df[feats], train_df[TARGET])
            refitted.append(name)
        models[name] = m
    return models, refitted


def evaluate(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)
    scores = score_predictions(y, pred, proba, class_labels=list(model.classes_)) if y is not None else None
    return pred, proba, scores


def plot_confusion(cm_df, title):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="mako_r", cbar=False, linewidths=0.4, ax=ax)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_digit_grid(X, y_true, y_pred, how_many):
    wrong = np.where(y_true != y_pred)[0][:how_many]
    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        ax.imshow(X[idx].reshape(8, 8), cmap="gray_r")
        ax.set_title(f"true {y_true[idx]} → pred {y_pred[idx]}", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- UI
feature_cols = load_feature_columns()
models, refitted = load_models()

st.title("🔢 Handwritten Digit Classifier Lab")
st.write(
    "Compare five classical machine-learning classifiers on 8×8 handwritten-digit "
    "images (UCI *Optical Recognition of Handwritten Digits*, 64 pixel features, "
    "10 classes). Upload a test CSV, pick a model and inspect how it behaves."
)

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help="64 pixel columns (pixel_0_0 … pixel_7_7) and an optional 'target' column.")
    st.caption("No file uploaded? The bundled `test_data.csv` (450 held-out images) is used.")
    st.header("2 · Model")
    chosen = st.selectbox("Choose a classifier", MODEL_NAMES, index=MODEL_NAMES.index("Support Vector Machine"))
    if refitted:
        st.caption("ℹ️ Re-fitted on start-up: " + ", ".join(refitted))

# ----- read data
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        source_note = f"Uploaded file: **{uploaded.name}**"
    except Exception as exc:
        st.error(f"Could not read that file as CSV ({exc}).")
        st.stop()
else:
    data = load_bundled_test_data()
    source_note = "Bundled sample: **test_data.csv**"

missing = [c for c in feature_cols if c not in data.columns]
if missing:
    st.error(f"The file is missing {len(missing)} required feature column(s), e.g. `{missing[0]}`. "
             "Expected the 64 columns pixel_0_0 … pixel_7_7.")
    st.stop()

X_df = data[feature_cols].apply(pd.to_numeric, errors="coerce")
bad_rows = X_df.isna().any(axis=1)
if bad_rows.any():
    st.warning(f"Dropped {int(bad_rows.sum())} row(s) containing missing / non-numeric pixel values.")
data = data.loc[~bad_rows].reset_index(drop=True)
X_df = X_df.loc[~bad_rows].reset_index(drop=True)

has_labels = TARGET in data.columns
y_true = None
if has_labels:
    y_series = pd.to_numeric(data[TARGET], errors="coerce")
    if y_series.isna().any():
        st.warning("The 'target' column has non-numeric values, so metrics are disabled.")
        has_labels = False
    else:
        y_true = y_series.astype(int).to_numpy()

st.caption(f"{source_note} · {len(data)} rows · {len(feature_cols)} features · "
           f"labels {'found' if has_labels else 'not found (predictions only)'}")

model = models[chosen]
X = X_df.to_numpy()
pred, proba, scores = evaluate(model, X_df, y_true)

tab_scores, tab_cm, tab_wrong, tab_cmp, tab_data = st.tabs(
    ["📊 Scores", "🧮 Confusion matrix & report", "🔍 Misclassified digits",
     "🏁 Compare all models", "🗂 Data & predictions"]
)

with tab_scores:
    st.subheader(f"{chosen} — evaluation on the selected data")
    if not has_labels:
        st.info("Add a 'target' column with the true digit (0–9) to see evaluation metrics.")
    else:
        row1 = st.columns(3)
        row2 = st.columns(3)
        for col, key in zip(row1 + row2, METRIC_ORDER):
            val = scores[key]
            col.metric(key, "n/a" if np.isnan(val) else f"{val:.4f}")
        st.caption("Precision, Recall and F1 are macro-averaged over the 10 digits; AUC is "
                   "one-vs-rest macro; MCC is the multi-class Matthews correlation coefficient.")
        if np.isnan(scores["AUC"]):
            st.caption("AUC shows n/a because the uploaded file does not contain all 10 classes.")

with tab_cm:
    if not has_labels:
        st.info("Confusion matrix and classification report need a 'target' column.")
    else:
        labels = list(model.classes_)
        left, right = st.columns([1, 1])
        with left:
            st.pyplot(plot_confusion(confusion_frame(y_true, pred, labels), f"{chosen}"))
        with right:
            st.markdown("**Classification report**")
            st.dataframe(report_frame(y_true, pred, labels))

with tab_wrong:
    if not has_labels:
        st.info("Needs a 'target' column to know which predictions are wrong.")
    else:
        n_wrong = int((pred != y_true).sum())
        st.write(f"{chosen} misclassified **{n_wrong}** of {len(y_true)} images.")
        if n_wrong == 0:
            st.success("Nothing to show — every image was classified correctly.")
        else:
            how_many = st.slider("Images to display", 1, min(24, n_wrong), min(12, n_wrong))
            st.pyplot(plot_digit_grid(X, y_true, pred, how_many))

with tab_cmp:
    st.subheader("All five models on the same data")
    if not has_labels:
        st.info("Add a 'target' column to compare models.")
    else:
        rows = []
        for name in MODEL_NAMES:
            _, _, s = evaluate(models[name], X_df, y_true)
            rows.append({"ML Model Name": name, **s})
        table = pd.DataFrame(rows).set_index("ML Model Name")
        st.dataframe(table.round(4))
        st.bar_chart(table[["Accuracy", "F1", "MCC"]])
        best_name = table["F1"].idxmax()
        st.success(f"Highest macro-F1 on this data: **{best_name}** ({table.loc[best_name, 'F1']:.4f})")

with tab_data:
    st.markdown("**First rows of the data in use**")
    st.dataframe(data.head(20))
    out = pd.DataFrame({"row": np.arange(len(data)), "predicted": pred})
    out["confidence"] = proba.max(axis=1).round(4)
    if has_labels:
        out["actual"] = y_true
        out["correct"] = out["predicted"] == out["actual"]
    st.markdown(f"**Predictions from {chosen}**")
    st.dataframe(out)
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{FILE_STEMS[chosen]}.csv", mime="text/csv")
