"""Train, tune, evaluate and save all five classifiers.

Run from the repository root:

    python -m model.train_models

Outputs
  data/digits_full.csv        complete dataset (1797 rows)
  data/train_data.csv         75 % stratified training split
  test_data.csv               25 % held-out split (upload this file in the app)
  model/*.joblib              one fitted pipeline per classifier
  model/best_params.json      hyper-parameters picked by 5-fold CV
  model/feature_columns.json  ordered list of feature names
  model/metrics_summary.csv   the comparison table used in the README
  assets/metrics_comparison.png
"""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from model.evaluation import METRIC_ORDER, score_predictions
from model.model_definitions import (
    FILE_STEMS, MODEL_NAMES, SEARCH_GRIDS, SEED, make_pipeline,
)

ROOT = Path(__file__).resolve().parent.parent
TARGET = "target"


def main():
    # ---------------------------------------------------------------- data
    frame = load_digits(as_frame=True).frame
    feature_cols = [c for c in frame.columns if c != TARGET]

    train_df, test_df = train_test_split(
        frame, test_size=0.25, stratify=frame[TARGET], random_state=SEED
    )
    (ROOT / "data").mkdir(exist_ok=True)
    frame.to_csv(ROOT / "data" / "digits_full.csv", index=False)
    train_df.to_csv(ROOT / "data" / "train_data.csv", index=False)
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    (ROOT / "model" / "feature_columns.json").write_text(json.dumps(feature_cols))

    X_tr, y_tr = train_df[feature_cols], train_df[TARGET]
    X_te, y_te = test_df[feature_cols], test_df[TARGET]
    print(f"train={X_tr.shape}  test={X_te.shape}  classes={sorted(y_tr.unique())}")

    # ------------------------------------------------------- fit & evaluate
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}

    for name in MODEL_NAMES:
        t0 = time.time()
        search = GridSearchCV(
            make_pipeline(name), SEARCH_GRIDS[name],
            scoring="f1_macro", cv=cv, n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        fit_secs = time.time() - t0

        best = search.best_estimator_
        pred = best.predict(X_te)
        proba = best.predict_proba(X_te)
        s = score_predictions(y_te, pred, proba, class_labels=list(best.classes_))
        s["CV F1 (train, 5-fold)"] = search.best_score_
        s["Tuning time (s)"] = fit_secs
        rows.append({"ML Model Name": name, **s})

        best_params[name] = dict(search.best_params_)
        joblib.dump(best, ROOT / "model" / f"{FILE_STEMS[name]}.joblib")
        print(f"{name:24s} acc={s['Accuracy']:.4f} auc={s['AUC']:.4f} "
              f"f1={s['F1']:.4f} mcc={s['MCC']:.4f}  best={search.best_params_}")

    (ROOT / "model" / "best_params.json").write_text(json.dumps(best_params, indent=2))
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "model" / "metrics_summary.csv", index=False)

    # -------------------------------------------------------------- figure
    (ROOT / "assets").mkdir(exist_ok=True)
    ax = table.set_index("ML Model Name")[METRIC_ORDER].plot.bar(
        figsize=(10, 4.6), width=0.8, colormap="viridis"
    )
    ax.set_ylim(0.6, 1.1)
    ax.set_ylabel("score on held-out test set")
    ax.set_title("Test-set metrics for the five classifiers (digits dataset)")
    ax.legend(ncol=6, loc="upper center", fontsize=8, frameon=False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(ROOT / "assets" / "metrics_comparison.png", dpi=150)
    print("saved comparison figure")


if __name__ == "__main__":
    main()
