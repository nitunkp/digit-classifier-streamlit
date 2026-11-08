"""The five classifiers required by the assignment, plus their search grids.

Distance / margin / gradient based models (LR, kNN, SVM) get a StandardScaler
inside a Pipeline so scaling is learnt on training folds only. Decision Tree and
Gaussian Naive Bayes are scale-insensitive and are used without scaling.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 2026

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes",
    "Support Vector Machine",
]

FILE_STEMS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}


def make_pipeline(name, **params):
    """Build an (unfitted) pipeline for `name`. `params` use the 'clf__' prefix."""
    if name == "Logistic Regression":
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=SEED)),
        ])
    elif name == "Decision Tree":
        pipe = Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))])
    elif name == "kNN":
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ])
    elif name == "Naive Bayes":
        pipe = Pipeline([("clf", GaussianNB())])
    elif name == "Support Vector Machine":
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=SEED)),
        ])
    else:
        raise ValueError(f"Unknown model name: {name}")
    if params:
        pipe.set_params(**params)
    return pipe


SEARCH_GRIDS = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [6, 10, 14, None],
        "clf__min_samples_leaf": [1, 3, 5],
    },
    "kNN": {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
    },
    # several pixels are always 0 in the training data -> zero variance, so the
    # smoothing term matters a lot for Gaussian NB on this dataset
    "Naive Bayes": {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
    "Support Vector Machine": {
        "clf__C": [1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.001],
    },
}
