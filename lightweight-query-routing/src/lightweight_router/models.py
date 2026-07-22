from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import TFIDF_PARAMS
from .structural_features import StructuralDocumented18

CLASSIFIER_FACTORIES = {
    "logistic_regression": lambda: LogisticRegression(penalty="l2", C=1.0, max_iter=5000, class_weight=None),
    "svm": lambda: SVC(kernel="rbf", gamma="scale", probability=False),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "knn": lambda: KNeighborsClassifier(n_neighbors=7, metric="cosine", algorithm="brute"),
}


def configuration_names() -> tuple[str, ...]:
    return tuple(f"tfidf_{name}" for name in CLASSIFIER_FACTORIES) + tuple(f"structural_documented_18_{name}" for name in CLASSIFIER_FACTORIES)


def build_pipeline(name: str) -> Pipeline:
    if name.startswith("tfidf_"):
        classifier_name = name.removeprefix("tfidf_")
        if classifier_name not in CLASSIFIER_FACTORIES:
            raise ValueError(f"Unknown configuration: {name}")
        return Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("classifier", CLASSIFIER_FACTORIES[classifier_name]())])
    prefix = "structural_documented_18_"
    if name.startswith(prefix):
        classifier_name = name.removeprefix(prefix)
        if classifier_name not in CLASSIFIER_FACTORIES:
            raise ValueError(f"Unknown configuration: {name}")
        return Pipeline([("structural", StructuralDocumented18()), ("scaler", StandardScaler()), ("classifier", CLASSIFIER_FACTORIES[classifier_name]())])
    raise ValueError(f"Unknown configuration: {name}")


def resolved_parameters(name: str) -> dict[str, Any]:
    return build_pipeline(name).get_params(deep=True)
