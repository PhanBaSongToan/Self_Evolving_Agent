from __future__ import annotations

from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .structural_features import StructuralDocumented18

WORD_BASE = {"ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True}
ONE_CHAR_PATTERN = r"(?u)\b\w+\b"

FEATURE_CONFIGS: dict[str, dict[str, Any]] = {
    "word_3000": {"kind": "word", "max_features": 3000},
    "word_3000_onechar": {"kind": "word", "max_features": 3000, "token_pattern": ONE_CHAR_PATTERN},
    "word_10000": {"kind": "word", "max_features": 10000},
    "word_20000": {"kind": "word", "max_features": 20000},
    "char_15000": {"kind": "char", "max_features": 15000},
    "word_char": {"kind": "word_char"},
    "word_char_structural": {"kind": "word_char_structural"},
}

CLASSIFIER_CONFIGS = (
    "linear_svc", "logistic_regression", "rbf_svc", "random_forest",
    "calibrated_linear_svc", "calibrated_logistic_regression",
)


def build_features(name: str):
    if name not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown improvement feature configuration: {name}")
    config = FEATURE_CONFIGS[name]
    kind = config["kind"]
    if kind == "word":
        params = {**WORD_BASE, "max_features": config["max_features"]}
        if "token_pattern" in config:
            params["token_pattern"] = config["token_pattern"]
        return TfidfVectorizer(**params)
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                               sublinear_tf=True, max_features=config["max_features"])
    word = TfidfVectorizer(**WORD_BASE, max_features=20000)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           sublinear_tf=True, max_features=15000)
    transformers = [("word", word), ("char", char)]
    if kind == "word_char_structural":
        structural = Pipeline([("documented_18", StructuralDocumented18()), ("scale", StandardScaler())])
        transformers.append(("structural", structural))
    return FeatureUnion(transformers)


def _base_classifier(name: str):
    if name == "linear_svc":
        return LinearSVC(C=1.0, max_iter=10000, random_state=42)
    if name == "logistic_regression":
        return LogisticRegression(penalty="l2", C=1.0, max_iter=5000, class_weight=None)
    if name == "rbf_svc":
        return SVC(kernel="rbf", gamma="scale", C=1.0, probability=False)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown uncalibrated classifier: {name}")


def build_improvement_estimator(feature_name: str, classifier_name: str):
    if classifier_name not in CLASSIFIER_CONFIGS:
        raise ValueError(f"Unknown improvement classifier: {classifier_name}")
    if classifier_name == "calibrated_linear_svc":
        base = Pipeline([("features", build_features(feature_name)),
                         ("classifier", LinearSVC(C=1.0, max_iter=10000, random_state=42))])
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=-1)
    if classifier_name == "calibrated_logistic_regression":
        base = Pipeline([("features", build_features(feature_name)),
                         ("classifier", LogisticRegression(penalty="l2", C=1.0, max_iter=5000, class_weight=None))])
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3, n_jobs=-1)
    return Pipeline([("features", build_features(feature_name)), ("classifier", _base_classifier(classifier_name))])


def supports_probabilities(classifier_name: str) -> bool:
    return classifier_name in ("logistic_regression", "calibrated_linear_svc", "calibrated_logistic_regression")
