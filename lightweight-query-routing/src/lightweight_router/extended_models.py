from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD

from .config import CANONICAL_LABELS
from .improvement_models import build_features
from .structural_features import StructuralDocumented18

ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2)
NB_ALPHAS = (.01, .1, .5, 1.0)
NBSVM_CS = (.25, .5, 1.0, 2.0, 4.0)

EXTENDED_FEATURES = (
    "word_3000", "word_3000_onechar", "char_15000", "word_char", "word_char_structural",
    "binary_count", "word_1_3", "char_2_6", "structural_only",
    "chi2_1000", "chi2_3000", "chi2_5000",
    "svd_100", "svd_200", "svd_400", "svd_structural_100", "svd_structural_200", "svd_structural_400",
)


def _word_char_union():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 sublinear_tf=True, max_features=15000)),
    ])


def build_extended_features(name: str):
    if name in {"word_3000", "word_3000_onechar", "char_15000", "word_char", "word_char_structural"}:
        return build_features(name)
    if name == "binary_count":
        return CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, binary=True)
    if name == "word_1_3":
        return TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=25000)
    if name == "char_2_6":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=2,
                               sublinear_tf=True, max_features=20000)
    if name == "structural_only":
        return StructuralDocumented18()
    if name.startswith("chi2_"):
        k = int(name.rsplit("_", 1)[1])
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)),
            ("chi2", SelectKBest(chi2, k=k)),
        ])
    if name.startswith("svd_structural_"):
        components = int(name.rsplit("_", 1)[1])
        return FeatureUnion([
            ("svd", Pipeline([("word_char", _word_char_union()),
                              ("svd", TruncatedSVD(n_components=components, random_state=42))])),
            ("structural", Pipeline([("documented_18", StructuralDocumented18()),
                                     ("scale", StandardScaler())])),
        ])
    if name.startswith("svd_"):
        components = int(name.rsplit("_", 1)[1])
        return Pipeline([("word_char", _word_char_union()),
                         ("svd", TruncatedSVD(n_components=components, random_state=42))])
    raise ValueError(f"Unknown extended feature family: {name}")


class MulticlassNBSVM(ClassifierMixin, BaseEstimator):
    """Fold-local multiclass NB-SVM with one log-count-ratio model per class."""

    def __init__(self, feature_kind: str = "word", classifier: str = "linear_svc", C: float = 1.0):
        self.feature_kind = feature_kind
        self.classifier = classifier
        self.C = C

    def _vectorizer(self):
        word = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, binary=True)
        if self.feature_kind == "word":
            return word
        if self.feature_kind == "word_char":
            char = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                   max_features=15000, binary=True)
            return FeatureUnion([("word", word), ("char", char)])
        raise ValueError(f"Unknown NB-SVM feature kind: {self.feature_kind}")

    def _classifier(self):
        if self.classifier == "linear_svc":
            return LinearSVC(C=self.C, max_iter=10000, random_state=42)
        if self.classifier == "logistic_regression":
            return LogisticRegression(C=self.C, max_iter=5000, random_state=42)
        raise ValueError(f"Unknown NB-SVM classifier: {self.classifier}")

    @staticmethod
    def _ratio(matrix, mask: np.ndarray) -> np.ndarray:
        positive = np.asarray(matrix[mask].sum(axis=0)).ravel() + 1.0
        negative = np.asarray(matrix[~mask].sum(axis=0)).ravel() + 1.0
        positive /= positive.sum()
        negative /= negative.sum()
        return np.log(positive / negative)

    def fit(self, X, y):
        values = np.asarray(X, dtype=object)
        labels = np.asarray(y, dtype=object)
        self.vectorizer_ = self._vectorizer()
        matrix = self.vectorizer_.fit_transform(values)
        observed = set(labels)
        self.classes_ = np.asarray([label for label in CANONICAL_LABELS if label in observed], dtype=object)
        if len(self.classes_) < 2:
            raise ValueError("NB-SVM requires at least two classes.")
        self.log_count_ratios_ = []
        self.classifiers_ = []
        for label in self.classes_:
            mask = labels == label
            ratio = self._ratio(matrix, mask)
            classifier = self._classifier()
            classifier.fit(matrix.multiply(ratio), mask.astype(int))
            self.log_count_ratios_.append(ratio)
            self.classifiers_.append(classifier)
        self.ratio_training_row_count_ = len(labels)
        return self

    def decision_function(self, X):
        matrix = self.vectorizer_.transform(np.asarray(X, dtype=object))
        columns = []
        for ratio, classifier in zip(self.log_count_ratios_, self.classifiers_):
            columns.append(np.asarray(classifier.decision_function(matrix.multiply(ratio))).reshape(-1))
        return np.column_stack(columns)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)


class LabelEncodingClassifier(ClassifierMixin, BaseEstimator):
    """Allow optional boosting libraries to consume canonical string labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        labels = np.asarray(y, dtype=object)
        self.classes_ = np.asarray([label for label in CANONICAL_LABELS if label in set(labels)], dtype=object)
        positions = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([positions[label] for label in labels])
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, encoded)
        return self

    def predict(self, X):
        return self.classes_[np.asarray(self.estimator_.predict(X), dtype=int).reshape(-1)]

    def predict_proba(self, X):
        return np.asarray(self.estimator_.predict_proba(X), dtype=float)


def build_extended_estimator(feature: str, model: str, params: dict[str, Any] | None = None):
    params = dict(params or {})
    if model.startswith("nbsvm_"):
        classifier = "linear_svc" if model.endswith("linear_svc") else "logistic_regression"
        return MulticlassNBSVM(feature_kind="word_char" if feature == "nbsvm_word_char" else "word",
                              classifier=classifier, C=float(params.get("C", 1.0)))
    if model == "ridge":
        classifier = RidgeClassifier(alpha=float(params.get("alpha", 1.0)), solver="lsqr",
                                     tol=1e-3, max_iter=2000)
    elif model == "ridge_cv":
        classifier = RidgeClassifierCV(alphas=ALPHAS, cv=3)
    elif model.startswith("sgd_"):
        loss = model.removeprefix("sgd_").removesuffix("_averaged")
        classifier = SGDClassifier(loss=loss, alpha=float(params.get("alpha", 1e-4)),
                                   average=model.endswith("_averaged"), max_iter=3000,
                                   tol=1e-4, random_state=42)
    elif model == "multinomial_nb":
        classifier = MultinomialNB(alpha=float(params.get("alpha", 1.0)))
    elif model == "complement_nb":
        classifier = ComplementNB(alpha=float(params.get("alpha", 1.0)))
    elif model == "bernoulli_nb":
        classifier = BernoulliNB(alpha=float(params.get("alpha", 1.0)), binarize=0.0)
    elif model == "linear_svc_reference":
        classifier = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    elif model == "extra_trees":
        classifier = ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, max_features="sqrt",
                                          n_jobs=-1, random_state=42)
    elif model == "xgboost":
        from xgboost import XGBClassifier
        classifier = LabelEncodingClassifier(XGBClassifier(
            n_estimators=300, learning_rate=.05, max_depth=6, min_child_weight=2,
            subsample=.8, colsample_bytree=.8, objective="multi:softprob",
            eval_metric="mlogloss", n_jobs=4, random_state=42,
        ))
    elif model == "lightgbm":
        from lightgbm import LGBMClassifier
        classifier = LabelEncodingClassifier(LGBMClassifier(
            n_estimators=300, learning_rate=.05, num_leaves=31, max_depth=-1,
            subsample=.8, colsample_bytree=.8, reg_lambda=1.0,
            n_jobs=4, random_state=42, verbosity=-1,
        ))
    elif model == "catboost":
        from catboost import CatBoostClassifier
        classifier = LabelEncodingClassifier(CatBoostClassifier(
            iterations=300, learning_rate=.05, depth=6, loss_function="MultiClass",
            random_seed=42, verbose=False, allow_writing_files=False, thread_count=4,
        ))
    else:
        raise ValueError(f"Unknown extended classical model: {model}")
    return Pipeline([("features", build_extended_features(feature)), ("classifier", classifier)])


def aligned_scores(estimator, X, classes=CANONICAL_LABELS) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(X), dtype=float)
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
    else:
        predicted = estimator.predict(X)
        scores = np.column_stack([np.asarray(predicted) == label for label in estimator.classes_]).astype(float)
    aligned = np.full((len(scores), len(classes)), -1e9 if not hasattr(estimator, "predict_proba") else 0.0)
    positions = {label: index for index, label in enumerate(estimator.classes_)}
    for target, label in enumerate(classes):
        if label in positions:
            aligned[:, target] = scores[:, positions[label]]
    return aligned


class VotingTextClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.asarray(CANONICAL_LABELS, dtype=object)
        self.estimators_ = [clone(estimator).fit(X, y) for estimator in self.estimators]
        return self

    def predict_proba(self, X):
        weights = np.ones(len(self.estimators_)) if self.weights is None else np.asarray(self.weights, dtype=float)
        votes = np.zeros((len(X), len(self.classes_)), dtype=float)
        for weight, estimator in zip(weights, self.estimators_):
            predicted = estimator.predict(X)
            for column, label in enumerate(self.classes_):
                votes[:, column] += weight * (predicted == label)
        return votes / weights.sum()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class CalibratedScoreAveragingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.classes_ = np.asarray(CANONICAL_LABELS, dtype=object)
        self.estimators_ = []
        for estimator in self.estimators:
            calibrated = CalibratedClassifierCV(estimator=clone(estimator), method="sigmoid", cv=3, n_jobs=1)
            calibrated.fit(X, y)
            self.estimators_.append(calibrated)
        return self

    def predict_proba(self, X):
        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
        for estimator in self.estimators_:
            probabilities += aligned_scores(estimator, X)
        return probabilities / len(self.estimators_)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class OOFStackingTextClassifier(ClassifierMixin, BaseEstimator):
    """Train the meta-model exclusively from inner out-of-fold base scores."""

    def __init__(self, estimators, meta_model: str = "logistic_regression", cv: int = 3):
        self.estimators = estimators
        self.meta_model = meta_model
        self.cv = cv

    def _meta(self):
        if self.meta_model == "logistic_regression":
            return LogisticRegression(max_iter=5000, random_state=42)
        if self.meta_model == "ridge":
            return RidgeClassifier(alpha=1.0)
        raise ValueError(f"Unknown stacking meta-model: {self.meta_model}")

    def fit(self, X, y):
        values = np.asarray(X, dtype=object)
        labels = np.asarray(y, dtype=object)
        self.classes_ = np.asarray(CANONICAL_LABELS, dtype=object)
        splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=2027)
        meta_blocks = []
        self.estimators_ = []
        assignment = np.zeros(len(values), dtype=int)
        for base in self.estimators:
            oof = np.zeros((len(values), len(self.classes_)), dtype=float)
            for train, validation in splitter.split(values, labels):
                fitted = clone(base).fit(values[train], labels[train])
                oof[validation] = aligned_scores(fitted, values[validation])
                assignment[validation] += 1
            meta_blocks.append(oof)
            self.estimators_.append(clone(base).fit(values, labels))
        if not np.all(assignment == len(self.estimators)):
            raise RuntimeError("Stacking invariant failed: meta rows were not exclusively inner OOF.")
        meta_features = np.hstack(meta_blocks)
        self.meta_estimator_ = self._meta().fit(meta_features, labels)
        self.meta_training_source_ = "inner_oof_only"
        self.meta_oof_assignment_count_ = assignment
        return self

    def _meta_features(self, X):
        return np.hstack([aligned_scores(estimator, X) for estimator in self.estimators_])

    def decision_function(self, X):
        if hasattr(self.meta_estimator_, "decision_function"):
            return self.meta_estimator_.decision_function(self._meta_features(X))
        return self.meta_estimator_.predict_proba(self._meta_features(X))

    def predict(self, X):
        return self.meta_estimator_.predict(self._meta_features(X))

    def predict_proba(self, X):
        if hasattr(self.meta_estimator_, "predict_proba"):
            return self.meta_estimator_.predict_proba(self._meta_features(X))
        scores = np.asarray(self.decision_function(X), dtype=float)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
