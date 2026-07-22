from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from .config import CANONICAL_LABELS
from .models import build_pipeline


@dataclass
class EvaluationResult:
    name: str
    fold_metrics: list[dict[str, Any]]
    oof: pd.DataFrame
    report: dict[str, Any]
    raw_confusion: np.ndarray
    normalized_confusion: np.ndarray
    fold_indices: dict[str, list[dict[str, list[int]]]]

    @property
    def summary(self) -> dict[str, Any]:
        accuracy = [row["accuracy"] for row in self.fold_metrics]
        macro_f1 = [row["macro_f1"] for row in self.fold_metrics]
        return {
            "configuration": self.name, "protocol": str(self.oof["protocol"].iloc[0]),
            "mean_accuracy": float(np.mean(accuracy)), "std_accuracy": float(np.std(accuracy)),
            "mean_macro_f1": float(np.mean(macro_f1)), "std_macro_f1": float(np.std(macro_f1)),
            "pooled_accuracy": float(accuracy_score(self.oof["true_label"], self.oof["predicted_label"])),
            "pooled_macro_f1": float(f1_score(self.oof["true_label"], self.oof["predicted_label"], labels=CANONICAL_LABELS, average="macro", zero_division=0)),
            "fit_time_seconds": float(sum(row["fit_time_seconds"] for row in self.fold_metrics)),
            "prediction_time_seconds": float(sum(row["prediction_time_seconds"] for row in self.fold_metrics)),
        }


def evaluate_configuration(frame: pd.DataFrame, name: str, *, n_splits: int = 5, shuffle: bool = False,
                           random_state: int | None = None) -> EvaluationResult:
    if len(frame) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    counts = frame["label"].value_counts()
    if counts.min() < n_splits:
        raise ValueError(f"Every class requires at least {n_splits} records for {n_splits}-fold stratified CV; counts: {counts.to_dict()}")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    queries = frame["query"].to_numpy()
    labels = frame["label"].to_numpy()
    if "protocol" not in frame.columns or frame["protocol"].nunique() != 1:
        raise ValueError("Evaluation requires exactly one protocol identifier on every input row.")
    protocol = str(frame["protocol"].iloc[0])
    pipeline = build_pipeline(name)
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    indices: list[dict[str, list[int]]] = []
    assigned = np.zeros(len(frame), dtype=int)
    for fold, (train_index, test_index) in enumerate(splitter.split(queries, labels), 1):
        model = clone(pipeline)
        start = time.perf_counter()
        model.fit(queries[train_index], labels[train_index])
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        predicted = model.predict(queries[test_index])
        predict_time = time.perf_counter() - start
        assigned[test_index] += 1
        metrics.append({"configuration": name, "fold": fold, "accuracy": float(accuracy_score(labels[test_index], predicted)),
                        "macro_f1": float(f1_score(labels[test_index], predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0)),
                        "fit_time_seconds": fit_time, "prediction_time_seconds": predict_time})
        indices.append({"fold": fold, "train_record_numbers": frame.iloc[train_index]["record_number"].astype(int).tolist(),
                        "validation_record_numbers": frame.iloc[test_index]["record_number"].astype(int).tolist()})
        for index, prediction in zip(test_index, predicted):
            row = frame.iloc[index]
            rows.append({"configuration": name, "record_number": int(row.record_number), "id": row.id,
                         "domain": row.domain, "query": row.query, "true_label": row.label,
                         "raw_true_label": row.raw_label, "predicted_label": prediction,
                         "protocol": protocol, "fold": fold})
    if not np.all(assigned == 1):
        raise RuntimeError("OOF invariant violated: each valid record must receive exactly one prediction.")
    oof = pd.DataFrame(rows).sort_values("record_number").reset_index(drop=True)
    raw = confusion_matrix(oof["true_label"], oof["predicted_label"], labels=CANONICAL_LABELS)
    normalized = confusion_matrix(oof["true_label"], oof["predicted_label"], labels=CANONICAL_LABELS, normalize="true")
    report = classification_report(oof["true_label"], oof["predicted_label"], labels=CANONICAL_LABELS,
                                   target_names=CANONICAL_LABELS, output_dict=True, zero_division=0)
    report["predicted_class_distribution"] = oof["predicted_label"].value_counts(normalize=True).reindex(CANONICAL_LABELS, fill_value=0).to_dict()
    report["support"] = int(len(oof))
    return EvaluationResult(name, metrics, oof, report, raw, normalized, {name: indices})


def evaluate_all(frame: pd.DataFrame) -> list[EvaluationResult]:
    from .models import configuration_names
    return [evaluate_configuration(frame, name) for name in configuration_names()]
