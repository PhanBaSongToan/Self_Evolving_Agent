from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from .config import CANONICAL_LABELS, COST_MAPPING

COMPLEXITY = {"single_hop": 0, "multi_hop": 1, "summary": 2}
EXPENSIVE_COST = 3.5


def routing_error_metrics(true_labels, predicted_labels) -> dict[str, float]:
    true = pd.Series(true_labels).reset_index(drop=True)
    predicted = pd.Series(predicted_labels).reset_index(drop=True)
    if len(true) != len(predicted):
        raise ValueError("True and predicted routing labels must align.")
    true_complexity = true.map(COMPLEXITY).to_numpy()
    predicted_complexity = predicted.map(COMPLEXITY).to_numpy()
    if np.isnan(true_complexity).any() or np.isnan(predicted_complexity).any():
        raise ValueError("Routing-error calculation received an unknown label.")
    def conditional(source: str, target: str) -> float:
        mask = true.eq(source)
        return float((mask & predicted.eq(target)).sum() / mask.sum()) if mask.sum() else 0.0
    return {
        "multi_hop_to_single_hop_rate": conditional("multi_hop", "single_hop"),
        "summary_to_single_hop_rate": conditional("summary", "single_hop"),
        "summary_to_multi_hop_rate": conditional("summary", "multi_hop"),
        "total_under_routing_rate": float(np.mean(predicted_complexity < true_complexity)),
        "total_over_routing_rate": float(np.mean(predicted_complexity > true_complexity)),
    }


def cost_metrics(predicted_labels) -> dict[str, Any]:
    predicted = pd.Series(predicted_labels)
    total = float(sum(COST_MAPPING[label]["cost_ratio"] for label in predicted))
    n = len(predicted)
    baseline = EXPENSIVE_COST * n
    label_distribution = predicted.value_counts(normalize=True).reindex(CANONICAL_LABELS, fill_value=0).to_dict()
    paradigm_distribution = {COST_MAPPING[label]["paradigm"]: float(label_distribution[label]) for label in CANONICAL_LABELS}
    return {"average_simulated_route_cost": total / n if n else 0.0,
            "simulated_savings_percent": (baseline - total) / baseline * 100 if n else 0.0,
            "class_routing_distribution": label_distribution, "paradigm_routing_distribution": paradigm_distribution}


def classification_and_routing_metrics(true_labels, predicted_labels) -> dict[str, Any]:
    true = pd.Series(true_labels)
    predicted = pd.Series(predicted_labels)
    report = classification_report(true, predicted, labels=CANONICAL_LABELS, output_dict=True, zero_division=0)
    result = {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(true, predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, predicted, labels=CANONICAL_LABELS).tolist(),
        "per_class": {label: report[label] for label in CANONICAL_LABELS},
    }
    result.update(routing_error_metrics(true, predicted))
    result.update(cost_metrics(predicted))
    return result


def policy_is_valid(candidate: dict[str, float], default: dict[str, float]) -> bool:
    return (candidate["accuracy"] >= default["accuracy"] - .005
            and candidate["macro_f1"] >= default["macro_f1"] - .005
            and candidate["summary_recall"] >= default["summary_recall"]
            and candidate["multi_hop_recall"] >= default["multi_hop_recall"] - .01
            and candidate["total_under_routing_rate"] <= default["total_under_routing_rate"])


def pareto_frontier(table: pd.DataFrame) -> pd.DataFrame:
    required = {"accuracy", "macro_f1", "simulated_savings_percent", "total_under_routing_rate"}
    if not required.issubset(table.columns):
        raise ValueError(f"Pareto table is missing {sorted(required - set(table.columns))}.")
    keep = []
    for index, row in table.iterrows():
        dominated = False
        for other_index, other in table.iterrows():
            if index == other_index:
                continue
            no_worse = (other.accuracy >= row.accuracy and other.macro_f1 >= row.macro_f1
                        and other.simulated_savings_percent >= row.simulated_savings_percent
                        and other.total_under_routing_rate <= row.total_under_routing_rate)
            strict = (other.accuracy > row.accuracy or other.macro_f1 > row.macro_f1
                      or other.simulated_savings_percent > row.simulated_savings_percent
                      or other.total_under_routing_rate < row.total_under_routing_rate)
            if no_worse and strict:
                dominated = True
                break
        keep.append(not dominated)
    return table.loc[keep].copy().reset_index(drop=True)
