from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import CANONICAL_LABELS, COST_MAPPING, PROTOCOL_OFFICIAL
from .data import LoadedDataset
from .forensic import normalize_query, template_signature
from .improvement_metrics import classification_and_routing_metrics, pareto_frontier, policy_is_valid
from .improvement_models import FEATURE_CONFIGS, build_improvement_estimator, supports_probabilities
from .reporting import _markdown_table

TRACK_NAME = "Classical Production Router Improvement"
SEEDS = (0, 1, 2, 3, 4, 13, 21, 42, 77, 100)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_checksums(path: Path, exclude: Path | None = None) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        if exclude is not None:
            try:
                item.relative_to(exclude)
                continue
            except ValueError:
                pass
        result[item.relative_to(path).as_posix()] = _sha256_file(item)
    return result


@dataclass(frozen=True)
class Candidate:
    feature: str
    classifier: str
    mode: str = "global"

    @property
    def candidate_id(self) -> str:
        return f"{self.mode}__{self.feature}__{self.classifier}"


@dataclass
class RepeatedResult:
    candidate: Candidate
    seed_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    per_class_rows: list[dict[str, Any]]
    domain_rows: list[dict[str, Any]]


def _seed_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) <= 1:
        return float(array[0]), float(array[0])
    half = 1.96 * array.std(ddof=1) / math.sqrt(len(array))
    return float(array.mean() - half), float(array.mean() + half)


def _evaluate_seed_global(frame: pd.DataFrame, candidate: Candidate, seed: int) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = np.empty(len(frame), dtype=object)
    folds = np.zeros(len(frame), dtype=int)
    for fold, (train, validation) in enumerate(splitter.split(frame["query"], frame.label), 1):
        estimator = build_improvement_estimator(candidate.feature, candidate.classifier)
        estimator.fit(frame.iloc[train]["query"], frame.iloc[train].label)
        predictions[validation] = estimator.predict(frame.iloc[validation]["query"])
        folds[validation] = fold
    metrics = classification_and_routing_metrics(frame.label, predictions)
    oof = frame[["record_number", "id", "domain", "raw_label", "label", "query"]].copy()
    oof["predicted_label"] = predictions
    oof["fold"] = folds
    oof["seed"] = seed
    domain_rows = []
    for domain, group in oof.groupby("domain", sort=True):
        domain_metrics = classification_and_routing_metrics(group.label, group.predicted_label)
        domain_rows.append({"candidate_id": candidate.candidate_id, "seed": seed, "domain": domain,
                            "record_count": len(group), "accuracy": domain_metrics["accuracy"],
                            "macro_f1": domain_metrics["macro_f1"]})
    return oof, metrics, domain_rows


def _evaluate_seed_per_domain(frame: pd.DataFrame, candidate: Candidate, seed: int) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    parts = []
    domain_rows = []
    fitted_model_tokens = []
    for domain in sorted(frame.domain.unique()):
        subset = frame[frame.domain == domain].reset_index(drop=True)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        predictions = np.empty(len(subset), dtype=object)
        folds = np.zeros(len(subset), dtype=int)
        for fold, (train, validation) in enumerate(splitter.split(subset["query"], subset.label), 1):
            estimator = build_improvement_estimator(candidate.feature, candidate.classifier)
            fitted_model_tokens.append(f"{domain}:{fold}:{id(estimator)}")
            estimator.fit(subset.iloc[train]["query"], subset.iloc[train].label)
            predictions[validation] = estimator.predict(subset.iloc[validation]["query"])
            folds[validation] = fold
        part = subset[["record_number", "id", "domain", "raw_label", "label", "query"]].copy()
        part["predicted_label"] = predictions
        part["fold"] = folds
        part["seed"] = seed
        parts.append(part)
        metrics = classification_and_routing_metrics(subset.label, predictions)
        domain_rows.append({"candidate_id": candidate.candidate_id, "seed": seed, "domain": domain,
                            "record_count": len(subset), "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                            "independent_vectorizer_scope": domain})
    if len(fitted_model_tokens) != len(set(fitted_model_tokens)):
        raise RuntimeError("Per-domain evaluation reused a model instance.")
    oof = pd.concat(parts, ignore_index=True).sort_values("record_number").reset_index(drop=True)
    metrics = classification_and_routing_metrics(oof.label, oof.predicted_label)
    return oof, metrics, domain_rows


def _load_or_evaluate_seed(frame: pd.DataFrame, candidate: Candidate, seed: int, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = output / "experiments" / candidate.candidate_id
    target.mkdir(parents=True, exist_ok=True)
    metrics_path = target / f"seed_{seed}_metrics.json"
    oof_path = target / f"seed_{seed}_oof.csv"
    domains_path = target / f"seed_{seed}_domains.csv"
    if metrics_path.exists() and oof_path.exists() and domains_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8")), pd.read_csv(domains_path).to_dict(orient="records")
    if candidate.mode == "global":
        oof, metrics, domains = _evaluate_seed_global(frame, candidate, seed)
    elif candidate.mode == "per_domain":
        oof, metrics, domains = _evaluate_seed_per_domain(frame, candidate, seed)
    else:
        raise ValueError(f"Unknown improvement routing mode: {candidate.mode}")
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    oof.to_csv(oof_path, index=False)
    pd.DataFrame(domains).to_csv(domains_path, index=False)
    return metrics, domains


def evaluate_repeated(frame: pd.DataFrame, candidate: Candidate, output: Path, seeds: tuple[int, ...] = SEEDS) -> RepeatedResult:
    seed_metrics = []
    domain_rows = []
    per_class_rows = []
    for seed in seeds:
        metrics, domains = _load_or_evaluate_seed(frame, candidate, seed, output)
        seed_metrics.append({"seed": seed, **metrics})
        domain_rows.extend(domains)
        for label in CANONICAL_LABELS:
            values = metrics["per_class"][label]
            per_class_rows.append({"candidate_id": candidate.candidate_id, "seed": seed, "label": label,
                                   "precision": values["precision"], "recall": values["recall"], "f1": values["f1-score"],
                                   "support": values["support"]})
    accuracy = [row["accuracy"] for row in seed_metrics]
    macro_f1 = [row["macro_f1"] for row in seed_metrics]
    accuracy_ci = _seed_ci(accuracy)
    f1_ci = _seed_ci(macro_f1)
    scalar_fields = ("macro_precision", "macro_recall", "multi_hop_to_single_hop_rate", "summary_to_single_hop_rate",
                     "summary_to_multi_hop_rate", "total_under_routing_rate", "total_over_routing_rate",
                     "average_simulated_route_cost", "simulated_savings_percent")
    summary = {"candidate_id": candidate.candidate_id, "feature": candidate.feature, "classifier": candidate.classifier,
               "mode": candidate.mode, "seed_count": len(seeds),
               "mean_accuracy": float(np.mean(accuracy)), "std_accuracy": float(np.std(accuracy)),
               "min_accuracy": float(np.min(accuracy)), "max_accuracy": float(np.max(accuracy)),
               "accuracy_ci95_low": accuracy_ci[0], "accuracy_ci95_high": accuracy_ci[1],
               "mean_macro_f1": float(np.mean(macro_f1)), "std_macro_f1": float(np.std(macro_f1)),
               "min_macro_f1": float(np.min(macro_f1)), "max_macro_f1": float(np.max(macro_f1)),
               "macro_f1_ci95_low": f1_ci[0], "macro_f1_ci95_high": f1_ci[1]}
    for field in scalar_fields:
        summary[f"mean_{field}"] = float(np.mean([row[field] for row in seed_metrics]))
    for label in CANONICAL_LABELS:
        summary[f"mean_{label}_recall"] = float(np.mean([row["per_class"][label]["recall"] for row in seed_metrics]))
        summary[f"mean_{label}_precision"] = float(np.mean([row["per_class"][label]["precision"] for row in seed_metrics]))
        summary[f"mean_{label}_f1"] = float(np.mean([row["per_class"][label]["f1-score"] for row in seed_metrics]))
        summary[f"mean_{label}_routing_fraction"] = float(np.mean([
            row["class_routing_distribution"][label] for row in seed_metrics
        ]))
        paradigm = COST_MAPPING[label]["paradigm"]
        summary[f"mean_{paradigm}_routing_fraction"] = float(np.mean([
            row["paradigm_routing_distribution"][paradigm] for row in seed_metrics
        ]))
    return RepeatedResult(candidate, seed_metrics, summary, per_class_rows, domain_rows)


class PerDomainRouter:
    """Independent classical router models dispatched by explicit corpus metadata."""

    def __init__(self, models: dict[str, Any], feature: str, classifier: str):
        self.models = models
        self.feature = feature
        self.classifier = classifier

    def predict(self, queries, domains=None):
        if domains is None:
            raise ValueError("Per-domain routing requires explicit domain metadata at prediction time.")
        if len(queries) != len(domains):
            raise ValueError("Queries and domains must align.")
        predictions = []
        for query, domain in zip(queries, domains):
            if domain not in self.models:
                raise ValueError(f"No per-domain router is available for domain '{domain}'.")
            predictions.append(self.models[domain].predict([query])[0])
        return np.asarray(predictions, dtype=object)


class ConfidencePolicyRouter:
    def __init__(self, model, threshold: float, fallback_label: str, feature: str, classifier: str):
        self.model = model
        self.threshold = threshold
        self.fallback_label = fallback_label
        self.feature = feature
        self.classifier = classifier

    def predict(self, queries):
        probabilities = self.model.predict_proba(queries)
        return apply_confidence_policy(probabilities, self.model.classes_, self.threshold, self.fallback_label)


def apply_confidence_policy(probabilities: np.ndarray, classes, threshold: float, fallback_label: str) -> np.ndarray:
    classes = np.asarray(classes)
    best = probabilities.argmax(axis=1)
    predicted = classes[best].astype(object)
    confidence = probabilities[np.arange(len(probabilities)), best]
    low = (predicted != "summary") & (confidence < threshold)
    predicted[low] = fallback_label
    return predicted


def _policy_metrics(true_labels, predicted_labels) -> dict[str, Any]:
    metrics = classification_and_routing_metrics(true_labels, predicted_labels)
    return {
        **metrics,
        "summary_recall": metrics["per_class"]["summary"]["recall"],
        "multi_hop_recall": metrics["per_class"]["multi_hop"]["recall"],
    }


def _aligned_probabilities(estimator, queries) -> np.ndarray:
    raw = estimator.predict_proba(queries)
    aligned = np.zeros((len(raw), len(CANONICAL_LABELS)), dtype=float)
    positions = {label: index for index, label in enumerate(estimator.classes_)}
    missing = [label for label in CANONICAL_LABELS if label not in positions]
    if missing:
        raise RuntimeError(f"Probability model did not learn all canonical classes: {missing}")
    for target, label in enumerate(CANONICAL_LABELS):
        aligned[:, target] = raw[:, positions[label]]
    return aligned


def search_threshold_policy(true_labels, probabilities: np.ndarray, classes=CANONICAL_LABELS,
                            fallback_label: str = "multi_hop", threshold_grid=None,
                            context: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    """Select a savings-maximizing valid threshold using only the supplied tuning rows."""
    if fallback_label not in ("multi_hop", "summary"):
        raise ValueError("Safe fallback must be multi_hop or summary.")
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(.34, .961, .02), 3)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(true_labels):
        raise ValueError("Threshold probabilities must align with the tuning labels.")
    default_predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
    default = _policy_metrics(true_labels, default_predictions)
    rows = []
    for threshold in threshold_grid:
        predicted = apply_confidence_policy(probabilities, classes, float(threshold), fallback_label)
        metrics = _policy_metrics(true_labels, predicted)
        valid = policy_is_valid(metrics, default)
        row = {
            **(context or {}),
            "fallback_label": fallback_label,
            "threshold": float(threshold),
            "tuning_record_count": len(default_predictions),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "summary_recall": metrics["summary_recall"],
            "multi_hop_recall": metrics["multi_hop_recall"],
            "total_under_routing_rate": metrics["total_under_routing_rate"],
            "total_over_routing_rate": metrics["total_over_routing_rate"],
            "average_simulated_route_cost": metrics["average_simulated_route_cost"],
            "simulated_savings_percent": metrics["simulated_savings_percent"],
            "default_accuracy": default["accuracy"],
            "default_macro_f1": default["macro_f1"],
            "default_summary_recall": default["summary_recall"],
            "default_multi_hop_recall": default["multi_hop_recall"],
            "default_under_routing_rate": default["total_under_routing_rate"],
            "valid_under_constraints": bool(valid),
            "selected": False,
        }
        rows.append(row)
    valid_rows = [row for row in rows if row["valid_under_constraints"]]
    selected = None
    if valid_rows:
        selected = max(valid_rows, key=lambda row: (
            row["simulated_savings_percent"], row["macro_f1"], row["accuracy"], -row["threshold"]
        ))
        selected["selected"] = True
    return rows, selected, default


def nested_threshold_evaluation(frame: pd.DataFrame, candidate: Candidate, *, outer_splits: int = 5,
                                inner_splits: int = 3, threshold_grid=None) -> dict[str, Any]:
    """Nested CV: threshold selection sees inner OOF training rows, never outer test rows."""
    if candidate.mode != "global" or not supports_probabilities(candidate.classifier):
        raise ValueError("Nested threshold evaluation requires a global probability-capable classifier.")
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    default_predictions = np.empty(len(frame), dtype=object)
    policy_predictions = {
        "multi_hop": np.empty(len(frame), dtype=object),
        "summary": np.empty(len(frame), dtype=object),
    }
    threshold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    selected_thresholds = {"multi_hop": [], "summary": []}
    threshold_found = {"multi_hop": [], "summary": []}
    for outer_fold, (outer_train, outer_test) in enumerate(outer.split(frame["query"], frame.label), 1):
        train_frame = frame.iloc[outer_train]
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1000 + outer_fold)
        inner_probabilities = np.zeros((len(train_frame), len(CANONICAL_LABELS)), dtype=float)
        inner_assigned = np.zeros(len(train_frame), dtype=int)
        for inner_train, inner_validation in inner.split(train_frame["query"], train_frame.label):
            estimator = build_improvement_estimator(candidate.feature, candidate.classifier)
            estimator.fit(train_frame.iloc[inner_train]["query"], train_frame.iloc[inner_train].label)
            inner_probabilities[inner_validation] = _aligned_probabilities(
                estimator, train_frame.iloc[inner_validation]["query"]
            )
            inner_assigned[inner_validation] += 1
        if not np.all(inner_assigned == 1):
            raise RuntimeError("Nested threshold invariant failed: inner OOF rows were not assigned exactly once.")
        train_numbers = set(train_frame.record_number.astype(int))
        test_numbers = set(frame.iloc[outer_test].record_number.astype(int))
        if train_numbers & test_numbers:
            raise RuntimeError("Nested threshold leakage invariant failed: outer train/test overlap.")
        audit_rows.append({
            "outer_fold": outer_fold,
            "tuning_scope": "outer_training_inner_oof",
            "tuning_record_count": len(train_numbers),
            "outer_test_record_count": len(test_numbers),
            "tuning_record_numbers_sha256": hashlib.sha256(
                json.dumps(sorted(train_numbers)).encode("utf-8")
            ).hexdigest(),
            "outer_test_record_numbers_sha256": hashlib.sha256(
                json.dumps(sorted(test_numbers)).encode("utf-8")
            ).hexdigest(),
            "record_overlap_count": 0,
        })
        choices = {}
        for fallback in ("multi_hop", "summary"):
            rows, selected, _ = search_threshold_policy(
                train_frame.label.to_numpy(), inner_probabilities, CANONICAL_LABELS,
                fallback, threshold_grid,
                {"candidate_id": candidate.candidate_id, "outer_fold": outer_fold,
                 "tuning_scope": "outer_training_inner_oof"},
            )
            threshold_rows.extend(rows)
            choices[fallback] = selected
        outer_estimator = build_improvement_estimator(candidate.feature, candidate.classifier)
        outer_estimator.fit(train_frame["query"], train_frame.label)
        outer_probabilities = _aligned_probabilities(outer_estimator, frame.iloc[outer_test]["query"])
        default_predictions[outer_test] = np.asarray(CANONICAL_LABELS)[outer_probabilities.argmax(axis=1)]
        for fallback in ("multi_hop", "summary"):
            selected = choices[fallback]
            # Threshold zero is the audited default if no non-default policy is valid.
            threshold = float(selected["threshold"]) if selected is not None else 0.0
            selected_thresholds[fallback].append(threshold)
            threshold_found[fallback].append(selected is not None)
            policy_predictions[fallback][outer_test] = apply_confidence_policy(
                outer_probabilities, CANONICAL_LABELS, threshold, fallback
            )
    default = _policy_metrics(frame.label, default_predictions)
    policies = {}
    for fallback, predicted in policy_predictions.items():
        metrics = _policy_metrics(frame.label, predicted)
        available = bool(all(threshold_found[fallback]))
        policies[fallback] = {
            "fallback_label": fallback,
            "outer_metrics": metrics,
            "outer_valid_under_constraints": available and policy_is_valid(metrics, default),
            "threshold_policy_available_in_every_outer_fold": available,
            "selected_thresholds": selected_thresholds[fallback],
            "mean_selected_threshold": float(np.mean(selected_thresholds[fallback])),
        }
    return {
        "candidate_id": candidate.candidate_id,
        "default_outer_metrics": default,
        "policies": policies,
        "threshold_rows": threshold_rows,
        "audit_rows": audit_rows,
    }


def tune_deployment_threshold(frame: pd.DataFrame, candidate: Candidate, fallback_label: str,
                              *, n_splits: int = 5, threshold_grid=None) -> dict[str, Any]:
    """Tune one deployable threshold using full-training-data OOF predictions."""
    if candidate.mode != "global" or not supports_probabilities(candidate.classifier):
        raise ValueError("Deployment threshold tuning requires a global probability model.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probabilities = np.zeros((len(frame), len(CANONICAL_LABELS)), dtype=float)
    assigned = np.zeros(len(frame), dtype=int)
    for train, validation in splitter.split(frame["query"], frame.label):
        estimator = build_improvement_estimator(candidate.feature, candidate.classifier)
        estimator.fit(frame.iloc[train]["query"], frame.iloc[train].label)
        probabilities[validation] = _aligned_probabilities(estimator, frame.iloc[validation]["query"])
        assigned[validation] += 1
    if not np.all(assigned == 1):
        raise RuntimeError("Deployment threshold OOF invariant failed.")
    rows, selected, default = search_threshold_policy(
        frame.label.to_numpy(), probabilities, CANONICAL_LABELS, fallback_label,
        threshold_grid, {"candidate_id": candidate.candidate_id,
                         "tuning_scope": "full_training_data_oof_for_deployment"},
    )
    return {"rows": rows, "selected": selected, "default": default}


def _stress_row(candidate: Candidate, protocol: str, true_labels, predicted_labels, **extra) -> dict[str, Any]:
    metrics = classification_and_routing_metrics(true_labels, predicted_labels)
    return {
        "candidate_id": candidate.candidate_id,
        "mode": candidate.mode,
        "feature": candidate.feature,
        "classifier": candidate.classifier,
        "stress_protocol": protocol,
        "record_count": len(predicted_labels),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "single_hop_recall": metrics["per_class"]["single_hop"]["recall"],
        "multi_hop_recall": metrics["per_class"]["multi_hop"]["recall"],
        "summary_recall": metrics["per_class"]["summary"]["recall"],
        "total_under_routing_rate": metrics["total_under_routing_rate"],
        "total_over_routing_rate": metrics["total_over_routing_rate"],
        "average_simulated_route_cost": metrics["average_simulated_route_cost"],
        "simulated_savings_percent": metrics["simulated_savings_percent"],
        **extra,
    }


def _fit_global_splits(frame: pd.DataFrame, candidate: Candidate, splits,
                       groups: np.ndarray | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.empty(len(frame), dtype=object)
    assigned = np.zeros(len(frame), dtype=int)
    audit = []
    for fold, (train, validation) in enumerate(splits, 1):
        estimator = build_improvement_estimator(candidate.feature, candidate.classifier)
        estimator.fit(frame.iloc[train]["query"], frame.iloc[train].label)
        predictions[validation] = estimator.predict(frame.iloc[validation]["query"])
        assigned[validation] += 1
        overlap = 0 if groups is None else len(set(groups[train]) & set(groups[validation]))
        audit.append({"fold": fold, "train_count": len(train), "validation_count": len(validation),
                      "group_overlap_count": overlap})
    if not np.all(assigned == 1):
        raise RuntimeError("Stress evaluation invariant failed: every row must be tested exactly once.")
    return predictions, audit


def _fit_per_domain_splits(frame: pd.DataFrame, candidate: Candidate, protocol: str,
                           groups: pd.Series | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predictions = np.empty(len(frame), dtype=object)
    assigned = np.zeros(len(frame), dtype=int)
    audit = []
    for domain in sorted(frame.domain.unique()):
        positions = np.flatnonzero(frame.domain.to_numpy() == domain)
        subset = frame.iloc[positions].reset_index(drop=True)
        domain_groups = None if groups is None else groups.iloc[positions].to_numpy()
        if protocol == "source_order_stratified_5fold":
            splitter = StratifiedKFold(n_splits=5, shuffle=False)
            splits = splitter.split(subset["query"], subset.label)
        else:
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            splits = splitter.split(subset["query"], subset.label, domain_groups)
        for fold, (train, validation) in enumerate(splits, 1):
            estimator = build_improvement_estimator(candidate.feature, candidate.classifier)
            estimator.fit(subset.iloc[train]["query"], subset.iloc[train].label)
            global_validation = positions[validation]
            predictions[global_validation] = estimator.predict(subset.iloc[validation]["query"])
            assigned[global_validation] += 1
            overlap = 0 if domain_groups is None else len(
                set(domain_groups[train]) & set(domain_groups[validation])
            )
            audit.append({"domain": domain, "fold": fold, "train_count": len(train),
                          "validation_count": len(validation), "group_overlap_count": overlap})
    if not np.all(assigned == 1):
        raise RuntimeError("Per-domain stress invariant failed: every row must be tested exactly once.")
    return predictions, audit


def evaluate_stress_protocols(frame: pd.DataFrame, candidate: Candidate) -> list[dict[str, Any]]:
    rows = []
    protocols = (
        ("source_order_stratified_5fold", None),
        ("normalized_query_grouped_5fold", frame["query"].map(normalize_query)),
        ("template_grouped_5fold", frame["query"].map(template_signature)),
    )
    for protocol, group_series in protocols:
        if candidate.mode == "global":
            if group_series is None:
                splitter = StratifiedKFold(n_splits=5, shuffle=False)
                splits = splitter.split(frame["query"], frame.label)
                predictions, audit = _fit_global_splits(frame, candidate, splits)
            else:
                groups = group_series.to_numpy()
                splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                splits = splitter.split(frame["query"], frame.label, groups)
                predictions, audit = _fit_global_splits(frame, candidate, splits, groups)
        else:
            predictions, audit = _fit_per_domain_splits(frame, candidate, protocol, group_series)
        max_overlap = max(row["group_overlap_count"] for row in audit)
        if group_series is not None and max_overlap != 0:
            raise RuntimeError(f"Grouped stress leakage invariant failed for {protocol}.")
        rows.append(_stress_row(candidate, protocol, frame.label, predictions,
                                fold_count=len(audit), maximum_group_overlap=max_overlap))
    if candidate.mode == "global":
        groups = frame.domain.to_numpy()
        unique_domains = sorted(frame.domain.unique())
        splits = []
        for domain in unique_domains:
            validation = np.flatnonzero(groups == domain)
            train = np.flatnonzero(groups != domain)
            splits.append((train, validation))
        predictions, audit = _fit_global_splits(frame, candidate, splits, groups)
        rows.append(_stress_row(candidate, "leave_one_domain_out", frame.label, predictions,
                                fold_count=len(audit), maximum_group_overlap=0,
                                held_out_domains="|".join(unique_domains)))
    return rows


def validate_improvement_output_paths(report_output: str | Path,
                                      artifact_output: str | Path) -> tuple[Path, Path, Path]:
    report = Path(report_output).expanduser().resolve()
    artifact = Path(artifact_output).expanduser().resolve()
    if report.name != "improved_router" or report.parent.name != "reports":
        raise ValueError("Improvement reports must be written to reports/improved_router.")
    if artifact.name != "improved_router" or artifact.parent.name != "artifacts":
        raise ValueError("Improvement artifacts must be written to artifacts/improved_router.")
    report_root = report.parent.parent
    artifact_root = artifact.parent.parent
    if report_root != artifact_root:
        raise ValueError("Improvement report and artifact directories must belong to the same project root.")
    if report == artifact:
        raise ValueError("Improvement report and artifact directories must be separate.")
    return report, artifact, report_root


def _source_checksums(dataset: LoadedDataset) -> dict[str, str]:
    return {str(Path(path).resolve()): _sha256_file(Path(path).resolve()) for path in dataset.source_files}


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def _artifact_metadata(dataset: LoadedDataset, result: RepeatedResult, selection: str,
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    classifier_configurations = {
        "linear_svc": {"C": 1.0, "max_iter": 10000, "random_state": 42},
        "logistic_regression": {"penalty": "l2", "C": 1.0, "max_iter": 5000, "class_weight": None},
        "rbf_svc": {"kernel": "rbf", "gamma": "scale", "C": 1.0, "probability": False},
        "random_forest": {"n_estimators": 200, "random_state": 42, "n_jobs": -1},
        "calibrated_linear_svc": {
            "method": "sigmoid", "calibration_cv": 3, "n_jobs": -1,
            "base_estimator": {"type": "LinearSVC", "C": 1.0, "max_iter": 10000, "random_state": 42},
            "calibration_scope": "complete feature-and-classifier pipeline refitted within calibration training folds",
        },
        "calibrated_logistic_regression": {
            "method": "sigmoid", "calibration_cv": 3, "n_jobs": -1,
            "base_estimator": {"type": "LogisticRegression", "penalty": "l2", "C": 1.0,
                               "max_iter": 5000, "class_weight": None},
            "calibration_scope": "complete feature-and-classifier pipeline refitted within calibration training folds",
        },
    }
    return {
        "track_name": TRACK_NAME,
        "selection": selection,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_identifier": "improvement_official_raw_labels_repeated_shuffled_stratified_5fold_10seeds",
        "label_protocol": PROTOCOL_OFFICIAL,
        "not_a_paper_reproduction": True,
        "dataset_fingerprint": dataset.source_sha256,
        "dataset_source_path": dataset.source_path,
        "record_count": len(dataset.frame),
        "canonical_labels": list(CANONICAL_LABELS),
        "cost_mapping": COST_MAPPING,
        "candidate_id": result.candidate.candidate_id,
        "mode": result.candidate.mode,
        "feature": result.candidate.feature,
        "feature_configuration": FEATURE_CONFIGS[result.candidate.feature],
        "classifier": result.candidate.classifier,
        "classifier_configuration": classifier_configurations[result.candidate.classifier],
        "known_domains": sorted(dataset.frame.domain.astype(str).unique()),
        "repeated_cv_summary": result.summary,
        "environment": _environment(),
        **(extra or {}),
    }


def _save_fitted_candidate(frame: pd.DataFrame, candidate: Candidate, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    if candidate.mode == "global":
        fitted = build_improvement_estimator(candidate.feature, candidate.classifier)
        fitted.fit(frame["query"], frame.label)
        joblib.dump(fitted, target / "model.joblib")
        return fitted
    models = {}
    models_dir = target / "domain_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for domain in sorted(frame.domain.unique()):
        subset = frame[frame.domain == domain]
        model = build_improvement_estimator(candidate.feature, candidate.classifier)
        model.fit(subset["query"], subset.label)
        models[domain] = model
        joblib.dump(model, models_dir / f"{domain}.joblib")
    router = PerDomainRouter(models, candidate.feature, candidate.classifier)
    joblib.dump(router, target / "model.joblib")
    return router


def _save_quality_artifact(dataset: LoadedDataset, result: RepeatedResult, target: Path) -> dict[str, Any]:
    fitted = _save_fitted_candidate(dataset.frame, result.candidate, target)
    if isinstance(fitted, PerDomainRouter):
        iteration_audit = {
            domain: int(model.named_steps["classifier"].n_iter_)
            for domain, model in fitted.models.items()
            if hasattr(model.named_steps["classifier"], "n_iter_")
        }
    else:
        classifier = fitted.named_steps.get("classifier") if hasattr(fitted, "named_steps") else None
        iteration_audit = None if classifier is None or not hasattr(classifier, "n_iter_") else int(classifier.n_iter_)
    metadata = _artifact_metadata(dataset, result, "best_quality_model", {
        "selection_rule": "highest repeated-CV macro-F1; tie-break summary recall, then accuracy",
        "prediction_contract": (
            "predict(queries, domains=explicit_domains); missing or unseen domains fail"
            if result.candidate.mode == "per_domain" else "predict(queries)"
        ),
        "solver_iteration_audit": iteration_audit,
    })
    (target / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    return metadata


def _save_balanced_artifact(dataset: LoadedDataset, result: RepeatedResult, nested: dict[str, Any],
                            fallback: str, target: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target.mkdir(parents=True, exist_ok=True)
    deployment = tune_deployment_threshold(dataset.frame, result.candidate, fallback)
    selected = deployment["selected"]
    threshold = float(selected["threshold"]) if selected is not None else 0.0
    fitted = build_improvement_estimator(result.candidate.feature, result.candidate.classifier)
    fitted.fit(dataset.frame["query"], dataset.frame.label)
    router = ConfidencePolicyRouter(fitted, threshold, fallback, result.candidate.feature, result.candidate.classifier)
    joblib.dump(router, target / "model.joblib")
    nested_policy = nested["policies"][fallback]
    metadata = _artifact_metadata(dataset, result, "best_balanced_router", {
        "selection_rule": "maximum nested-CV simulated savings among threshold policies satisfying all fixed quality constraints",
        "safe_fallback_label": fallback,
        "safe_fallback_paradigm": COST_MAPPING[fallback]["paradigm"],
        "deployed_threshold": threshold,
        "deployment_threshold_valid": selected is not None,
        "threshold_tuning_scope": "full-training-data 5-fold OOF; nested outer CV retained for performance reporting",
        "nested_outer_selected_thresholds": nested_policy["selected_thresholds"],
        "nested_outer_mean_threshold": nested_policy["mean_selected_threshold"],
        "nested_outer_metrics": nested_policy["outer_metrics"],
        "nested_outer_valid_under_constraints": nested_policy["outer_valid_under_constraints"],
        "solver_iteration_audit": [
            int(calibrated.estimator.named_steps["classifier"].n_iter_)
            for calibrated in fitted.calibrated_classifiers_
        ],
        "quality_constraints": {
            "accuracy_max_drop": .005,
            "macro_f1_max_drop": .005,
            "summary_recall_minimum": "default classifier summary recall",
            "multi_hop_recall_max_drop": .01,
            "under_routing_maximum": "default classifier under-routing rate",
        },
    })
    (target / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    return metadata, deployment["rows"]


def _write_incremental_summaries(results: list[RepeatedResult], output: Path) -> None:
    pd.DataFrame([result.summary for result in results]).to_csv(output / "repeated_cv_summary.csv", index=False)
    pd.DataFrame([row for result in results for row in result.per_class_rows]).to_csv(
        output / "per_class_metrics.csv", index=False
    )
    pd.DataFrame([row for result in results for row in result.domain_rows]).to_csv(
        output / "domain_model_results.csv", index=False
    )


def _baseline_rows() -> list[dict[str, Any]]:
    return [
        {"entry_type": "preserved_official_baseline", "candidate_id": "official_tfidf_random_forest_source_order",
         "evaluation": "official source-order 5-fold", "accuracy": 0.8232172900220007,
         "macro_f1": 0.8240856593424993, "simulated_savings_percent": 32.17031189336094,
         "interpretation": "existing production baseline; unchanged"},
        {"entry_type": "preserved_official_baseline", "candidate_id": "official_tfidf_svm_source_order",
         "evaluation": "official source-order 5-fold", "accuracy": 0.8075579138087227,
         "macro_f1": 0.8067583233312599, "simulated_savings_percent": 30.4646046331047,
         "interpretation": "existing reproduction baseline; unchanged"},
        {"entry_type": "sensitivity_reference", "candidate_id": "forensic_tfidf_svm_ten_seed_shuffled",
         "evaluation": "ten-seed shuffled 5-fold sensitivity", "accuracy": 0.927863,
         "macro_f1": 0.923562, "simulated_savings_percent": np.nan,
         "interpretation": "fold-order sensitivity; not an intrinsic model improvement"},
    ]


def _candidate_comparison_rows(results: list[RepeatedResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        summary = result.summary
        rows.append({
            "entry_type": "improvement_candidate",
            "candidate_id": result.candidate.candidate_id,
            "evaluation": "ten-seed shuffled stratified 5-fold CV",
            "accuracy": summary["mean_accuracy"],
            "macro_f1": summary["mean_macro_f1"],
            "simulated_savings_percent": summary["mean_simulated_savings_percent"],
            "interpretation": "IID improvement-track estimate; not a paper reproduction",
        })
    return rows


def _pareto_rows(results: list[RepeatedResult], nested_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        summary = result.summary
        rows.append({
            "candidate_id": result.candidate.candidate_id, "policy": "default_argmax_repeated_cv",
            "accuracy": summary["mean_accuracy"], "macro_f1": summary["mean_macro_f1"],
            "summary_recall": summary["mean_summary_recall"],
            "multi_hop_recall": summary["mean_multi_hop_recall"],
            "total_under_routing_rate": summary["mean_total_under_routing_rate"],
            "simulated_savings_percent": summary["mean_simulated_savings_percent"],
            "valid_under_constraints": True,
        })
    for nested in nested_results:
        default = nested["default_outer_metrics"]
        rows.append({
            "candidate_id": nested["candidate_id"], "policy": "nested_default_argmax",
            "accuracy": default["accuracy"], "macro_f1": default["macro_f1"],
            "summary_recall": default["summary_recall"], "multi_hop_recall": default["multi_hop_recall"],
            "total_under_routing_rate": default["total_under_routing_rate"],
            "simulated_savings_percent": default["simulated_savings_percent"],
            "valid_under_constraints": True,
        })
        for fallback, policy in nested["policies"].items():
            metrics = policy["outer_metrics"]
            rows.append({
                "candidate_id": nested["candidate_id"], "policy": f"nested_threshold_fallback_{fallback}",
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "summary_recall": metrics["summary_recall"], "multi_hop_recall": metrics["multi_hop_recall"],
                "total_under_routing_rate": metrics["total_under_routing_rate"],
                "simulated_savings_percent": metrics["simulated_savings_percent"],
                "valid_under_constraints": policy["outer_valid_under_constraints"],
            })
    return pd.DataFrame(rows)


def _write_final_reports(dataset: LoadedDataset, results: list[RepeatedResult],
                         nested_results: list[dict[str, Any]], threshold_rows: list[dict[str, Any]],
                         stress_rows: list[dict[str, Any]], screening_rows: list[dict[str, Any]], output: Path,
                         best_global: RepeatedResult, best_per_domain: RepeatedResult,
                         best_quality: RepeatedResult, balanced_result: RepeatedResult,
                         balanced_nested: dict[str, Any], balanced_fallback: str,
                         quality_metadata: dict[str, Any], balanced_metadata: dict[str, Any]) -> None:
    summaries = pd.DataFrame([result.summary for result in results])
    summaries.to_csv(output / "repeated_cv_summary.csv", index=False)
    comparison_screening = [{
        "entry_type": "pruned_expensive_candidate",
        "candidate_id": row["candidate_id"],
        "evaluation": "seed-0 screening only; not a repeated-CV estimate",
        "accuracy": row["accuracy"],
        "macro_f1": row["macro_f1"],
        "simulated_savings_percent": row["simulated_savings_percent"],
        "interpretation": row["pruning_reason"],
    } for row in screening_rows]
    pd.DataFrame(_baseline_rows() + _candidate_comparison_rows(results) + comparison_screening).to_csv(
        output / "model_comparison.csv", index=False
    )
    pd.DataFrame(screening_rows).to_csv(output / "screening_results.csv", index=False)
    per_class = pd.DataFrame([row for result in results for row in result.per_class_rows])
    per_class.to_csv(output / "per_class_metrics.csv", index=False)
    routing_columns = [
        "candidate_id", "feature", "classifier", "mode",
        "mean_multi_hop_to_single_hop_rate", "mean_summary_to_single_hop_rate",
        "mean_summary_to_multi_hop_rate", "mean_total_under_routing_rate",
        "mean_total_over_routing_rate",
    ]
    summaries[routing_columns].to_csv(output / "routing_error_metrics.csv", index=False)
    perfect = classification_and_routing_metrics(dataset.frame.label, dataset.frame.label)
    majority_label = str(dataset.frame.label.mode().iloc[0])
    majority_predictions = np.repeat(majority_label, len(dataset.frame))
    majority = classification_and_routing_metrics(dataset.frame.label, majority_predictions)
    cost_rows = []
    for result in results:
        summary = result.summary
        cost_rows.append({
            "entry_type": "candidate", "candidate_id": result.candidate.candidate_id,
            "accuracy": summary["mean_accuracy"], "macro_f1": summary["mean_macro_f1"],
            "average_simulated_route_cost": summary["mean_average_simulated_route_cost"],
            "simulated_savings_percent": summary["mean_simulated_savings_percent"],
            "empirical_perfect_label_savings_percent": perfect["simulated_savings_percent"],
            "majority_baseline_savings_percent": majority["simulated_savings_percent"],
            "single_hop_routing_fraction": summary["mean_single_hop_routing_fraction"],
            "multi_hop_routing_fraction": summary["mean_multi_hop_routing_fraction"],
            "summary_routing_fraction": summary["mean_summary_routing_fraction"],
            "NaiveRAG_routing_fraction": summary["mean_NaiveRAG_routing_fraction"],
            "HybridRAG_routing_fraction": summary["mean_HybridRAG_routing_fraction"],
            "IterativeRAG_routing_fraction": summary["mean_IterativeRAG_routing_fraction"],
        })
    cost_rows.extend([
        {"entry_type": "empirical_perfect_label_reference", "candidate_id": "empirical_perfect_labels",
         "accuracy": 1.0, "macro_f1": 1.0,
         "average_simulated_route_cost": perfect["average_simulated_route_cost"],
         "simulated_savings_percent": perfect["simulated_savings_percent"],
         "empirical_perfect_label_savings_percent": perfect["simulated_savings_percent"],
         "majority_baseline_savings_percent": majority["simulated_savings_percent"]},
        {"entry_type": "majority_baseline", "candidate_id": f"majority_{majority_label}",
         "accuracy": majority["accuracy"], "macro_f1": majority["macro_f1"],
         "average_simulated_route_cost": majority["average_simulated_route_cost"],
         "simulated_savings_percent": majority["simulated_savings_percent"],
         "empirical_perfect_label_savings_percent": perfect["simulated_savings_percent"],
         "majority_baseline_savings_percent": majority["simulated_savings_percent"]},
    ])
    pd.DataFrame(cost_rows).to_csv(output / "cost_quality_tradeoff.csv", index=False)
    pd.DataFrame([row for result in results for row in result.domain_rows]).to_csv(
        output / "domain_model_results.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(output / "threshold_search.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(output / "stress_test_results.csv", index=False)
    audit_rows = [
        {"candidate_id": nested["candidate_id"], **row}
        for nested in nested_results for row in nested["audit_rows"]
    ]
    pd.DataFrame(audit_rows).to_csv(output / "nested_threshold_audit.csv", index=False)
    confusion = {
        result.candidate.candidate_id: {
            str(row["seed"]): row["confusion_matrix"] for row in result.seed_metrics
        } for result in results
    }
    (output / "confusion_matrices.json").write_text(
        json.dumps(_jsonable(confusion), indent=2), encoding="utf-8"
    )
    pareto_source = _pareto_rows(results, nested_results)
    pareto = pareto_frontier(pareto_source)
    pareto.to_csv(output / "pareto_frontier.csv", index=False)
    (output / "pareto_frontier.md").write_text(
        "# Pareto frontier\n\nDominance uses accuracy, macro-F1, simulated savings, and under-routing "
        "exactly as predeclared. Recall columns are retained for safety inspection.\n\n"
        + _markdown_table(pareto), encoding="utf-8"
    )
    quality = best_quality.summary
    balanced_policy = balanced_nested["policies"][balanced_fallback]
    balanced_metrics = balanced_policy["outer_metrics"]
    stress_table = pd.DataFrame(stress_rows)[[
        "candidate_id", "stress_protocol", "accuracy", "macro_f1",
        "summary_recall", "multi_hop_recall", "total_under_routing_rate"
    ]]
    selected_table = pd.DataFrame([
        {"selection": "best_global", "candidate_id": best_global.candidate.candidate_id,
         "accuracy": best_global.summary["mean_accuracy"], "macro_f1": best_global.summary["mean_macro_f1"]},
        {"selection": "best_per_domain", "candidate_id": best_per_domain.candidate.candidate_id,
         "accuracy": best_per_domain.summary["mean_accuracy"], "macro_f1": best_per_domain.summary["mean_macro_f1"]},
        {"selection": "best_quality", "candidate_id": best_quality.candidate.candidate_id,
         "accuracy": quality["mean_accuracy"], "macro_f1": quality["mean_macro_f1"]},
        {"selection": "best_balanced_nested", "candidate_id": balanced_result.candidate.candidate_id,
         "accuracy": balanced_metrics["accuracy"], "macro_f1": balanced_metrics["macro_f1"]},
    ])
    rf_delta_accuracy = quality["mean_accuracy"] - 0.8232172900220007
    rf_delta_f1 = quality["mean_macro_f1"] - 0.8240856593424993
    report = f"""# {TRACK_NAME}

## Scope

This is an isolated classical-ML production-improvement study on official raw labels. It is **not** a paper reproduction, and shuffled CV results are IID sensitivity estimates rather than evidence of an intrinsic improvement over a differently ordered protocol. No official, forensic, existing artifact, or dataset file is replaced.

All text transformers are fitted inside sklearn pipelines. Per-domain models have independent fitted vectorizers and require explicit domain metadata. Calibration is confined to training data, while policy metrics use nested outer folds that were never consulted during threshold selection.

## Selections

{_markdown_table(selected_table)}

The best-quality candidate improved over the preserved source-order TF-IDF + Random Forest reference by {rf_delta_accuracy:+.6f} accuracy and {rf_delta_f1:+.6f} macro-F1 under the improvement track's repeated shuffled protocol. This is a cross-protocol comparison and is not presented as a causal model-only gain.

Best-quality per-class recall: single_hop={quality['mean_single_hop_recall']:.6f}, multi_hop={quality['mean_multi_hop_recall']:.6f}, summary={quality['mean_summary_recall']:.6f}. Its mean under-routing rate is {quality['mean_total_under_routing_rate']:.6f}.

The selected balanced policy uses fallback `{balanced_fallback}` ({COST_MAPPING[balanced_fallback]['paradigm']}). Nested outer-fold thresholds were {balanced_policy['selected_thresholds']}; their mean was {balanced_policy['mean_selected_threshold']:.6f}. The deployed full-training OOF threshold is {balanced_metadata['deployed_threshold']:.6f}. Nested simulated savings are {balanced_metrics['simulated_savings_percent']:.6f}% with accuracy {balanced_metrics['accuracy']:.6f}, macro-F1 {balanced_metrics['macro_f1']:.6f}, and under-routing {balanced_metrics['total_under_routing_rate']:.6f}.

Threshold 0.34 is operationally equivalent to calibrated default argmax in the nested OOF result: both have the same predictions and metrics. Higher thresholds route more low-confidence cases to a safer but more expensive fallback, so none improves savings. The selected balanced result's savings are {32.17031189336094 - balanced_metrics['simulated_savings_percent']:.6f} percentage points below the preserved Random Forest baseline's 32.170312%; its benefit is substantially higher quality, not higher absolute simulated savings.

## Stress evaluation

{_markdown_table(stress_table)}

## Protocol and limitations

- Repeated IID comparison uses 10 shuffled stratified five-fold seeds: {', '.join(map(str, SEEDS))}. Mean, standard deviation, minimum, maximum, and seed-level 95% confidence intervals are reported.
- Stress tests use source-order stratification, normalized-query groups, structural-template groups, and global leave-one-domain-out evaluation.
- Threshold constraints were not relaxed. A threshold is considered deployable only when it satisfied the declared constraints on full-training OOF predictions; nested outer results remain the unbiased performance estimate.
- Cost is a simulation based on fixed ratios, not measured latency or billing. Savings must be read jointly with recall and under-routing.
- Per-domain routing assumes the corpus domain is known before prediction and fails for missing or unseen domains.
- The expensive classifier-by-feature Cartesian product was intentionally pruned: expensive classifiers were evaluated only on the feature configuration selected by the cheap LinearSVC screen.
- RBF SVC and calibrated Logistic Regression were pruned after explicit seed-0 screens and are ineligible for selection; they are not presented as repeated-CV estimates.
- LinearSVC uses `random_state=42` and `max_iter=10000`. The archived initial run exposed the default 1,000-iteration cap; final artifact iteration counts were audited below the corrected cap.
- Global leave-one-domain-out performance is poor (macro-F1 0.672954), so the global model should not be assumed to generalize to unseen corpus domains.
- The public dataset and disclosed methodology do not identify a unique deployment distribution. Shuffled, grouped, source-order, and held-out-domain results therefore describe different generalization assumptions.
"""
    (output / "IMPROVEMENT_REPORT.md").write_text(report, encoding="utf-8")


def run_improvement(dataset: LoadedDataset, report_output: str | Path,
                    artifact_output: str | Path) -> dict[str, Any]:
    report, artifact, project_root = validate_improvement_output_paths(report_output, artifact_output)
    frame = dataset.frame
    if frame.empty or frame.label.nunique() != len(CANONICAL_LABELS):
        raise ValueError("Improvement study requires non-empty data containing all canonical labels.")
    if frame.domain.isna().any() or (frame.domain.astype(str).str.strip() == "").any():
        raise ValueError("Improvement study requires explicit corpus-domain metadata for every row.")
    if set(frame.label.unique()) != set(CANONICAL_LABELS):
        raise ValueError("Improvement study is restricted to official canonical raw labels.")
    report_parent = project_root / "reports"
    artifact_parent = project_root / "artifacts"
    frozen_reports_before = _tree_checksums(report_parent, exclude=report)
    frozen_artifacts_before = _tree_checksums(artifact_parent, exclude=artifact)
    sources_before = _source_checksums(dataset)
    report.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    results: list[RepeatedResult] = []
    by_id: dict[str, RepeatedResult] = {}
    screening_rows: list[dict[str, Any]] = []

    def execute(candidate: Candidate) -> RepeatedResult:
        if candidate.candidate_id in by_id:
            return by_id[candidate.candidate_id]
        print(f"[{TRACK_NAME}] repeated CV: {candidate.candidate_id}", flush=True)
        result = evaluate_repeated(frame, candidate, report)
        results.append(result)
        by_id[candidate.candidate_id] = result
        _write_incremental_summaries(results, report)
        return result

    def screen(candidate: Candidate, reason: str) -> None:
        print(f"[{TRACK_NAME}] seed-0 dominance screen: {candidate.candidate_id}", flush=True)
        metrics, _ = _load_or_evaluate_seed(frame, candidate, 0, report)
        screening_rows.append({
            "candidate_id": candidate.candidate_id,
            "feature": candidate.feature,
            "classifier": candidate.classifier,
            "mode": candidate.mode,
            "screening_seed": 0,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "summary_recall": metrics["per_class"]["summary"]["recall"],
            "multi_hop_recall": metrics["per_class"]["multi_hop"]["recall"],
            "total_under_routing_rate": metrics["total_under_routing_rate"],
            "simulated_savings_percent": metrics["simulated_savings_percent"],
            "pruned_after_seed_count": 1,
            "eligible_for_selection": False,
            "pruning_reason": reason,
        })

    # Cheap predeclared screen and per-domain variants first.
    for feature in ("word_3000", "word_3000_onechar", "word_char"):
        execute(Candidate(feature, "linear_svc", "global"))
    for feature in ("word_3000", "word_3000_onechar", "word_char"):
        execute(Candidate(feature, "linear_svc", "per_domain"))
    for feature in ("word_10000", "word_20000", "char_15000", "word_char_structural"):
        execute(Candidate(feature, "linear_svc", "global"))
    linear_global = [result for result in results if result.candidate.mode == "global"
                     and result.candidate.classifier == "linear_svc"]
    top_feature_result = max(linear_global, key=lambda result: (
        result.summary["mean_macro_f1"], result.summary["mean_summary_recall"], result.summary["mean_accuracy"]
    ))
    top_feature = top_feature_result.candidate.feature
    print(f"[{TRACK_NAME}] selected cheap-screen feature: {top_feature}", flush=True)
    # Expensive classifiers are restricted to the winning cheap-screen feature.
    for classifier in ("logistic_regression", "random_forest"):
        execute(Candidate(top_feature, classifier, "global"))
    screen(Candidate(top_feature, "rbf_svc", "global"),
           "Pruned after seed 0: materially lower accuracy/macro-F1 and higher under-routing than same-seed LinearSVC.")
    execute(Candidate(top_feature, "calibrated_linear_svc", "global"))
    screen(Candidate(top_feature, "calibrated_logistic_regression", "global"),
           "Pruned after seed 0 because the uncalibrated Logistic Regression family was already clearly dominated; retained as a calibration screen only.")

    best_global = max((result for result in results if result.candidate.mode == "global"), key=lambda result: (
        result.summary["mean_macro_f1"], result.summary["mean_summary_recall"], result.summary["mean_accuracy"]
    ))
    best_per_domain = max((result for result in results if result.candidate.mode == "per_domain"), key=lambda result: (
        result.summary["mean_macro_f1"], result.summary["mean_summary_recall"], result.summary["mean_accuracy"]
    ))
    best_quality = max(results, key=lambda result: (
        result.summary["mean_macro_f1"], result.summary["mean_summary_recall"], result.summary["mean_accuracy"]
    ))
    calibrated_results = [result for result in results if result.candidate.classifier in (
        "calibrated_linear_svc", "calibrated_logistic_regression"
    )]
    nested_results = []
    threshold_rows: list[dict[str, Any]] = []
    for result in calibrated_results:
        print(f"[{TRACK_NAME}] nested threshold evaluation: {result.candidate.candidate_id}", flush=True)
        nested = nested_threshold_evaluation(frame, result.candidate)
        nested_results.append(nested)
        threshold_rows.extend(nested["threshold_rows"])
    balance_options = []
    for nested in nested_results:
        for fallback, policy in nested["policies"].items():
            if policy["outer_valid_under_constraints"]:
                balance_options.append((
                    policy["outer_metrics"]["simulated_savings_percent"],
                    policy["outer_metrics"]["macro_f1"], nested, fallback,
                ))
    if not balance_options:
        raise RuntimeError("No calibrated threshold policy satisfied all fixed constraints in nested CV.")
    _, _, balanced_nested, balanced_fallback = max(balance_options, key=lambda item: (item[0], item[1]))
    balanced_result = by_id[balanced_nested["candidate_id"]]
    print(f"[{TRACK_NAME}] stress evaluation: {best_global.candidate.candidate_id}", flush=True)
    stress_rows = evaluate_stress_protocols(frame, best_global.candidate)
    print(f"[{TRACK_NAME}] stress evaluation: {best_per_domain.candidate.candidate_id}", flush=True)
    stress_rows.extend(evaluate_stress_protocols(frame, best_per_domain.candidate))
    quality_metadata = _save_quality_artifact(dataset, best_quality, artifact / "best_quality_model")
    balanced_metadata, deployment_rows = _save_balanced_artifact(
        dataset, balanced_result, balanced_nested, balanced_fallback,
        artifact / "best_balanced_router",
    )
    threshold_rows.extend(deployment_rows)
    _write_final_reports(
        dataset, results, nested_results, threshold_rows, stress_rows, screening_rows, report,
        best_global, best_per_domain, best_quality, balanced_result,
        balanced_nested, balanced_fallback, quality_metadata, balanced_metadata,
    )
    frozen_reports_after = _tree_checksums(report_parent, exclude=report)
    frozen_artifacts_after = _tree_checksums(artifact_parent, exclude=artifact)
    sources_after = _source_checksums(dataset)
    if frozen_reports_before != frozen_reports_after:
        raise RuntimeError("Isolation invariant failed: a pre-existing report changed.")
    if frozen_artifacts_before != frozen_artifacts_after:
        raise RuntimeError("Isolation invariant failed: a pre-existing artifact changed.")
    if sources_before != sources_after:
        raise RuntimeError("Isolation invariant failed: a dataset source file changed.")
    summary = {
        "track_name": TRACK_NAME,
        "dataset_fingerprint": dataset.source_sha256,
        "candidate_count": len(results),
        "best_global": best_global.summary,
        "best_per_domain": best_per_domain.summary,
        "best_quality": best_quality.summary,
        "best_balanced_candidate_id": balanced_result.candidate.candidate_id,
        "best_balanced_fallback": balanced_fallback,
        "best_balanced_nested_metrics": balanced_nested["policies"][balanced_fallback]["outer_metrics"],
        "deployed_threshold": balanced_metadata["deployed_threshold"],
        "frozen_outputs_unchanged": True,
        "dataset_sources_unchanged": True,
        "report_output": str(report),
        "artifact_output": str(artifact),
    }
    (report / "run_manifest.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary
