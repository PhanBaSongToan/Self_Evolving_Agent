from __future__ import annotations

import hashlib
import io
import json
import math
import pickle
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .config import CANONICAL_LABELS, COST_MAPPING, PROTOCOL_OFFICIAL
from .data import LoadedDataset
from .extended_models import (
    ALPHAS, NB_ALPHAS, NBSVM_CS, CalibratedScoreAveragingClassifier,
    OOFStackingTextClassifier, VotingTextClassifier, aligned_scores, build_extended_estimator,
)
from .forensic import normalize_query, template_signature
from .improvement import SEEDS, _jsonable, _sha256_file, _tree_checksums
from .improvement_metrics import classification_and_routing_metrics, pareto_frontier
from .reporting import _markdown_table

TRACK_NAME = "Extended Non-Deep-Learning Model Benchmark"
HALVING_SEEDS = (0, 21, 42)
FEATURE_SCREEN_SEEDS = (0, 42)
LAMBDA_VALUES = (0.0, .05, .10, .20, .30, .50, .75, 1.0)
LOSS_SCALES = (.5, 1.0, 2.0)
QUALITY_LOSS = np.asarray([
    [0.0, .3, .6],
    [4.0, 0.0, .4],
    [8.0, 4.0, 0.0],
])
ROUTE_COSTS = np.asarray([COST_MAPPING[label]["cost_ratio"] for label in CANONICAL_LABELS], dtype=float)


def _clean(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(character if character.isalnum() or character == "_" else "_" for character in text)


@dataclass(frozen=True)
class ExtendedCandidate:
    feature: str
    model: str
    params_json: str = "{}"
    mode: str = "global"

    @classmethod
    def create(cls, feature: str, model: str, params: dict[str, Any] | None = None, mode: str = "global"):
        return cls(feature, model, json.dumps(params or {}, sort_keys=True, separators=(",", ":")), mode)

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.params_json)

    @property
    def candidate_id(self) -> str:
        suffix = ""
        if self.params:
            readable = "__".join(f"{_clean(key)}_{_clean(value)}" for key, value in sorted(self.params.items()))
            if len(readable) > 100 or any(isinstance(value, (list, dict)) for value in self.params.values()):
                readable = "cfg_" + hashlib.sha256(self.params_json.encode("utf-8")).hexdigest()[:12]
            suffix = "__" + readable
        return f"{self.mode}__{self.feature}__{self.model}{suffix}"


@dataclass
class ExtendedResult:
    candidate: ExtendedCandidate
    seed_metrics: list[dict[str, Any]]
    summary: dict[str, Any]
    per_class_rows: list[dict[str, Any]]
    domain_rows: list[dict[str, Any]]
    selectable: bool = True
    stage: str = "repeated_cv"


def validate_extended_paths(report_output: str | Path, artifact_output: str | Path) -> tuple[Path, Path, Path]:
    report = Path(report_output).expanduser().resolve()
    artifact = Path(artifact_output).expanduser().resolve()
    if report.name != "extended_classical_benchmark" or report.parent.name != "reports":
        raise ValueError("Extended benchmark reports must be written to reports/extended_classical_benchmark.")
    if artifact.name != "extended_classical_benchmark" or artifact.parent.name != "artifacts":
        raise ValueError("Extended benchmark artifacts must be written to artifacts/extended_classical_benchmark.")
    if report.parent.parent != artifact.parent.parent:
        raise ValueError("Extended benchmark report and artifact paths must share a project root.")
    return report, artifact, report.parent.parent


def _source_checksums(dataset: LoadedDataset) -> dict[str, str]:
    return {str(Path(path).resolve()): _sha256_file(Path(path).resolve()) for path in dataset.source_files}


def _candidate_builder(candidate: ExtendedCandidate, registry: dict[str, ExtendedCandidate]):
    params = candidate.params
    if candidate.model.startswith("ensemble_"):
        base_ids = params["base_ids"]
        bases = [_candidate_builder(registry[candidate_id], registry) for candidate_id in base_ids]
        if candidate.model == "ensemble_calibrated_score_average":
            return CalibratedScoreAveragingClassifier(bases)
        if candidate.model == "ensemble_stack_logistic":
            return OOFStackingTextClassifier(bases, meta_model="logistic_regression", cv=3)
        if candidate.model == "ensemble_stack_ridge":
            return OOFStackingTextClassifier(bases, meta_model="ridge", cv=3)
        if candidate.model == "ensemble_hard":
            return VotingTextClassifier(bases)
        if candidate.model == "ensemble_weighted":
            return VotingTextClassifier(bases, weights=params["weights"])
        raise ValueError(f"Unknown ensemble model: {candidate.model}")
    return build_extended_estimator(candidate.feature, candidate.model, params)


def _estimate_size(estimator) -> int:
    try:
        return len(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        buffer = io.BytesIO()
        joblib.dump(estimator, buffer, compress=0)
        return buffer.tell()


def _evaluate_global_seed(frame: pd.DataFrame, candidate: ExtendedCandidate, seed: int,
                          registry: dict[str, ExtendedCandidate]) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    predictions = np.empty(len(frame), dtype=object)
    scores = np.zeros((len(frame), len(CANONICAL_LABELS)), dtype=float)
    folds = np.zeros(len(frame), dtype=int)
    fit_seconds = 0.0
    predict_seconds = 0.0
    size_bytes = 0
    for fold, (train, validation) in enumerate(splitter.split(frame["query"], frame.label), 1):
        estimator = _candidate_builder(candidate, registry)
        started = time.perf_counter()
        estimator.fit(frame.iloc[train]["query"], frame.iloc[train].label)
        fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        predictions[validation] = estimator.predict(frame.iloc[validation]["query"])
        scores[validation] = aligned_scores(estimator, frame.iloc[validation]["query"])
        predict_seconds += time.perf_counter() - started
        folds[validation] = fold
        if fold == 5:
            size_bytes = _estimate_size(estimator)
    metrics = classification_and_routing_metrics(frame.label, predictions)
    metrics.update({"fit_time_seconds": fit_seconds, "prediction_time_seconds": predict_seconds,
                    "prediction_latency_ms_per_record": predict_seconds / len(frame) * 1000,
                    "estimated_artifact_size_bytes": size_bytes})
    oof = frame[["record_number", "id", "domain", "raw_label", "label", "query"]].copy()
    oof["predicted_label"] = predictions
    for column, label in enumerate(CANONICAL_LABELS):
        oof[f"score_{label}"] = scores[:, column]
    oof["fold"] = folds
    oof["seed"] = seed
    domains = []
    for domain, group in oof.groupby("domain", sort=True):
        values = classification_and_routing_metrics(group.label, group.predicted_label)
        domains.append({"candidate_id": candidate.candidate_id, "seed": seed, "domain": domain,
                        "accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
                        "summary_recall": values["per_class"]["summary"]["recall"],
                        "under_routing_rate": values["total_under_routing_rate"], "record_count": len(group)})
    return oof, metrics, domains


def _evaluate_per_domain_seed(frame: pd.DataFrame, candidate: ExtendedCandidate, seed: int,
                              registry: dict[str, ExtendedCandidate]) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    predictions = np.empty(len(frame), dtype=object)
    scores = np.zeros((len(frame), len(CANONICAL_LABELS)), dtype=float)
    folds = np.zeros(len(frame), dtype=int)
    fit_seconds = 0.0
    predict_seconds = 0.0
    size_bytes = 0
    domain_rows = []
    assigned = np.zeros(len(frame), dtype=int)
    for domain in sorted(frame.domain.unique()):
        positions = np.flatnonzero(frame.domain.to_numpy() == domain)
        subset = frame.iloc[positions].reset_index(drop=True)
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        domain_predictions = np.empty(len(subset), dtype=object)
        for fold, (train, validation) in enumerate(splitter.split(subset["query"], subset.label), 1):
            estimator = _candidate_builder(ExtendedCandidate(candidate.feature, candidate.model,
                                                               candidate.params_json, "global"), registry)
            started = time.perf_counter()
            estimator.fit(subset.iloc[train]["query"], subset.iloc[train].label)
            fit_seconds += time.perf_counter() - started
            started = time.perf_counter()
            predicted = estimator.predict(subset.iloc[validation]["query"])
            fold_scores = aligned_scores(estimator, subset.iloc[validation]["query"])
            predict_seconds += time.perf_counter() - started
            global_validation = positions[validation]
            predictions[global_validation] = predicted
            domain_predictions[validation] = predicted
            scores[global_validation] = fold_scores
            folds[global_validation] = fold
            assigned[global_validation] += 1
            if fold == 5:
                size_bytes += _estimate_size(estimator)
        values = classification_and_routing_metrics(subset.label, domain_predictions)
        domain_rows.append({"candidate_id": candidate.candidate_id, "seed": seed, "domain": domain,
                            "accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
                            "summary_recall": values["per_class"]["summary"]["recall"],
                            "under_routing_rate": values["total_under_routing_rate"], "record_count": len(subset),
                            "independent_vectorizer_scope": domain})
    if not np.all(assigned == 1):
        raise RuntimeError("Per-domain extended CV failed one-prediction-per-row invariant.")
    metrics = classification_and_routing_metrics(frame.label, predictions)
    metrics.update({"fit_time_seconds": fit_seconds, "prediction_time_seconds": predict_seconds,
                    "prediction_latency_ms_per_record": predict_seconds / len(frame) * 1000,
                    "estimated_artifact_size_bytes": size_bytes})
    oof = frame[["record_number", "id", "domain", "raw_label", "label", "query"]].copy()
    oof["predicted_label"] = predictions
    for column, label in enumerate(CANONICAL_LABELS):
        oof[f"score_{label}"] = scores[:, column]
    oof["fold"] = folds
    oof["seed"] = seed
    return oof, metrics, domain_rows


def _seed_paths(output: Path, candidate: ExtendedCandidate, seed: int) -> tuple[Path, Path, Path]:
    directory = output / "experiments" / candidate.candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    return (directory / f"seed_{seed}_metrics.json", directory / f"seed_{seed}_oof.csv",
            directory / f"seed_{seed}_domains.csv")


def _load_or_evaluate_seed(frame: pd.DataFrame, candidate: ExtendedCandidate, seed: int, output: Path,
                           registry: dict[str, ExtendedCandidate]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics_path, oof_path, domain_path = _seed_paths(output, candidate, seed)
    if metrics_path.exists() and oof_path.exists() and domain_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8")), pd.read_csv(domain_path).to_dict(orient="records")
    if candidate.mode == "per_domain":
        oof, metrics, domains = _evaluate_per_domain_seed(frame, candidate, seed, registry)
    else:
        oof, metrics, domains = _evaluate_global_seed(frame, candidate, seed, registry)
    metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
    oof.to_csv(oof_path, index=False)
    pd.DataFrame(domains).to_csv(domain_path, index=False)
    return metrics, domains


def _ci95(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return float(array[0]), float(array[0])
    half = 1.96 * array.std(ddof=1) / math.sqrt(len(array))
    return float(array.mean() - half), float(array.mean() + half)


def evaluate_candidate(frame: pd.DataFrame, candidate: ExtendedCandidate, output: Path,
                       registry: dict[str, ExtendedCandidate], seeds: tuple[int, ...] = SEEDS,
                       *, selectable: bool = True, stage: str = "repeated_cv") -> ExtendedResult:
    registry[candidate.candidate_id] = candidate
    metrics_rows = []
    per_class = []
    domains = []
    for seed in seeds:
        metrics, seed_domains = _load_or_evaluate_seed(frame, candidate, seed, output, registry)
        metrics_rows.append({"seed": seed, **metrics})
        domains.extend(seed_domains)
        for label in CANONICAL_LABELS:
            values = metrics["per_class"][label]
            per_class.append({"candidate_id": candidate.candidate_id, "seed": seed, "label": label,
                              "precision": values["precision"], "recall": values["recall"],
                              "f1": values["f1-score"], "support": values["support"], "stage": stage})
    accuracies = [row["accuracy"] for row in metrics_rows]
    macro_f1 = [row["macro_f1"] for row in metrics_rows]
    accuracy_ci = _ci95(accuracies)
    f1_ci = _ci95(macro_f1)
    summary = {
        "candidate_id": candidate.candidate_id, "mode": candidate.mode, "feature": candidate.feature,
        "model": candidate.model, "params_json": candidate.params_json, "seed_count": len(seeds),
        "seeds": "|".join(map(str, seeds)), "selectable": selectable, "stage": stage,
        "mean_accuracy": float(np.mean(accuracies)), "std_accuracy": float(np.std(accuracies)),
        "min_accuracy": float(np.min(accuracies)), "max_accuracy": float(np.max(accuracies)),
        "accuracy_ci95_low": accuracy_ci[0], "accuracy_ci95_high": accuracy_ci[1],
        "mean_macro_f1": float(np.mean(macro_f1)), "std_macro_f1": float(np.std(macro_f1)),
        "min_macro_f1": float(np.min(macro_f1)), "max_macro_f1": float(np.max(macro_f1)),
        "macro_f1_ci95_low": f1_ci[0], "macro_f1_ci95_high": f1_ci[1],
    }
    for field in ("macro_precision", "macro_recall", "total_under_routing_rate", "total_over_routing_rate",
                  "average_simulated_route_cost", "simulated_savings_percent", "fit_time_seconds",
                  "prediction_time_seconds", "prediction_latency_ms_per_record", "estimated_artifact_size_bytes"):
        summary[f"mean_{field}"] = float(np.mean([row[field] for row in metrics_rows]))
    for label in CANONICAL_LABELS:
        summary[f"mean_{label}_recall"] = float(np.mean([row["per_class"][label]["recall"] for row in metrics_rows]))
        summary[f"mean_{label}_f1"] = float(np.mean([row["per_class"][label]["f1-score"] for row in metrics_rows]))
    return ExtendedResult(candidate, metrics_rows, summary, per_class, domains, selectable, stage)


def load_oof(output: Path, candidate: ExtendedCandidate, seed: int) -> pd.DataFrame:
    return pd.read_csv(_seed_paths(output, candidate, seed)[1])


def derive_voting_result(frame: pd.DataFrame, candidate: ExtendedCandidate, bases: list[ExtendedResult],
                         output: Path, registry: dict[str, ExtendedCandidate], weights=None) -> ExtendedResult:
    registry[candidate.candidate_id] = candidate
    for seed_index, seed in enumerate(SEEDS):
        metrics_path, oof_path, domain_path = _seed_paths(output, candidate, seed)
        if metrics_path.exists() and oof_path.exists() and domain_path.exists():
            continue
        base_frames = [load_oof(output, result.candidate, seed).sort_values("record_number").reset_index(drop=True)
                       for result in bases]
        reference = base_frames[0]
        if any(not reference.record_number.equals(other.record_number) for other in base_frames[1:]):
            raise RuntimeError("Voting candidates do not share identical OOF rows.")
        vote_weights = np.ones(len(bases)) if weights is None else np.asarray(weights, dtype=float)
        votes = np.zeros((len(reference), len(CANONICAL_LABELS)), dtype=float)
        for weight, base in zip(vote_weights, base_frames):
            for column, label in enumerate(CANONICAL_LABELS):
                votes[:, column] += weight * (base.predicted_label.to_numpy() == label)
        predicted = np.asarray(CANONICAL_LABELS)[votes.argmax(axis=1)]
        metrics = classification_and_routing_metrics(reference.label, predicted)
        metrics.update({
            "fit_time_seconds": float(sum(result.seed_metrics[seed_index]["fit_time_seconds"] for result in bases)),
            "prediction_time_seconds": 0.0,
            "prediction_latency_ms_per_record": 0.0,
            "estimated_artifact_size_bytes": float(sum(
                result.seed_metrics[seed_index]["estimated_artifact_size_bytes"] for result in bases
            )),
        })
        oof = reference[["record_number", "id", "domain", "raw_label", "label", "query", "fold", "seed"]].copy()
        oof["predicted_label"] = predicted
        probabilities = votes / vote_weights.sum()
        for column, label in enumerate(CANONICAL_LABELS):
            oof[f"score_{label}"] = probabilities[:, column]
        domains = []
        for domain, group in oof.groupby("domain", sort=True):
            values = classification_and_routing_metrics(group.label, group.predicted_label)
            domains.append({"candidate_id": candidate.candidate_id, "seed": seed, "domain": domain,
                            "accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
                            "summary_recall": values["per_class"]["summary"]["recall"],
                            "under_routing_rate": values["total_under_routing_rate"], "record_count": len(group)})
        metrics_path.write_text(json.dumps(_jsonable(metrics), indent=2), encoding="utf-8")
        oof.to_csv(oof_path, index=False)
        pd.DataFrame(domains).to_csv(domain_path, index=False)
    return evaluate_candidate(frame, candidate, output, registry, selectable=True, stage="ensemble_repeated_cv")


def error_diversity(results: list[ExtendedResult], output: Path) -> pd.DataFrame:
    rows = []
    complexity = {"single_hop": 0, "multi_hop": 1, "summary": 2}
    for first, second in combinations(results, 2):
        aggregate = []
        for seed in SEEDS:
            a = load_oof(output, first.candidate, seed).sort_values("record_number").reset_index(drop=True)
            b = load_oof(output, second.candidate, seed).sort_values("record_number").reset_index(drop=True)
            if not a.record_number.equals(b.record_number):
                raise RuntimeError("Diversity candidates do not align.")
            true = a.label.to_numpy()
            pa = a.predicted_label.to_numpy()
            pb = b.predicted_label.to_numpy()
            correct_a = pa == true
            correct_b = pb == true
            under_a = np.asarray([complexity[p] < complexity[t] for p, t in zip(pa, true)])
            under_b = np.asarray([complexity[p] < complexity[t] for p, t in zip(pb, true)])
            row = {
                "seed": seed,
                "disagreement_rate": float(np.mean(pa != pb)),
                "both_correct_rate": float(np.mean(correct_a & correct_b)),
                "both_wrong_rate": float(np.mean(~correct_a & ~correct_b)),
                "model_a_only_correct_rate": float(np.mean(correct_a & ~correct_b)),
                "model_b_only_correct_rate": float(np.mean(~correct_a & correct_b)),
                "under_routing_disagreement_rate": float(np.mean(under_a != under_b)),
            }
            for label in CANONICAL_LABELS:
                mask = true == label
                row[f"{label}_disagreement_rate"] = float(np.mean(pa[mask] != pb[mask]))
            aggregate.append(row)
        rows.append({
            "model_a": first.candidate.candidate_id, "model_b": second.candidate.candidate_id,
            **{f"mean_{key}": float(np.mean([row[key] for row in aggregate]))
               for key in aggregate[0] if key != "seed"},
        })
    return pd.DataFrame(rows)


def _stress_metrics(candidate: ExtendedCandidate, protocol: str, true, predicted, fit_time: float,
                    predict_time: float, **extra) -> dict[str, Any]:
    metrics = classification_and_routing_metrics(true, predicted)
    return {
        "candidate_id": candidate.candidate_id, "mode": candidate.mode,
        "stress_protocol": protocol, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"], "macro_recall": metrics["macro_recall"],
        "single_hop_recall": metrics["per_class"]["single_hop"]["recall"],
        "multi_hop_recall": metrics["per_class"]["multi_hop"]["recall"],
        "summary_recall": metrics["per_class"]["summary"]["recall"],
        "under_routing_rate": metrics["total_under_routing_rate"],
        "over_routing_rate": metrics["total_over_routing_rate"],
        "simulated_savings_percent": metrics["simulated_savings_percent"],
        "fit_time_seconds": fit_time, "prediction_time_seconds": predict_time, **extra,
    }


def evaluate_stress(frame: pd.DataFrame, candidate: ExtendedCandidate,
                    registry: dict[str, ExtendedCandidate]) -> list[dict[str, Any]]:
    rows = []
    protocols = (
        ("source_order_5fold", None),
        ("normalized_query_grouped_5fold", frame["query"].map(normalize_query).to_numpy()),
        ("template_grouped_5fold", frame["query"].map(template_signature).to_numpy()),
    )
    for protocol, groups in protocols:
        predicted = np.empty(len(frame), dtype=object)
        assigned = np.zeros(len(frame), dtype=int)
        fit_time = 0.0
        predict_time = 0.0
        overlap_max = 0
        if candidate.mode == "global":
            if groups is None:
                splits = StratifiedKFold(n_splits=5, shuffle=False).split(frame["query"], frame.label)
            else:
                splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
                    frame["query"], frame.label, groups
                )
            for train, validation in splits:
                estimator = _candidate_builder(candidate, registry)
                started = time.perf_counter(); estimator.fit(frame.iloc[train]["query"], frame.iloc[train].label)
                fit_time += time.perf_counter() - started
                started = time.perf_counter(); predicted[validation] = estimator.predict(frame.iloc[validation]["query"])
                predict_time += time.perf_counter() - started
                assigned[validation] += 1
                if groups is not None:
                    overlap_max = max(overlap_max, len(set(groups[train]) & set(groups[validation])))
        else:
            for domain in sorted(frame.domain.unique()):
                positions = np.flatnonzero(frame.domain.to_numpy() == domain)
                subset = frame.iloc[positions].reset_index(drop=True)
                domain_groups = None if groups is None else groups[positions]
                if domain_groups is None:
                    splits = StratifiedKFold(n_splits=5, shuffle=False).split(subset["query"], subset.label)
                else:
                    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
                        subset["query"], subset.label, domain_groups
                    )
                for train, validation in splits:
                    global_candidate = ExtendedCandidate(candidate.feature, candidate.model, candidate.params_json, "global")
                    estimator = _candidate_builder(global_candidate, registry)
                    started = time.perf_counter(); estimator.fit(subset.iloc[train]["query"], subset.iloc[train].label)
                    fit_time += time.perf_counter() - started
                    started = time.perf_counter(); values = estimator.predict(subset.iloc[validation]["query"])
                    predict_time += time.perf_counter() - started
                    global_validation = positions[validation]
                    predicted[global_validation] = values
                    assigned[global_validation] += 1
                    if domain_groups is not None:
                        overlap_max = max(overlap_max, len(set(domain_groups[train]) & set(domain_groups[validation])))
        if not np.all(assigned == 1) or (groups is not None and overlap_max):
            raise RuntimeError(f"Stress invariant failed for {candidate.candidate_id}: {protocol}")
        rows.append(_stress_metrics(candidate, protocol, frame.label, predicted, fit_time, predict_time,
                                    maximum_group_overlap=overlap_max))
    if candidate.mode == "global":
        predicted = np.empty(len(frame), dtype=object)
        fit_time = predict_time = 0.0
        for domain in sorted(frame.domain.unique()):
            train = np.flatnonzero(frame.domain.to_numpy() != domain)
            validation = np.flatnonzero(frame.domain.to_numpy() == domain)
            estimator = _candidate_builder(candidate, registry)
            started = time.perf_counter(); estimator.fit(frame.iloc[train]["query"], frame.iloc[train].label)
            fit_time += time.perf_counter() - started
            started = time.perf_counter(); predicted[validation] = estimator.predict(frame.iloc[validation]["query"])
            predict_time += time.perf_counter() - started
        rows.append(_stress_metrics(candidate, "leave_one_domain_out", frame.label, predicted,
                                    fit_time, predict_time, held_out_domain_count=frame.domain.nunique()))
    return rows


def _risk_predictions(probabilities: np.ndarray, loss_scale: float, lambda_cost: float) -> np.ndarray:
    expected = probabilities @ (QUALITY_LOSS * loss_scale) + lambda_cost * ROUTE_COSTS.reshape(1, -1)
    return np.asarray(CANONICAL_LABELS)[expected.argmin(axis=1)]


def _risk_policy_valid(candidate: dict[str, Any], default: dict[str, Any]) -> bool:
    return (candidate["accuracy"] >= default["accuracy"] - .005
            and candidate["macro_f1"] >= default["macro_f1"] - .005
            and candidate["summary_recall"] >= default["summary_recall"]
            and candidate["multi_hop_recall"] >= default["multi_hop_recall"] - .005
            and candidate["total_under_routing_rate"] <= default["total_under_routing_rate"])


def _policy_metrics(true, predicted) -> dict[str, Any]:
    metrics = classification_and_routing_metrics(true, predicted)
    metrics["summary_recall"] = metrics["per_class"]["summary"]["recall"]
    metrics["multi_hop_recall"] = metrics["per_class"]["multi_hop"]["recall"]
    return metrics


def search_expected_risk(true, probabilities: np.ndarray, context=None):
    default_predictions = np.asarray(CANONICAL_LABELS)[probabilities.argmax(axis=1)]
    default = _policy_metrics(true, default_predictions)
    rows = []
    for scale in LOSS_SCALES:
        for lambda_cost in LAMBDA_VALUES:
            predicted = _risk_predictions(probabilities, scale, lambda_cost)
            metrics = _policy_metrics(true, predicted)
            valid = _risk_policy_valid(metrics, default)
            rows.append({
                **(context or {}), "loss_scale": scale, "lambda_cost": lambda_cost,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "summary_recall": metrics["summary_recall"], "multi_hop_recall": metrics["multi_hop_recall"],
                "under_routing_rate": metrics["total_under_routing_rate"],
                "over_routing_rate": metrics["total_over_routing_rate"],
                "simulated_savings_percent": metrics["simulated_savings_percent"],
                "default_accuracy": default["accuracy"], "default_macro_f1": default["macro_f1"],
                "default_summary_recall": default["summary_recall"],
                "default_multi_hop_recall": default["multi_hop_recall"],
                "default_under_routing_rate": default["total_under_routing_rate"],
                "valid_under_constraints": bool(valid), "selected": False,
            })
    valid_rows = [row for row in rows if row["valid_under_constraints"]]
    selected = max(valid_rows, key=lambda row: (row["simulated_savings_percent"], row["macro_f1"])) if valid_rows else None
    if selected:
        selected["selected"] = True
    return rows, selected, default


def _calibrated(estimator):
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=3, n_jobs=-1)


def nested_expected_risk(frame: pd.DataFrame, candidate: ExtendedCandidate,
                         registry: dict[str, ExtendedCandidate]) -> dict[str, Any]:
    if candidate.mode != "global":
        raise ValueError("Expected-risk calibration currently requires a global candidate.")
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    default_predictions = np.empty(len(frame), dtype=object)
    policy_predictions = np.empty(len(frame), dtype=object)
    rows = []
    audit = []
    selections = []
    all_found = True
    for outer_fold, (outer_train, outer_test) in enumerate(outer.split(frame["query"], frame.label), 1):
        training = frame.iloc[outer_train]
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=3000 + outer_fold)
        probabilities = np.zeros((len(training), len(CANONICAL_LABELS)), dtype=float)
        assigned = np.zeros(len(training), dtype=int)
        for inner_train, inner_validation in inner.split(training["query"], training.label):
            estimator = _calibrated(_candidate_builder(candidate, registry))
            estimator.fit(training.iloc[inner_train]["query"], training.iloc[inner_train].label)
            probabilities[inner_validation] = aligned_scores(estimator, training.iloc[inner_validation]["query"])
            assigned[inner_validation] += 1
        if not np.all(assigned == 1):
            raise RuntimeError("Nested risk tuning did not assign every inner OOF row exactly once.")
        fold_rows, selected, _ = search_expected_risk(
            training.label.to_numpy(), probabilities,
            {"candidate_id": candidate.candidate_id, "outer_fold": outer_fold,
             "tuning_scope": "outer_training_inner_oof"},
        )
        rows.extend(fold_rows)
        all_found &= selected is not None
        selection = {"loss_scale": 1.0, "lambda_cost": 0.0} if selected is None else selected
        selections.append({"outer_fold": outer_fold, "loss_scale": selection["loss_scale"],
                           "lambda_cost": selection["lambda_cost"], "valid_policy_found": selected is not None})
        estimator = _calibrated(_candidate_builder(candidate, registry))
        estimator.fit(training["query"], training.label)
        outer_probabilities = aligned_scores(estimator, frame.iloc[outer_test]["query"])
        default_predictions[outer_test] = np.asarray(CANONICAL_LABELS)[outer_probabilities.argmax(axis=1)]
        policy_predictions[outer_test] = _risk_predictions(
            outer_probabilities, selection["loss_scale"], selection["lambda_cost"]
        ) if selected is not None else default_predictions[outer_test]
        train_numbers = set(training.record_number.astype(int))
        test_numbers = set(frame.iloc[outer_test].record_number.astype(int))
        audit.append({"candidate_id": candidate.candidate_id, "outer_fold": outer_fold,
                      "tuning_scope": "outer_training_inner_oof", "tuning_count": len(train_numbers),
                      "test_count": len(test_numbers), "overlap_count": len(train_numbers & test_numbers)})
    default = _policy_metrics(frame.label, default_predictions)
    policy = _policy_metrics(frame.label, policy_predictions)
    return {"candidate_id": candidate.candidate_id, "default_metrics": default, "policy_metrics": policy,
            "outer_valid_under_constraints": all_found and _risk_policy_valid(policy, default),
            "selections": selections, "search_rows": rows, "audit_rows": audit}


class ExpectedRiskRouter:
    def __init__(self, model, loss_scale: float, lambda_cost: float):
        self.model = model
        self.loss_scale = loss_scale
        self.lambda_cost = lambda_cost
        self.quality_loss = QUALITY_LOSS.copy()
        self.route_costs = ROUTE_COSTS.copy()

    def predict(self, queries):
        probabilities = aligned_scores(self.model, queries)
        return _risk_predictions(probabilities, self.loss_scale, self.lambda_cost)


def tune_deployment_risk(frame: pd.DataFrame, candidate: ExtendedCandidate,
                         registry: dict[str, ExtendedCandidate]) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = np.zeros((len(frame), len(CANONICAL_LABELS)), dtype=float)
    for train, validation in splitter.split(frame["query"], frame.label):
        estimator = _calibrated(_candidate_builder(candidate, registry))
        estimator.fit(frame.iloc[train]["query"], frame.iloc[train].label)
        probabilities[validation] = aligned_scores(estimator, frame.iloc[validation]["query"])
    rows, selected, default = search_expected_risk(
        frame.label.to_numpy(), probabilities,
        {"candidate_id": candidate.candidate_id, "outer_fold": "deployment",
         "tuning_scope": "full_training_data_oof_for_deployment"},
    )
    return {"rows": rows, "selected": selected, "default": default}


class ExtendedPerDomainRouter:
    def __init__(self, models: dict[str, Any], candidate_id: str):
        self.models = models
        self.candidate_id = candidate_id

    def predict(self, queries, domains=None):
        if domains is None:
            raise ValueError("Extended per-domain routing requires explicit domain metadata.")
        if len(queries) != len(domains):
            raise ValueError("Queries and domains must align.")
        predictions = []
        for query, domain in zip(queries, domains):
            if domain not in self.models:
                raise ValueError(f"No extended per-domain model exists for '{domain}'.")
            predictions.append(self.models[domain].predict([query])[0])
        return np.asarray(predictions, dtype=object)


def _write_progress(all_results: list[ExtendedResult], full_results: dict[str, ExtendedResult], output: Path):
    pd.DataFrame([result.summary for result in all_results]).to_csv(output / "all_model_results.csv", index=False)
    pd.DataFrame([result.summary for result in full_results.values()]).to_csv(
        output / "repeated_cv_results.csv", index=False
    )


def _best(results, key="mean_macro_f1"):
    return max(results, key=lambda result: (
        result.summary[key], result.summary["mean_summary_recall"], result.summary["mean_accuracy"]
    ))


def _best_global(results, key="mean_macro_f1"):
    global_results = [result for result in results if result.candidate.mode == "global"]
    if not global_results:
        raise ValueError("At least one global candidate is required.")
    return _best(global_results, key=key)


def _save_candidate_artifact(frame: pd.DataFrame, candidate: ExtendedCandidate, target: Path,
                             registry: dict[str, ExtendedCandidate]):
    target.mkdir(parents=True, exist_ok=True)
    if candidate.mode == "global":
        model = _candidate_builder(candidate, registry)
        model.fit(frame["query"], frame.label)
        joblib.dump(model, target / "model.joblib")
        return model
    models = {}
    domain_dir = target / "domain_models"
    domain_dir.mkdir(parents=True, exist_ok=True)
    for domain in sorted(frame.domain.unique()):
        subset = frame[frame.domain == domain]
        global_candidate = ExtendedCandidate(candidate.feature, candidate.model, candidate.params_json, "global")
        model = _candidate_builder(global_candidate, registry)
        model.fit(subset["query"], subset.label)
        models[domain] = model
        joblib.dump(model, domain_dir / f"{domain}.joblib")
    router = ExtendedPerDomainRouter(models, candidate.candidate_id)
    joblib.dump(router, target / "model.joblib")
    return router


def _metadata(dataset: LoadedDataset, result: ExtendedResult, selection: str, artifact_file: Path,
              extra: dict[str, Any] | None = None):
    return {
        "track_name": TRACK_NAME, "selection": selection, "not_a_paper_reproduction": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL_OFFICIAL,
        "evaluation_protocol": "ten_seed_shuffled_stratified_5fold",
        "dataset_fingerprint": dataset.source_sha256, "record_count": len(dataset.frame),
        "candidate_id": result.candidate.candidate_id, "mode": result.candidate.mode,
        "feature": result.candidate.feature, "model": result.candidate.model,
        "parameters": result.candidate.params, "repeated_cv_summary": result.summary,
        "known_domains": sorted(dataset.frame.domain.unique()), "cost_mapping": COST_MAPPING,
        "artifact_size_bytes": artifact_file.stat().st_size,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "scikit_learn": sklearn.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__}, **(extra or {}),
    }


def _pareto_table(full_results: list[ExtendedResult], risk_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in full_results:
        summary = result.summary
        rows.append({
            "candidate_id": result.candidate.candidate_id, "policy": "default_repeated_cv",
            "accuracy": summary["mean_accuracy"], "macro_f1": summary["mean_macro_f1"],
            "summary_recall": summary["mean_summary_recall"],
            "multi_hop_recall": summary["mean_multi_hop_recall"],
            "total_under_routing_rate": summary["mean_total_under_routing_rate"],
            "simulated_savings_percent": summary["mean_simulated_savings_percent"],
        })
    for risk in risk_results:
        for name, metrics in (("nested_argmax", risk["default_metrics"]),
                              ("nested_expected_risk", risk["policy_metrics"])):
            rows.append({
                "candidate_id": risk["candidate_id"], "policy": name,
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "summary_recall": metrics["summary_recall"],
                "multi_hop_recall": metrics["multi_hop_recall"],
                "total_under_routing_rate": metrics["total_under_routing_rate"],
                "simulated_savings_percent": metrics["simulated_savings_percent"],
            })
    return pareto_frontier(pd.DataFrame(rows))


def _write_reports(dataset: LoadedDataset, all_results: list[ExtendedResult],
                   full_results: dict[str, ExtendedResult], ensemble_results: list[ExtendedResult],
                   diversity: pd.DataFrame, stress: pd.DataFrame, risk_results: list[dict[str, Any]],
                   risk_rows: list[dict[str, Any]], output: Path, selections: dict[str, Any],
                   dependency_versions: dict[str, str]):
    all_table = pd.DataFrame([result.summary for result in all_results])
    all_table.to_csv(output / "all_model_results.csv", index=False)
    repeated = pd.DataFrame([result.summary for result in full_results.values()])
    repeated.to_csv(output / "repeated_cv_results.csv", index=False)
    pd.DataFrame([row for result in all_results for row in result.per_class_rows]).to_csv(
        output / "per_class_metrics.csv", index=False
    )
    cost = repeated[["candidate_id", "mean_accuracy", "mean_macro_f1", "mean_summary_recall",
                     "mean_multi_hop_recall", "mean_total_under_routing_rate",
                     "mean_total_over_routing_rate", "mean_average_simulated_route_cost",
                     "mean_simulated_savings_percent"]].copy()
    for risk in risk_results:
        metrics = risk["policy_metrics"]
        cost.loc[len(cost)] = [f"{risk['candidate_id']}__nested_expected_risk", metrics["accuracy"],
                               metrics["macro_f1"], metrics["summary_recall"], metrics["multi_hop_recall"],
                               metrics["total_under_routing_rate"], metrics["total_over_routing_rate"],
                               metrics["average_simulated_route_cost"], metrics["simulated_savings_percent"]]
    cost.to_csv(output / "cost_quality_results.csv", index=False)
    diversity.to_csv(output / "error_diversity.csv", index=False)
    pd.DataFrame([result.summary for result in ensemble_results]).to_csv(output / "ensemble_results.csv", index=False)
    repeated[["candidate_id", "mean_fit_time_seconds", "mean_prediction_time_seconds",
              "mean_prediction_latency_ms_per_record", "mean_estimated_artifact_size_bytes"]].to_csv(
        output / "model_runtime_size.csv", index=False
    )
    stress.to_csv(output / "stress_results.csv", index=False)
    pareto = _pareto_table(list(full_results.values()), risk_results)
    pareto.to_csv(output / "pareto_frontier.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output / "expected_risk_policy.csv", index=False)
    best_single = selections["best_single"]
    best_ensemble = selections.get("best_ensemble")
    best_global = selections["best_global"]
    best_per_domain = selections["best_per_domain"]
    risk_selection = selections["risk_selection"]
    selected = pd.DataFrame([
        {"selection": "best_single", "candidate_id": best_single.candidate.candidate_id,
         "accuracy": best_single.summary["mean_accuracy"], "macro_f1": best_single.summary["mean_macro_f1"]},
        {"selection": "best_ensemble", "candidate_id": best_ensemble.candidate.candidate_id if best_ensemble else "none",
         "accuracy": best_ensemble.summary["mean_accuracy"] if best_ensemble else np.nan,
         "macro_f1": best_ensemble.summary["mean_macro_f1"] if best_ensemble else np.nan},
        {"selection": "best_global", "candidate_id": best_global.candidate.candidate_id,
         "accuracy": best_global.summary["mean_accuracy"], "macro_f1": best_global.summary["mean_macro_f1"]},
        {"selection": "best_per_domain", "candidate_id": best_per_domain.candidate.candidate_id,
         "accuracy": best_per_domain.summary["mean_accuracy"], "macro_f1": best_per_domain.summary["mean_macro_f1"]},
    ])
    stress_display = stress[["candidate_id", "stress_protocol", "accuracy", "macro_f1",
                             "summary_recall", "multi_hop_recall", "under_routing_rate"]]
    exceeded = selections["best_accuracy"].summary["mean_accuracy"] > 0.9481040507312022
    report = f"""# {TRACK_NAME}

## Scope

This is an isolated, strictly non-deep-learning benchmark on official raw labels. It does not modify or replace reproduction, forensic, improved-router, production-artifact, or dataset files. Answers and supporting facts are never model inputs.

Hyperparameter grids were fixed before execution. Successive-halving screens use seeds {HALVING_SEEDS}; expensive feature screens use seeds {FEATURE_SCREEN_SEEDS}. Screen-only configurations are ineligible for final selection. Every selectable result uses the identical ten shuffled five-fold seeds {SEEDS}.

Optional boosting dependencies: {json.dumps(dependency_versions, sort_keys=True)}.

## Selected results

{_markdown_table(selected)}

Previous best per-domain accuracy 0.948104 was {'exceeded' if exceeded else 'not exceeded'}. The best selectable accuracy is {selections['best_accuracy'].summary['mean_accuracy']:.6f}; best macro-F1 is {selections['best_macro_f1'].summary['mean_macro_f1']:.6f}.

Best model per-class recall: single_hop={selections['best_macro_f1'].summary['mean_single_hop_recall']:.6f}, multi_hop={selections['best_macro_f1'].summary['mean_multi_hop_recall']:.6f}, summary={selections['best_macro_f1'].summary['mean_summary_recall']:.6f}. Under-routing is {selections['best_macro_f1'].summary['mean_total_under_routing_rate']:.6f}.

The cost-quality selection is `{risk_selection['candidate_id']}`. Nested expected-risk savings are {risk_selection['metrics']['simulated_savings_percent']:.6f}% with accuracy {risk_selection['metrics']['accuracy']:.6f}, macro-F1 {risk_selection['metrics']['macro_f1']:.6f}, and under-routing {risk_selection['metrics']['total_under_routing_rate']:.6f}. Constraints were not relaxed.

## Stress results

{_markdown_table(stress_display)}

## Limitations

- Shuffled IID CV, source-order CV, grouped CV, per-domain CV, and leave-one-domain-out answer different deployment questions; their numbers are not interchangeable.
- Boosting uses conservative, fixed model sizes. Early-stopping evaluation sets are not passed through the composite pipelines because doing so would bypass the fold-local feature transforms; feature selection and dimensionality reduction remain inside every fold.
- Meta-models are trained only on inner out-of-fold base scores. They never receive in-sample base predictions.
- Expected-risk loss scales and cost weights are illustrative simulation parameters, not measured harm or billing data.
- A higher simulated saving is not considered an improvement unless all predeclared quality and under-routing constraints hold.
"""
    (output / "MODEL_BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")


def run_extended_benchmark(dataset: LoadedDataset, report_output: str | Path,
                           artifact_output: str | Path) -> dict[str, Any]:
    report, artifact, project_root = validate_extended_paths(report_output, artifact_output)
    frame = dataset.frame
    if frame.empty or set(frame.label.unique()) != set(CANONICAL_LABELS):
        raise ValueError("Extended benchmark requires all official canonical labels.")
    if frame.domain.isna().any():
        raise ValueError("Extended benchmark requires explicit domain metadata.")
    reports_before = _tree_checksums(project_root / "reports", exclude=report)
    artifacts_before = _tree_checksums(project_root / "artifacts", exclude=artifact)
    sources_before = _source_checksums(dataset)
    report.mkdir(parents=True, exist_ok=True)
    artifact.mkdir(parents=True, exist_ok=True)
    registry: dict[str, ExtendedCandidate] = {}
    all_results: list[ExtendedResult] = []
    full_results: dict[str, ExtendedResult] = {}

    def execute(candidate: ExtendedCandidate, seeds=SEEDS, selectable=True, stage="repeated_cv"):
        print(f"[{TRACK_NAME}] {stage}: {candidate.candidate_id} seeds={seeds}", flush=True)
        result = evaluate_candidate(frame, candidate, report, registry, tuple(seeds),
                                    selectable=selectable, stage=stage)
        all_results.append(result)
        if selectable and tuple(seeds) == SEEDS:
            full_results[candidate.candidate_id] = result
        _write_progress(all_results, full_results, report)
        return result

    global_reference = execute(ExtendedCandidate.create("word_char_structural", "linear_svc_reference"))
    per_domain_reference = execute(ExtendedCandidate.create("word_char", "linear_svc_reference", mode="per_domain"))

    # Linear families: three-seed successive halving, then ten-seed finalists.
    ridge_screens = [execute(ExtendedCandidate.create("word_char_structural", "ridge", {"alpha": alpha}),
                             HALVING_SEEDS, False, "hyperparameter_halving") for alpha in ALPHAS]
    ridge_winner = _best(ridge_screens)
    ridge_full = execute(ridge_winner.candidate)
    # RidgeClassifierCV is paired with the 3k sparse word representation: its nested solver is
    # technically unsuitable for the 35k union (pre-result smoke exceeded six minutes/seed).
    ridge_cv = execute(ExtendedCandidate.create("word_3000", "ridge_cv"))
    linear_full = [ridge_full, ridge_cv]
    sgd_names = ("sgd_hinge", "sgd_log_loss", "sgd_modified_huber", "sgd_squared_hinge",
                 "sgd_hinge_averaged", "sgd_log_loss_averaged")
    for model in sgd_names:
        screens = [execute(ExtendedCandidate.create("word_char_structural", model, {"alpha": alpha}),
                           HALVING_SEEDS, False, "hyperparameter_halving") for alpha in ALPHAS]
        linear_full.append(execute(_best(screens).candidate))

    nb_full = []
    for model in ("multinomial_nb", "complement_nb", "bernoulli_nb"):
        screens = [execute(ExtendedCandidate.create("word_char", model, {"alpha": alpha}),
                           HALVING_SEEDS, False, "hyperparameter_halving") for alpha in NB_ALPHAS]
        nb_full.append(execute(_best(screens).candidate))

    nbsvm_full = []
    nbsvm_specs = (
        ("nbsvm_word", "nbsvm_word_linear_svc", "global"),
        ("nbsvm_word", "nbsvm_word_logistic_regression", "global"),
        ("nbsvm_word_char", "nbsvm_word_char_linear_svc", "global"),
        ("nbsvm_word_char", "nbsvm_word_char_linear_svc", "per_domain"),
    )
    for feature, model, mode in nbsvm_specs:
        screens = [execute(ExtendedCandidate.create(feature, model, {"C": C}, mode),
                           HALVING_SEEDS, False, "hyperparameter_halving") for C in NBSVM_CS]
        nbsvm_full.append(execute(_best(screens).candidate))

    # Feature-family halving for the strongest sparse linear and ComplementNB configurations.
    strongest_linear = _best(linear_full)
    feature_names = ("word_3000", "word_3000_onechar", "char_15000", "word_char",
                     "word_char_structural", "binary_count", "word_1_3", "char_2_6")
    feature_screens = [execute(ExtendedCandidate.create(feature, strongest_linear.candidate.model,
                                                        strongest_linear.candidate.params),
                               FEATURE_SCREEN_SEEDS, False, "feature_halving") for feature in feature_names]
    feature_winner = _best(feature_screens)
    if feature_winner.candidate.candidate_id not in full_results:
        linear_full.append(execute(feature_winner.candidate))
    complement = next(result for result in nb_full if result.candidate.model == "complement_nb")
    nb_feature_screens = [execute(ExtendedCandidate.create(feature, "complement_nb", complement.candidate.params),
                                  FEATURE_SCREEN_SEEDS, False, "feature_halving")
                          for feature in ("word_3000", "word_3000_onechar", "word_char", "binary_count",
                                          "word_1_3", "char_2_6")]
    nb_feature_winner = _best(nb_feature_screens)
    if nb_feature_winner.candidate.candidate_id not in full_results:
        nb_full.append(execute(nb_feature_winner.candidate))

    # Tree/boost feature screens. ExtraTrees establishes one representative per feature category.
    tree_features = ("structural_only", "chi2_1000", "chi2_3000", "chi2_5000", "svd_100", "svd_200",
                     "svd_400", "svd_structural_100", "svd_structural_200", "svd_structural_400")
    extra_screens = [execute(ExtendedCandidate.create(feature, "extra_trees"), FEATURE_SCREEN_SEEDS,
                             False, "tree_feature_halving") for feature in tree_features]
    representatives = [next(result for result in extra_screens if result.candidate.feature == "structural_only")]
    for prefix in ("chi2_", "svd_", "svd_structural_"):
        group = [result for result in extra_screens if result.candidate.feature.startswith(prefix)]
        if prefix == "svd_":
            group = [result for result in group if not result.candidate.feature.startswith("svd_structural_")]
        representatives.append(_best(group))
    tree_family_screens: dict[str, list[ExtendedResult]] = {"extra_trees": representatives}
    dependency_versions = {}
    for package in ("xgboost", "lightgbm", "catboost"):
        module = __import__(package)
        dependency_versions[package] = getattr(module, "__version__", "unknown")
        tree_family_screens[package] = [
            execute(ExtendedCandidate.create(result.candidate.feature, package), FEATURE_SCREEN_SEEDS,
                    False, "boosting_feature_halving") for result in representatives
        ]
    best_so_far = max(result.summary["mean_macro_f1"] for result in full_results.values())
    tree_full = []
    family_winners = [_best(results) for results in tree_family_screens.values()]
    forced = _best(family_winners)
    for winner in family_winners:
        stronger_recall = any(winner.summary[f"mean_{label}_recall"] >
                              max(result.summary[f"mean_{label}_recall"] for result in full_results.values()) + .02
                              for label in CANONICAL_LABELS)
        if winner is forced or winner.summary["mean_macro_f1"] >= best_so_far - .02 or stronger_recall:
            tree_full.append(execute(winner.candidate))

    # Retain at most five diverse role finalists.
    # Deployable query-only ensembles cannot safely refit a per-domain OOF base without
    # receiving domain metadata at inference time. Keep the per-domain NB-SVM as a
    # separately evaluated finalist, but use the strongest global NB-SVM in ensembles so
    # the fitted artifact exactly matches the protocol represented by its OOF predictions.
    best_nbsvm = _best_global(nbsvm_full)
    best_ridge = _best([ridge_full, ridge_cv])
    best_complement = _best([result for result in nb_full if result.candidate.model == "complement_nb"])
    best_tree = _best(tree_full) if tree_full else None
    role_candidates = [global_reference, best_nbsvm, best_ridge, best_complement] + ([best_tree] if best_tree else [])
    preliminary_diversity = error_diversity(role_candidates, report)
    retained = [global_reference]
    current_best = max(result.summary["mean_macro_f1"] for result in full_results.values())
    for candidate in role_candidates[1:]:
        pair = preliminary_diversity[
            ((preliminary_diversity.model_a == global_reference.candidate.candidate_id)
             & (preliminary_diversity.model_b == candidate.candidate.candidate_id))
            | ((preliminary_diversity.model_b == global_reference.candidate.candidate_id)
               & (preliminary_diversity.model_a == candidate.candidate.candidate_id))
        ].iloc[0]
        stronger_recall = any(candidate.summary[f"mean_{label}_recall"] >
                              global_reference.summary[f"mean_{label}_recall"] for label in CANONICAL_LABELS)
        if (candidate.summary["mean_macro_f1"] >= current_best - .01
                or pair.mean_disagreement_rate >= .08 or stronger_recall):
            retained.append(candidate)
    retained = retained[:5]
    diversity = error_diversity(retained, report)

    ensemble_results = []
    base_ids = [result.candidate.candidate_id for result in retained]
    hard_candidate = ExtendedCandidate.create("retained_oof", "ensemble_hard", {"base_ids": base_ids})
    hard = derive_voting_result(frame, hard_candidate, retained, report, registry)
    all_results.append(hard); full_results[hard.candidate.candidate_id] = hard; ensemble_results.append(hard)
    weights = [3, 3, 2, 1, 1][:len(retained)]
    weighted_candidate = ExtendedCandidate.create("retained_oof", "ensemble_weighted",
                                                  {"base_ids": base_ids, "weights": weights})
    weighted = derive_voting_result(frame, weighted_candidate, retained, report, registry, weights)
    all_results.append(weighted); full_results[weighted.candidate.candidate_id] = weighted; ensemble_results.append(weighted)
    _write_progress(all_results, full_results, report)
    nested_bases = retained[:min(4, len(retained))]
    nested_ids = [result.candidate.candidate_id for result in nested_bases]
    for model in ("ensemble_calibrated_score_average", "ensemble_stack_logistic", "ensemble_stack_ridge"):
        ensemble_results.append(execute(ExtendedCandidate.create("retained_nested", model,
                                                                  {"base_ids": nested_ids})))

    best_single = _best([result for result in full_results.values() if not result.candidate.model.startswith("ensemble_")])
    best_global_single = _best([result for result in full_results.values()
                                if result.candidate.mode == "global"
                                and not result.candidate.model.startswith("ensemble_")])
    best_ensemble = _best(ensemble_results)
    best_global = _best([result for result in full_results.values() if result.candidate.mode == "global"])
    best_per_domain = _best([result for result in full_results.values() if result.candidate.mode == "per_domain"])
    best_accuracy = max(full_results.values(), key=lambda result: result.summary["mean_accuracy"])
    best_macro = _best(full_results.values())

    risk_finalists = [best_global_single]
    if best_ensemble.summary["mean_macro_f1"] >= best_global_single.summary["mean_macro_f1"] - .01:
        risk_finalists.append(best_ensemble)
    risk_results = []
    risk_rows = []
    for finalist in risk_finalists:
        print(f"[{TRACK_NAME}] nested expected risk: {finalist.candidate.candidate_id}", flush=True)
        risk = nested_expected_risk(frame, finalist.candidate, registry)
        risk_results.append(risk); risk_rows.extend(risk["search_rows"])
    valid_risks = [risk for risk in risk_results if risk["outer_valid_under_constraints"]]
    if valid_risks:
        selected_risk = max(valid_risks, key=lambda risk: risk["policy_metrics"]["simulated_savings_percent"])
        risk_metrics = selected_risk["policy_metrics"]
    else:
        selected_risk = max(risk_results, key=lambda risk: risk["default_metrics"]["simulated_savings_percent"])
        risk_metrics = selected_risk["default_metrics"]
    risk_candidate = registry[selected_risk["candidate_id"]]

    stress_candidates = {best_global.candidate.candidate_id: best_global,
                         best_per_domain.candidate.candidate_id: best_per_domain,
                         best_ensemble.candidate.candidate_id: best_ensemble}
    stress_rows = []
    for result in stress_candidates.values():
        print(f"[{TRACK_NAME}] stress: {result.candidate.candidate_id}", flush=True)
        stress_rows.extend(evaluate_stress(frame, result.candidate, registry))
    stress_table = pd.DataFrame(stress_rows)

    # Artifacts.
    artifact_results = (("best_accuracy_model", best_accuracy), ("best_macro_f1_model", best_macro))
    for selection_name, result in artifact_results:
        target = artifact / selection_name
        _save_candidate_artifact(frame, result.candidate, target, registry)
        metadata = _metadata(dataset, result, selection_name, target / "model.joblib")
        (target / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")
    risk_target = artifact / "best_cost_quality_router"
    risk_target.mkdir(parents=True, exist_ok=True)
    deployment = tune_deployment_risk(frame, risk_candidate, registry)
    risk_rows.extend(deployment["rows"])
    selected_deployment = deployment["selected"]
    calibrated = _calibrated(_candidate_builder(risk_candidate, registry))
    calibrated.fit(frame["query"], frame.label)
    if selected_deployment and selected_risk["outer_valid_under_constraints"]:
        risk_router = ExpectedRiskRouter(calibrated, selected_deployment["loss_scale"],
                                         selected_deployment["lambda_cost"])
        deployed_policy = True
    else:
        risk_router = calibrated
        deployed_policy = False
    joblib.dump(risk_router, risk_target / "model.joblib")
    risk_result = full_results[risk_candidate.candidate_id]
    risk_metadata = _metadata(dataset, risk_result, "best_cost_quality_router", risk_target / "model.joblib", {
        "nested_expected_risk": selected_risk, "deployment_policy_valid": deployed_policy,
        "deployment_loss_scale": selected_deployment["loss_scale"] if selected_deployment else None,
        "deployment_lambda_cost": selected_deployment["lambda_cost"] if selected_deployment else None,
        "quality_loss_matrix": QUALITY_LOSS.tolist(), "lambda_grid": list(LAMBDA_VALUES),
        "loss_scale_grid": list(LOSS_SCALES),
    })
    (risk_target / "metadata.json").write_text(json.dumps(_jsonable(risk_metadata), indent=2), encoding="utf-8")

    selections = {"best_single": best_single, "best_ensemble": best_ensemble, "best_global": best_global,
                  "best_per_domain": best_per_domain, "best_accuracy": best_accuracy,
                  "best_macro_f1": best_macro,
                  "risk_selection": {"candidate_id": selected_risk["candidate_id"], "metrics": risk_metrics}}
    _write_reports(dataset, all_results, full_results, ensemble_results, diversity, stress_table,
                   risk_results, risk_rows, report, selections, dependency_versions)
    if reports_before != _tree_checksums(project_root / "reports", exclude=report):
        raise RuntimeError("Extended isolation failed: a pre-existing report changed.")
    if artifacts_before != _tree_checksums(project_root / "artifacts", exclude=artifact):
        raise RuntimeError("Extended isolation failed: a pre-existing artifact changed.")
    if sources_before != _source_checksums(dataset):
        raise RuntimeError("Extended isolation failed: a dataset source changed.")
    summary = {
        "track_name": TRACK_NAME, "dataset_fingerprint": dataset.source_sha256,
        "unique_configuration_count": len({result.candidate.candidate_id for result in all_results}),
        "full_repeated_configuration_count": len(full_results), "tests_pending": True,
        "best_single": best_single.summary, "best_ensemble": best_ensemble.summary,
        "best_global": best_global.summary, "best_per_domain": best_per_domain.summary,
        "best_accuracy": best_accuracy.summary, "best_macro_f1": best_macro.summary,
        "risk_selection": {"candidate_id": selected_risk["candidate_id"], "metrics": risk_metrics,
                           "deployment_policy_valid": deployed_policy},
        "previous_accuracy_exceeded": best_accuracy.summary["mean_accuracy"] > .9481040507312022,
        "protected_outputs_unchanged": True, "dataset_sources_unchanged": True,
    }
    (report / "run_manifest.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary
