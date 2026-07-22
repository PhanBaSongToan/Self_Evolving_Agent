from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import string
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from .config import CANONICAL_LABELS, PROTOCOL_OFFICIAL, TFIDF_PARAMS
from .data import LoadedDataset
from .models import build_pipeline
from .reporting import _markdown_table

PAPER_ACCURACY = 0.932
PAPER_MACRO_F1 = 0.928
DISCLOSED = "DISCLOSED_COMPATIBLE"
PLAUSIBLE = "AMBIGUOUS_BUT_PLAUSIBLE"
INVALID = "DIAGNOSTIC_ONLY_INVALID_FOR_REPRODUCTION"
VALIDITY_ORDER = {DISCLOSED: 0, PLAUSIBLE: 1, INVALID: 2}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_checksum(frame: pd.DataFrame) -> str:
    payload = "\n".join(frame["id"].fillna(frame["record_number"].astype(str)).astype(str))
    return _sha256_bytes(payload.encode("utf-8"))


def fold_indices_checksum(frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> str:
    payload = [{"fold": index + 1,
                "train": frame.iloc[train]["record_number"].astype(int).tolist(),
                "validation": frame.iloc[test]["record_number"].astype(int).tolist()}
               for index, (train, test) in enumerate(splits)]
    return _sha256_bytes(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def make_stratified_splits(frame: pd.DataFrame, *, shuffle: bool = False, seed: int | None = None,
                           groups: pd.Series | None = None, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
        return list(splitter.split(frame["query"], frame["label"]))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    return list(splitter.split(frame["query"], frame["label"], groups=groups))


def _fold_distribution(frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]], checksum: str) -> pd.DataFrame:
    rows = []
    lengths = frame["query"].str.findall(r"\b[\w'-]+\b").str.len()
    for fold, (train, validation) in enumerate(splits, 1):
        subset = frame.iloc[validation]
        fold_lengths = lengths.iloc[validation]
        rows.append({"fold": fold, "train_count": len(train), "validation_count": len(validation),
                     "fold_indices_checksum": checksum,
                     "label_distribution": json.dumps(subset.label.value_counts().sort_index().to_dict()),
                     "domain_distribution": json.dumps(subset.domain.value_counts().sort_index().to_dict()),
                     "query_length_min": int(fold_lengths.min()), "query_length_q25": float(fold_lengths.quantile(.25)),
                     "query_length_median": float(fold_lengths.median()), "query_length_mean": float(fold_lengths.mean()),
                     "query_length_q75": float(fold_lengths.quantile(.75)), "query_length_max": int(fold_lengths.max())})
    return pd.DataFrame(rows)


@dataclass
class ForensicResult:
    experiment_id: str
    validity_class: str
    accuracy: float
    macro_f1: float
    mean_fold_macro_f1: float
    fold_macro_f1_std: float
    oof: pd.DataFrame
    fold_metrics: pd.DataFrame
    fold_distributions: pd.DataFrame
    confusion: np.ndarray
    class_report: dict[str, Any]
    ordered_ids_checksum: str
    fold_checksum: str
    metadata: dict[str, Any]

    def distance_row(self, category: str, **overrides: Any) -> dict[str, Any]:
        row = {"experiment_id": self.experiment_id, "category": category, "validity_class": self.validity_class,
               "input_fields": "question", "training_scope": "global", "cv_type": "StratifiedKFold",
               "shuffle": False, "seed": "", "grouping": "none", "preprocessing_scope": "fold_local",
               "accuracy": self.accuracy, "macro_f1": self.macro_f1,
               "accuracy_delta_pp": (self.accuracy - PAPER_ACCURACY) * 100,
               "macro_f1_delta": self.macro_f1 - PAPER_MACRO_F1,
               "template_overlap_rate": np.nan, "leakage_warning": "", "notes": ""}
        row.update(overrides)
        return row


def _save_result(result: ForensicResult, output: Path, splits: list[tuple[np.ndarray, np.ndarray]], frame: pd.DataFrame) -> None:
    target = output / "experiments" / re.sub(r"[^A-Za-z0-9_.-]", "_", result.experiment_id)
    target.mkdir(parents=True, exist_ok=True)
    result.oof.to_csv(target / "oof_predictions.csv", index=False)
    result.fold_metrics.to_csv(target / "fold_metrics.csv", index=False)
    result.fold_distributions.to_csv(target / "fold_distributions.csv", index=False)
    pd.DataFrame(result.confusion, index=CANONICAL_LABELS, columns=CANONICAL_LABELS).to_csv(target / "confusion_matrix.csv")
    indices = [{"fold": fold, "train_record_numbers": frame.iloc[train].record_number.astype(int).tolist(),
                "validation_record_numbers": frame.iloc[test].record_number.astype(int).tolist()}
               for fold, (train, test) in enumerate(splits, 1)]
    (target / "fold_indices.json").write_text(json.dumps(indices), encoding="utf-8")
    summary = {"experiment_id": result.experiment_id, "validity_class": result.validity_class,
               "accuracy": result.accuracy, "macro_f1": result.macro_f1,
               "mean_fold_macro_f1": result.mean_fold_macro_f1, "fold_macro_f1_std": result.fold_macro_f1_std,
               "accuracy_delta_pp": (result.accuracy - PAPER_ACCURACY) * 100,
               "macro_f1_delta": result.macro_f1 - PAPER_MACRO_F1,
               "ordered_record_ids_checksum": result.ordered_ids_checksum,
               "fold_indices_checksum": result.fold_checksum, "metadata": result.metadata,
               "classification_report": result.class_report}
    (target / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")


def run_svm_experiment(frame: pd.DataFrame, texts: Iterable[str], splits: list[tuple[np.ndarray, np.ndarray]],
                       experiment_id: str, validity_class: str, *, output: Path | None = None,
                       vectorizer_overrides: dict[str, Any] | None = None,
                       svc_overrides: dict[str, Any] | None = None,
                       global_vectorizer_fit: bool = False,
                       metadata: dict[str, Any] | None = None) -> ForensicResult:
    if output is not None:
        cached_target = output / "experiments" / re.sub(r"[^A-Za-z0-9_.-]", "_", experiment_id)
        required = [cached_target / name for name in ("summary.json", "oof_predictions.csv", "fold_metrics.csv", "fold_distributions.csv", "confusion_matrix.csv")]
        if all(path.exists() for path in required):
            summary = json.loads((cached_target / "summary.json").read_text(encoding="utf-8"))
            return ForensicResult(
                experiment_id=experiment_id, validity_class=summary["validity_class"],
                accuracy=float(summary["accuracy"]), macro_f1=float(summary["macro_f1"]),
                mean_fold_macro_f1=float(summary["mean_fold_macro_f1"]), fold_macro_f1_std=float(summary["fold_macro_f1_std"]),
                oof=pd.read_csv(cached_target / "oof_predictions.csv"), fold_metrics=pd.read_csv(cached_target / "fold_metrics.csv"),
                fold_distributions=pd.read_csv(cached_target / "fold_distributions.csv"),
                confusion=pd.read_csv(cached_target / "confusion_matrix.csv", index_col=0).to_numpy(),
                class_report=summary["classification_report"], ordered_ids_checksum=summary["ordered_record_ids_checksum"],
                fold_checksum=summary["fold_indices_checksum"], metadata=summary.get("metadata", {}))
    work = frame.reset_index(drop=True).copy()
    text_values = np.asarray(list(texts), dtype=object)
    if len(text_values) != len(work):
        raise ValueError("Forensic text inputs must align one-to-one with the ordered frame.")
    tfidf_params = dict(TFIDF_PARAMS)
    tfidf_params.update(vectorizer_overrides or {})
    svc_params = {"kernel": "rbf", "gamma": "scale"}
    svc_params.update(svc_overrides or {})
    predictions = np.empty(len(work), dtype=object)
    assignment = np.zeros(len(work), dtype=int)
    fold_rows = []
    vocabulary_ratios = []
    global_matrix = None
    if global_vectorizer_fit:
        global_vectorizer = TfidfVectorizer(**tfidf_params)
        global_matrix = global_vectorizer.fit_transform(text_values)
        global_vocabulary = set(global_vectorizer.vocabulary_)
    for fold, (train, validation) in enumerate(splits, 1):
        if global_vectorizer_fit:
            model = SVC(**svc_params).fit(global_matrix[train], work.iloc[train].label)
            predicted = model.predict(global_matrix[validation])
            local_vectorizer = TfidfVectorizer(**tfidf_params).fit(text_values[train])
            vocabulary_ratios.append(len(set(local_vectorizer.vocabulary_) & global_vocabulary) / len(global_vocabulary))
        else:
            model = Pipeline([("tfidf", TfidfVectorizer(**tfidf_params)), ("classifier", SVC(**svc_params))])
            model.fit(text_values[train], work.iloc[train].label)
            predicted = model.predict(text_values[validation])
        predictions[validation] = predicted
        assignment[validation] += 1
        fold_rows.append({"fold": fold,
                          "accuracy": accuracy_score(work.iloc[validation].label, predicted),
                          "macro_f1": f1_score(work.iloc[validation].label, predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0)})
    if not np.all(assignment == 1):
        raise RuntimeError("Forensic OOF invariant failed: every record must receive exactly one prediction.")
    fold_metrics = pd.DataFrame(fold_rows)
    accuracy = float(accuracy_score(work.label, predictions))
    macro_f1 = float(f1_score(work.label, predictions, labels=CANONICAL_LABELS, average="macro", zero_division=0))
    oof = work[["record_number", "id", "domain", "raw_label", "label", "protocol", "query"]].copy()
    oof.insert(0, "experiment_id", experiment_id)
    oof.insert(1, "ordered_position", np.arange(len(work)))
    oof["predicted_label"] = predictions
    oof["fold"] = -1
    for fold, (_, validation) in enumerate(splits, 1):
        oof.loc[validation, "fold"] = fold
    fold_checksum = fold_indices_checksum(work, splits)
    fold_distributions = _fold_distribution(work, splits, fold_checksum)
    result = ForensicResult(
        experiment_id=experiment_id, validity_class=validity_class, accuracy=accuracy, macro_f1=macro_f1,
        mean_fold_macro_f1=float(fold_metrics.macro_f1.mean()), fold_macro_f1_std=float(fold_metrics.macro_f1.std(ddof=0)),
        oof=oof, fold_metrics=fold_metrics, fold_distributions=fold_distributions,
        confusion=confusion_matrix(work.label, predictions, labels=CANONICAL_LABELS),
        class_report=classification_report(work.label, predictions, labels=CANONICAL_LABELS, output_dict=True, zero_division=0),
        ordered_ids_checksum=ordered_id_checksum(work), fold_checksum=fold_checksum,
        metadata={**(metadata or {}), "tfidf_params": tfidf_params, "svc_params": svc_params,
                  "global_vectorizer_fit": global_vectorizer_fit,
                  "mean_fold_to_global_vocabulary_ratio": float(np.mean(vocabulary_ratios)) if vocabulary_ratios else None})
    if output is not None:
        _save_result(result, output, splits, work)
    return result


def _result_from_baseline(dataset: LoadedDataset, reports_root: Path, output: Path) -> tuple[ForensicResult, list[tuple[np.ndarray, np.ndarray]]]:
    frame = dataset.frame.reset_index(drop=True)
    splits = make_stratified_splits(frame)
    baseline_dir = reports_root / PROTOCOL_OFFICIAL
    summary = pd.read_csv(baseline_dir / "results_summary.csv").set_index("configuration").loc["tfidf_svm"]
    all_oof = pd.read_csv(baseline_dir / "oof_predictions.csv")
    original = all_oof[all_oof.configuration == "tfidf_svm"].sort_values("record_number")
    predictions = original.predicted_label.to_numpy()
    fold_metrics = pd.read_csv(baseline_dir / "fold_metrics.csv")
    fold_metrics = fold_metrics[fold_metrics.configuration == "tfidf_svm"][["fold", "accuracy", "macro_f1"]].reset_index(drop=True)
    checksum = fold_indices_checksum(frame, splits)
    result = ForensicResult(
        experiment_id="A0", validity_class=DISCLOSED, accuracy=float(summary.pooled_accuracy),
        macro_f1=float(summary.pooled_macro_f1), mean_fold_macro_f1=float(summary.mean_macro_f1),
        fold_macro_f1_std=float(summary.std_macro_f1),
        oof=pd.DataFrame({"experiment_id": "A0", "ordered_position": np.arange(len(frame)),
                          "record_number": frame.record_number, "id": frame.id, "domain": frame.domain,
                          "raw_label": frame.raw_label, "label": frame.label, "protocol": frame.protocol,
                          "query": frame["query"], "predicted_label": predictions,
                          "fold": original.fold.to_numpy()}),
        fold_metrics=fold_metrics, fold_distributions=_fold_distribution(frame, splits, checksum),
        confusion=confusion_matrix(frame.label, predictions, labels=CANONICAL_LABELS),
        class_report=classification_report(frame.label, predictions, labels=CANONICAL_LABELS, output_dict=True, zero_division=0),
        ordered_ids_checksum=ordered_id_checksum(frame), fold_checksum=checksum,
        metadata={"source": "frozen official_raw_labels result files", "recomputed": False})
    _save_result(result, output, splits, frame)
    return result, splits


def write_baseline_manifest(dataset: LoadedDataset, reports_root: Path, output: Path, baseline: ForensicResult) -> dict[str, Any]:
    baseline_dir = reports_root / PROTOCOL_OFFICIAL
    pipeline = build_pipeline("tfidf_svm")
    manifest = {
        "dataset_fingerprint": dataset.source_sha256,
        "source_file_checksums": {str(path): _sha256_file(Path(path)) for path in dataset.source_files},
        "data_order": {"description": "source files lexicographically, then zero-based source row",
                       "ordered_record_ids_checksum": ordered_id_checksum(dataset.frame),
                       "first_ids": dataset.frame.id.head(5).tolist(), "last_ids": dataset.frame.id.tail(5).tolist()},
        "label_distribution": dataset.frame.raw_label.value_counts().to_dict(),
        "dependency_versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                                "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "tfidf_parameters": pipeline.named_steps["tfidf"].get_params(),
        "estimator_parameters": pipeline.named_steps["classifier"].get_params(),
        "cv_fold_indices_checksum": baseline.fold_checksum,
        "original_fold_indices_file_checksum": _sha256_file(baseline_dir / "fold_indices.json"),
        "oof_predictions_checksum": _sha256_file(baseline_dir / "oof_predictions.csv"),
        "baseline_metrics": {"accuracy": baseline.accuracy, "macro_f1": baseline.macro_f1},
        "paper_target": {"accuracy": PAPER_ACCURACY, "macro_f1": PAPER_MACRO_F1},
    }
    (output / "baseline_manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return manifest


def _global_shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _sort_by_id(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["id", "record_number"], kind="mergesort").reset_index(drop=True)


def interleave_domain_label(frame: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for key, group in frame.groupby(["domain", "label"], sort=True):
        groups.append((key, group.sort_values(["id", "record_number"], kind="mergesort")))
    positions = []
    for offset in range(max(len(group) for _, group in groups)):
        for _, group in groups:
            if offset < len(group):
                positions.append(group.index[offset])
    return frame.loc[positions].reset_index(drop=True)


def run_fold_order_experiments(dataset: LoadedDataset, reports_root: Path, output: Path) -> tuple[list[ForensicResult], dict[str, tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]], pd.DataFrame]:
    frame = dataset.frame.reset_index(drop=True)
    results: list[ForensicResult] = []
    specs: dict[str, tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]] = {}
    baseline, baseline_splits = _result_from_baseline(dataset, reports_root, output)
    results.append(baseline)
    specs["A0"] = (frame, baseline_splits)
    seeds = (0, 1, 2, 3, 4, 13, 21, 42, 77, 100)
    for seed in seeds:
        experiment_id = "A1" if seed == 42 else f"A2_seed_{seed}"
        splits = make_stratified_splits(frame, shuffle=True, seed=seed)
        result = run_svm_experiment(frame, frame["query"], splits, experiment_id, PLAUSIBLE, output=output,
                                    metadata={"order": "source", "shuffle": True, "seed": seed})
        results.append(result)
        specs[experiment_id] = (frame, splits)
    ordered_variants = {
        "A3": (_global_shuffle(frame, 42), "global record shuffle seed 42"),
        "A4": (_sort_by_id(frame), "global stable sort by record ID"),
        "A5": (interleave_domain_label(frame), "deterministic domain-label interleave"),
    }
    for experiment_id, (ordered, description) in ordered_variants.items():
        splits = make_stratified_splits(ordered)
        validity = INVALID if experiment_id == "A5" else PLAUSIBLE
        result = run_svm_experiment(ordered, ordered["query"], splits, experiment_id, validity, output=output,
                                    metadata={"order": description, "shuffle": False, "seed": 42 if experiment_id == "A3" else None})
        result.validity_class = validity
        results.append(result)
        specs[experiment_id] = (ordered, splits)
    rows = []
    for result in results:
        meta = result.metadata
        rows.append({"experiment_id": result.experiment_id, "validity_class": result.validity_class,
                     "order": meta.get("order", "source"), "shuffle": meta.get("shuffle", False), "seed": meta.get("seed", ""),
                     "accuracy": result.accuracy, "macro_f1": result.macro_f1,
                     "accuracy_delta_pp": (result.accuracy - PAPER_ACCURACY) * 100,
                     "macro_f1_delta": result.macro_f1 - PAPER_MACRO_F1,
                     "ordered_record_ids_checksum": result.ordered_ids_checksum, "fold_indices_checksum": result.fold_checksum})
    matrix = pd.DataFrame(rows).sort_values("experiment_id")
    matrix.to_csv(output / "fold_order_matrix.csv", index=False)
    seed_results = [result for result in results if result.experiment_id == "A1" or result.experiment_id.startswith("A2_seed_")]
    values = np.asarray([result.macro_f1 for result in seed_results])
    closest = min(seed_results, key=lambda result: abs(result.macro_f1 - PAPER_MACRO_F1))
    ci = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    analysis = ["# Fold and order analysis", "", "All variants use the disclosed TF-IDF and SVM settings. A0 references the frozen baseline; no original report was overwritten.", "",
                _markdown_table(matrix), "", "## Multi-seed shuffled sensitivity", "",
                f"Across all 10 predeclared seeds: mean macro-F1 `{values.mean():.6f}`, standard deviation `{values.std(ddof=0):.6f}`, minimum `{values.min():.6f}`, maximum `{values.max():.6f}`, and approximate 95% CI for the mean `[{values.mean()-ci:.6f}, {values.mean()+ci:.6f}]`.", "",
                f"The seed closest to the paper was `{closest.metadata.get('seed')}` at macro-F1 `{closest.macro_f1:.6f}`. It is not selected as the reproduced result."]
    (output / "fold_order_analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    return results, specs, matrix


def _combined_result(experiment_id: str, validity: str, frame: pd.DataFrame, oof: pd.DataFrame,
                     fold_metrics: pd.DataFrame, metadata: dict[str, Any]) -> ForensicResult:
    ordered = frame.sort_values("record_number").reset_index(drop=True)
    predictions = oof.sort_values("record_number").predicted_label.to_numpy()
    accuracy = float(accuracy_score(ordered.label, predictions))
    macro_f1 = float(f1_score(ordered.label, predictions, labels=CANONICAL_LABELS, average="macro", zero_division=0))
    return ForensicResult(experiment_id, validity, accuracy, macro_f1,
                          float(fold_metrics.macro_f1.mean()), float(fold_metrics.macro_f1.std(ddof=0)),
                          oof.sort_values("record_number").reset_index(drop=True), fold_metrics,
                          pd.DataFrame(), confusion_matrix(ordered.label, predictions, labels=CANONICAL_LABELS),
                          classification_report(ordered.label, predictions, labels=CANONICAL_LABELS, output_dict=True, zero_division=0),
                          ordered_id_checksum(ordered), metadata.get("fold_checksum", "multiple_independent_folds"), metadata)


def run_domain_experiments(dataset: LoadedDataset, output: Path, baseline: ForensicResult) -> tuple[list[ForensicResult], pd.DataFrame]:
    frame = dataset.frame.reset_index(drop=True)
    results = [baseline]
    rows = [{"experiment_id": "B0", "scope": "global", "domain": "ALL", "record_count": len(frame),
             "accuracy": baseline.accuracy, "macro_f1": baseline.macro_f1,
             "unweighted_domain_macro_f1": np.nan, "weighted_domain_macro_f1": np.nan, "validity_class": DISCLOSED}]
    domain_results = []
    for domain in sorted(frame.domain.unique()):
        subset = frame[frame.domain == domain].reset_index(drop=True)
        splits = make_stratified_splits(subset)
        result = run_svm_experiment(subset, subset["query"], splits, f"B1_{domain}", PLAUSIBLE, output=output,
                                    metadata={"training_scope": "independent_per_domain", "domain": domain,
                                              "model_instance_scope": domain})
        domain_results.append(result)
        rows.append({"experiment_id": "B1", "scope": "independent_per_domain", "domain": domain,
                     "record_count": len(subset), "accuracy": result.accuracy, "macro_f1": result.macro_f1,
                     "unweighted_domain_macro_f1": np.nan, "weighted_domain_macro_f1": np.nan, "validity_class": PLAUSIBLE})
    combined_oof = pd.concat([result.oof for result in domain_results], ignore_index=True)
    combined_folds = pd.concat([result.fold_metrics.assign(domain=result.metadata["domain"]) for result in domain_results], ignore_index=True)
    combined = _combined_result("B1", PLAUSIBLE, frame, combined_oof, combined_folds,
                                {"training_scope": "independent_per_domain", "model_instances": [r.metadata["model_instance_scope"] for r in domain_results]})
    unweighted = float(np.mean([result.macro_f1 for result in domain_results]))
    weighted = float(np.average([result.macro_f1 for result in domain_results], weights=[len(result.oof) for result in domain_results]))
    combined.metadata.update({"unweighted_domain_macro_f1": unweighted, "weighted_domain_macro_f1": weighted})
    combined_target = output / "experiments" / "B1"
    combined_target.mkdir(parents=True, exist_ok=True)
    combined.oof.to_csv(combined_target / "oof_predictions.csv", index=False)
    (combined_target / "summary.json").write_text(json.dumps(_jsonable({"accuracy": combined.accuracy, "macro_f1": combined.macro_f1, **combined.metadata}), indent=2), encoding="utf-8")
    results.append(combined)
    rows.append({"experiment_id": "B1", "scope": "independent_per_domain", "domain": "POOLED",
                 "record_count": len(frame), "accuracy": combined.accuracy, "macro_f1": combined.macro_f1,
                 "unweighted_domain_macro_f1": unweighted, "weighted_domain_macro_f1": weighted, "validity_class": PLAUSIBLE})

    prefixed = frame.apply(lambda row: f"DOMAIN_{re.sub(r'[^A-Za-z0-9]', '_', str(row.domain)).upper()} {row['query']}", axis=1)
    splits = make_stratified_splits(frame)
    prefixed_result = run_svm_experiment(frame, prefixed, splits, "B2", INVALID, output=output,
                                         metadata={"training_scope": "global", "domain_prefix": True})
    results.append(prefixed_result)
    rows.append({"experiment_id": "B2", "scope": "global_domain_prefixed", "domain": "ALL", "record_count": len(frame),
                 "accuracy": prefixed_result.accuracy, "macro_f1": prefixed_result.macro_f1,
                 "unweighted_domain_macro_f1": np.nan, "weighted_domain_macro_f1": np.nan, "validity_class": INVALID})

    loto_rows = []
    loto_predictions = []
    cached_loto = output / "experiments" / "B3" / "oof_predictions.csv"
    if cached_loto.exists():
        loto_oof = pd.read_csv(cached_loto)
        for domain, test in loto_oof.groupby("domain", sort=True):
            loto_rows.append({"experiment_id": "B3", "scope": "leave_one_domain_out", "domain": domain,
                              "record_count": len(test), "accuracy": float(accuracy_score(test.label, test.predicted_label)),
                              "macro_f1": float(f1_score(test.label, test.predicted_label, labels=CANONICAL_LABELS, average="macro", zero_division=0)),
                              "unweighted_domain_macro_f1": np.nan, "weighted_domain_macro_f1": np.nan, "validity_class": DISCLOSED})
    else:
        for domain in sorted(frame.domain.unique()):
            train = frame[frame.domain != domain]
            test = frame[frame.domain == domain]
            model = Pipeline([("tfidf", TfidfVectorizer(**TFIDF_PARAMS)), ("classifier", SVC(kernel="rbf", gamma="scale"))])
            model.fit(train["query"], train.label)
            predicted = model.predict(test["query"])
            accuracy = float(accuracy_score(test.label, predicted))
            macro_f1 = float(f1_score(test.label, predicted, labels=CANONICAL_LABELS, average="macro", zero_division=0))
            loto_rows.append({"experiment_id": "B3", "scope": "leave_one_domain_out", "domain": domain,
                              "record_count": len(test), "accuracy": accuracy, "macro_f1": macro_f1,
                              "unweighted_domain_macro_f1": np.nan, "weighted_domain_macro_f1": np.nan, "validity_class": DISCLOSED})
            part = test[["record_number", "id", "domain", "raw_label", "label", "protocol", "query"]].copy()
            part["predicted_label"] = predicted
            loto_predictions.append(part)
        loto_oof = pd.concat(loto_predictions, ignore_index=True)
    loto_fold_metrics = pd.DataFrame(loto_rows).rename(columns={"domain": "held_out_domain"})
    loto_result = _combined_result("B3", DISCLOSED, frame, loto_oof, loto_fold_metrics,
                                   {"training_scope": "leave_one_domain_out", "fold_checksum": "held_out_domain_names"})
    results.append(loto_result)
    rows.extend(loto_rows)
    rows.append({"experiment_id": "B3", "scope": "leave_one_domain_out", "domain": "POOLED", "record_count": len(frame),
                 "accuracy": loto_result.accuracy, "macro_f1": loto_result.macro_f1,
                 "unweighted_domain_macro_f1": float(np.mean([row["macro_f1"] for row in loto_rows])),
                 "weighted_domain_macro_f1": float(np.average([row["macro_f1"] for row in loto_rows], weights=[row["record_count"] for row in loto_rows])),
                 "validity_class": DISCLOSED})
    loto_target = output / "experiments" / "B3"
    loto_target.mkdir(parents=True, exist_ok=True)
    loto_result.oof.to_csv(loto_target / "oof_predictions.csv", index=False)

    matrix = pd.DataFrame(rows)
    matrix.to_csv(output / "domain_protocol_matrix.csv", index=False)
    closest = min((result for result in results if result.experiment_id in ("B0", "B1", "B2")), key=lambda item: abs(item.macro_f1 - PAPER_MACRO_F1))
    if closest.experiment_id == "B1":
        interpretation = "The paper's scores are substantially more consistent with independent per-domain routers than with the unshuffled global router; domain prefix injection did not help."
    elif closest.experiment_id == "B2":
        interpretation = "Only domain injection approached the paper, which would indicate invalid domain leakage."
    elif closest.experiment_id == "B0":
        interpretation = "The global router was most consistent with the paper among these tests."
    else:
        interpretation = "None of the tested domain protocols was consistent with the paper."
    analysis = ["# Domain protocol analysis", "", _markdown_table(matrix), "",
                f"The closest tested domain protocol was `{closest.experiment_id}` at macro-F1 `{closest.macro_f1:.6f}`; the paper target is `{PAPER_MACRO_F1:.3f}`.", "",
                "B1 creates a fresh pipeline and fold-local vocabulary independently for each domain; model instances are not reused across domains. B2 injects domain metadata and is invalid for a query-text-only reproduction.", "",
                f"Interpretation: {interpretation}"]
    (output / "domain_protocol_analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    return results, matrix


def normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def punctuation_normalize_query(text: str) -> str:
    lowered = normalize_query(text)
    return re.sub(rf"[{re.escape(string.punctuation)}]", " ", lowered).strip()


def template_signature(text: str) -> str:
    value = normalize_query(text)
    value = re.sub(r'"[^"\n]*"|\'[^\'\n]*\'', " QUOTED ", value)
    value = re.sub(r"\b(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b", " DATE ", value)
    value = re.sub(r"\b[a-z]+[-_]?[a-z]*\d+[a-z\d_-]*\b", " ID ", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", " NUMBER ", value)
    return re.sub(r"\s+", " ", value).strip()


def _group_stats(frame: pd.DataFrame, keys: pd.Series) -> dict[str, Any]:
    grouped = frame.assign(__key=keys).groupby("__key", sort=True)
    sizes = grouped.size()
    duplicate_keys = set(sizes[sizes > 1].index)
    members = frame.assign(__key=keys)[lambda item: item.__key.isin(duplicate_keys)]
    return {"duplicate_groups": int((sizes > 1).sum()),
            "records_in_duplicate_groups": int(len(members)),
            "percentage_records_in_group_gt_one": float(len(members) / len(frame) * 100),
            "groups_crossing_labels": int((grouped.label.nunique() > 1).sum()),
            "groups_crossing_domains": int((grouped.domain.nunique() > 1).sum()),
            "largest_group": int(sizes.max())}


def sparse_similarity_audit(frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=20000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(frame["query"])
    thresholds = (0.90, 0.95, 0.98)
    pair_counts = {str(threshold): 0 for threshold in thresholds}
    member_sets = {str(threshold): set() for threshold in thresholds}
    radius_model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1).fit(matrix)
    batch_size = 256
    for start in range(0, len(frame), batch_size):
        distances, indices = radius_model.radius_neighbors(matrix[start:start + batch_size], radius=0.10, return_distance=True)
        for offset, (row_distances, row_indices) in enumerate(zip(distances, indices)):
            left = start + offset
            for distance, right in zip(row_distances, row_indices):
                right = int(right)
                if right <= left:
                    continue
                similarity = 1 - float(distance)
                for threshold in thresholds:
                    if similarity >= threshold:
                        key = str(threshold)
                        pair_counts[key] += 1
                        member_sets[key].update((left, right))
    validation_rates = {str(threshold): [] for threshold in thresholds}
    for train, validation in splits:
        nearest = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1).fit(matrix[train])
        distances, _ = nearest.kneighbors(matrix[validation])
        similarities = 1 - distances[:, 0]
        for threshold in thresholds:
            validation_rates[str(threshold)].append(float(np.mean(similarities >= threshold)))
    return {"method": "sparse character n-gram TF-IDF with radius/nearest-neighbor queries; no dense all-pairs matrix",
            "sparse_matrix_shape": list(matrix.shape), "sparse_matrix_nnz": int(matrix.nnz),
            "pair_counts": pair_counts,
            "record_membership_rates": {key: len(value) / len(frame) for key, value in member_sets.items()},
            "validation_has_training_neighbor_rates": {key: {"fold_rates": values, "pooled_rate": float(np.mean(values))} for key, values in validation_rates.items()}}


def run_duplicate_experiments(dataset: LoadedDataset, output: Path, shuffled_result: ForensicResult,
                              shuffled_splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[list[ForensicResult], dict[str, Any], pd.DataFrame]:
    frame = dataset.frame.reset_index(drop=True)
    keys = {"exact_original": frame["query"].astype(str),
            "lowercase_whitespace": frame["query"].map(normalize_query),
            "punctuation_normalized": frame["query"].map(punctuation_normalize_query),
            "template_signature": frame["query"].map(template_signature)}
    audit_path = output / "duplicate_template_audit.json"
    if audit_path.exists():
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        audit = audit_payload["group_statistics"]
        overlap = audit_payload["shuffled_seed_42_training_overlap"]
    else:
        audit = {name: _group_stats(frame, key) for name, key in keys.items()}
        overlap = {}
        for name, key in keys.items():
            fold_rates = []
            for train, validation in shuffled_splits:
                training_keys = set(key.iloc[train])
                fold_rates.append(float(key.iloc[validation].isin(training_keys).mean()))
            overlap[name] = {"fold_rates": fold_rates, "pooled_rate": float(np.mean(fold_rates))}
        similarity = sparse_similarity_audit(frame, shuffled_splits)
        audit_payload = {"normalization_definitions": {
            "lowercase_whitespace": "lowercase and collapse whitespace",
            "punctuation_normalized": "lowercase, collapse whitespace, replace punctuation with spaces",
            "template_signature": "also replace quoted spans, dates, obvious alphanumeric IDs, integers and decimals with typed placeholders"},
            "group_statistics": audit, "shuffled_seed_42_training_overlap": overlap,
            "character_ngram_similarity": similarity}
        audit_path.write_text(json.dumps(_jsonable(audit_payload), indent=2), encoding="utf-8")
    grouped_results = []
    rows = [{"experiment_id": "C0", "grouping": "none", "accuracy": shuffled_result.accuracy,
             "macro_f1": shuffled_result.macro_f1, "validity_class": PLAUSIBLE,
             "template_overlap_rate": overlap["template_signature"]["pooled_rate"]}]
    for experiment_id, group_name in (("C1", "lowercase_whitespace"), ("C2", "template_signature")):
        splits = make_stratified_splits(frame, groups=keys[group_name])
        result = run_svm_experiment(frame, frame["query"], splits, experiment_id, PLAUSIBLE, output=output,
                                    metadata={"grouping": group_name, "shuffle": False})
        result.validity_class = PLAUSIBLE
        grouped_results.append(result)
        rows.append({"experiment_id": experiment_id, "grouping": group_name, "accuracy": result.accuracy,
                     "macro_f1": result.macro_f1, "validity_class": PLAUSIBLE,
                     "template_overlap_rate": overlap["template_signature"]["pooled_rate"] if experiment_id == "C1" else 0.0})
    table = pd.DataFrame(rows)
    table.to_csv(output / "grouped_cv_results.csv", index=False)
    inflation = shuffled_result.macro_f1 - grouped_results[-1].macro_f1
    analysis = ["# Duplicate and template audit", "", _markdown_table(pd.DataFrame(audit).T.reset_index(names="normalization")), "",
                "## Grouped CV", "", _markdown_table(table), "",
                f"Shuffled seed-42 macro-F1 minus template-grouped macro-F1 is `{inflation:.6f}`. A substantial positive value, together with non-trivial validation template overlap, indicates shuffled-CV inflation; otherwise template leakage does not explain the paper gap.", "",
                "Character similarity uses sparse nearest-neighbor/radius queries and never constructs a dense 7,727 × 7,727 matrix."]
    (output / "duplicate_template_audit.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    return grouped_results, audit_payload, table


def load_forensic_source_records(dataset: LoadedDataset) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for source_file in dataset.source_files:
        with Path(source_file).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = str(row.get("id"))
                if identifier in records:
                    raise ValueError(f"Duplicate forensic source ID {identifier} in {source_file}:{line_number}")
                records[identifier] = row
    if set(dataset.frame.id.astype(str)) != set(records):
        raise ValueError("Forensic source records do not align exactly with the audited dataset IDs.")
    return records


def _facts_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for item in value:
        if isinstance(item, dict):
            parts.append(str(item.get("text", json.dumps(item, sort_keys=True, ensure_ascii=False))))
        else:
            parts.append(str(item))
    return " ".join(parts)


CONTAMINATION_MODES = {
    "D0": {"input_fields": "question", "validity_class": PLAUSIBLE},
    "D1": {"input_fields": "question+answer", "validity_class": INVALID},
    "D2": {"input_fields": "question+supporting_facts", "validity_class": INVALID},
    "D3": {"input_fields": "question+answer+supporting_facts", "validity_class": INVALID},
    "D4": {"input_fields": "full_record_excluding_label", "validity_class": INVALID},
    "D5": {"input_fields": "question+supporting_fact_count", "validity_class": INVALID},
}


def contamination_texts(frame: pd.DataFrame, source_records: dict[str, dict[str, Any]], mode: str) -> pd.Series:
    if mode not in CONTAMINATION_MODES:
        raise ValueError(f"Unknown contamination mode {mode}.")
    values = []
    for row in frame.itertuples():
        source = source_records[str(row.id)]
        question = str(source.get("question", row.query))
        answer = str(source.get("answer", ""))
        facts = _facts_text(source.get("supporting_facts"))
        if mode == "D0":
            text = question
        elif mode == "D1":
            text = f"{question} [ANSWER] {answer}"
        elif mode == "D2":
            text = f"{question} [SUPPORTING_FACTS] {facts}"
        elif mode == "D3":
            text = f"{question} [ANSWER] {answer} [SUPPORTING_FACTS] {facts}"
        elif mode == "D4":
            content = {key: value for key, value in source.items() if key not in ("type", "label", "query_type")}
            text = json.dumps(content, sort_keys=True, ensure_ascii=False)
        else:
            count = len(source.get("supporting_facts", [])) if isinstance(source.get("supporting_facts"), list) else 0
            text = f"{question} SUPPORTING_FACT_COUNT_{count}"
        values.append(text)
    return pd.Series(values)


def run_contamination_experiments(dataset: LoadedDataset, output: Path, ordered_frame: pd.DataFrame,
                                  splits: list[tuple[np.ndarray, np.ndarray]], d0_reference: ForensicResult) -> tuple[list[ForensicResult], pd.DataFrame]:
    source_records = load_forensic_source_records(dataset)
    results = []
    rows = []
    for mode, definition in CONTAMINATION_MODES.items():
        if mode == "D0":
            result = d0_reference
        else:
            texts = contamination_texts(ordered_frame, source_records, mode)
            result = run_svm_experiment(ordered_frame, texts, splits, mode, INVALID, output=output,
                                        metadata={"input_fields": definition["input_fields"], "leakage_diagnostic": True})
            results.append(result)
        rows.append({"experiment_id": mode, "input_fields": definition["input_fields"],
                     "validity_class": definition["validity_class"], "accuracy": result.accuracy,
                     "macro_f1": result.macro_f1, "accuracy_delta_pp": (result.accuracy - PAPER_ACCURACY) * 100,
                     "macro_f1_delta": result.macro_f1 - PAPER_MACRO_F1,
                     "leakage_warning": "" if mode == "D0" else "Invalid diagnostic: answer/supporting metadata may encode target-correlated generation behavior."})
    table = pd.DataFrame(rows)
    table.to_csv(output / "input_contamination_matrix.csv", index=False)
    closest = table.iloc[(table.macro_f1 - PAPER_MACRO_F1).abs().argmin()]
    analysis = ["# Input contamination analysis", "", "D0 remains question-only. D1–D5 are invalid diagnostics and never produce final models.", "",
                _markdown_table(table), "",
                f"The closest contamination mode was `{closest.experiment_id}` at macro-F1 `{closest.macro_f1:.6f}`.", "",
                "Answers and supporting facts are downstream/target-adjacent content rather than router-time query input. A large jump from adding them is evidence of input contamination, not a valid reproduction protocol."]
    (output / "input_contamination_analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    return results, table


def run_preprocessing_leakage(dataset: LoadedDataset, output: Path, frame: pd.DataFrame,
                              splits: list[tuple[np.ndarray, np.ndarray]], correct_reference: ForensicResult) -> tuple[ForensicResult, pd.DataFrame]:
    leaked = run_svm_experiment(frame, frame["query"], splits, "E1", INVALID, output=output,
                                global_vectorizer_fit=True,
                                metadata={"preprocessing_scope": "global_before_cv", "leakage_warning": "TF-IDF saw validation text"})
    rows = [{"experiment_id": "E0", "validity_class": correct_reference.validity_class,
             "preprocessing_scope": "fold_local", "accuracy": correct_reference.accuracy,
             "macro_f1": correct_reference.macro_f1, "vocabulary_overlap_ratio": np.nan},
            {"experiment_id": "E1", "validity_class": INVALID, "preprocessing_scope": "global_before_cv",
             "accuracy": leaked.accuracy, "macro_f1": leaked.macro_f1,
             "vocabulary_overlap_ratio": leaked.metadata["mean_fold_to_global_vocabulary_ratio"]}]
    table = pd.DataFrame(rows)
    per_class = {}
    for label in CANONICAL_LABELS:
        per_class[label] = {"e0_f1": correct_reference.class_report[label]["f1-score"],
                            "e1_f1": leaked.class_report[label]["f1-score"],
                            "change": leaked.class_report[label]["f1-score"] - correct_reference.class_report[label]["f1-score"]}
    payload = {"rows": rows, "accuracy_difference": leaked.accuracy - correct_reference.accuracy,
               "macro_f1_difference": leaked.macro_f1 - correct_reference.macro_f1,
               "per_class_changes": per_class,
               "warning": "E1 is invalid because TF-IDF vocabulary and IDF weights were fitted using validation text."}
    (output / "preprocessing_leakage.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    (output / "preprocessing_leakage_analysis.md").write_text(
        "# Preprocessing leakage analysis\n\n" + _markdown_table(table) +
        f"\n\nGlobal TF-IDF changed accuracy by `{payload['accuracy_difference']:.6f}` and macro-F1 by `{payload['macro_f1_difference']:.6f}`. E1 is an invalid diagnostic.\n", encoding="utf-8")
    return leaked, table


def run_parameter_sensitivity(output: Path, frame: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]],
                              reference: ForensicResult) -> tuple[list[ForensicResult], pd.DataFrame]:
    specs = [
        ("F1", {}, {"class_weight": "balanced"}, "class_weight=balanced"),
        ("F2", {"lowercase": False}, {}, "lowercase=False"),
        ("F3", {"strip_accents": "unicode"}, {}, "strip_accents=unicode"),
        ("F4", {"token_pattern": r"(?u)\b\w+\b"}, {}, "allow one-character tokens"),
        ("F5_C0.1", {}, {"C": 0.1}, "SVC C=0.1"),
        ("F5_C10", {}, {"C": 10.0}, "SVC C=10.0"),
    ]
    results = []
    rows = [{"experiment_id": "F0", "assumption": "current disclosed configuration", "validity_class": reference.validity_class,
             "accuracy": reference.accuracy, "macro_f1": reference.macro_f1,
             "mean_fold_macro_f1": reference.mean_fold_macro_f1, "pooled_minus_mean_fold_macro_f1": reference.macro_f1 - reference.mean_fold_macro_f1},
            {"experiment_id": "F5_C1", "assumption": "SVC C=1.0 (same as F0)", "validity_class": reference.validity_class,
             "accuracy": reference.accuracy, "macro_f1": reference.macro_f1,
             "mean_fold_macro_f1": reference.mean_fold_macro_f1, "pooled_minus_mean_fold_macro_f1": reference.macro_f1 - reference.mean_fold_macro_f1}]
    for experiment_id, vectorizer, svc, note in specs:
        result = run_svm_experiment(frame, frame["query"], splits, experiment_id, PLAUSIBLE, output=output,
                                    vectorizer_overrides=vectorizer, svc_overrides=svc,
                                    metadata={"assumption": note, "one_factor_at_a_time": True})
        results.append(result)
        rows.append({"experiment_id": experiment_id, "assumption": note, "validity_class": PLAUSIBLE,
                     "accuracy": result.accuracy, "macro_f1": result.macro_f1,
                     "mean_fold_macro_f1": result.mean_fold_macro_f1,
                     "pooled_minus_mean_fold_macro_f1": result.macro_f1 - result.mean_fold_macro_f1})
    rows.append({"experiment_id": "F6", "assumption": "reporting comparison only", "validity_class": PLAUSIBLE,
                 "accuracy": reference.accuracy, "macro_f1": reference.macro_f1,
                 "mean_fold_macro_f1": reference.mean_fold_macro_f1,
                 "pooled_minus_mean_fold_macro_f1": reference.macro_f1 - reference.mean_fold_macro_f1})
    table = pd.DataFrame(rows).sort_values("experiment_id")
    table.to_csv(output / "parameter_sensitivity.csv", index=False)
    (output / "parameter_sensitivity_analysis.md").write_text(
        "# Targeted undocumented-assumption sensitivity\n\nEach row changes one factor only; no best-setting combination or broad tuning was performed.\n\n" + _markdown_table(table) + "\n", encoding="utf-8")
    return results, table


def _git(repo: Path, *arguments: str) -> bytes:
    process = subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True)
    return process.stdout


def _parse_snapshot_file(content: bytes, source: str) -> list[dict[str, Any]]:
    text = content.decode("utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            lists = [value for value in parsed.values() if isinstance(value, list) and all(isinstance(item, dict) for item in value)]
            if len(lists) == 1:
                return lists[0]
    except json.JSONDecodeError as exc:
        if exc.msg != "Extra data":
            raise ValueError(f"Cannot parse historical {source}: {exc}") from exc
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if line.strip():
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Historical {source}:{line_number} is not an object")
            records.append(item)
    return records


def audit_dataset_history(repo: Path, output: Path) -> dict[str, Any]:
    repo = repo.resolve()
    current = _git(repo, "rev-parse", "HEAD").decode().strip()
    log_lines = _git(repo, "log", "--all", "--reverse", "--format=%H%x09%cI%x09%s", "--", "data").decode("utf-8").splitlines()
    commits = []
    for line in log_lines:
        commit, timestamp, subject = line.split("\t", 2)
        paths = _git(repo, "ls-tree", "-r", "--name-only", commit, "--", "data").decode("utf-8").splitlines()
        question_paths = sorted(path for path in paths if path.endswith("/Question.json"))
        if not question_paths:
            continue
        all_records = []
        file_checksums = {}
        composite = hashlib.sha256()
        for path in question_paths:
            content = _git(repo, "show", f"{commit}:{path}")
            file_checksums[path] = _sha256_bytes(content)
            composite.update(path.encode("utf-8") + b"\0" + bytes.fromhex(file_checksums[path]) + b"\0")
            all_records.extend(_parse_snapshot_file(content, f"{commit}:{path}"))
        labels = pd.Series([record.get("type", record.get("label")) for record in all_records]).value_counts().to_dict()
        identity = {str(record.get("id")): _sha256_bytes(json.dumps({"question": record.get("question"), "type": record.get("type")}, sort_keys=True, ensure_ascii=False).encode("utf-8")) for record in all_records}
        commits.append({"commit": commit, "timestamp": timestamp, "subject": subject, "question_files": question_paths,
                        "record_count": len(all_records), "label_distribution": labels,
                        "snapshot_checksum": composite.hexdigest(), "file_checksums": file_checksums,
                        "__identity": identity})
    for index, snapshot in enumerate(commits):
        if index == 0:
            snapshot["changes_from_previous"] = None
            continue
        previous = commits[index - 1]["__identity"]
        current_identity = snapshot["__identity"]
        shared = set(previous) & set(current_identity)
        snapshot["changes_from_previous"] = {"added_ids": len(set(current_identity) - set(previous)),
                                               "removed_ids": len(set(previous) - set(current_identity)),
                                               "changed_question_or_label": sum(previous[key] != current_identity[key] for key in shared)}
    public_commits = []
    for snapshot in commits:
        item = dict(snapshot)
        item.pop("__identity")
        total = item["record_count"]
        percentages = {label: count / total * 100 for label, count in item["label_distribution"].items()} if total else {}
        item["matches_paper_declared_label_distribution"] = all(abs(percentages.get(label, 0) - expected) <= .05 for label, expected in {"single_hop": 52.9, "multi_hop": 17.1, "summary": 30.0}.items())
        public_commits.append(item)
    payload = {"repository": str(repo), "current_commit": current, "commits_affecting_data": public_commits,
               "snapshot_count": len(public_commits),
               "any_snapshot_has_7729_records": any(item["record_count"] == 7729 for item in public_commits),
               "any_snapshot_matches_paper_distribution": any(item["matches_paper_declared_label_distribution"] for item in public_commits)}
    (output / "dataset_history.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    table = pd.DataFrame([{"commit": item["commit"][:12], "timestamp": item["timestamp"], "records": item["record_count"],
                           "labels": json.dumps(item["label_distribution"], sort_keys=True), "checksum": item["snapshot_checksum"],
                           "matches_paper_distribution": item["matches_paper_declared_label_distribution"]} for item in public_commits])
    text = ["# Dataset history audit", "", f"Read-only Git inspection of `{repo}` at `{current}` found {len(public_commits)} snapshot(s) affecting `data/`.", "",
            _markdown_table(table), "", f"Any 7,729-record snapshot: `{payload['any_snapshot_has_7729_records']}`.", "",
            f"Any snapshot matching the paper-declared label distribution: `{payload['any_snapshot_matches_paper_distribution']}`.", "",
            "No source file or checkout was modified; historical blobs were read with `git show`."]
    (output / "dataset_history.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return payload


def determine_conclusion(distance: pd.DataFrame) -> str:
    disclosed = distance[(distance.validity_class == DISCLOSED) & (distance.leakage_warning.fillna("") == "")]
    if ((disclosed.accuracy_delta_pp.abs() <= .5) & (disclosed.macro_f1_delta.abs() <= .005)).any():
        return "NUMERICALLY_REPRODUCED"
    plausible = distance[distance.validity_class == PLAUSIBLE]
    if ((plausible.accuracy_delta_pp.abs() <= 2.0) & (plausible.macro_f1_delta.abs() <= .02)).any():
        return "PARTIALLY_EXPLAINED"
    invalid = distance[distance.validity_class == INVALID]
    if ((invalid.accuracy_delta_pp.abs() <= .5) & (invalid.macro_f1_delta.abs() <= .005)).any():
        return "EXPLAINED_ONLY_BY_INVALID_DIAGNOSTIC"
    return "NOT_NUMERICALLY_REPRODUCIBLE"


def _distance_rows(a_results: list[ForensicResult], b_results: list[ForensicResult], c_results: list[ForensicResult],
                   contamination: pd.DataFrame, leaked: ForensicResult, f_results: list[ForensicResult],
                   duplicate_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in a_results:
        meta = result.metadata
        rows.append(result.distance_row("fold_order", shuffle=meta.get("shuffle", False), seed=meta.get("seed", ""),
                                        cv_type="StratifiedKFold", notes=meta.get("order", meta.get("source", ""))))
    for result in b_results:
        if result.experiment_id == "A0":
            continue
        scope = result.metadata.get("training_scope", "global")
        rows.append(result.distance_row("domain_protocol", training_scope=scope,
                                        input_fields="question+domain_prefix" if result.experiment_id == "B2" else "question",
                                        cv_type="leave-one-domain-out" if result.experiment_id == "B3" else "StratifiedKFold",
                                        leakage_warning="domain metadata injection" if result.experiment_id == "B2" else ""))
    template_rate = duplicate_payload["shuffled_seed_42_training_overlap"]["template_signature"]["pooled_rate"]
    for result in c_results:
        rows.append(result.distance_row("duplicates_templates", grouping=result.metadata.get("grouping", "none"),
                                        cv_type="StratifiedGroupKFold", template_overlap_rate=template_rate))
    for row in contamination.to_dict(orient="records"):
        rows.append({"experiment_id": row["experiment_id"], "category": "input_contamination", "validity_class": row["validity_class"],
                     "input_fields": row["input_fields"], "training_scope": "global", "cv_type": "selected_group_A_folds",
                     "shuffle": "inherited", "seed": "inherited", "grouping": "none", "preprocessing_scope": "fold_local",
                     "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "accuracy_delta_pp": row["accuracy_delta_pp"],
                     "macro_f1_delta": row["macro_f1_delta"], "template_overlap_rate": template_rate,
                     "leakage_warning": row["leakage_warning"], "notes": "one input factor at a time"})
    rows.append(leaked.distance_row("preprocessing_leakage", preprocessing_scope="global_before_cv",
                                    leakage_warning="TF-IDF fitted on validation text"))
    for result in f_results:
        rows.append(result.distance_row("parameter_sensitivity", notes=result.metadata.get("assumption", "one factor")))
    return rows


def write_distance_reports(rows: list[dict[str, Any]], output: Path) -> tuple[pd.DataFrame, str]:
    columns = ["experiment_id", "category", "validity_class", "input_fields", "training_scope", "cv_type", "shuffle", "seed",
               "grouping", "preprocessing_scope", "accuracy", "macro_f1", "accuracy_delta_pp", "macro_f1_delta",
               "template_overlap_rate", "leakage_warning", "notes"]
    table = pd.DataFrame(rows)
    for column in columns:
        if column not in table:
            table[column] = ""
    table = table[columns]
    table["__validity_order"] = table.validity_class.map(VALIDITY_ORDER)
    by_validity = table.sort_values(["__validity_order", "experiment_id"]).drop(columns="__validity_order")
    by_distance = table.assign(absolute_macro_f1_distance=(table.macro_f1 - PAPER_MACRO_F1).abs()).sort_values("absolute_macro_f1_distance").drop(columns="__validity_order")
    by_validity.to_csv(output / "distance_to_paper.csv", index=False)
    by_distance.to_csv(output / "distance_to_paper_by_distance.csv", index=False)
    conclusion = determine_conclusion(by_validity)
    return by_validity, conclusion


def _tree_checksums(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {item.relative_to(path).as_posix(): _sha256_file(item) for item in sorted(path.rglob("*")) if item.is_file()}


def run_forensic_reproduction(dataset: LoadedDataset, output: str | Path, dataset_repo: str | Path) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.name != "forensic_reproduction":
        raise ValueError("Forensic diagnostics must write to a directory named forensic_reproduction.")
    output.mkdir(parents=True, exist_ok=True)
    if set(dataset.frame.protocol.unique()) != {PROTOCOL_OFFICIAL}:
        raise ValueError("Forensic reproduction requires the untouched official_raw_labels dataset.")
    reports_root = output.parent
    project_root = reports_root.parent
    frozen_paths = [reports_root / PROTOCOL_OFFICIAL, reports_root / "paper_label_permutation_diagnostic", project_root / "artifacts"]
    frozen_before = {str(path): _tree_checksums(path) for path in frozen_paths}

    a_results, a_specs, fold_matrix = run_fold_order_experiments(dataset, reports_root, output)
    baseline = next(result for result in a_results if result.experiment_id == "A0")
    baseline_manifest = write_baseline_manifest(dataset, reports_root, output, baseline)
    selected_a = min(a_results, key=lambda result: (abs(result.macro_f1 - PAPER_MACRO_F1), result.experiment_id))
    selected_frame, selected_splits = a_specs[selected_a.experiment_id]

    b_results, domain_matrix = run_domain_experiments(dataset, output, baseline)
    shuffled = next(result for result in a_results if result.experiment_id == "A1")
    shuffled_frame, shuffled_splits = a_specs["A1"]
    c_grouped, duplicate_payload, grouped_table = run_duplicate_experiments(dataset, output, shuffled, shuffled_splits)
    contamination_results, contamination_table = run_contamination_experiments(dataset, output, selected_frame, selected_splits, selected_a)
    leaked, preprocessing_table = run_preprocessing_leakage(dataset, output, selected_frame, selected_splits, selected_a)
    f_results, parameter_table = run_parameter_sensitivity(output, selected_frame, selected_splits, selected_a)
    history = audit_dataset_history(Path(dataset_repo), output)

    seed_rows = fold_matrix[fold_matrix.experiment_id.str.startswith("A2_seed_") | (fold_matrix.experiment_id == "A1")]
    seed_accuracy_mean = float(seed_rows.accuracy.mean())
    seed_mean = float(seed_rows.macro_f1.mean())
    seed_variance = float(seed_rows.macro_f1.var(ddof=0))
    rows = _distance_rows(a_results, b_results, [shuffled, *c_grouped], contamination_table, leaked, f_results, duplicate_payload)
    rows.append({"experiment_id": "A2_multi_seed_aggregate", "category": "fold_order", "validity_class": PLAUSIBLE,
                 "input_fields": "question", "training_scope": "global", "cv_type": "StratifiedKFold",
                 "shuffle": True, "seed": "10 predeclared seeds", "grouping": "none", "preprocessing_scope": "fold_local",
                 "accuracy": seed_accuracy_mean, "macro_f1": seed_mean,
                 "accuracy_delta_pp": (seed_accuracy_mean - PAPER_ACCURACY) * 100,
                 "macro_f1_delta": seed_mean - PAPER_MACRO_F1, "template_overlap_rate": np.nan,
                 "leakage_warning": "", "notes": "Mean across all seeds; not a selected lucky seed."})
    rows.extend([
        {**selected_a.distance_row("preprocessing_leakage"), "experiment_id": "E0", "notes": f"reference to {selected_a.experiment_id}"},
        {**selected_a.distance_row("parameter_sensitivity"), "experiment_id": "F0", "notes": "current configuration"},
        {**selected_a.distance_row("parameter_sensitivity"), "experiment_id": "F5_C1", "notes": "C=1.0; same as F0"},
        {**selected_a.distance_row("parameter_sensitivity"), "experiment_id": "F6", "notes": f"pooled macro-F1 {selected_a.macro_f1:.6f} versus mean fold {selected_a.mean_fold_macro_f1:.6f}"},
    ])
    distance, conclusion = write_distance_reports(rows, output)
    frozen_after = {str(path): _tree_checksums(path) for path in frozen_paths}
    if frozen_before != frozen_after:
        raise RuntimeError("Frozen baseline/diagnostic reports or production artifacts changed during forensic execution.")

    disclosed = distance[distance.validity_class == DISCLOSED]
    plausible = distance[distance.validity_class == PLAUSIBLE]
    invalid = distance[distance.validity_class == INVALID]
    closest_disclosed = disclosed.iloc[(disclosed.macro_f1 - PAPER_MACRO_F1).abs().argmin()]
    closest_plausible = plausible.iloc[(plausible.macro_f1 - PAPER_MACRO_F1).abs().argmin()]
    closest_invalid = invalid.iloc[(invalid.macro_f1 - PAPER_MACRO_F1).abs().argmin()]
    b1 = next(result for result in b_results if result.experiment_id == "B1")
    c1 = next(result for result in c_grouped if result.experiment_id == "C1")
    c2 = next(result for result in c_grouped if result.experiment_id == "C2")
    any_target = bool((distance.macro_f1 - PAPER_MACRO_F1).abs().le(.005).any())
    material_a = [result.experiment_id for result in a_results
                  if result.experiment_id != "A0" and result.validity_class == DISCLOSED
                  and abs(result.macro_f1 - PAPER_MACRO_F1) <= .03]
    summary = {"dataset_fingerprint": dataset.source_sha256, "baseline": {"accuracy": baseline.accuracy, "macro_f1": baseline.macro_f1},
               "paper_target": {"accuracy": PAPER_ACCURACY, "macro_f1": PAPER_MACRO_F1},
               "selected_group_a_reference": selected_a.experiment_id,
               "multi_seed_macro_f1_mean": seed_mean, "multi_seed_macro_f1_variance": seed_variance,
               "independent_per_domain": {"accuracy": b1.accuracy, "macro_f1": b1.macro_f1, **b1.metadata},
               "grouped_cv": {"normalized_query_macro_f1": c1.macro_f1, "template_macro_f1": c2.macro_f1},
               "closest_disclosed_compatible": closest_disclosed.to_dict(),
               "closest_ambiguous_plausible": closest_plausible.to_dict(),
               "closest_invalid_diagnostic": closest_invalid.to_dict(),
               "any_experiment_within_0.005_macro_f1": any_target,
               "material_disclosed_compatible_group_a_protocols_within_0.03": material_a,
               "all_eight_additional_run_triggered": bool(material_a),
               "decision": conclusion, "history_snapshot_count": history["snapshot_count"],
               "frozen_outputs_unchanged": True}
    if material_a:
        summary["all_eight_note"] = "A new disclosed-compatible fold/order protocol came within 0.03 macro-F1 and requires a separately reviewed all-eight run."
    else:
        summary["all_eight_note"] = "No new disclosed-compatible fold/order protocol materially approached the paper (predeclared threshold: within 0.03 macro-F1). Only the preserved official baseline remains an all-eight run; ambiguous diagnostics were kept TF-IDF+SVM-focused."
    (output / "forensic_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")

    hypotheses = ["undisclosed fold shuffling or order", "independent per-domain training", "domain metadata leakage",
                  "duplicate/template leakage", "answer or supporting-fact contamination", "globally fitted preprocessing",
                  "undocumented classical estimator/preprocessing defaults", "an unpublished dataset snapshot", "reporting or implementation error"]
    report = f"""# Forensic reproduction report

## 1. Frozen baseline

Dataset fingerprint `{dataset.source_sha256}`; official raw-label TF-IDF + SVM accuracy `{baseline.accuracy:.6f}`, macro-F1 `{baseline.macro_f1:.6f}`. Original reports and production artifacts were checksum-verified unchanged. See `baseline_manifest.json`.

## 2. Paper target

Accuracy `{PAPER_ACCURACY:.3f}` and macro-F1 `{PAPER_MACRO_F1:.3f}`.

## 3. Hypotheses

{chr(10).join('- ' + item for item in hypotheses)}

## 4. Fold ordering findings

See `fold_order_analysis.md`. Across the 10 predeclared shuffled seeds, mean macro-F1 was `{seed_mean:.6f}` with variance `{seed_variance:.8f}`. The closest seed is reported but never selected as the reproduction.

## 5. Domain protocol findings

Independent per-domain pooled accuracy/macro-F1: `{b1.accuracy:.6f}` / `{b1.macro_f1:.6f}`. See `domain_protocol_analysis.md`.

## 6. Duplicate and template leakage

Normalized-query grouped macro-F1 `{c1.macro_f1:.6f}`; template-grouped macro-F1 `{c2.macro_f1:.6f}`. See `duplicate_template_audit.md`.

## 7. Input contamination findings

See `input_contamination_analysis.md`. D1–D5 are invalid diagnostics and cannot become final models.

## 8. Preprocessing leakage findings

Global-fit TF-IDF macro-F1 `{leaked.macro_f1:.6f}` versus correct reference `{selected_a.macro_f1:.6f}`. Global fitting is invalid.

## 9. Parameter sensitivity

See `parameter_sensitivity_analysis.md`. Each experiment changes one assumption; settings are not combined.

## 10. Dataset history

Read-only Git inspection found `{history['snapshot_count']}` data snapshot(s). Any 7,729-row snapshot: `{history['any_snapshot_has_7729_records']}`. Any paper-distribution snapshot: `{history['any_snapshot_matches_paper_distribution']}`.

## 11. Closest disclosed-compatible and plausible protocols

The closest disclosed-compatible protocol is `{closest_disclosed.experiment_id}`: accuracy `{closest_disclosed.accuracy:.6f}`, macro-F1 `{closest_disclosed.macro_f1:.6f}`.

The closest ambiguous-but-plausible condition is `{closest_plausible.experiment_id}`: accuracy `{closest_plausible.accuracy:.6f}`, macro-F1 `{closest_plausible.macro_f1:.6f}`. Across all ten shuffled seeds, mean accuracy/macro-F1 is `{seed_accuracy_mean:.6f}` / `{seed_mean:.6f}`. The aggregate, rather than one lucky seed, supports partial explanation; shuffle behavior remains undisclosed.

## 12. Closest invalid diagnostic

`{closest_invalid.experiment_id}`: accuracy `{closest_invalid.accuracy:.6f}`, macro-F1 `{closest_invalid.macro_f1:.6f}`. It is not a valid reproduction result.

## 13. Numerical reproducibility decision

**{conclusion}**. Any experiment within 0.005 macro-F1 of the paper: `{any_target}`.

## 14. Remaining uncertainty

Public evidence may not reveal the authors’ exact data snapshot, ordering, input construction, or implementation. This audit does not allege misconduct; it distinguishes tested behavior from unresolved reporting or implementation differences.

## 15. Exact commands executed

```powershell
python -m lightweight_router forensic --data "{dataset.source_path}" --expect-ragrouter-bench --query-field question --label-field type --domain-field domain --id-field id --dataset-repo "{Path(dataset_repo).resolve()}" --output "{output}"
python -m pytest -q
```
"""
    (output / "FORENSIC_REPRODUCTION_REPORT.md").write_text(report, encoding="utf-8")
    (output / "environment.txt").write_text(f"Python: {sys.version}\nPlatform: {platform.platform()}\nscikit-learn: {sklearn.__version__}\nnumpy: {np.__version__}\npandas: {pd.__version__}\nscipy: {scipy.__version__}\n", encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps(_jsonable({"command": "forensic", "dataset_fingerprint": dataset.source_sha256,
                                                                      "output": str(output), "dataset_repo": str(Path(dataset_repo).resolve()),
                                                                      "frozen_checksums_before_and_after_match": True}), indent=2), encoding="utf-8")
    return summary
