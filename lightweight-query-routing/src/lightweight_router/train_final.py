from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from .config import CANONICAL_LABELS, COST_MAPPING, PROTOCOL_DIAGNOSTIC, PROTOCOL_OFFICIAL
from .data import LoadedDataset
from .evaluation import evaluate_all
from .models import build_pipeline, resolved_parameters


def train_final_models(dataset: LoadedDataset, output: str | Path, results=None, protocol: str = PROTOCOL_OFFICIAL) -> dict:
    observed = set(dataset.frame["protocol"].dropna().astype(str).unique())
    if protocol == PROTOCOL_DIAGNOSTIC or observed == {PROTOCOL_DIAGNOSTIC}:
        raise ValueError("Final artifact training rejects the diagnostic label-permutation protocol.")
    if protocol != PROTOCOL_OFFICIAL or observed != {PROTOCOL_OFFICIAL}:
        raise ValueError(f"Final artifacts require protocol {PROTOCOL_OFFICIAL}; observed {sorted(observed)}.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = results if results is not None else evaluate_all(dataset.frame)
    best = max(results, key=lambda item: item.summary["pooled_macro_f1"])
    best_model = build_pipeline(best.name).fit(dataset.frame["query"], dataset.frame["label"])
    svm_model = build_pipeline("tfidf_svm").fit(dataset.frame["query"], dataset.frame["label"])
    joblib.dump(best_model, output / "best_model.joblib")
    joblib.dump(svm_model, output / "tfidf_svm.joblib")
    metadata = {
        "scope": "Reproduction of the non-deep-learning subset of the paper.",
        "protocol": PROTOCOL_OFFICIAL,
        "source_dataset_checksum": dataset.source_sha256, "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version, "platform": platform.platform(), "dependency_versions": {"scikit_learn": sklearn.__version__},
        "selected_best_configuration": best.name, "feature_configuration": "tfidf" if best.name.startswith("tfidf") else "structural_documented_18",
        "model_resolved_parameters": resolved_parameters(best.name), "labels": list(CANONICAL_LABELS),
        "dataset_counts": {"parsed": dataset.total_parsed_records, "valid": len(dataset.frame), "invalid": len(dataset.invalid_records)},
        "cv_results": [result.summary for result in results],
        "known_limitations": ["Only 18 of the paper's claimed 23 structural features are specified.", "Primary CV uses unshuffled folds because shuffle behavior was unspecified."],
    }
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (output / "label_mapping.json").write_text(json.dumps(COST_MAPPING, indent=2), encoding="utf-8")
    return metadata
