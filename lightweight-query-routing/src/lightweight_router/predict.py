from __future__ import annotations

from typing import Any

import joblib
import numpy as np

from .config import COST_MAPPING, PROTOCOL_OFFICIAL


def predict_query(model_path: str, query: str, domain: str | None = None) -> dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("--query must be non-empty.")
    model = joblib.load(model_path)
    is_per_domain = hasattr(model, "models") and hasattr(model, "feature") and hasattr(model, "classifier")
    if is_per_domain:
        if not domain or not domain.strip():
            raise ValueError("This per-domain model requires --domain with explicit corpus metadata.")
        label = str(model.predict([query], [domain])[0])
    else:
        label = str(model.predict([query])[0])
    result: dict[str, Any] = {"query": query, "predicted_label": label,
                              "recommended_paradigm": COST_MAPPING[label]["paradigm"],
                              "simulated_cost_ratio": COST_MAPPING[label]["cost_ratio"],
                              "model": "tfidf_svm" if "tfidf_svm" in str(model_path).lower() else "saved_model",
                              "protocol": PROTOCOL_OFFICIAL}
    if is_per_domain:
        result["domain"] = domain
    if hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function([query])).reshape(-1)
        result["uncalibrated_decision_score"] = {str(key): float(value) for key, value in zip(model.classes_, score)}
    return result
