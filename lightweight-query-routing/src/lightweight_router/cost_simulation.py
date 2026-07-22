from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import CANONICAL_LABELS, COST_MAPPING, PAPER_LABEL_PERCENTAGES, PROTOCOLS

ALWAYS_EXPENSIVE_COST = 3.5


def _distribution(labels: pd.Series) -> dict[str, float]:
    return labels.value_counts(normalize=True).reindex(CANONICAL_LABELS, fill_value=0).astype(float).to_dict()


def _cost(labels: pd.Series) -> float:
    return round(float(sum(COST_MAPPING[label]["cost_ratio"] for label in labels)), 10)


def _savings(total_cost: float, n: int) -> float:
    baseline = ALWAYS_EXPENSIVE_COST * n
    return (baseline - total_cost) / baseline * 100 if n else 0.0


def validate_protocol_predictions(oof: pd.DataFrame, protocol: str | None) -> str:
    if not protocol:
        raise ValueError("Cost simulation requires an explicit protocol identifier.")
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown cost protocol '{protocol}'.")
    if "protocol" not in oof.columns:
        raise ValueError("OOF predictions are missing their protocol identifier.")
    observed = set(oof["protocol"].dropna().astype(str).unique())
    if observed != {protocol}:
        raise ValueError(f"Protocol mixing rejected: requested {protocol}, OOF predictions contain {sorted(observed)}.")
    if "raw_true_label" not in oof.columns:
        raise ValueError("OOF predictions are missing raw_true_label provenance.")
    return protocol


def cost_context(true_labels: pd.Series, raw_labels: pd.Series, protocol: str) -> dict[str, Any]:
    if not protocol:
        raise ValueError("Cost context requires a protocol identifier.")
    n = len(true_labels)
    effective_distribution = _distribution(true_labels)
    raw_distribution = _distribution(raw_labels)
    majority_class = str(true_labels.value_counts().idxmax())
    majority_cost = round(n * COST_MAPPING[majority_class]["cost_ratio"], 10)
    empirical_cost = _cost(true_labels)
    paper_fixed_distribution = {label: PAPER_LABEL_PERCENTAGES[label] / 100 for label in CANONICAL_LABELS}
    paper_fixed_average = sum(paper_fixed_distribution[label] * COST_MAPPING[label]["cost_ratio"] for label in CANONICAL_LABELS)
    paper_fixed_cost = round(n * paper_fixed_average, 10)
    distribution_matches_paper = all(abs(effective_distribution[label] - paper_fixed_distribution[label]) <= 0.001 for label in CANONICAL_LABELS)
    return {
        "protocol_name": protocol,
        "raw_label_distribution": raw_distribution,
        "effective_label_distribution": effective_distribution,
        "label_to_paradigm_mapping": {label: COST_MAPPING[label]["paradigm"] for label in CANONICAL_LABELS},
        "paradigm_to_cost_mapping": {COST_MAPPING[label]["paradigm"]: COST_MAPPING[label]["cost_ratio"] for label in CANONICAL_LABELS},
        "majority_class": majority_class, "majority_baseline_cost": majority_cost,
        "majority_baseline_average_cost": majority_cost / n if n else 0.0,
        "majority_baseline_savings_percent": _savings(majority_cost, n),
        "empirical_perfect_label_cost": empirical_cost,
        "empirical_perfect_label_average_cost": empirical_cost / n if n else 0.0,
        "empirical_perfect_label_savings_percent": _savings(empirical_cost, n),
        "paper_fixed_perfect_label_reference": {
            "distribution": paper_fixed_distribution, "total_cost": paper_fixed_cost,
            "average_cost": paper_fixed_average, "savings_percent": _savings(paper_fixed_cost, n),
        },
        "paper_fixed_distribution_warning": "" if distribution_matches_paper else "WARNING: The paper-fixed label distribution does not match the effective local dataset distribution.",
    }


def cost_summary(oof: pd.DataFrame, configuration: str, protocol: str | None = None) -> dict[str, Any]:
    if oof.empty:
        raise ValueError("Cost simulation requires OOF predictions.")
    protocol = validate_protocol_predictions(oof, protocol)
    context = cost_context(oof["true_label"], oof["raw_true_label"], protocol)
    cost = _cost(oof["predicted_label"])
    baseline = ALWAYS_EXPENSIVE_COST * len(oof)
    return {
        "configuration": configuration, **context,
        "macro_f1": float(f1_score(oof["true_label"], oof["predicted_label"], labels=CANONICAL_LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(oof["true_label"], oof["predicted_label"])),
        "predicted_routing_distribution": _distribution(oof["predicted_label"]),
        "total_simulated_cost": cost, "average_simulated_cost_per_query": round(cost / len(oof), 10),
        "always_expensive_baseline_cost": baseline, "simulated_savings_percent": (baseline - cost) / baseline * 100,
        "source": "out_of_fold_predictions_only",
    }


def majority_baseline(true_labels: pd.Series, raw_labels: pd.Series, protocol: str) -> dict[str, Any]:
    majority = str(true_labels.value_counts().idxmax())
    oof = pd.DataFrame({"true_label": true_labels.to_numpy(), "raw_true_label": raw_labels.to_numpy(),
                        "predicted_label": majority, "protocol": protocol})
    return cost_summary(oof, f"majority_{majority}", protocol)


def perfect_label_references(true_labels: pd.Series, raw_labels: pd.Series, protocol: str) -> list[dict[str, Any]]:
    n = len(true_labels)
    empirical = pd.DataFrame({"true_label": true_labels.to_numpy(), "raw_true_label": raw_labels.to_numpy(),
                              "predicted_label": true_labels.to_numpy(), "protocol": protocol})
    empirical_row = cost_summary(empirical, "empirical_perfect_label_reference", protocol)
    context = cost_context(true_labels, raw_labels, protocol)
    paper = context["paper_fixed_perfect_label_reference"]
    paper_row = {"configuration": "paper_fixed_perfect_label_reference", **context, "macro_f1": 1.0, "accuracy": 1.0,
                 "predicted_routing_distribution": paper["distribution"], "total_simulated_cost": paper["total_cost"],
                 "average_simulated_cost_per_query": paper["average_cost"], "always_expensive_baseline_cost": ALWAYS_EXPENSIVE_COST * n,
                 "simulated_savings_percent": paper["savings_percent"], "source": "paper_fixed_label_distribution"}
    return [paper_row, empirical_row]
