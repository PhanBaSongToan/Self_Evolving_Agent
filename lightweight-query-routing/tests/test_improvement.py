from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from lightweight_router.improvement import (
    Candidate,
    PerDomainRouter,
    _save_fitted_candidate,
    nested_threshold_evaluation,
    search_threshold_policy,
    validate_improvement_output_paths,
)
from lightweight_router.improvement_metrics import pareto_frontier, policy_is_valid, routing_error_metrics
from lightweight_router.improvement_models import ONE_CHAR_PATTERN, build_features, build_improvement_estimator
from lightweight_router.predict import predict_query


def _synthetic_frame(records_per_class_domain: int = 9) -> pd.DataFrame:
    rows = []
    terms = {"single_hop": "fact", "multi_hop": "reason chain", "summary": "summarize overview"}
    record_number = 0
    for domain in ("domain_alpha", "domain_beta"):
        domain_term = "alphacorpus" if domain == "domain_alpha" else "betacorpus"
        for label, label_term in terms.items():
            for index in range(records_per_class_domain):
                rows.append({
                    "record_number": record_number,
                    "id": f"q{record_number}",
                    "domain": domain,
                    "raw_label": label,
                    "label": label,
                    "query": f"{domain_term} {label_term} common token variation{index % 3}",
                })
                record_number += 1
    return pd.DataFrame(rows)


def test_one_character_token_configuration():
    vectorizer = build_features("word_3000_onechar")
    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.token_pattern == ONE_CHAR_PATTERN
    vectorizer.fit(["x repeated token", "x another token"])
    assert "x" in vectorizer.vocabulary_


def test_word_character_and_structural_feature_unions():
    word_char = build_features("word_char")
    assert isinstance(word_char, FeatureUnion)
    assert [name for name, _ in word_char.transformer_list] == ["word", "char"]
    extended = build_features("word_char_structural")
    assert [name for name, _ in extended.transformer_list] == ["word", "char", "structural"]
    assert isinstance(dict(extended.transformer_list)["structural"], Pipeline)


def test_per_domain_models_have_independent_vectorizers_and_no_domain_leakage(tmp_path):
    frame = _synthetic_frame()
    router = _save_fitted_candidate(frame, Candidate("word_3000", "linear_svc", "per_domain"), tmp_path)
    assert isinstance(router, PerDomainRouter)
    alpha = router.models["domain_alpha"].named_steps["features"]
    beta = router.models["domain_beta"].named_steps["features"]
    assert alpha is not beta
    assert alpha.vocabulary_ is not beta.vocabulary_
    assert "alphacorpus" in alpha.vocabulary_ and "betacorpus" not in alpha.vocabulary_
    assert "betacorpus" in beta.vocabulary_ and "alphacorpus" not in beta.vocabulary_
    with pytest.raises(ValueError, match="explicit domain"):
        router.predict(["a query"])
    with pytest.raises(ValueError, match="No per-domain router"):
        router.predict(["a query"], ["unseen_domain"])
    with pytest.raises(ValueError, match="requires --domain"):
        predict_query(str(tmp_path / "model.joblib"), "a query")
    cli_prediction = predict_query(str(tmp_path / "model.joblib"), "alphacorpus fact common token",
                                   "domain_alpha")
    assert cli_prediction["domain"] == "domain_alpha"


def test_calibration_wraps_the_complete_fold_local_pipeline():
    calibrated = build_improvement_estimator("word_char", "calibrated_linear_svc")
    assert isinstance(calibrated, CalibratedClassifierCV)
    assert calibrated.cv == 3 and calibrated.method == "sigmoid"
    assert isinstance(calibrated.estimator, Pipeline)
    assert "features" in calibrated.estimator.named_steps
    assert not hasattr(calibrated.estimator.named_steps["features"], "vocabulary_")
    classifier = calibrated.estimator.named_steps["classifier"]
    assert classifier.max_iter == 10000 and classifier.random_state == 42


def test_nested_threshold_selection_uses_only_outer_training_rows():
    frame = _synthetic_frame(8)
    nested = nested_threshold_evaluation(
        frame, Candidate("word_3000", "logistic_regression", "global"),
        outer_splits=3, inner_splits=2, threshold_grid=[.4, .6],
    )
    assert len(nested["audit_rows"]) == 3
    assert all(row["tuning_scope"] == "outer_training_inner_oof" for row in nested["audit_rows"])
    assert all(row["record_overlap_count"] == 0 for row in nested["audit_rows"])
    assert all(row["tuning_record_count"] < len(frame) for row in nested["audit_rows"])
    assert all(row["tuning_scope"] == "outer_training_inner_oof" for row in nested["threshold_rows"])


def test_threshold_search_selects_only_a_constrained_valid_policy():
    true = np.asarray(["single_hop", "multi_hop", "summary"] * 3)
    probabilities = np.asarray([
        [.8, .1, .1], [.1, .8, .1], [.1, .1, .8],
        [.7, .2, .1], [.2, .7, .1], [.1, .2, .7],
        [.9, .05, .05], [.05, .9, .05], [.05, .05, .9],
    ])
    rows, selected, _ = search_threshold_policy(true, probabilities, fallback_label="multi_hop",
                                                 threshold_grid=[.4, .75])
    assert selected is not None
    assert selected["valid_under_constraints"] is True
    assert selected["selected"] is True
    assert sum(row["selected"] for row in rows) == 1


def test_under_and_over_routing_calculation():
    true = ["single_hop", "multi_hop", "summary", "summary"]
    predicted = ["multi_hop", "single_hop", "single_hop", "summary"]
    metrics = routing_error_metrics(true, predicted)
    assert metrics["total_under_routing_rate"] == .5
    assert metrics["total_over_routing_rate"] == .25
    assert metrics["multi_hop_to_single_hop_rate"] == 1.0
    assert metrics["summary_to_single_hop_rate"] == .5


def test_policy_constraints_and_invalid_policy_rejection():
    default = {"accuracy": .90, "macro_f1": .89, "summary_recall": .80,
               "multi_hop_recall": .82, "total_under_routing_rate": .10}
    valid = {"accuracy": .895, "macro_f1": .885, "summary_recall": .80,
             "multi_hop_recall": .81, "total_under_routing_rate": .10}
    invalid = dict(valid, summary_recall=.799)
    assert policy_is_valid(valid, default)
    assert not policy_is_valid(invalid, default)


def test_pareto_dominance_keeps_only_non_dominated_rows():
    table = pd.DataFrame([
        {"name": "dominant", "accuracy": .9, "macro_f1": .9,
         "simulated_savings_percent": 30.0, "total_under_routing_rate": .05},
        {"name": "dominated", "accuracy": .8, "macro_f1": .8,
         "simulated_savings_percent": 20.0, "total_under_routing_rate": .10},
        {"name": "tradeoff", "accuracy": .88, "macro_f1": .88,
         "simulated_savings_percent": 35.0, "total_under_routing_rate": .06},
    ])
    assert set(pareto_frontier(table).name) == {"dominant", "tradeoff"}


def test_improvement_output_guards_reject_existing_artifact_targets(tmp_path):
    reports = tmp_path / "reports" / "improved_router"
    artifacts = tmp_path / "artifacts" / "improved_router"
    assert validate_improvement_output_paths(reports, artifacts)[:2] == (reports.resolve(), artifacts.resolve())
    with pytest.raises(ValueError, match="artifacts/improved_router"):
        validate_improvement_output_paths(reports, tmp_path / "artifacts")
    with pytest.raises(ValueError, match="reports/improved_router"):
        validate_improvement_output_paths(tmp_path / "reports" / "official_raw_labels", artifacts)
