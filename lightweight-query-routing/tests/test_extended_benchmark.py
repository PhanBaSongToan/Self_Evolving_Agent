from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from lightweight_router.extended_benchmark import (
    ExtendedCandidate,
    ExtendedResult,
    LAMBDA_VALUES,
    _best_global,
    _risk_policy_valid,
    nested_expected_risk,
    search_expected_risk,
    validate_extended_paths,
)
from lightweight_router.extended_models import (
    OOFStackingTextClassifier,
    MulticlassNBSVM,
    build_extended_estimator,
    build_extended_features,
)


def _texts(n_per_class=12):
    labels = []
    texts = []
    terms = {"single_hop": "fact direct", "multi_hop": "reason chain", "summary": "summary overview"}
    for label, term in terms.items():
        for index in range(n_per_class):
            labels.append(label)
            texts.append(f"{term} common repeated variant{index % 3}")
    return np.asarray(texts, dtype=object), np.asarray(labels, dtype=object)


def _frame(n_per_class=15):
    texts, labels = _texts(n_per_class)
    return pd.DataFrame({
        "record_number": np.arange(len(texts)), "id": [f"q{i}" for i in range(len(texts))],
        "domain": ["domain_a"] * len(texts), "raw_label": labels, "label": labels, "query": texts,
    })


def test_nbsvm_ratios_and_vocabulary_are_training_local():
    texts, labels = _texts()
    model = MulticlassNBSVM(feature_kind="word", classifier="linear_svc", C=1.0)
    model.fit(texts, labels)
    assert model.ratio_training_row_count_ == len(texts)
    assert len(model.log_count_ratios_) == 3
    assert all(ratio.shape[0] == len(model.vectorizer_.vocabulary_) for ratio in model.log_count_ratios_)
    assert "validationonlytoken" not in model.vectorizer_.vocabulary_
    model.predict(["validationonlytoken fact direct"])
    assert "validationonlytoken" not in model.vectorizer_.vocabulary_


def test_multiclass_nbsvm_scores_and_predictions():
    texts, labels = _texts()
    model = MulticlassNBSVM(feature_kind="word_char", classifier="logistic_regression", C=.5).fit(texts, labels)
    scores = model.decision_function(texts[:4])
    probabilities = model.predict_proba(texts[:4])
    assert scores.shape == (4, 3)
    assert probabilities.shape == (4, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(model.predict(texts)) <= {"single_hop", "multi_hop", "summary"}


def test_ridge_complement_and_sgd_configurations():
    ridge = build_extended_estimator("word_3000", "ridge", {"alpha": 1e-3})
    assert ridge.named_steps["classifier"].alpha == 1e-3
    complement = build_extended_estimator("word_char", "complement_nb", {"alpha": .1})
    assert complement.named_steps["classifier"].alpha == .1
    losses = {
        "sgd_hinge": ("hinge", False), "sgd_log_loss": ("log_loss", False),
        "sgd_modified_huber": ("modified_huber", False), "sgd_squared_hinge": ("squared_hinge", False),
        "sgd_hinge_averaged": ("hinge", True), "sgd_log_loss_averaged": ("log_loss", True),
    }
    for name, (loss, averaged) in losses.items():
        classifier = build_extended_estimator("word_3000", name, {"alpha": 1e-4}).named_steps["classifier"]
        assert classifier.loss == loss and bool(classifier.average) is averaged


def test_svd_and_chi_square_are_unfitted_fold_pipeline_steps():
    svd = build_extended_features("svd_100")
    assert isinstance(svd, Pipeline)
    assert "svd" in svd.named_steps and not hasattr(svd.named_steps["svd"], "components_")
    selected = build_extended_features("chi2_1000")
    assert isinstance(selected, Pipeline)
    assert "chi2" in selected.named_steps and not hasattr(selected.named_steps["chi2"], "scores_")
    estimator = build_extended_estimator("svd_100", "extra_trees")
    assert estimator.named_steps["features"] is svd or isinstance(estimator.named_steps["features"], Pipeline)


def test_stacking_meta_model_uses_only_inner_oof_scores():
    texts, labels = _texts(15)
    bases = [build_extended_estimator("word_3000", "ridge", {"alpha": 1e-3}),
             build_extended_estimator("word_3000", "complement_nb", {"alpha": .1})]
    stack = OOFStackingTextClassifier(bases, meta_model="logistic_regression", cv=3).fit(texts, labels)
    assert stack.meta_training_source_ == "inner_oof_only"
    assert np.all(stack.meta_oof_assignment_count_ == len(bases))
    assert len(stack.predict(texts[:5])) == 5


def test_expected_risk_policy_and_fixed_lambda_grid():
    true = np.asarray(["single_hop", "multi_hop", "summary"] * 4)
    probabilities = np.asarray([[.8, .1, .1], [.1, .8, .1], [.05, .1, .85]] * 4)
    rows, selected, default = search_expected_risk(true, probabilities)
    assert {row["lambda_cost"] for row in rows} == set(LAMBDA_VALUES)
    assert selected is not None and selected["valid_under_constraints"]
    assert default["accuracy"] == 1.0


def test_nested_lambda_tuning_has_zero_outer_test_overlap():
    frame = _frame(15)
    candidate = ExtendedCandidate.create("word_3000", "ridge", {"alpha": 1e-3})
    registry = {candidate.candidate_id: candidate}
    result = nested_expected_risk(frame, candidate, registry)
    assert len(result["audit_rows"]) == 5
    assert all(row["tuning_scope"] == "outer_training_inner_oof" for row in result["audit_rows"])
    assert all(row["overlap_count"] == 0 for row in result["audit_rows"])


def test_invalid_expected_risk_policy_is_rejected():
    default = {"accuracy": .9, "macro_f1": .9, "summary_recall": .8,
               "multi_hop_recall": .8, "total_under_routing_rate": .1}
    invalid = {"accuracy": .895, "macro_f1": .895, "summary_recall": .8,
               "multi_hop_recall": .794, "total_under_routing_rate": .1}
    assert not _risk_policy_valid(invalid, default)


def test_extended_artifact_isolation_paths(tmp_path):
    report = tmp_path / "reports" / "extended_classical_benchmark"
    artifact = tmp_path / "artifacts" / "extended_classical_benchmark"
    assert validate_extended_paths(report, artifact)[:2] == (report.resolve(), artifact.resolve())
    with pytest.raises(ValueError, match="extended_classical_benchmark"):
        validate_extended_paths(tmp_path / "reports" / "improved_router", artifact)
    with pytest.raises(ValueError, match="extended_classical_benchmark"):
        validate_extended_paths(report, tmp_path / "artifacts" / "improved_router")


def test_query_only_ensemble_retention_selects_global_candidate():
    summary = {"mean_macro_f1": .90, "mean_summary_recall": .90, "mean_accuracy": .90}
    global_result = ExtendedResult(
        ExtendedCandidate.create("nbsvm_word", "nbsvm_word_linear_svc", {"C": .25}),
        [], summary, [], [],
    )
    per_domain_result = ExtendedResult(
        ExtendedCandidate.create(
            "nbsvm_word_char", "nbsvm_word_char_linear_svc", {"C": .25}, mode="per_domain"
        ),
        [], {**summary, "mean_macro_f1": .99}, [], [],
    )
    assert _best_global([per_domain_result, global_result]) is global_result
    with pytest.raises(ValueError, match="global candidate"):
        _best_global([per_domain_result])
