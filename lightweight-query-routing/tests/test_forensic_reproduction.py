from pathlib import Path

import pandas as pd
import pytest

from lightweight_router.config import PROTOCOL_OFFICIAL
from lightweight_router.forensic import (
    DISCLOSED, INVALID, PLAUSIBLE, CONTAMINATION_MODES, determine_conclusion,
    fold_indices_checksum, interleave_domain_label, make_stratified_splits,
    ordered_id_checksum, run_domain_experiments, run_forensic_reproduction,
    run_svm_experiment, sparse_similarity_audit, template_signature,
)


def _frame(domains=("a", "b", "c", "d"), per_label_domain=5):
    rows = []
    for domain in domains:
        for label in ("single_hop", "multi_hop", "summary"):
            for index in range(per_label_domain):
                rows.append({"record_number": len(rows), "id": f"{domain}-{label}-{index}", "domain": domain,
                             "raw_label": label, "label": label, "protocol": PROTOCOL_OFFICIAL,
                             "query": f"{label} repeated class text {index} in {domain}"})
    return pd.DataFrame(rows)


def test_stable_order_and_fold_checksums():
    frame = _frame()
    first = make_stratified_splits(frame, shuffle=True, seed=42)
    second = make_stratified_splits(frame, shuffle=True, seed=42)
    assert fold_indices_checksum(frame, first) == fold_indices_checksum(frame, second)
    assert ordered_id_checksum(interleave_domain_label(frame)) == ordered_id_checksum(interleave_domain_label(frame))


def test_deterministic_multi_seed_execution():
    frame = _frame(domains=("a",), per_label_domain=5)
    values = []
    for _ in range(2):
        run = []
        for seed in (0, 42):
            splits = make_stratified_splits(frame, shuffle=True, seed=seed)
            run.append(run_svm_experiment(frame, frame["query"], splits, f"seed-{seed}", PLAUSIBLE).macro_f1)
        values.append(run)
    assert values[0] == values[1]


def test_independent_per_domain_cv_has_distinct_model_scopes(tmp_path):
    frame = _frame()
    splits = make_stratified_splits(frame)
    baseline = run_svm_experiment(frame, frame["query"], splits, "A0", DISCLOSED)
    from lightweight_router.data import LoadedDataset, FieldSelection
    dataset = LoadedDataset(frame, [], FieldSelection("question", "type", "domain", "id"), "synthetic", "x", "json", len(frame), [], [])
    results, _ = run_domain_experiments(dataset, tmp_path, baseline)
    combined = next(result for result in results if result.experiment_id == "B1")
    assert combined.metadata["model_instances"] == ["a", "b", "c", "d"]
    assert len(set(combined.metadata["model_instances"])) == 4
    assert len(combined.oof) == len(frame)


def test_grouped_duplicate_cv_and_template_stability():
    frame = _frame(domains=("a", "b"), per_label_domain=10)
    groups = frame.apply(lambda row: f"{row.label}-{int(row.record_number) % 5}", axis=1)
    splits = make_stratified_splits(frame, groups=groups)
    for train, validation in splits:
        assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[validation]))
    result = run_svm_experiment(frame, frame["query"], splits, "C1", DISCLOSED)
    assert len(result.oof) == len(frame)
    text = 'On 2024-01-03 item ABC123 said "hello" and cost 42.5'
    assert template_signature(text) == template_signature(text)
    assert "NUMBER" in template_signature(text)


def test_sparse_similarity_avoids_dense_all_pairs():
    frame = _frame(domains=("a",), per_label_domain=5)
    result = sparse_similarity_audit(frame, make_stratified_splits(frame))
    assert result["method"].startswith("sparse")
    assert result["sparse_matrix_nnz"] > 0
    source = Path(__file__).resolve().parents[1].joinpath("src/lightweight_router/forensic.py").read_text(encoding="utf-8")
    assert ".toarray(" not in source and ".todense(" not in source


def test_invalid_diagnostic_markers_and_global_fit():
    assert CONTAMINATION_MODES["D0"]["validity_class"] == PLAUSIBLE
    assert all(CONTAMINATION_MODES[mode]["validity_class"] == INVALID for mode in ("D1", "D2", "D3", "D4", "D5"))
    frame = _frame(domains=("a",), per_label_domain=5)
    result = run_svm_experiment(frame, frame["query"], make_stratified_splits(frame), "E1", INVALID, global_vectorizer_fit=True)
    assert result.validity_class == INVALID
    assert result.metadata["global_vectorizer_fit"] is True


def test_forensic_output_guard_preserves_existing_artifact(tmp_path):
    artifact = tmp_path / "best_model.joblib"
    artifact.write_bytes(b"frozen")
    with pytest.raises(ValueError, match="forensic_reproduction"):
        run_forensic_reproduction(None, tmp_path / "not-allowed", tmp_path)  # guard fires before dataset access
    assert artifact.read_bytes() == b"frozen"


def test_distance_decision_rules():
    base = {"experiment_id": "x", "category": "x", "input_fields": "question", "training_scope": "global",
            "cv_type": "cv", "shuffle": False, "seed": "", "grouping": "none", "preprocessing_scope": "fold_local",
            "template_overlap_rate": 0.0, "leakage_warning": "", "notes": ""}
    disclosed = pd.DataFrame([{**base, "validity_class": DISCLOSED, "accuracy": .931, "macro_f1": .926,
                               "accuracy_delta_pp": -.1, "macro_f1_delta": -.002}])
    assert determine_conclusion(disclosed) == "NUMERICALLY_REPRODUCED"
    plausible = disclosed.assign(validity_class=PLAUSIBLE, accuracy_delta_pp=-1.0, macro_f1_delta=-.01)
    assert determine_conclusion(plausible) == "PARTIALLY_EXPLAINED"
    invalid = disclosed.assign(validity_class=INVALID)
    assert determine_conclusion(invalid) == "EXPLAINED_ONLY_BY_INVALID_DIAGNOSTIC"
    far = disclosed.assign(accuracy_delta_pp=-10.0, macro_f1_delta=-.1)
    assert determine_conclusion(far) == "NOT_NUMERICALLY_REPRODUCIBLE"
