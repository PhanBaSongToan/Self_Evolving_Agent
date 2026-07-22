import json

import pandas as pd
import pytest

from lightweight_router.cli import build_parser, main
from lightweight_router.config import PROTOCOL_DIAGNOSTIC, PROTOCOL_OFFICIAL
from lightweight_router.cost_simulation import cost_context, cost_summary, majority_baseline, perfect_label_references
from lightweight_router.data import apply_protocol, load_dataset
from lightweight_router.evaluation import evaluate_configuration
from lightweight_router.predict import predict_query
from lightweight_router.train_final import train_final_models


def _row(domain, index, label):
    return {"id": f"{domain}_{index:04d}", "question": f"{label} question {index} for {domain}",
            "type": label, "answer": "excluded", "supporting_facts": ["a", "b"]}


def _directory_fixture(tmp_path):
    root = tmp_path / "data"
    counts = {}
    for domain_index, domain in enumerate(("medical", "musique", "quality", "legal")):
        directory = root / domain
        directory.mkdir(parents=True)
        rows = []
        for label in ("single_hop", "multi_hop", "summary"):
            for offset in range(5):
                rows.append(_row(domain, domain_index * 100 + len(rows), label))
        (directory / "Question.json").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        counts[domain] = len(rows)
    return root, counts


def test_jsonl_records_inside_json_extension(tmp_path):
    path = tmp_path / "Question.json"
    rows = [_row("d", index, label) for index, label in enumerate(("single_hop", "multi_hop", "summary"))]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    loaded = load_dataset(path, query_field="question", label_field="type", id_field="id")
    assert loaded.file_format == "jsonl_in_json_extension"
    assert loaded.frame.source_row_index.tolist() == [0, 1, 2]


def test_four_directory_question_files_and_combined_count(tmp_path):
    root, counts = _directory_fixture(tmp_path)
    loaded = load_dataset(root, query_field="question", label_field="type", domain_field="domain", id_field="id")
    assert len(loaded.source_files) == 4
    assert len(loaded.frame) == sum(counts.values()) == 60
    assert set(loaded.frame.source_domain) == set(counts)
    assert loaded.frame.supporting_fact_count.eq(2).all()


def test_official_distribution_majority_and_empirical_costs():
    raw = pd.Series(["multi_hop"] * 4086 + ["single_hop"] * 2320 + ["summary"] * 1321)
    context = cost_context(raw, raw, PROTOCOL_OFFICIAL)
    assert context["majority_class"] == "multi_hop"
    assert context["raw_label_distribution"]["multi_hop"] == pytest.approx(4086 / 7727)
    expected_cost = 4086 * 2.8 + 2320 * 1.4 + 1321 * 3.5
    assert context["empirical_perfect_label_cost"] == pytest.approx(expected_cost)
    assert context["empirical_perfect_label_savings_percent"] == pytest.approx((3.5 * 7727 - expected_cost) / (3.5 * 7727) * 100)
    majority = majority_baseline(raw, raw, PROTOCOL_OFFICIAL)
    assert majority["simulated_savings_percent"] == pytest.approx(20.0)
    paper, empirical = perfect_label_references(raw, raw, PROTOCOL_OFFICIAL)
    assert paper["source"] == "paper_fixed_label_distribution"
    assert empirical["source"] == "out_of_fold_predictions_only"


def test_diagnostic_disabled_by_default_and_requires_flag(data_file, tmp_path):
    args = build_parser().parse_args(["reproduce", "--data", str(data_file), "--output", str(tmp_path)])
    assert args.protocol == PROTOCOL_OFFICIAL
    assert args.paper_label_permutation_diagnostic is False
    with pytest.raises(SystemExit) as exc:
        main(["reproduce", "--data", str(data_file), "--protocol", PROTOCOL_DIAGNOSTIC, "--output", str(tmp_path)])
    assert exc.value.code == 2


def test_apply_diagnostic_keeps_raw_labels_and_separate_protocol(data_file):
    loaded = load_dataset(data_file)
    diagnostic = apply_protocol(loaded, PROTOCOL_DIAGNOSTIC)
    assert set(diagnostic.frame.protocol) == {PROTOCOL_DIAGNOSTIC}
    assert diagnostic.frame.loc[diagnostic.frame.raw_label == "multi_hop", "label"].eq("single_hop").all()
    assert loaded.frame.label.equals(loaded.frame.raw_label)


def test_final_training_rejects_diagnostic(data_file, tmp_path):
    diagnostic = apply_protocol(load_dataset(data_file), PROTOCOL_DIAGNOSTIC)
    with pytest.raises(ValueError, match="rejects"):
        train_final_models(diagnostic, tmp_path, results=[], protocol=PROTOCOL_DIAGNOSTIC)


def test_oof_protocol_and_cost_mixing_rejection(data_file):
    frame = load_dataset(data_file).frame
    result = evaluate_configuration(frame, "tfidf_logistic_regression")
    assert set(result.oof.protocol) == {PROTOCOL_OFFICIAL}
    with pytest.raises(ValueError, match="Protocol mixing"):
        cost_summary(result.oof, result.name, PROTOCOL_DIAGNOSTIC)
    with pytest.raises(ValueError, match="explicit protocol"):
        cost_summary(result.oof, result.name)


def test_prediction_uses_raw_semantic_mapping(data_file, tmp_path):
    from lightweight_router.models import build_pipeline
    import joblib
    frame = load_dataset(data_file).frame
    model_path = tmp_path / "tfidf_svm.joblib"
    joblib.dump(build_pipeline("tfidf_svm").fit(frame["query"], frame["label"]), model_path)
    result = predict_query(str(model_path), "Compare the two causes")
    assert result["protocol"] == PROTOCOL_OFFICIAL
    expected = {"single_hop": ("NaiveRAG", 1.4), "multi_hop": ("HybridRAG", 2.8), "summary": ("IterativeRAG", 3.5)}
    assert (result["recommended_paradigm"], result["simulated_cost_ratio"]) == expected[result["predicted_label"]]
