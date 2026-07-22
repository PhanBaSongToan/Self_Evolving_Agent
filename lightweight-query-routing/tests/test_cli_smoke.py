import json

from lightweight_router.cli import main


def test_cli_audit_smoke(data_file, tmp_path, capsys):
    output = tmp_path / "reports"
    main(["audit", "--data", str(data_file), "--output", str(output)])
    assert (output / "dataset_audit.json").exists()
    assert "Wrote audit" in capsys.readouterr().out


def test_cli_reproduce_smoke(data_file, tmp_path):
    reports_root = tmp_path / "reports"
    main(["reproduce", "--data", str(data_file), "--output", str(reports_root)])
    output = reports_root / "official_raw_labels"
    for name in ("results_summary.csv", "fold_metrics.csv", "classification_reports.json", "oof_predictions.csv", "cost_simulation.csv", "REPRODUCTION_REPORT.md"):
        assert (output / name).exists()
    reports = json.loads((output / "classification_reports.json").read_text(encoding="utf-8"))
    assert len(reports) == 8
    raw_summary = (output / "results_summary.csv").read_bytes()
    main(["reproduce", "--data", str(data_file), "--protocol", "paper_label_permutation_diagnostic",
          "--paper-label-permutation-diagnostic", "--output", str(reports_root)])
    diagnostic = reports_root / "paper_label_permutation_diagnostic"
    assert (diagnostic / "results_summary.csv").exists()
    assert (output / "results_summary.csv").read_bytes() == raw_summary


def test_train_final_and_three_prediction_smokes(data_file, tmp_path, capsys):
    artifacts = tmp_path / "artifacts"
    main(["train-final", "--data", str(data_file), "--output", str(artifacts)])
    capsys.readouterr()
    assert (artifacts / "best_model.joblib").exists()
    assert (artifacts / "tfidf_svm.joblib").exists()
    for query in ("What happened?", "Compare the two causes.", "Summarize the report."):
        main(["predict", "--model", str(artifacts / "tfidf_svm.joblib"), "--query", query])
        prediction = json.loads(capsys.readouterr().out)
        assert prediction["predicted_label"] in {"single_hop", "multi_hop", "summary"}
