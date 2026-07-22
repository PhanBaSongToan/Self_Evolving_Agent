import json

import pytest

from lightweight_router.data import DataValidationError, load_dataset


def test_json_list_loader(data_file):
    loaded = load_dataset(data_file)
    assert len(loaded.frame) == 18
    assert loaded.fields.query == "question"
    assert set(loaded.frame.label) == {"single_hop", "multi_hop", "summary"}


def test_nested_json_records_loader(tmp_path, records):
    path = tmp_path / "nested.json"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    assert len(load_dataset(path).frame) == 18


def test_jsonl_loader(tmp_path, records):
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    assert load_dataset(path).file_format == "jsonl"


def test_ambiguous_field_detection(tmp_path, records):
    for row in records:
        row["query"] = row["question"]
    path = tmp_path / "ambiguous.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(DataValidationError, match="Ambiguous query"):
        load_dataset(path)


def test_directory_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="directory"):
        load_dataset(tmp_path)
