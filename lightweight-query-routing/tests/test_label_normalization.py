import json

from lightweight_router.data import load_dataset


def test_alias_normalization_is_recorded(data_file):
    loaded = load_dataset(data_file)
    assert len(loaded.normalizations) >= 3
    assert "factual" not in set(loaded.frame.label)


def test_unknown_label_rejected_to_invalid_records(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps([{"query": "x", "label": "unexpected"}]), encoding="utf-8")
    loaded = load_dataset(path)
    assert loaded.frame.empty
    assert loaded.invalid_records[0]["reason"] == "unknown_label"
