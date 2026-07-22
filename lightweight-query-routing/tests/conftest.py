import json

import pytest


@pytest.fixture
def records():
    rows = []
    labels = [("single_hop", "factual"), ("multi_hop", "reasoning"), ("summary", "summarization")]
    for label, alias in labels:
        for index in range(6):
            rows.append({"id": f"{label}-{index}", "question": f"{label} query number {index} what happened?", "label": label if index else alias, "domain": "a" if index < 3 else "b"})
    return rows


@pytest.fixture
def data_file(tmp_path, records):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path
