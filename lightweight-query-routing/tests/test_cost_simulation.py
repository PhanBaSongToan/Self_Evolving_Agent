import pandas as pd

from lightweight_router.cost_simulation import cost_summary, majority_baseline, perfect_label_references


def test_cost_simulation_arithmetic():
    labels = ["single_hop", "multi_hop", "summary"]
    oof = pd.DataFrame({"true_label": labels, "raw_true_label": labels, "predicted_label": labels, "protocol": "official_raw_labels"})
    result = cost_summary(oof, "x", "official_raw_labels")
    assert result["total_simulated_cost"] == 7.7
    assert round(result["simulated_savings_percent"], 6) == round((10.5 - 7.7) / 10.5 * 100, 6)


def test_majority_and_references():
    labels = pd.Series(["single_hop", "multi_hop", "summary"])
    majority = majority_baseline(labels, labels, "official_raw_labels")
    assert majority["total_simulated_cost"] == 4.2
    paper, empirical = perfect_label_references(labels, labels, "official_raw_labels")
    assert paper["configuration"] == "paper_fixed_perfect_label_reference"
    assert empirical["macro_f1"] == 1.0
