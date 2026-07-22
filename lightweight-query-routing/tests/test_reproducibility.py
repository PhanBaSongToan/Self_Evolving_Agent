from lightweight_router.data import load_dataset
from lightweight_router.evaluation import evaluate_configuration


def test_repeatable_random_forest_cv(data_file):
    frame = load_dataset(data_file).frame
    first = evaluate_configuration(frame, "tfidf_random_forest")
    second = evaluate_configuration(frame, "tfidf_random_forest")
    assert first.oof.predicted_label.tolist() == second.oof.predicted_label.tolist()
