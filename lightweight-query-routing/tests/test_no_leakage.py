from lightweight_router.evaluation import evaluate_configuration
from lightweight_router.models import build_pipeline


def test_tfidf_parameters_and_fold_local_fit(data_file):
    from lightweight_router.data import load_dataset
    loaded = load_dataset(data_file)
    pipeline = build_pipeline("tfidf_svm")
    assert pipeline.named_steps["tfidf"].get_params()["ngram_range"] == (1, 2)
    assert pipeline.named_steps["tfidf"].get_params()["max_features"] == 3000
    assert not hasattr(pipeline.named_steps["tfidf"], "vocabulary_")
    result = evaluate_configuration(loaded.frame, "tfidf_svm")
    assert len(result.oof) == len(loaded.frame)
    assert result.oof.record_number.nunique() == len(loaded.frame)
    assert len(result.fold_indices["tfidf_svm"]) == 5


def test_requested_model_hyperparameters():
    assert build_pipeline("tfidf_logistic_regression").named_steps["classifier"].get_params()["max_iter"] == 5000
    assert build_pipeline("tfidf_logistic_regression").named_steps["classifier"].get_params()["class_weight"] is None
    svm = build_pipeline("tfidf_svm").named_steps["classifier"]
    assert svm.kernel == "rbf" and svm.gamma == "scale" and svm.probability is False
    forest = build_pipeline("tfidf_random_forest").named_steps["classifier"]
    assert forest.n_estimators == 200 and forest.random_state == 42 and forest.n_jobs == -1
    knn = build_pipeline("tfidf_knn").named_steps["classifier"]
    assert knn.n_neighbors == 7 and knn.metric == "cosine" and knn.algorithm == "brute"
