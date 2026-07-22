# Lightweight Query Routing

An auditable **reproduction of the non-deep-learning subset of the paper** “Lightweight Query Routing for Adaptive RAG: A Baseline Study on RAGRouter-Bench”. It implements exactly eight classical feature/classifier combinations: TF-IDF or `structural_documented_18`, each with Logistic Regression, SVM, Random Forest, and KNN. It is not a full reproduction of all paper configurations.

No deep-learning libraries, embeddings, external APIs, or LLM calls are used.

## Input contract

Pass one `.json`/`.jsonl` file or a directory containing domain-level `Question.json` files. Standard JSON lists, nested record lists, JSONL, and NDJSON content stored with a `.json` suffix are supported. Directory inputs preserve source file, parent-domain, and zero-based source-row provenance. Records must contain one query field and one label field. Labels must be `single_hop`, `multi_hop`, or `summary`; the unambiguous aliases `factual`, `reasoning`, and `summarization` are accepted and logged.

The primary protocol is always `official_raw_labels`. The optional `paper_label_permutation_diagnostic` requires both its protocol name and `--paper-label-permutation-diagnostic`; it writes to a separate report directory and can never produce final model artifacts.

## Windows PowerShell

```powershell
cd lightweight-query-routing
python -m pip install -e ".[dev]"
python -m lightweight_router audit --data "C:\path\to\ragrouter\data" --expect-ragrouter-bench --query-field question --label-field type --domain-field domain --id-field id --output reports
python -m lightweight_router reproduce --data "C:\path\to\ragrouter\data" --expect-ragrouter-bench --query-field question --label-field type --domain-field domain --id-field id --protocol official_raw_labels --output reports
python -m lightweight_router reproduce --data "C:\path\to\ragrouter\data" --expect-ragrouter-bench --query-field question --label-field type --domain-field domain --id-field id --protocol paper_label_permutation_diagnostic --paper-label-permutation-diagnostic --output reports
python -m lightweight_router robustness --data "C:\path\to\routing_records.json" --output reports\robustness
python -m lightweight_router train-final --data "C:\path\to\ragrouter\data" --expect-ragrouter-bench --query-field question --label-field type --domain-field domain --id-field id --protocol official_raw_labels --output artifacts
python -m lightweight_router predict --model artifacts\tfidf_svm.joblib --query "What caused the policy change?"
python -m lightweight_router forensic --data "C:\path\to\ragrouter\data" --expect-ragrouter-bench --query-field question --label-field type --domain-field domain --id-field id --dataset-repo "C:\path\to\ragrouter-dataset" --output reports\forensic_reproduction
python -m lightweight_router improve --data "C:\path\to\ragrouter\data" --expect-ragrouter-bench --query-field question --label-field type --domain-field domain --id-field id --output reports\improved_router --artifact-output artifacts\improved_router
python -m lightweight_router predict --model artifacts\improved_router\best_quality_model\model.joblib --domain musique --query "What caused both events?"
python -m lightweight_router extended-benchmark --data "C:\path\to\ragrouter\data" --expect-ragrouter-bench --query-field question --label-field type --domain-field domain --id-field id --output reports\extended_classical_benchmark --artifact-output artifacts\extended_classical_benchmark
```

`reproduce` runs unshuffled 5-fold stratified CV. Raw and diagnostic artifacts are written under `reports/official_raw_labels/` and `reports/paper_label_permutation_diagnostic/`. `robustness` is explicitly an additional audit and never replaces primary results.

`improve` runs the separate **Classical Production Router Improvement** track. It uses only classical sklearn models, writes only below `reports/improved_router/` and `artifacts/improved_router/`, and does not replace the official reproduction, forensic evidence, or existing production model. Its repeated shuffled results are IID improvement estimates and are not claimed as reproductions of the paper.

`extended-benchmark` runs the isolated **Extended Non-Deep-Learning Model Benchmark**. Optional XGBoost, LightGBM, and CatBoost dependencies can be installed with `pip install -e ".[extended-classical]"`. All preprocessing, NB ratios, calibration, dimensionality reduction, and meta-model training remain fold-local.
