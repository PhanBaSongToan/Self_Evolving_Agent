from __future__ import annotations

from pathlib import Path

CANONICAL_LABELS = ("single_hop", "multi_hop", "summary")
LABEL_ALIASES = {"factual": "single_hop", "reasoning": "multi_hop", "summarization": "summary"}
FIELD_CANDIDATES = {
    "query": ("question", "query", "text", "input"),
    "label": ("label", "type", "query_type"),
    "domain": ("domain", "dataset", "corpus", "source"),
    "id": ("id", "query_id", "uid"),
}
PAPER_TOTAL = 7727
PAPER_LABEL_PERCENTAGES = {"single_hop": 52.9, "multi_hop": 17.1, "summary": 30.0}
PAPER_DOMAIN_COUNTS = (3356, 1200, 1277, 1896)
PROTOCOL_OFFICIAL = "official_raw_labels"
PROTOCOL_DIAGNOSTIC = "paper_label_permutation_diagnostic"
PROTOCOLS = (PROTOCOL_OFFICIAL, PROTOCOL_DIAGNOSTIC)
DIAGNOSTIC_LABEL_PERMUTATION = {"multi_hop": "single_hop", "single_hop": "summary", "summary": "multi_hop"}
DIAGNOSTIC_WARNING = (
    "WARNING: This protocol applies an inferred label-name permutation based only on matching class proportions. "
    "It is not verified by author code and must not be treated as canonical dataset semantics."
)
EXPECTED_RAGROUTER_COUNTS = {
    "graphragBench_medical": {"multi_hop": 509, "single_hop": 1098, "summary": 289},
    "musique": {"multi_hop": 2590, "single_hop": 398, "summary": 368},
    "quality": {"multi_hop": 461, "single_hop": 454, "summary": 283},
    "ultraDomain_legal": {"multi_hop": 526, "single_hop": 370, "summary": 381},
}
PAPER_RESULTS = {
    "tfidf_logistic_regression": (0.921, 0.918),
    "tfidf_svm": (0.932, 0.928),
    "tfidf_random_forest": (0.919, 0.914),
    "tfidf_knn": (0.854, 0.855),
    "structural_documented_18_logistic_regression": (0.781, 0.763),
    "structural_documented_18_svm": (0.791, 0.774),
    "structural_documented_18_random_forest": (0.778, 0.752),
    "structural_documented_18_knn": (0.783, 0.762),
}
COST_MAPPING = {
    "single_hop": {"paradigm": "NaiveRAG", "cost_ratio": 1.4},
    "multi_hop": {"paradigm": "HybridRAG", "cost_ratio": 2.8},
    "summary": {"paradigm": "IterativeRAG", "cost_ratio": 3.5},
}
TFIDF_PARAMS = {"ngram_range": (1, 2), "max_features": 3000, "min_df": 2, "sublinear_tf": True}
PATTERNS_PATH = Path(__file__).resolve().parents[2] / "configs" / "structural_patterns.json"
