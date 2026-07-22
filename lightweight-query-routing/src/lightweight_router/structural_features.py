from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .config import PATTERNS_PATH

FEATURE_NAMES = (
    "token_count", "character_count", "average_word_length", "question_word_who",
    "question_word_what", "question_word_when", "question_word_where", "question_word_why",
    "question_word_how", "question_word_which", "has_negation", "approximate_named_entity_count",
    "clause_count", "comparative_pattern", "temporal_pattern", "aggregation_pattern",
    "causal_pattern", "procedural_pattern",
)
WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
CAPITALIZED_SPAN_RE = re.compile(r"(?<![\w])(?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*)*", re.UNICODE)
SENTENCE_START_RE = re.compile(r"(?:^|[.!?]\s+)([A-Z][\w'-]*)")


def _patterns(path: Path | None = None) -> dict:
    with (path or PATTERNS_PATH).open(encoding="utf-8") as handle:
        return json.load(handle)


class StructuralDocumented18(BaseEstimator, TransformerMixin):
    """The 18 structural features explicitly described by the paper.

    Named entities are contiguous capitalized spans after excluding a likely first
    sentence word. Clauses are commas/semicolons/colons plus coordinating or
    subordinating conjunction occurrences. These are documented approximations.
    """

    def __init__(self, patterns_path: str | None = None):
        self.patterns_path = patterns_path

    def fit(self, X: Iterable[str], y=None):
        self._patterns = _patterns(Path(self.patterns_path) if self.patterns_path else None)
        self._regexes = {key: re.compile(value, re.IGNORECASE) for key, value in self._patterns.items() if key.endswith("_regex")}
        return self

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)

    def _row(self, value: object) -> list[float]:
        text = str(value) if value is not None else ""
        words = WORD_RE.findall(text)
        lower_words = [word.lower() for word in words]
        word_set = set(lower_words)
        p = self._patterns
        entities = []
        for match in CAPITALIZED_SPAN_RE.finditer(text):
            span_words = match.group(0).split()
            prefix = text[:match.start()]
            at_sentence_start = not prefix.strip() or bool(re.search(r"[.!?]\s*$", prefix))
            # "Compare Alice" should count Alice, but a lone ordinary first word
            # such as "What" should not be treated as an entity.
            if at_sentence_start:
                span_words = span_words[1:]
            if span_words:
                entities.append(" ".join(span_words))
        clause_marks = len(re.findall(r"[,;:]", text))
        conjunctions = sum(lower_words.count(word) for word in p["clause_conjunctions"])
        return [
            float(len(words)), float(len(text)), float(sum(len(word) for word in words) / len(words) if words else 0.0),
            *[float(word in word_set) for word in p["question_words"]],
            float(any(word in word_set for word in p["negation_words"])), float(len(entities)),
            float(clause_marks + conjunctions),
            *[float(bool(self._regexes[key].search(text))) for key in ("comparative_regex", "temporal_regex", "aggregation_regex", "causal_regex", "procedural_regex")],
        ]

    def transform(self, X: Iterable[str]) -> np.ndarray:
        if not hasattr(self, "_patterns"):
            self.fit(X)
        matrix = np.asarray([self._row(value) for value in X], dtype=float)
        return matrix.reshape(0, len(FEATURE_NAMES)) if matrix.size == 0 else matrix
