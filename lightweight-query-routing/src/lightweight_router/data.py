from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CANONICAL_LABELS, DIAGNOSTIC_LABEL_PERMUTATION, EXPECTED_RAGROUTER_COUNTS,
    FIELD_CANDIDATES, LABEL_ALIASES, PROTOCOL_DIAGNOSTIC, PROTOCOL_OFFICIAL, PROTOCOLS,
)


class DataValidationError(ValueError):
    """Raised for an input that cannot be safely interpreted."""


@dataclass(frozen=True)
class FieldSelection:
    query: str
    label: str
    domain: str | None = None
    identifier: str | None = None


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    invalid_records: list[dict[str, Any]]
    fields: FieldSelection
    source_path: str
    source_sha256: str
    file_format: str
    total_parsed_records: int
    normalizations: list[dict[str, Any]]
    source_files: list[str]
    expected_ragrouter_bench: bool = False

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self.fields)
        payload.update({
            "source_path": self.source_path, "source_sha256": self.source_sha256,
            "dataset_fingerprint": self.source_sha256, "source_files": self.source_files,
            "file_format": self.file_format, "total_parsed_records": self.total_parsed_records,
            "valid_records": len(self.frame), "invalid_records": len(self.invalid_records),
            "normalizations": self.normalizations, "expected_ragrouter_bench": self.expected_ragrouter_bench,
        })
        return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_fingerprint(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_json_lines(text: str, path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataValidationError(f"Malformed JSON object in {path} at line {line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise DataValidationError(f"JSONL record in {path} at line {line_number} is not an object.")
        item = dict(item)
        item["__source_row_index"] = len(records)
        records.append(item)
    return records


def _parse_one_file(path: Path) -> tuple[list[dict[str, Any]], str]:
    suffix = path.suffix.lower()
    if suffix not in (".json", ".jsonl", ".ndjson"):
        raise DataValidationError(f"Unsupported data format '{suffix}' for {path}. Use .json, .jsonl, or .ndjson.")
    text = path.read_text(encoding="utf-8")
    if suffix in (".jsonl", ".ndjson"):
        return _parse_json_lines(text, path), "jsonl"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # NDJSON stored with .json parses its first object and then raises Extra data.
        if exc.msg != "Extra data":
            raise DataValidationError(f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        return _parse_json_lines(text, path), "jsonl_in_json_extension"
    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict):
        possible = [(key, value) for key, value in parsed.items() if isinstance(value, list) and all(isinstance(row, dict) for row in value)]
        if len(possible) != 1:
            keys = [key for key, _ in possible]
            raise DataValidationError(
                "JSON object must contain exactly one list of record objects; "
                f"found candidate list keys in {path}: {keys or 'none'}."
            )
        records = possible[0][1]
    else:
        raise DataValidationError(f"JSON input {path} must be a record list or contain one record list.")
    if not all(isinstance(row, dict) for row in records):
        raise DataValidationError(f"Every record in {path} must be an object.")
    enriched = []
    for index, row in enumerate(records):
        item = dict(row)
        item["__source_row_index"] = index
        enriched.append(item)
    return enriched, "json"


def _parse_path(path: Path) -> tuple[list[dict[str, Any]], str, list[Path], str]:
    if not path.exists():
        raise DataValidationError(f"Data path does not exist: {path}")
    if path.is_file():
        records, file_format = _parse_one_file(path)
        for row in records:
            row["__source_file"] = str(path)
            row["__source_domain"] = None
        return records, file_format, [path], sha256_file(path)
    files = sorted(path.rglob("Question.json"), key=lambda item: item.relative_to(path).as_posix())
    if not files:
        raise DataValidationError(f"Directory contains no Question.json files: {path}")
    records: list[dict[str, Any]] = []
    formats = set()
    for source in files:
        domain = source.parent.name
        source_records, source_format = _parse_one_file(source)
        formats.add(source_format)
        for row in source_records:
            row["__source_file"] = str(source)
            row["__source_domain"] = domain
            row.setdefault("domain", domain)
            records.append(row)
    file_format = "directory:" + "+".join(sorted(formats))
    return records, file_format, files, _directory_fingerprint(path, files)


def _override_or_detect(records: list[dict[str, Any]], kind: str, supplied: str | None, required: bool) -> str | None:
    if supplied:
        if not any(supplied in record for record in records):
            raise DataValidationError(f"Explicit --{kind}-field '{supplied}' is absent from every record.")
        return supplied
    matches = [name for name in FIELD_CANDIDATES[kind] if any(name in record and record[name] not in (None, "") for record in records)]
    if not matches:
        if required:
            raise DataValidationError(f"Could not detect a {kind} field. Supply --{kind}-field explicitly.")
        return None
    if len(matches) > 1:
        raise DataValidationError(f"Ambiguous {kind} fields {matches}. Supply --{kind}-field explicitly.")
    return matches[0]


def _select_fields(records: list[dict[str, Any]], query_field: str | None, label_field: str | None,
                   domain_field: str | None, id_field: str | None) -> FieldSelection:
    if not records:
        raise DataValidationError("Input contains zero records.")
    return FieldSelection(
        query=_override_or_detect(records, "query", query_field, True),
        label=_override_or_detect(records, "label", label_field, True),
        domain=_override_or_detect(records, "domain", domain_field, False),
        identifier=_override_or_detect(records, "id", id_field, False),
    )


def _normalized_text(value: Any) -> tuple[str | None, bool]:
    if not isinstance(value, str):
        return None, False
    cleaned = value.replace("\x00", "").strip()
    return cleaned, cleaned != value


def _normal_label(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "missing_label"
    incoming = value.strip().lower()
    if incoming in CANONICAL_LABELS:
        return incoming, None
    if incoming in LABEL_ALIASES:
        return LABEL_ALIASES[incoming], "alias"
    return None, "unknown_label"


def _supporting_fact_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _validate_expected_ragrouter(frame: pd.DataFrame, files: list[Path]) -> None:
    expected_domains = set(EXPECTED_RAGROUTER_COUNTS)
    actual_domains = set(frame["domain"].astype(str).unique())
    if len(files) != 4 or actual_domains != expected_domains:
        raise DataValidationError(f"RAGRouter-Bench expectation failed: expected four domains {sorted(expected_domains)}, got {sorted(actual_domains)} from {len(files)} files.")
    for domain, expected in EXPECTED_RAGROUTER_COUNTS.items():
        actual = frame.loc[frame.domain == domain, "raw_label"].value_counts().to_dict()
        if actual != expected:
            raise DataValidationError(f"RAGRouter-Bench count mismatch for {domain}: expected {expected}, got {actual}.")


def load_dataset(data_path: str | Path, *, query_field: str | None = None, label_field: str | None = None,
                 domain_field: str | None = None, id_field: str | None = None,
                 expect_ragrouter_bench: bool = False) -> LoadedDataset:
    path = Path(data_path).expanduser().resolve()
    records, file_format, source_files, fingerprint = _parse_path(path)
    fields = _select_fields(records, query_field, label_field, domain_field, id_field)
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    normalizations: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        query, query_changed = _normalized_text(record.get(fields.query))
        label, label_issue = _normal_label(record.get(fields.label))
        issue = None
        if not query:
            issue = "missing_or_empty_query"
        elif label_issue == "missing_label":
            issue = "missing_label"
        elif label_issue == "unknown_label":
            issue = "unknown_label"
        if issue:
            invalid.append({"record_number": position, "source_file": record.get("__source_file"),
                            "source_row_index": record.get("__source_row_index"), "reason": issue, "record": record})
            continue
        if query_changed:
            normalizations.append({"record_number": position, "field": fields.query, "operation": "removed_nulls_and_trimmed_whitespace"})
        if label_issue == "alias":
            normalizations.append({"record_number": position, "field": fields.label, "from": record[fields.label], "to": label})
        valid.append({
            "record_number": position, "query": query, "label": label, "raw_label": label,
            "protocol": PROTOCOL_OFFICIAL,
            "domain": None if not fields.domain else record.get(fields.domain),
            "id": None if not fields.identifier else record.get(fields.identifier),
            "source_file": record.get("__source_file"), "source_domain": record.get("__source_domain"),
            "source_row_index": int(record.get("__source_row_index", position)),
            "supporting_fact_count": _supporting_fact_count(record.get("supporting_facts")),
        })
    columns = ["record_number", "query", "label", "raw_label", "protocol", "domain", "id", "source_file", "source_domain", "source_row_index", "supporting_fact_count"]
    frame = pd.DataFrame(valid, columns=columns)
    if expect_ragrouter_bench:
        _validate_expected_ragrouter(frame, source_files)
    return LoadedDataset(frame=frame, invalid_records=invalid, fields=fields, source_path=str(path),
                         source_sha256=fingerprint, file_format=file_format,
                         total_parsed_records=len(records), normalizations=normalizations,
                         source_files=[str(item) for item in source_files], expected_ragrouter_bench=expect_ragrouter_bench)


def apply_protocol(dataset: LoadedDataset, protocol: str) -> LoadedDataset:
    if protocol not in PROTOCOLS:
        raise DataValidationError(f"Unknown protocol '{protocol}'. Expected one of {PROTOCOLS}.")
    frame = dataset.frame.copy()
    frame["label"] = frame["raw_label"]
    if protocol == PROTOCOL_DIAGNOSTIC:
        frame["label"] = frame["raw_label"].map(DIAGNOSTIC_LABEL_PERMUTATION)
    frame["protocol"] = protocol
    return LoadedDataset(frame=frame, invalid_records=dataset.invalid_records, fields=dataset.fields,
                         source_path=dataset.source_path, source_sha256=dataset.source_sha256,
                         file_format=dataset.file_format, total_parsed_records=dataset.total_parsed_records,
                         normalizations=dataset.normalizations, source_files=dataset.source_files,
                         expected_ragrouter_bench=dataset.expected_ragrouter_bench)
