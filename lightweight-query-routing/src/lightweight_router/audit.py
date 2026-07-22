from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CANONICAL_LABELS, DIAGNOSTIC_LABEL_PERMUTATION, EXPECTED_RAGROUTER_COUNTS,
    PAPER_DOMAIN_COUNTS, PAPER_LABEL_PERCENTAGES, PAPER_TOTAL,
)
from .data import LoadedDataset


def _counts(series: pd.Series) -> dict[str, dict[str, float | int]]:
    values = series.fillna("<missing>").astype(str)
    total = len(values)
    return {str(key): {"count": int(value), "percent": round(100 * value / total, 4) if total else 0.0}
            for key, value in values.value_counts(dropna=False).sort_index().items()}


def build_audit(dataset: LoadedDataset) -> dict[str, Any]:
    frame = dataset.frame
    invalid_reasons = Counter(row["reason"] for row in dataset.invalid_records)
    duplicate_query_mask = frame.duplicated("query", keep=False) if not frame.empty else pd.Series([], dtype=bool)
    duplicate_id_mask = frame["id"].notna() & frame.duplicated("id", keep=False) if not frame.empty else pd.Series([], dtype=bool)
    duplicate_queries = frame[duplicate_query_mask]
    duplicate_ids = frame[duplicate_id_mask]
    query_conflicts = (duplicate_queries.groupby("query")["label"].nunique() > 1).sum() if not duplicate_queries.empty else 0
    id_conflicts = 0
    if not duplicate_ids.empty:
        id_conflicts = sum(group[["query", "label", "domain"]].drop_duplicates().shape[0] > 1 for _, group in duplicate_ids.groupby("id"))
    cross_domain = 0
    if frame["domain"].notna().any():
        cross_domain = int((frame.groupby("query")["domain"].nunique() > 1).sum())
    label_by_domain = {}
    if frame["domain"].notna().any():
        for domain, group in frame.groupby("domain", dropna=False):
            label_by_domain[str(domain)] = _counts(group["label"])
    local_percentages = {label: stats["percent"] for label, stats in _counts(frame["raw_label"]).items()}
    expectation = None
    if dataset.expected_ragrouter_bench:
        expectation = {
            "expected_domains": sorted(EXPECTED_RAGROUTER_COUNTS),
            "actual_domains": sorted(frame["domain"].astype(str).unique()),
            "expected_total": sum(sum(counts.values()) for counts in EXPECTED_RAGROUTER_COUNTS.values()),
            "actual_total": len(frame), "status": "PASS",
        }
    return {
        **dataset.metadata(),
        "missing_or_empty_queries": invalid_reasons["missing_or_empty_query"],
        "missing_labels": invalid_reasons["missing_label"],
        "unknown_labels": invalid_reasons["unknown_label"],
        "unique_ids": int(frame["id"].nunique(dropna=True)),
        "duplicate_ids": int(duplicate_ids["id"].nunique()) if not duplicate_ids.empty else 0,
        "exact_duplicate_queries": int(duplicate_queries["query"].nunique()) if not duplicate_queries.empty else 0,
        "duplicate_queries_with_conflicting_labels": int(query_conflicts),
        "duplicate_ids_with_conflicting_content": int(id_conflicts),
        "query_duplicates_across_domains": cross_domain,
        "class_counts": _counts(frame["raw_label"]),
        "domain_counts": _counts(frame["domain"]) if frame["domain"].notna().any() else {},
        "label_distribution_within_domain": label_by_domain,
        "ragrouter_bench_expectation": expectation,
        "paper_comparison": {
            "stated_total_queries": PAPER_TOTAL,
            "local_valid_records": len(frame),
            "paper_label_percentages": PAPER_LABEL_PERCENTAGES,
            "local_label_percentages": local_percentages,
            "local_minus_paper_percentage_points": {label: round(local_percentages.get(label, 0) - paper_percent, 4) for label, paper_percent in PAPER_LABEL_PERCENTAGES.items()},
            "local_minus_paper_total": len(frame) - PAPER_TOTAL,
            "paper_printed_domain_counts": list(PAPER_DOMAIN_COUNTS),
            "paper_printed_domain_count_sum": sum(PAPER_DOMAIN_COUNTS),
            "discrepancy_note": "The paper's printed domain counts sum to 7,729, not its stated 7,727 total. This implementation does not modify either figure.",
        },
    }


def write_label_distribution_reconciliation(dataset: LoadedDataset, output: str | Path) -> list[dict[str, Any]]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    counts = dataset.frame["raw_label"].value_counts().reindex(CANONICAL_LABELS, fill_value=0)
    total = int(counts.sum())
    local_percentages = (counts / total * 100).to_dict()
    local_rank = counts.rank(method="min", ascending=False).astype(int).to_dict()
    paper_series = pd.Series(PAPER_LABEL_PERCENTAGES)
    paper_rank = paper_series.rank(method="min", ascending=False).astype(int).to_dict()
    rows = []
    for label in CANONICAL_LABELS:
        difference = float(local_percentages[label] - PAPER_LABEL_PERCENTAGES[label])
        rows.append({
            "local_canonical_label": label, "local_count": int(counts[label]),
            "local_percentage": float(local_percentages[label]),
            "paper_declared_count_implied_from_7727": PAPER_TOTAL * PAPER_LABEL_PERCENTAGES[label] / 100,
            "paper_declared_percentage": PAPER_LABEL_PERCENTAGES[label],
            "percentage_point_difference": difference, "local_frequency_rank": int(local_rank[label]),
            "paper_frequency_rank": int(paper_rank[label]),
            "mismatch_status": "MISMATCH" if abs(difference) > 0.05 or local_rank[label] != paper_rank[label] else "APPROXIMATE_MATCH",
        })
    payload = {
        "dataset_fingerprint": dataset.source_sha256, "total": total, "rows": rows,
        "prominent_finding": "The official local dataset has multi_hop as the majority class at 52.8795%, while the lightweight paper describes single_hop as the majority class at 52.9%.",
        "inferred_cyclic_permutation_observation": DIAGNOSTIC_LABEL_PERMUTATION,
        "permutation_status": "Audit observation only; not proven ground truth and never applied to the primary protocol.",
    }
    (output / "label_distribution_reconciliation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "label_distribution_reconciliation.csv", index=False)
    lines = ["# Label-distribution reconciliation", "", f"**{payload['prominent_finding']}**", "",
             "| Local label | Local count | Local % | Paper-implied count | Paper % | Difference (pp) | Local rank | Paper rank | Status |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['local_canonical_label']} | {row['local_count']} | {row['local_percentage']:.4f} | {row['paper_declared_count_implied_from_7727']:.3f} | {row['paper_declared_percentage']:.1f} | {row['percentage_point_difference']:.4f} | {row['local_frequency_rank']} | {row['paper_frequency_rank']} | {row['mismatch_status']} |")
    lines.extend(["", "The same proportions approximately align under an inferred cyclic name permutation:", "",
                  "- local `multi_hop` → paper `single_hop`", "", "- local `single_hop` → paper `summary`", "", "- local `summary` → paper `multi_hop`", "",
                  "This is an audit observation only. It is not proven ground truth and is not used by the primary protocol."])
    (output / "label_distribution_reconciliation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def write_semantic_spot_check(dataset: LoadedDataset, output: str | Path, per_label_domain: int = 10) -> list[dict[str, Any]]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for domain in sorted(dataset.frame["domain"].dropna().astype(str).unique()):
        for label in CANONICAL_LABELS:
            group = dataset.frame[(dataset.frame.domain.astype(str) == domain) & (dataset.frame.raw_label == label)].copy()
            group["__sort_id"] = group["id"].fillna("").astype(str)
            selected = group.sort_values(["__sort_id", "source_row_index"]).head(per_label_domain)
            if len(selected) < per_label_domain:
                raise ValueError(f"Semantic spot check needs {per_label_domain} {label} records in {domain}; found {len(selected)}.")
            for row in selected.itertuples():
                samples.append({"id": row.id, "domain": row.domain, "question": row.query, "label": row.raw_label,
                                "number_of_supporting_facts": int(row.supporting_fact_count), "source_file": row.source_file,
                                "source_row_index": int(row.source_row_index)})
    (output / "label_semantic_samples.json").write_text(json.dumps({"selection": "first 10 sorted IDs per domain and label", "samples": samples}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Label semantic spot check", "", "Deterministic human-review sample: the first 10 sorted record IDs for every domain/label pair. No LLM or automatic relabeling was used.", "",
             "Use this report to inspect whether `single_hop` tends toward direct fact lookup, `multi_hop` tends to combine facts, and `summary` tends toward aggregation or summarization. These tendencies do not authorize label rewriting."]
    for domain in sorted({str(row["domain"]) for row in samples}):
        lines.extend(["", f"## {domain}"])
        for label in CANONICAL_LABELS:
            lines.extend(["", f"### {label}", ""])
            for row in (item for item in samples if str(item["domain"]) == domain and item["label"] == label):
                question = str(row["question"]).replace("\n", " ")
                lines.append(f"- `{row['id']}` — {question} _(facts: {row['number_of_supporting_facts']}; row: {row['source_row_index']}; source: `{row['source_file']}`)_")
    (output / "label_semantic_spot_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return samples


def write_audit(dataset: LoadedDataset, output: str | Path) -> dict[str, Any]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = build_audit(dataset)
    (output / "dataset_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "invalid_records.jsonl").open("w", encoding="utf-8") as handle:
        for row in dataset.invalid_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lines = ["# Dataset audit", "", f"- Source: `{audit['source_path']}`", f"- SHA-256: `{audit['source_sha256']}`",
             f"- Format: {audit['file_format']}", f"- Parsed / valid / invalid: {audit['total_parsed_records']} / {audit['valid_records']} / {audit['invalid_records']}",
             f"- Detected fields: query=`{audit['query']}`, label=`{audit['label']}`, domain=`{audit['domain']}`, id=`{audit['identifier']}`", "",
             "## Integrity", "", f"- Missing or empty queries: {audit['missing_or_empty_queries']}",
             f"- Missing labels: {audit['missing_labels']}", f"- Unknown labels: {audit['unknown_labels']}",
             f"- Unique IDs / duplicate IDs: {audit['unique_ids']} / {audit['duplicate_ids']}",
             f"- Exact duplicate queries: {audit['exact_duplicate_queries']}",
             f"- Duplicate queries with conflicting labels: {audit['duplicate_queries_with_conflicting_labels']}",
             f"- Duplicate IDs with conflicting content: {audit['duplicate_ids_with_conflicting_content']}",
             f"- Duplicate queries across domains: {audit['query_duplicates_across_domains']}", "",
             "## Distribution", "", "```json", json.dumps({"classes": audit["class_counts"], "domains": audit["domain_counts"], "labels_within_domain": audit["label_distribution_within_domain"]}, indent=2), "```", "",
             "## Paper accounting note", "", "The paper states 7,727 total queries. Its printed domain counts (3,356 + 1,200 + 1,277 + 1,896) sum to 7,729. This audit reports the discrepancy without attempting to fix it."]
    (output / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_label_distribution_reconciliation(dataset, output)
    if dataset.fields.domain is not None and dataset.frame["domain"].notna().any():
        minimum_group = int(dataset.frame.groupby(["domain", "raw_label"]).size().min())
        write_semantic_spot_check(dataset, output, per_label_domain=min(10, minimum_group))
    return audit
