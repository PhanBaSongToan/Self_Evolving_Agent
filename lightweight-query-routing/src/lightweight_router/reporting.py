from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from .config import CANONICAL_LABELS, DIAGNOSTIC_WARNING, PAPER_RESULTS, PROTOCOL_DIAGNOSTIC, PROTOCOL_OFFICIAL
from .cost_simulation import cost_summary, majority_baseline, perfect_label_references
from .data import LoadedDataset
from .evaluation import EvaluationResult


def _markdown_table(table: pd.DataFrame) -> str:
    """Render a compact markdown table without adding a tabulate dependency."""
    if table.empty:
        return "_No rows._"
    headers = [str(column) for column in table.columns]
    def cell(value: Any) -> str:
        if isinstance(value, float):
            value = f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False, name=None))
    return "\n".join(rows)


def _write_confusion(output: Path, result: EvaluationResult) -> None:
    target = output / "confusion_matrices"
    target.mkdir(parents=True, exist_ok=True)
    for name, matrix in (("raw", result.raw_confusion), ("row_normalized", result.normalized_confusion)):
        pd.DataFrame(matrix, index=CANONICAL_LABELS, columns=CANONICAL_LABELS).to_csv(target / f"{result.name}_{name}.csv")
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(xticks=np.arange(3), yticks=np.arange(3), xticklabels=CANONICAL_LABELS, yticklabels=CANONICAL_LABELS,
                 xlabel="Predicted label", ylabel="True label", title=f"{result.name}: {name.replace('_', ' ')}")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{matrix[i, j]:.3f}" if name != "raw" else str(int(matrix[i, j])), ha="center", va="center")
        figure.tight_layout()
        figure.savefig(target / f"{result.name}_{name}.png", dpi=150)
        plt.close(figure)


def write_primary_outputs(dataset: LoadedDataset, results: list[EvaluationResult], output: str | Path) -> list[dict[str, Any]]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = [result.summary for result in results]
    pd.DataFrame(summaries).to_csv(output / "results_summary.csv", index=False)
    pd.DataFrame([metric for result in results for metric in result.fold_metrics]).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame([row for result in results for _, row in result.oof.iterrows()]).to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame([{**{"configuration": row["configuration"]}, "fit_time_seconds": row["fit_time_seconds"], "prediction_time_seconds": row["prediction_time_seconds"]}
                  for result in results for row in result.fold_metrics]).to_csv(output / "timing.csv", index=False)
    (output / "classification_reports.json").write_text(json.dumps({result.name: result.report for result in results}, indent=2), encoding="utf-8")
    (output / "fold_indices.json").write_text(json.dumps({key: value for result in results for key, value in result.fold_indices.items()}, indent=2), encoding="utf-8")
    for result in results:
        _write_confusion(output, result)
    return summaries


def write_paper_comparison(summaries: list[dict[str, Any]], output: str | Path, protocol: str = PROTOCOL_OFFICIAL) -> None:
    output = Path(output)
    rows = []
    for row in summaries:
        paper_accuracy, paper_f1 = PAPER_RESULTS[row["configuration"]]
        accuracy_diff = row["pooled_accuracy"] - paper_accuracy
        f1_diff = row["pooled_macro_f1"] - paper_f1
        if abs(f1_diff) <= .005 and abs(accuracy_diff) <= .005:
            parity = "GREEN"
        elif abs(f1_diff) <= .020 and abs(accuracy_diff) <= .020:
            parity = "YELLOW"
        else:
            parity = "RED"
        rows.append({"configuration": row["configuration"], "paper_accuracy": paper_accuracy, "reproduced_accuracy": row["pooled_accuracy"],
                     "accuracy_absolute_difference": abs(accuracy_diff), "accuracy_percentage_point_difference": accuracy_diff * 100,
                     "paper_macro_f1": paper_f1, "reproduced_macro_f1": row["pooled_macro_f1"], "macro_f1_absolute_difference": abs(f1_diff),
                     "parity_status": parity, "structural_warning": "Only 18 documented structural features were implemented." if row["configuration"].startswith("structural") else ""})
    table = pd.DataFrame(rows)
    table.to_csv(output / "paper_comparison.csv", index=False)
    warning = f"**{DIAGNOSTIC_WARNING}**\n\n" if protocol == PROTOCOL_DIAGNOSTIC else ""
    output.joinpath("paper_comparison.md").write_text("# Paper comparison\n\n" + warning + _markdown_table(table) + "\n\nStructural rows use `structural_documented_18`, not an unspecified 23-feature implementation. Matching macro-F1 does not prove semantic label reproduction.\n", encoding="utf-8")


def write_cost_outputs(results: list[EvaluationResult], dataset: LoadedDataset, output: str | Path, protocol: str) -> list[dict[str, Any]]:
    output = Path(output)
    rows = [cost_summary(result.oof, result.name, protocol) for result in results]
    rows.extend([majority_baseline(dataset.frame["label"], dataset.frame["raw_label"], protocol),
                 *perfect_label_references(dataset.frame["label"], dataset.frame["raw_label"], protocol)])
    nested = ("predicted_routing_distribution", "raw_label_distribution", "effective_label_distribution",
              "label_to_paradigm_mapping", "paradigm_to_cost_mapping", "paper_fixed_perfect_label_reference")
    serializable = [{**row, **{key: json.dumps(row[key], sort_keys=True) for key in nested}} for row in rows]
    pd.DataFrame(serializable).to_csv(output / "cost_simulation.csv", index=False)
    output.joinpath("cost_accuracy_tradeoff.md").write_text(
        f"# Cost–accuracy trade-off\n\nProtocol: `{protocol}`.\n\n" + (f"**{DIAGNOSTIC_WARNING}**\n\n" if protocol == PROTOCOL_DIAGNOSTIC else "") +
        "All figures are post-hoc relative-cost simulations using out-of-fold predictions only. No RAG execution or LLM API calls occurred.\n\n" +
        _markdown_table(pd.DataFrame(serializable)) + "\n\nThe paper-fixed reference uses 0.529 / 0.171 / 0.300; the empirical reference uses the observed local labels.\n", encoding="utf-8")
    return rows


def write_domain_outputs(results: list[EvaluationResult], dataset: LoadedDataset, output: str | Path) -> None:
    output = Path(output)
    if dataset.fields.domain is None or not dataset.frame["domain"].notna().any():
        output.joinpath("domain_breakdown.md").write_text("# Domain breakdown\n\nNo domain field was supplied or detected; domain metrics were skipped rather than fabricated.\n", encoding="utf-8")
        return
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    rows = []
    best_tfidf = max((result for result in results if result.name.startswith("tfidf_")), key=lambda item: item.summary["pooled_macro_f1"])
    best_structural = max((result for result in results if result.name.startswith("structural_")), key=lambda item: item.summary["pooled_macro_f1"])
    for result in results:
        for domain, group in result.oof.groupby("domain", dropna=False):
            class_report = classification_report(group.true_label, group.predicted_label, labels=CANONICAL_LABELS, output_dict=True, zero_division=0)
            rows.append({"configuration": result.name, "domain": domain, "record_count": len(group), "label_distribution": json.dumps(group.true_label.value_counts(normalize=True).to_dict()),
                         "accuracy": accuracy_score(group.true_label, group.predicted_label), "macro_f1": f1_score(group.true_label, group.predicted_label, labels=CANONICAL_LABELS, average="macro", zero_division=0),
                         "per_class_metrics": json.dumps({label: class_report[label] for label in CANONICAL_LABELS})})
    table = pd.DataFrame(rows)
    table.to_csv(output / "domain_metrics.csv", index=False)
    protocol = str(results[0].oof["protocol"].iloc[0])
    warning = f"**{DIAGNOSTIC_WARNING}**\n\n" if protocol == PROTOCOL_DIAGNOSTIC else ""
    output.joinpath("domain_breakdown.md").write_text("# Domain breakdown\n\n" + warning + "Best allowed TF-IDF classifier: `" + best_tfidf.name + "`. Best allowed structural classifier: `" + best_structural.name + "`.\n\n" + _markdown_table(table) + "\n", encoding="utf-8")


def write_core_documents(dataset: LoadedDataset, summaries: list[dict[str, Any]], output: str | Path, protocol: str = PROTOCOL_OFFICIAL) -> None:
    output = Path(output)
    gaps_warning = f"\n**{DIAGNOSTIC_WARNING}**\n" if protocol == PROTOCOL_DIAGNOSTIC else ""
    gaps = f"""# Reproduction gaps
{gaps_warning}

The paper claims 23 structural features, but its written enumeration identifies only 18. This package deliberately implements only those 18 as `structural_documented_18`; it does not invent five undocumented features. Its named-entity and clause-count heuristics are deterministic documented approximations because the paper does not specify them.

Other possible numerical differences include unknown original ordering, unknown CV shuffle behavior and seed, unknown scikit-learn version, unspecified text-preprocessing defaults, the five missing structural definitions, and the paper’s count discrepancy: it states 7,727 queries while printed domain counts total 7,729.
"""
    output.joinpath("REPRODUCTION_GAPS.md").write_text(gaps, encoding="utf-8")
    table = _markdown_table(pd.DataFrame(summaries))
    diagnostic_warning = f"\n**{DIAGNOSTIC_WARNING}**\n" if protocol == PROTOCOL_DIAGNOSTIC else ""
    report = f"""# Reproduction report

## Objective

Reproduction of the non-deep-learning subset of the paper.
{diagnostic_warning}
Protocol: `{protocol}`.

## Exact scope

Eight classical combinations only: TF-IDF or `structural_documented_18` with Logistic Regression, SVM, Random Forest, or KNN. It is not a full reproduction of all paper configurations.

## Explicit deep-learning exclusion

No deep-learning components, embeddings, external embedding APIs, LLM APIs, or transformer pipelines are present. The static prohibition test scans imports and dependency declarations.

## Dataset audit

Source checksum: `{dataset.source_sha256}`. Valid records: {len(dataset.frame)}. Detected fields: query `{dataset.fields.query}`, label `{dataset.fields.label}`, domain `{dataset.fields.domain}`, id `{dataset.fields.identifier}`. See `dataset_audit.md`.

## Reproduction protocol

Primary evaluation is `StratifiedKFold(n_splits=5, shuffle=False)`. TF-IDF uses `(1,2)` n-grams, 3,000 max features, `min_df=2`, and sublinear TF. TF-IDF and scaling are fitted inside each pipeline fold.

## Feature definitions

The TF-IDF branch applies no stemming, lemmatization, stop-word removal, or custom text cleaning. The structural branch is `structural_documented_18`: token/character counts, average word length, seven question-word flags, negation, approximate capitalized-span entities, clause heuristic, and five documented pattern flags.

## Classifier definitions

Logistic Regression uses L2, C=1.0, `max_iter=5000`, and no class weights. SVM uses RBF with `gamma=scale`, default C, and disabled probability estimation. Random Forest uses 200 trees, `random_state=42`, and `n_jobs=-1`. KNN uses 7 neighbours, cosine distance, and brute-force search. Resolved estimator parameters and sklearn version are recorded in saved model metadata.

## Cross-validation protocol

Every valid record receives exactly one OOF prediction. The saved fold indices, metrics, confusion matrices, and fit/prediction times make fold behavior auditable. Unshuffled CV is a minimally assumed choice because the paper does not state shuffle behavior or a seed.

## Main results

{table}

## Paper comparison

See `paper_comparison.md`; its GREEN/YELLOW/RED rubric reports numerical parity separately from protocol match.

A consistent class-name permutation does not necessarily alter accuracy or macro-F1. It does change per-class names, confusion-matrix interpretation, majority-class identity, and routing decisions when label names map to different RAG costs. It therefore changes simulated token savings. Matching macro-F1 does not prove semantic reproduction.

## Cost simulation

See `cost_accuracy_tradeoff.md`. All costs are post-hoc relative-cost simulations, not RAG or LLM measurements.

## Domain breakdown

See `domain_breakdown.md`; it uses pooled OOF predictions when a domain field is present.

## Robustness audit

See `robustness/` when the optional command is run. It is explicitly an additional audit and not part of the primary protocol.

## Reproduction gaps

See `REPRODUCTION_GAPS.md`.

## Conclusions

Protocol match does not by itself establish numerical parity; use the comparison and gaps reports together.

## Commands to reproduce

`python -m lightweight_router reproduce --data <labeled.json> --output reports`\n"""
    output.joinpath("REPRODUCTION_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": dataset.source_sha256,
                "python": sys.version, "platform": platform.platform(), "scikit_learn": sklearn.__version__,
                "primary_cv": {"n_splits": 5, "shuffle": False}, "protocol": protocol,
                "diagnostic_warning": DIAGNOSTIC_WARNING if protocol == PROTOCOL_DIAGNOSTIC else None,
                "scope": "non-deep-learning subset"}
    output.joinpath("run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output.joinpath("environment.txt").write_text((DIAGNOSTIC_WARNING + "\n" if protocol == PROTOCOL_DIAGNOSTIC else "") + f"Python: {sys.version}\nPlatform: {platform.platform()}\nscikit-learn: {sklearn.__version__}\n", encoding="utf-8")
    if protocol == PROTOCOL_DIAGNOSTIC:
        output.joinpath("WARNING.md").write_text(f"# WARNING\n\n{DIAGNOSTIC_WARNING}\n", encoding="utf-8")
