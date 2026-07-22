from __future__ import annotations

from pathlib import Path

import pandas as pd

from .evaluation import evaluate_configuration
from .models import configuration_names
from .reporting import _markdown_table


AUDIT_NOTE = "Additional audit — not part of the paper's primary reproduction protocol."


def run_robustness(frame: pd.DataFrame, output: str | Path) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    deduplicated = frame.drop_duplicates("query", keep="first")
    for configuration in configuration_names():
        if deduplicated.label.value_counts().min() >= 5:
            result = evaluate_configuration(deduplicated, configuration)
            rows.append({"audit": "duplicate_free", "configuration": configuration, **result.summary, "note": AUDIT_NOTE})
        sensitivity = []
        for seed in (0, 1, 2, 3, 4, 13, 21, 42, 77, 100):
            result = evaluate_configuration(frame, configuration, shuffle=True, random_state=seed)
            sensitivity.append(result.summary["pooled_macro_f1"])
        rows.append({"audit": "shuffled_cv_sensitivity", "configuration": configuration, "mean_macro_f1": float(pd.Series(sensitivity).mean()),
                     "std_macro_f1": float(pd.Series(sensitivity).std(ddof=0)), "min_macro_f1": float(min(sensitivity)), "max_macro_f1": float(max(sensitivity)), "note": AUDIT_NOTE})
    if frame.domain.notna().any():
        for configuration in configuration_names():
            for domain, test in frame.groupby("domain", dropna=False):
                train = frame[frame.domain != domain]
                from .models import build_pipeline
                from sklearn.metrics import accuracy_score, f1_score
                model = build_pipeline(configuration).fit(train["query"], train["label"])
                prediction = model.predict(test["query"])
                rows.append({"audit": "leave_one_domain_out", "configuration": configuration, "held_out_domain": domain,
                             "record_count": len(test), "accuracy": accuracy_score(test.label, prediction),
                             "macro_f1": f1_score(test.label, prediction, average="macro", zero_division=0), "note": AUDIT_NOTE})
    token_count = frame["query"].str.findall(r"\b[\w'-]+\b").str.len()
    quartiles = pd.qcut(token_count, q=4, duplicates="drop")
    # Slices use primary OOF predictions to avoid a second trained prediction pass.
    for configuration in configuration_names():
        result = evaluate_configuration(frame, configuration)
        merged = result.oof.assign(length_quartile=quartiles.to_numpy())
        for interval, group in merged.groupby("length_quartile", observed=True):
            from sklearn.metrics import accuracy_score, f1_score
            rows.append({"audit": "query_length_slice", "configuration": configuration, "length_quartile": str(interval), "record_count": len(group),
                         "accuracy": accuracy_score(group.true_label, group.predicted_label), "macro_f1": f1_score(group.true_label, group.predicted_label, average="macro", zero_division=0), "note": AUDIT_NOTE})
    table = pd.DataFrame(rows)
    table.to_csv(output / "robustness_results.csv", index=False)
    (output / "README.md").write_text("# Robustness audit\n\n" + AUDIT_NOTE + "\n\n" + _markdown_table(table) + "\n", encoding="utf-8")
