from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audit import write_audit
from .config import PROTOCOL_DIAGNOSTIC, PROTOCOL_OFFICIAL, PROTOCOLS
from .data import DataValidationError, apply_protocol, load_dataset
from .evaluation import evaluate_all
from .extended_benchmark import run_extended_benchmark
from .forensic import run_forensic_reproduction
from .improvement import run_improvement
from .predict import predict_query
from .reporting import write_core_documents, write_cost_outputs, write_domain_outputs, write_paper_comparison, write_primary_outputs
from .robustness import run_robustness
from .train_final import train_final_models


def _data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", required=True, help="Path to a labeled file or directory of domain Question.json files.")
    parser.add_argument("--query-field")
    parser.add_argument("--label-field")
    parser.add_argument("--domain-field")
    parser.add_argument("--id-field")
    parser.add_argument("--expect-ragrouter-bench", action="store_true", help="Require the verified four-domain, 7,727-record local dataset shape.")


def _load(args):
    return load_dataset(args.data, query_field=args.query_field, label_field=args.label_field,
                        domain_field=args.domain_field, id_field=args.id_field,
                        expect_ragrouter_bench=args.expect_ragrouter_bench)


def _protocol_arguments(parser: argparse.ArgumentParser, diagnostic_flag: bool = False) -> None:
    parser.add_argument("--protocol", choices=PROTOCOLS, default=PROTOCOL_OFFICIAL)
    if diagnostic_flag:
        parser.add_argument("--paper-label-permutation-diagnostic", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lightweight_router", description="Auditable classical query-routing reproduction.")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit one labeled input file.")
    _data_arguments(audit)
    audit.add_argument("--output", required=True)
    reproduce = commands.add_parser("reproduce", help="Run the eight-configuration primary reproduction.")
    _data_arguments(reproduce)
    _protocol_arguments(reproduce, diagnostic_flag=True)
    reproduce.add_argument("--output", required=True)
    robust = commands.add_parser("robustness", help="Run additional robustness audits.")
    _data_arguments(robust)
    _protocol_arguments(robust)
    robust.add_argument("--output", required=True)
    train = commands.add_parser("train-final", help="CV-select and train final permitted models.")
    _data_arguments(train)
    _protocol_arguments(train)
    train.add_argument("--output", required=True)
    forensic = commands.add_parser("forensic", help="Run the controlled classical forensic reproduction audit.")
    _data_arguments(forensic)
    forensic.add_argument("--dataset-repo", required=True, help="Read-only Git repository containing the dataset history.")
    forensic.add_argument("--output", required=True, help="Must end in reports/forensic_reproduction.")
    improve = commands.add_parser("improve", help="Run the isolated classical production-router improvement study.")
    _data_arguments(improve)
    improve.add_argument("--output", required=True, help="Must be reports/improved_router.")
    improve.add_argument("--artifact-output", required=True, help="Must be artifacts/improved_router.")
    extended = commands.add_parser("extended-benchmark", help="Run the isolated extended non-deep-learning benchmark.")
    _data_arguments(extended)
    extended.add_argument("--output", required=True, help="Must be reports/extended_classical_benchmark.")
    extended.add_argument("--artifact-output", required=True,
                          help="Must be artifacts/extended_classical_benchmark.")
    predict = commands.add_parser("predict", help="Predict a routing label from a saved model.")
    predict.add_argument("--model", required=True)
    predict.add_argument("--query", required=True)
    predict.add_argument("--domain", help="Explicit corpus domain required by per-domain improvement models.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "predict":
            print(json.dumps(predict_query(args.model, args.query, args.domain), ensure_ascii=False))
            return
        dataset = _load(args)
        if args.command == "audit":
            write_audit(dataset, args.output)
            print(f"Wrote audit to {Path(args.output).resolve()}")
            return
        if args.command == "reproduce":
            write_audit(dataset, args.output)
            if args.protocol == PROTOCOL_DIAGNOSTIC and not args.paper_label_permutation_diagnostic:
                raise DataValidationError("The diagnostic protocol requires --paper-label-permutation-diagnostic.")
            if args.paper_label_permutation_diagnostic and args.protocol != PROTOCOL_DIAGNOSTIC:
                raise DataValidationError("--paper-label-permutation-diagnostic requires --protocol paper_label_permutation_diagnostic.")
            protocol_dataset = apply_protocol(dataset, args.protocol)
            protocol_output = Path(args.output) / args.protocol
            protocol_output.mkdir(parents=True, exist_ok=True)
            results = evaluate_all(protocol_dataset.frame)
            summaries = write_primary_outputs(protocol_dataset, results, protocol_output)
            write_paper_comparison(summaries, protocol_output, args.protocol)
            write_cost_outputs(results, protocol_dataset, protocol_output, args.protocol)
            write_domain_outputs(results, protocol_dataset, protocol_output)
            write_core_documents(protocol_dataset, summaries, protocol_output, args.protocol)
            print(f"Completed eight classical configurations for {args.protocol}; reports: {protocol_output.resolve()}")
            return
        if args.command == "robustness":
            if args.protocol == PROTOCOL_DIAGNOSTIC:
                raise DataValidationError("Robustness audits default to official raw labels; diagnostic robustness is not enabled.")
            run_robustness(apply_protocol(dataset, PROTOCOL_OFFICIAL).frame, args.output)
            print(f"Completed additional robustness audit: {Path(args.output).resolve()}")
            return
        if args.command == "train-final":
            protocol_dataset = apply_protocol(dataset, args.protocol)
            metadata = train_final_models(protocol_dataset, args.output, protocol=args.protocol)
            print(json.dumps({"selected_best_configuration": metadata["selected_best_configuration"], "output": str(Path(args.output).resolve())}))
            return
        if args.command == "forensic":
            summary = run_forensic_reproduction(dataset, args.output, args.dataset_repo)
            print(json.dumps({"decision": summary["decision"], "output": str(Path(args.output).resolve())}))
            return
        if args.command == "improve":
            summary = run_improvement(dataset, args.output, args.artifact_output)
            print(json.dumps({"track_name": summary["track_name"],
                              "best_quality": summary["best_quality"]["candidate_id"],
                              "output": str(Path(args.output).resolve()),
                              "artifact_output": str(Path(args.artifact_output).resolve())}))
            return
        if args.command == "extended-benchmark":
            summary = run_extended_benchmark(dataset, args.output, args.artifact_output)
            print(json.dumps({"track_name": summary["track_name"],
                              "best_accuracy": summary["best_accuracy"]["candidate_id"],
                              "output": str(Path(args.output).resolve()),
                              "artifact_output": str(Path(args.artifact_output).resolve())}))
            return
    except (DataValidationError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
