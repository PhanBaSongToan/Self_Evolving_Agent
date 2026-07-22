# Extended Non-Deep-Learning Model Benchmark

## Scope

This is an isolated, strictly non-deep-learning benchmark on official raw labels. It does not modify or replace reproduction, forensic, improved-router, production-artifact, or dataset files. Answers and supporting facts are never model inputs.

Hyperparameter grids were fixed before execution. Successive-halving screens use seeds (0, 21, 42); expensive feature screens use seeds (0, 42). Screen-only configurations are ineligible for final selection. Every selectable result uses the identical ten shuffled five-fold seeds (0, 1, 2, 3, 4, 13, 21, 42, 77, 100).

Optional boosting dependencies: {"catboost": "1.2.10", "lightgbm": "4.7.0", "xgboost": "3.3.0"}.

## Selected results

| selection | candidate_id | accuracy | macro_f1 |
| --- | --- | --- | --- |
| best_single | per_domain__word_char__linear_svc_reference | 0.948104 | 0.943432 |
| best_ensemble | global__retained_nested__ensemble_stack_logistic__cfg_0d463254676f | 0.952103 | 0.947794 |
| best_global | global__retained_nested__ensemble_stack_logistic__cfg_0d463254676f | 0.952103 | 0.947794 |
| best_per_domain | per_domain__word_char__linear_svc_reference | 0.948104 | 0.943432 |

Previous best per-domain accuracy 0.948104 was exceeded. The best selectable accuracy is 0.952103; best macro-F1 is 0.947794.

Best model per-class recall: single_hop=0.949267, multi_hop=0.960524, summary=0.931037. Under-routing is 0.025197.

The cost-quality selection is `global__word_char_structural__linear_svc_reference`. Nested expected-risk savings are 27.806393% with accuracy 0.925974, macro-F1 0.920587, and under-routing 0.023424. Constraints were not relaxed.

## Stress results

| candidate_id | stress_protocol | accuracy | macro_f1 | summary_recall | multi_hop_recall | under_routing_rate |
| --- | --- | --- | --- | --- | --- | --- |
| global__retained_nested__ensemble_stack_logistic__cfg_0d463254676f | source_order_5fold | 0.845477 | 0.844831 | 0.881151 | 0.819139 | 0.095768 |
| global__retained_nested__ensemble_stack_logistic__cfg_0d463254676f | normalized_query_grouped_5fold | 0.945257 | 0.941325 | 0.925057 | 0.962800 | 0.025236 |
| global__retained_nested__ensemble_stack_logistic__cfg_0d463254676f | template_grouped_5fold | 0.943574 | 0.939318 | 0.923543 | 0.960597 | 0.026272 |
| global__retained_nested__ensemble_stack_logistic__cfg_0d463254676f | leave_one_domain_out | 0.640611 | 0.672924 | 0.894777 | 0.484826 | 0.257927 |
| per_domain__word_char__linear_svc_reference | source_order_5fold | 0.943445 | 0.940409 | 0.937169 | 0.953500 | 0.028989 |
| per_domain__word_char__linear_svc_reference | normalized_query_grouped_5fold | 0.937751 | 0.933881 | 0.921272 | 0.964758 | 0.025366 |
| per_domain__word_char__linear_svc_reference | template_grouped_5fold | 0.939304 | 0.935697 | 0.928085 | 0.967205 | 0.023295 |

## Limitations

- Shuffled IID CV, source-order CV, grouped CV, per-domain CV, and leave-one-domain-out answer different deployment questions; their numbers are not interchangeable.
- Boosting uses conservative, fixed model sizes. Feature selection and dimensionality reduction remain inside every fold.
- Meta-models are trained only on inner out-of-fold base scores. They never receive in-sample base predictions.
- Expected-risk loss scales and cost weights are illustrative simulation parameters, not measured harm or billing data.
- A higher simulated saving is not considered an improvement unless all predeclared quality and under-routing constraints hold.
