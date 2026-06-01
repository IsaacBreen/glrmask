# Chunk 13-30 all-remaining cleanup package

This directory documents the consolidated pass that applies the remaining publication-cleanup chunks after Chunk 12.  The pass is intentionally architectural: it optimizes the source tree for what a reader should see, not for immediate compilation.

## Chunks covered

| Chunk | Work area | Detailed file |
|---:|---|---|
| 13 | Commit runtime, scanner/parser separation finalization | `docs/refactor/chunk_13_to_30/13_commit_runtime_scanner_parser_separation.md` |
| 14 | Template-DFA subsystem | `docs/refactor/chunk_13_to_30/14_template_dfa_subsystem.md` |
| 15 | Weights, masks, pair sets, and Boolean relation algebra | `docs/refactor/chunk_13_to_30/15_sets_weights_pair_relations.md` |
| 16 | Leveled GSS and parser stack data structures | `docs/refactor/chunk_13_to_30/16_gss_stack_structures.md` |
| 17 | Automata module cleanup | `docs/refactor/chunk_13_to_30/17_automata_module_cleanup.md` |
| 18 | Configuration, profiling, diagnostics, and logging | `docs/refactor/chunk_13_to_30/18_configuration_profiling_diagnostics_logging.md` |
| 19 | Error handling and invariant policy | `docs/refactor/chunk_13_to_30/19_error_handling_invariant_policy.md` |
| 20 | Python bindings publication cleanup | `docs/refactor/chunk_13_to_30/20_python_bindings_publication_cleanup.md` |
| 21 | Tests, examples, benchmarks, and documentation | `docs/refactor/chunk_13_to_30/21_tests_examples_benchmarks_docs.md` |
| 22 | Serialization and cache compatibility | `docs/refactor/chunk_13_to_30/22_serialization_cache_compatibility.md` |
| 23 | ID-space naming and invariants | `docs/refactor/chunk_13_to_30/23_id_space_naming_invariants.md` |
| 24 | Mapped artifact and compaction cleanup | `docs/refactor/chunk_13_to_30/24_mapped_artifact_compaction_cleanup.md` |
| 25 | Small naming, comment, and style cleanup sweep | `docs/refactor/chunk_13_to_30/25_small_naming_comment_style_sweep.md` |
| 26 | One-shot execution order without premature compilation | `docs/refactor/chunk_13_to_30/26_one_shot_execution_order.md` |
| 27 | Post-refactor validation and acceptance criteria | `docs/refactor/chunk_13_to_30/27_post_refactor_validation.md` |
| 28 | Frontend importers other than JSON Schema | `docs/refactor/chunk_13_to_30/28_frontend_importers_non_json.md` |
| 29 | Publication documentation skeleton and content | `docs/refactor/chunk_13_to_30/29_docs_skeleton_content.md` |
| 30 | Performance preservation after structural cleanup | `docs/refactor/chunk_13_to_30/30_performance_preservation.md` |

## Global target shape

The target crate has these top-level mathematical domains:

```text
api/              public facade
import/           external language frontends -> grammar_ir
grammar_ir/       grammar syntax and lowering
parser/           GLR tables, parser advance, graph-structured stacks
compile/          compile-time automata, scan relations, id-space quotients
sets/             set/weight algebra
automata/         generic finite automata
runtime/          immutable artifact, mutable state, Mask, Commit, token-space projection
diagnostics/      user-visible diagnostics and cache maintenance
config/           configuration vocabulary and environment helpers
invariants/       assertion policy
bindings/python/  PyO3 binding boundary
```

The mathematical invariant behind the whole cleanup is separation of denotation from representation.  A module name should tell the reader what language, relation, quotient, or transition system it denotes.  Implementation tricks should be nested under that denotation rather than defining the public mental model.
