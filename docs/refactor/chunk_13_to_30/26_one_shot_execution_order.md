# Chunk 26: One-shot execution order without premature compilation

## Purpose

This chunk completes one remaining publication-cleanup area after the first twelve structural chunks.  It is deliberately self-contained: a reader should not need the historical plan to understand what changed, what the target architecture is, and how to continue compile repair later.

## Files and directories in scope

- `docs/api_boundary.md`
- `docs/architecture.md`
- `docs/chunk_02_terminology_alignment.md`
- `docs/compile_pipeline.md`
- `docs/configuration.md`
- `docs/json_schema.md`
- `docs/json_schema_support.md`
- `docs/paper_mapping.md`
- `docs/parser_dwa.md`
- `docs/performance.md`
- `docs/performance_preservation.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_CHANGESET.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_CHECKS.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_FILES.csv`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_LOC.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_SYMBOLS.csv`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_SYMBOLS.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_TREE.txt`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_patch_stats.txt`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_ARTIFACT_TREE.txt`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_CHANGESET.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_CHECKS.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_FILE_MANIFEST.txt`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_patch_stats.txt`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_CHANGESET.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_CHECKS.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_COMMIT_LOC.md`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_COMMIT_TREE.txt`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_FILE_MANIFEST.txt`
- `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_patch_stats.txt`
- `docs/refactor/chunk_02/implementation_manual.md`
- `docs/refactor/chunk_02/mathematical_contracts.md`
- `docs/refactor/chunk_02/review_checklist.md`
- `docs/refactor/chunk_03/00_implementation_manual.md`
- `docs/refactor/chunk_03/01_mathematical_contracts.md`
- `docs/refactor/chunk_03/02_application_instructions.md`
- `docs/refactor/chunk_03/03_review_checklist.md`
- `docs/refactor/chunk_03/04_dependency_graph_and_parallelism.md`
- `docs/refactor/chunk_03/05_symbol_move_table.md`
- `docs/refactor/chunk_03/06_file_level_surgery.md`
- `docs/refactor/chunk_03/07_phase_by_phase_deep_dive.md`
- `docs/refactor/chunk_03/08_invariant_catalogue.md`
- `docs/refactor/chunk_03/09_basic_implementer_manual.md`
- `docs/refactor/chunk_03/10_follow_up_backlog.md`
- `docs/refactor/chunk_04/00_implementation_manual.md`
- `docs/refactor/chunk_04/01_mathematical_contracts.md`
- `docs/refactor/chunk_04/02_algorithm_deep_dive.md`
- `docs/refactor/chunk_04/03_file_and_symbol_surgery.md`
- `docs/refactor/chunk_04/04_partitioning_theory_and_policy.md`
- `docs/refactor/chunk_04/05_merge_and_id_map_contracts.md`
- `docs/refactor/chunk_04/06_review_checklist.md`
- `docs/refactor/chunk_04/07_basic_implementer_manual.md`
- `docs/refactor/chunk_04/08_risk_register_and_follow_up.md`
- `docs/refactor/chunk_04/09_exhaustive_symbol_inventory.md`
- `docs/refactor/chunk_04/10_manual_application_instructions.md`
- `docs/refactor/chunk_04/11_proof_obligations_by_file.md`
- `docs/refactor/chunk_04/12_direct_partition_split_blueprint.md`
- `docs/refactor/chunk_04/13_pair_partition_split_blueprint.md`
- `docs/refactor/chunk_04/14_reviewer_question_bank.md`
- `docs/refactor/chunk_05/basic_implementer_walkthrough.md`
- `docs/refactor/chunk_05/file_by_file_change_ledger.md`
- `docs/refactor/chunk_05/followup_backlog.md`
- `docs/refactor/chunk_05/function_proof_obligations.md`
- `docs/refactor/chunk_05/implementation_manual.md`
- `docs/refactor/chunk_05/manual_application_notes.md`
- `docs/refactor/chunk_05/mathematical_contracts.md`
- `docs/refactor/chunk_05/mathematical_deep_dive.md`
- `docs/refactor/chunk_05/phase_graph.md`
- `docs/refactor/chunk_05/review_checklist.md`
- `docs/refactor/chunk_05/risk_register.md`
- `docs/refactor/chunk_05/symbol_map.md`
- `docs/refactor/chunk_05/tables/file_line_counts.md`
- `docs/refactor/chunk_06/00_overview.md`
- `docs/refactor/chunk_06/01_mathematical_contract.md`
- `docs/refactor/chunk_06/02_file_by_file_implementation.md`
- `docs/refactor/chunk_06/03_algorithm_walkthrough.md`
- `docs/refactor/chunk_06/04_invariant_catalogue.md`
- `docs/refactor/chunk_06/05_exact_edit_log.md`
- `docs/refactor/chunk_06/06_review_checklist.md`
- `docs/refactor/chunk_06/07_risk_register.md`

## Priority

Priority level: **publication-shaping / high**.  These changes are primarily about making the mathematical architecture visible.  They should be completed before detailed compile repair because compile errors are much easier to repair once the target module boundaries are correct.

## Target abstraction

The target abstraction for this chunk is not “a set of Rust files”.  It is a named mathematical object or policy boundary.  The source tree should encode that object directly.  Names that describe accidents of implementation, old experiments, or temporary benchmark harnesses should be demoted to compatibility shims or deleted.


## Mathematical reading discipline

For every function in this area, classify it before editing:

1. **Denotation constructor** — builds a language, relation, quotient, automaton, or transition system.
2. **Representation transformer** — changes storage while preserving denotation.
3. **Evaluator** — applies an already-built object to a state, token, byte, or stack.
4. **Policy reader** — chooses an algorithm or diagnostic mode.
5. **Reporter** — formats diagnostics or profiles without changing semantics.

Functions from different classes should not be interleaved unless a module is explicitly an orchestrator.

## Definition of done

A chunk is done when a beginner can answer these questions by looking only at file names and short module headers:

- What mathematical object does this directory own?
- What does each child file own?
- Which file is the public boundary?
- Which files are compatibility shims?
- Which operations are semantic, and which are optimizations?
- Which invariants must tests check after compile repair?


## Concrete application rules

1. Keep compatibility shims small and visibly marked with `#[doc(hidden)]` where possible.
2. Move canonical code into a module named after its denotation.
3. Update imports in canonical code to the new path.
4. Do not hide a semantic operation inside an optimization module.
5. Do not put environment-variable parsing inside a pure mathematical algorithm unless it is temporarily documented as legacy.
6. Every README must state both denotation and forbidden dependencies.
7. Old names may remain only as shims, not as the preferred path in new code.
8. Every large file left unsplit must be called out as a remaining mechanical extraction target.

## Invariants to test after compile repair

- Moving files did not change the recognized language or runtime transition relation.
- Every compatibility shim reexports exactly the canonical module and no new logic.
- Every quotient map is applied consistently to all ids in its artifact.
- Every fast path has a direct reference path with equal denotation.
- Diagnostic/profiling code cannot mutate semantic state except for cache/stat counters.

## Review checklist

- Read the directory README first.
- Confirm public names match paper names.
- Confirm no old path is used by canonical code.
- Confirm examples describe public API, not internal modules.
- Confirm future compile-repair notes are exact enough to execute mechanically.

## Deferred compile repair notes

This pass intentionally does not compile.  The repair order is: path imports, visibility, module declarations, formatting, clippy, unit tests, integration tests, serialization tests, benchmark parity.  Do not start by deleting compatibility shims; they are temporary scaffolding for the first compile repair pass.
