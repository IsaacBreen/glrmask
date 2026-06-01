# Chunk 29 implementation manual: docs_skeleton_content

## Scope

docs

## Exact files to open first

1. `docs/api_boundary.md`
2. `docs/architecture.md`
3. `docs/chunk_02_terminology_alignment.md`
4. `docs/compile_pipeline.md`
5. `docs/configuration.md`
6. `docs/json_schema.md`
7. `docs/json_schema_support.md`
8. `docs/paper_mapping.md`
9. `docs/parser_dwa.md`
10. `docs/performance.md`
11. `docs/performance_preservation.md`
12. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_CHANGESET.md`
13. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_CHECKS.md`
14. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_FILES.csv`
15. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_LOC.md`
16. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_SYMBOLS.csv`
17. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_SYMBOLS.md`
18. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_JSON_SCHEMA_TREE.txt`
19. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_08_patch_stats.txt`
20. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_ARTIFACT_TREE.txt`
21. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_CHANGESET.md`
22. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_CHECKS.md`
23. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_FILE_MANIFEST.txt`
24. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_10_patch_stats.txt`
25. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_CHANGESET.md`
26. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_CHECKS.md`
27. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_COMMIT_LOC.md`
28. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_COMMIT_TREE.txt`
29. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_FILE_MANIFEST.txt`
30. `docs/refactor/archive/prior_chunk_artifacts/glrmask_chunk_12/glrmask_chunk_12_patch_stats.txt`
31. `docs/refactor/chunk_02/implementation_manual.md`
32. `docs/refactor/chunk_02/mathematical_contracts.md`
33. `docs/refactor/chunk_02/review_checklist.md`
34. `docs/refactor/chunk_03/00_implementation_manual.md`
35. `docs/refactor/chunk_03/01_mathematical_contracts.md`
36. `docs/refactor/chunk_03/02_application_instructions.md`
37. `docs/refactor/chunk_03/03_review_checklist.md`
38. `docs/refactor/chunk_03/04_dependency_graph_and_parallelism.md`
39. `docs/refactor/chunk_03/05_symbol_move_table.md`
40. `docs/refactor/chunk_03/06_file_level_surgery.md`
41. `docs/refactor/chunk_03/07_phase_by_phase_deep_dive.md`
42. `docs/refactor/chunk_03/08_invariant_catalogue.md`
43. `docs/refactor/chunk_03/09_basic_implementer_manual.md`
44. `docs/refactor/chunk_03/10_follow_up_backlog.md`
45. `docs/refactor/chunk_04/00_implementation_manual.md`
46. `docs/refactor/chunk_04/01_mathematical_contracts.md`
47. `docs/refactor/chunk_04/02_algorithm_deep_dive.md`
48. `docs/refactor/chunk_04/03_file_and_symbol_surgery.md`
49. `docs/refactor/chunk_04/04_partitioning_theory_and_policy.md`
50. `docs/refactor/chunk_04/05_merge_and_id_map_contracts.md`
51. `docs/refactor/chunk_04/06_review_checklist.md`
52. `docs/refactor/chunk_04/07_basic_implementer_manual.md`
53. `docs/refactor/chunk_04/08_risk_register_and_follow_up.md`
54. `docs/refactor/chunk_04/09_exhaustive_symbol_inventory.md`
55. `docs/refactor/chunk_04/10_manual_application_instructions.md`
56. `docs/refactor/chunk_04/11_proof_obligations_by_file.md`
57. `docs/refactor/chunk_04/12_direct_partition_split_blueprint.md`
58. `docs/refactor/chunk_04/13_pair_partition_split_blueprint.md`
59. `docs/refactor/chunk_04/14_reviewer_question_bank.md`
60. `docs/refactor/chunk_05/basic_implementer_walkthrough.md`
61. `docs/refactor/chunk_05/file_by_file_change_ledger.md`
62. `docs/refactor/chunk_05/followup_backlog.md`
63. `docs/refactor/chunk_05/function_proof_obligations.md`
64. `docs/refactor/chunk_05/implementation_manual.md`
65. `docs/refactor/chunk_05/manual_application_notes.md`
66. `docs/refactor/chunk_05/mathematical_contracts.md`
67. `docs/refactor/chunk_05/mathematical_deep_dive.md`
68. `docs/refactor/chunk_05/phase_graph.md`
69. `docs/refactor/chunk_05/review_checklist.md`
70. `docs/refactor/chunk_05/risk_register.md`
71. `docs/refactor/chunk_05/symbol_map.md`
72. `docs/refactor/chunk_05/tables/file_line_counts.md`
73. `docs/refactor/chunk_06/00_overview.md`
74. `docs/refactor/chunk_06/01_mathematical_contract.md`
75. `docs/refactor/chunk_06/02_file_by_file_implementation.md`
76. `docs/refactor/chunk_06/03_algorithm_walkthrough.md`
77. `docs/refactor/chunk_06/04_invariant_catalogue.md`
78. `docs/refactor/chunk_06/05_exact_edit_log.md`
79. `docs/refactor/chunk_06/06_review_checklist.md`
80. `docs/refactor/chunk_06/07_risk_register.md`
81. `docs/refactor/chunk_06/08_deferred_followups.md`
82. `docs/refactor/chunk_06/09_partial_boundary_test_design.md`
83. `docs/refactor/chunk_06/10_for_basic_implementer.md`
84. `docs/refactor/chunk_06/11_patch_acceptance_criteria.md`
85. `docs/refactor/chunk_06/12_symbol_ledger.md`
86. `docs/refactor/chunk_06/13_blue_sky_target.md`
87. `docs/refactor/chunk_06/14_phase_by_phase_deep_dive.md`
88. `docs/refactor/chunk_06/15_function_level_ledger.md`
89. `docs/refactor/chunk_06/16_compile_repair_strategy.md`
90. `docs/refactor/chunk_06/17_relation_to_paper.md`
91. `docs/refactor/chunk_06/18_design_alternatives_rejected.md`
92. `docs/refactor/chunk_06/19_performance_notes.md`
93. `docs/refactor/chunk_06/20_quality_bar.md`
94. `docs/refactor/chunk_07/00_chunk_07_overview.md`
95. `docs/refactor/chunk_07/01_mathematical_contract.md`
96. `docs/refactor/chunk_07/02_file_by_file_surgery.md`
97. `docs/refactor/chunk_07/03_symbol_move_table.md`
98. `docs/refactor/chunk_07/04_lowering_deep_dive.md`
99. `docs/refactor/chunk_07/05_transform_deep_dive.md`
100. `docs/refactor/chunk_07/06_glrm_boundary.md`
101. `docs/refactor/chunk_07/07_invariant_catalogue.md`
102. `docs/refactor/chunk_07/08_review_checklist.md`
103. `docs/refactor/chunk_07/09_future_refactor_options.md`
104. `docs/refactor/chunk_07/10_basic_implementer_manual.md`
105. `docs/refactor/chunk_07/11_deferred_compile_repair_strategy.md`
106. `docs/refactor/chunk_07/12_proof_obligations_by_module.md`
107. `docs/refactor/chunk_07/13_manual_apply_instructions.md`
108. `docs/refactor/chunk_07/14_import_migration_backlog.md`
109. `docs/refactor/chunk_07/15_test_design_after_split.md`
110. `docs/refactor/chunk_07/16_risk_register.md`
111. `docs/refactor/chunk_07/17_visibility_policy.md`
112. `docs/refactor/chunk_07/18_exact_subtraction_two_levels.md`
113. `docs/refactor/chunk_07/19_separated_sequence_mathematics.md`
114. `docs/refactor/chunk_07/20_repeat_lowering_mathematics.md`
115. `docs/refactor/chunk_07/21_transform_pipeline_order.md`
116. `docs/refactor/chunk_07/22_reader_guide.md`
117. `docs/refactor/chunk_07/23_open_questions.md`
118. `docs/refactor/chunk_07/24_chunk_07_definition_of_done.md`
119. `docs/refactor/chunk_08/15_symbol_inventory.md`
120. `docs/refactor/chunk_08/16_file_metrics.md`

## Mechanical procedure

1. Open the canonical module boundary file before editing children.
2. Read the directory README and confirm the denotation it claims.
3. For every import that uses an old path, choose one of two actions: update to the canonical path, or leave only inside a compatibility shim.
4. For every public or crate-visible symbol, classify it as constructor, transformer, evaluator, policy, reporter, or compatibility.
5. Move constructors and evaluators into semantic modules; keep reporters in diagnostics/profiling modules.
6. Preserve old names only as `#[doc(hidden)]` shims.
7. Do not change algorithmic logic unless a move forces a path update.
8. Do not add environment-variable reads to pure files.
9. Add or update README text whenever a directory boundary changes.
10. Record every deliberate non-split large file as future mechanical extraction, not as forgotten work.

## Beginner-level edit recipe

- If you see a file whose name says only `mod.rs` and it is longer than 250 lines, look for obvious groups separated by comments.
- If a group contains option parsing, move it to `options.rs`.
- If a group contains print or profile formatting, move it to `profile.rs` or diagnostics.
- If a group contains helper structs used only by one algorithm, keep it near that algorithm.
- If a group defines a mathematical carrier type used by many algorithms, move it upward into a named domain module.
- After each move, search for the old path across `src`, `bindings`, `examples`, `tests`, and `benches`.

## Definition of complete for this chunk

- The target directory exists.
- The compatibility directory, if any, contains only shims.
- Canonical source files import canonical paths.
- Documentation names the denotation, forbidden dependencies, and validation checks.
- The changeset explains why the new grouping is mathematically better.
