# Proof obligations by Terminal-DWA file

This document states what each file must preserve.  A future compiler repair should use these obligations before making local fixes.

## `src/compile/terminal_dwa/builder.rs`

- Produce a `MappedArtifact<DWA>` whose weights are in the correct local-id coordinate space for downstream reconciliation.
- Union local partition relations by merge rather than by ad hoc mutation.
- Preserve the empty-vocabulary identity case.

## `src/compile/terminal_dwa/classify.rs`

- Maintain its declared support role without acquiring unrelated orchestration responsibilities.
- Keep coordinate-space assumptions documented at call boundaries.

## `src/compile/terminal_dwa/direct_partition/max_length.rs`

- Cover only direct/single-step terminal-path cases selected by `partition.rs`.
- Maintain exact correspondence between token ranges and weight masks.
- Update id maps whenever state/token classes are compacted.

## `src/compile/terminal_dwa/direct_partition/mod.rs`

- Cover only direct/single-step terminal-path cases selected by `partition.rs`.
- Maintain exact correspondence between token ranges and weight masks.
- Update id maps whenever state/token classes are compacted.

## `src/compile/terminal_dwa/global_state_map.rs`

- Compute a tokenizer-state quotient whose witness is a `ManyToOneIdMap`.
- Use the global equivalence scope, not a local direct/pair scope.
- Leave vocabulary partitioning and local DWA construction to other modules.

## `src/compile/terminal_dwa/grammar_helpers.rs`

- Maintain its declared support role without acquiring unrelated orchestration responsibilities.
- Keep coordinate-space assumptions documented at call boundaries.

## `src/compile/terminal_dwa/merge.rs`

- Reconcile every local tokenizer-state coordinate with a global coordinate.
- Reconcile every local token coordinate with a global internal-token coordinate.
- Remap every weight consistently with the id maps.

## `src/compile/terminal_dwa/mod.rs`

- Expose the subsystem without hiding implementation policy inside the boundary file.
- Make neighbouring phases able to call the stable Terminal-DWA entry points without knowing internal modules.
- Avoid any construction logic that could blur denotation and strategy.

## `src/compile/terminal_dwa/options.rs`

- Convert process-level strings and integers into typed internal policy values.
- Avoid constructing or mutating automata, vocabularies, id maps, parser tables, or runtime artifacts.
- Keep every default explicit so a publication reader can reconstruct the build route.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/combined.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/compat.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/disallowed_follows.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/mod.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/shared.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/fast.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/max_length.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state/mod.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/max_length.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/mod.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/state_equivalence/pipeline.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/fast.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/equivalence_analysis/vocab/mod.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/mod.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/pair_partition/nwa_builder.rs`

- Produce a `MappedArtifact<DWA>` whose weights are in the correct local-id coordinate space for downstream reconciliation.
- Union local partition relations by merge rather than by ad hoc mutation.
- Preserve the empty-vocabulary identity case.

## `src/compile/terminal_dwa/pair_partition/postprocess.rs`

- Cover multi-step terminal-path cases selected by `partition.rs`.
- Preserve compatibility between simplified tokenizer views and original tokenizer transitions.
- Apply disallowed-follow constraints as constraints on weights, not as parser semantics changes.

## `src/compile/terminal_dwa/partition.rs`

- Maintain its declared support role without acquiring unrelated orchestration responsibilities.
- Keep coordinate-space assumptions documented at call boundaries.

## `src/compile/terminal_dwa/types.rs`

- Maintain its declared support role without acquiring unrelated orchestration responsibilities.
- Keep coordinate-space assumptions documented at call boundaries.

## `src/compile/terminal_dwa/vocab_partition.rs`

- Return a disjoint, intended-to-be-exhaustive set of sub-vocabularies.
- Preserve token ids from the original caller vocabulary.
- Keep partition choice independent of final relation semantics.

