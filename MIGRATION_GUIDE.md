# JSON Migration Guide

## GrammarConstraint

The JSON serialization format for `GrammarConstraint` has been overhauled to reduce size and complexity. This is a breaking change.

### Removed Fields

The following fields are no longer serialized and will be initialized with default/empty values upon deserialization:
- `precomputed1` (Replaced by `precomputed4`)
- `trie1_god`
- `run_precompute4` (Defaults to `true`)
- `post_commit_allow_check_mode` (Defaults to `StepProbe`)
- `terminal_map_by_llm`
- `original_to_dummy_map`

### Changed Fields

#### `precompute4_vocab` (StageVocab)
**Old Format:** A JSON object containing `internal_to_original`, `original_to_internal`, `internal_max_llm_token`, etc.
**New Format:** A simple JSON map representing `internal_to_original`. All other fields (inverse map, sparse matrix, max IDs) are recomputed during deserialization.

#### `original_llm_vocab` (LLMVocab)
**Old Format:** A JSON object containing `llm_token_map` and `max_original_llm_token_id`.
**New Format:** The JSON representation of `llm_token_map` directly. `max_original_llm_token_id` is derived from the map keys.

### Action Required

If you are manually constructing or parsing these JSON files outside of the `GrammarConstraint::to_json` / `from_json` methods, you must update your schemas to match the new reduced structure. Old JSON files may fail to load if they are missing the new direct map structures, though `GrammarConstraint` does not support backward compatibility for the old format.