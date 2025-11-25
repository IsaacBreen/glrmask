# JSON Migration Guide

## GrammarConstraint

The JSON serialization format for `GrammarConstraint` has been overhauled to reduce size and improve parsing performance. **This is a breaking change** - old JSON files will not deserialize.

### Removed Fields

The following fields are no longer serialized:
- `precomputed1` (functionality replaced by `precomputed4`)
- `trie1_god` (runtime-only structure)
- `run_precompute4` (always true)
- `post_commit_allow_check_mode` (runtime-only)
- `terminal_map_by_llm` (runtime-only)
- `original_to_dummy_map` (dummy terminal support removed)

### Removed from GrammarConstraintConfig

- `use_dummy_terminals`
- `dummy_terminal_map`
- `dummy_terminal_penalties`
- `run_precompute4`

**Note:** Dummy terminal support has been completely removed from the codebase.

### Compact Array Formats

The following types now use compact array formats instead of verbose objects:

#### CharTransitions
```diff
- {"97": target1, "98": target2}
+ [[97, target1], [98, target2]]
```

#### HybridBitset
```diff
- [[start1, end1], [start2, end2]]
+ [start1, end1, start2, end2]
```

#### Stage7ShiftsAndReducesLookaheadValue
```diff
- {"variant": "Shift", "state_id": X}
+ ["S", X]

- {"variant": "Reduce", "nonterminal_id": X, "len": Y, "production_ids": [...]}
+ ["R", X, Y]  // production_ids dropped

- {"variant": "Split", "shift": null, "reduces": {...}}
+ ["X", null, [[nt_id, len], ...]]
```

#### Goto
```diff
- {"state_id": X, "accept": false}
+ [X, false]
```

#### Row
```diff
- {"shifts_and_reduces_full": {...}, "gotos": {...}, "default_reduce": ...}
+ [[[tid, Action], ...], [[ntid, Goto], ...], default_reduce]
```

### Changed Fields

#### `precompute4_vocab` (StageVocab)
- **Old:** JSON object with all fields  
- **New:** Only `internal_to_original` map serialized; other fields recomputed on load

#### `original_llm_vocab` (LLMVocab)
- **Old:** Object with `llm_token_map` and `max_original_llm_token_id`
- **New:** Direct `llm_token_map`; `max_original_llm_token_id` derived from keys

### Migration Path

**Old JSON files are not compatible**. To migrate:
1. Load grammar with old code version
2. Re-serialize with new code version
3. Use new JSON going forward

Alternatively, rebuild grammars from source EBNF definitions.