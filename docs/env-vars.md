# GLRMASK Environment Variables

This document lists all `GLRMASK_*` environment variables used in this crate, grouped by functional area.

## Value Parsers

- **Strict `"1"` bool**: only value `1` enables; anything else disables.
- **Truthy bool**: enabled unless value is empty or one of `0`, `false`, `no`, `off` (case-insensitive).
- **Presence toggle**: enabled when variable is present, regardless of value.
- **Compact mode**: `none|0|off|skip`, `fast`, `full|1|on`.
- **Minimize strategy**: `full`, `fast`, or `threshold:<n>`.

## GLR Parser / Table

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_DISABLE_REPLACE` | strict `1` bool | off |
| `GLRMASK_DISABLE_REPLACE_SHIFT` | strict `1` bool | off |
| `GLRMASK_DISABLE_REPLACE_GOTO` | strict `1` bool | off |
| `GLRMASK_ENABLE_LOCAL_FORWARD_REPLACE` | strict `1` bool | off |

## Compiler Pipeline

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_COMPILE_THREADS` | positive integer (`usize > 0`) | auto (rayon/macOS logic) |
| `GLRMASK_PROFILE_COMPILE` | truthy bool | off |
| `GLRMASK_PROFILE_COMPILE_SUMMARY` | truthy bool | off |
| `GLRMASK_PROFILE_PHASES` | truthy bool | off |
| `GLRMASK_DEBUG_PROFILE` | truthy bool | off |
| `GLRMASK_DEBUG_VERBOSE` | truthy bool | off |
| `GLRMASK_WARN_PROBLEMATIC_BYTE_TERMINALS` | truthy bool | off |
| `GLRMASK_DISABLE_TERMINAL_COLORING` | truthy bool | off |
| `GLRMASK_L1_IDMAP` | strict `1` bool | off |
| `GLRMASK_NO_PARTITION` | strict `1` bool | off |
| `GLRMASK_COMPACT_FINAL` | compact mode | `full` |

## Oracle / File Inputs

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_ORACLE_LOAD` | filesystem path (read JSON) | unused |
| `GLRMASK_ORACLE_DUMP` | filesystem path (write JSON) | no dump |
| `GLRMASK_PARTITION_FILE` | filesystem path (partition JSON) | auto partitioning |
| `GLRMASK_EXIT_AFTER_L1` | partition label string | no early exit |

## Terminal DWA / ID Map (L1/L2P/Merge)

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_FORCE_ALL_L2P` | strict `1` bool | off |
| `GLRMASK_DEBUG_DWA_DUMP` | strict `1` bool | off |
| `GLRMASK_PROFILE_TERMINAL_DWA` | presence toggle | off |
| `GLRMASK_PROFILE_PARSER_DWA` | presence toggle | off |
| `GLRMASK_PROFILE_DETERMINIZE` | strict `1` bool | off |
| `GLRMASK_PROFILE_COMPACT` | presence toggle | off |
| `GLRMASK_DEBUG_CHARACTERIZE` | presence toggle | off |
| `GLRMASK_DEBUG_MAX_LENGTH` | truthy bool | off |
| `GLRMASK_DISABLE_DIVERSITY_STATE_ORDER` | truthy bool | off |
| `GLRMASK_DISABLE_TRIE_WALK` | truthy bool | off |
| `GLRMASK_VOCAB_UNGROUPED_BATCH` | truthy bool | off |
| `GLRMASK_VOCAB_EQUIV_BATCH_SIZE` | positive integer (`usize > 0`) | auto |
| `GLRMASK_SKIP_MAX_LENGTH_STATE_EQUIV` | truthy bool | off |
| `GLRMASK_SKIP_TOKEN_STATE_EQUIV` | truthy bool | off |
| `GLRMASK_USE_REFERENCE_EQUIV` | truthy bool | off |
| `GLRMASK_USE_SLOW_VOCAB_EQUIV` | truthy bool | off |
| `GLRMASK_FORCE_PRE_VOCAB_STATE_REDUCTION` | truthy bool | off |
| `GLRMASK_DISABLE_PRE_VOCAB_STATE_REDUCTION` | truthy bool | off |
| `GLRMASK_PROFILE_VOCAB_REACHABILITY` | truthy bool | off |

### Minimize strategy vars

| Variable | Valid values | Default behavior |
|---|---|---|
| `GLRMASK_MINIMIZE_BUNDLE` | minimize strategy | callsite default (`minimize_fast` for multi-group bundles) |
| `GLRMASK_MINIMIZE_L2P` | minimize strategy | callsite default (`minimize_with_threshold(..., 50)`) |
| `GLRMASK_MINIMIZE_MERGE` | minimize strategy | callsite default (`minimize`) |
| `GLRMASK_MINIMIZE_MERGE_GLOBAL` | minimize strategy | callsite default (`minimize`) |
| `GLRMASK_MINIMIZE_PARSER_DWA` | minimize strategy | callsite default (`minimize_fast`) |

### Compact mode vars

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_COMPACT_L1` | compact mode | `fast` |
| `GLRMASK_COMPACT_MERGE` | compact mode | `fast` |
| `GLRMASK_COMPACT_MERGE_GLOBAL` | compact mode | `fast` |

## JSON Schema Import

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_NO_ADDITIONAL_PROPERTIES` | truthy bool | off |
| `GLRMASK_AP_DEFAULT_FALSE` | truthy bool | off |
| `GLRMASK_AP_SHARED_EXCLUSIONS` | truthy bool | off |
| `GLRMASK_ADDPROP_NO_EXCLUSIONS` | truthy bool | off |
| `GLRMASK_AP_KEY_ANY_STRING` | truthy bool | off |
| `GLRMASK_MERGE_ANYOF` | strict `1` bool | off |
| `GLRMASK_DISABLE_EXACT_CLOSED_OBJECT_UNION` | truthy bool | off |
| `GLRMASK_ENABLE_FACTORED_CLOSED_OBJECT` | truthy bool | off |
| `GLRMASK_PROFILE_OBJECT_FUSION` | presence toggle | off |
| `GLRMASK_SPLIT_OPEN_QUOTE` | truthy bool | on |
| `GLRMASK_SPLIT_CLOSE_QUOTE` | truthy bool | off |
| `GLRMASK_SPLIT_COLON_SPACE` | truthy bool | on |
| `GLRMASK_SPLIT_COLON_FROM_SPACE` | truthy bool | off |
| `GLRMASK_STRING_REPEAT_CHUNK` | integer (`usize`) | `256` |
| `GLRMASK_MAX_STRING_LENGTH_CAP` | integer (`usize`) | unset (`None`) |

### Closed-object threshold knobs

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_ALTS` | integer (`usize`) | `128` |
| `GLRMASK_CLOSED_REQUIRED_OBJECT_FUSED_LITERAL_MAX_TOTAL_BYTES` | integer (`usize`) | `65536` |
| `GLRMASK_EXACT_CLOSED_OBJECT_UNION_MAX_VARIANTS` | integer (`usize`) | `8` |
| `GLRMASK_EXACT_CLOSED_OBJECT_UNION_MAX_KEYS` | integer (`usize`) | `16` |
| `GLRMASK_EXACT_CLOSED_OBJECT_SINGLE_MAX_KEYS` | integer (`usize`) | `16` |
| `GLRMASK_EXACT_CLOSED_OBJECT_UNION_MAX_STATES` | integer (`usize`) | `128` |
| `GLRMASK_FACTORED_OPEN_OBJECT_MAX_KEYS` | integer (`usize`) | `64` |

## Grammar AST Lowering

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_REPEAT_TREE_SHAPE` | `left`, `balanced`, `leftbalanced`, `left_balanced` (other set value falls to right) | `leftbalanced` when unset |
| `GLRMASK_MAX_RUNTIME_REDUCTION_LEN` | positive integer (`usize > 0`) | `5` |

## Runtime

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_PROFILE_VERBOSE` | presence toggle | off |

## Notes

- For minimize strategy vars, invalid set values panic with a validation error.
- For compact mode vars, unknown set values silently fall back to the per-callsite default.
- `GLRMASK_AP_KEY_ANY_STRING` is effectively enabled if either itself or `GLRMASK_ADDPROP_NO_EXCLUSIONS` is enabled.
