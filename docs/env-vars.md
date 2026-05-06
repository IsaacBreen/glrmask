# GLRMASK Environment Variables

This document lists all `GLRMASK_*` environment variables used in this crate, grouped by functional area.

## Value Parsers

- **Strict `"1"` bool**: only value `1` enables; anything else disables.
- **Truthy bool**: enabled unless value is empty or one of `0`, `false`, `no`, `off` (case-insensitive).
- **Presence toggle**: enabled when variable is present, regardless of value.
- **Compact mode**: `none|0|off|skip`, `fast`, `full|1|on`.
- **Minimize strategy**: `full`, `fast`, or `threshold:<n>`.

## Compiler Pipeline

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_COMPILE_THREADS` | positive integer (`usize > 0`) | auto (rayon/macOS logic) |
| `GLRMASK_PROFILE_COMPILE` | truthy bool | off |
| `GLRMASK_PROFILE_COMPILE_SUMMARY` | truthy bool | off |
| `GLRMASK_DISABLE_TERMINAL_COLORING` | truthy bool | off |

## Terminal DWA / ID Map (L1/L2P/Merge)

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_FORCE_ALL_L2P` | strict `1` bool | off |
| `GLRMASK_PROFILE_DETERMINIZE` | strict `1` bool | off |
| `GLRMASK_DISABLE_DIVERSITY_STATE_ORDER` | truthy bool | off |
| `GLRMASK_DISABLE_TRIE_WALK` | truthy bool | off |
| `GLRMASK_VOCAB_UNGROUPED_BATCH` | truthy bool | off |
| `GLRMASK_VOCAB_EQUIV_BATCH_SIZE` | positive integer (`usize > 0`) | auto |

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
| `GLRMASK_ADDPROP_NO_EXCLUSIONS` | truthy bool | off |
| `GLRMASK_AP_KEY_ANY_STRING` | truthy bool | off |
| `GLRMASK_MERGE_ANYOF` | strict `1` bool | off |
| `GLRMASK_STRING_REPEAT_CHUNK` | integer (`usize`) | `256` |

## Grammar AST Lowering

| Variable | Valid values | Default |
|---|---|---|
| `GLRMASK_REPEAT_TREE_SHAPE` | `left`, `balanced`, `leftbalanced`, `left_balanced` (other set value falls to right) | `leftbalanced` when unset |
| `GLRMASK_MAX_RUNTIME_REDUCTION_LEN` | positive integer (`usize > 0`) | `5` |

## Notes

- For minimize strategy vars, invalid set values panic with a validation error.
- For compact mode vars, unknown set values silently fall back to the per-callsite default.
- `GLRMASK_AP_KEY_ANY_STRING` is effectively enabled if either itself or `GLRMASK_ADDPROP_NO_EXCLUSIONS` is enabled.
