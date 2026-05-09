# 2026-05-09 - L2P Fast-Sound ID Map Ban

## Sticky Note

Do not remove this note.

Do not restore any L2P fast-sound, sampled, approximate, identity, lex-dedup,
or otherwise shortcut id-map path that bypasses full state equivalence or full
vocab equivalence analysis.

## Required Invariant

For L2P terminal-DWA construction, state equivalence analysis and vocab
equivalence analysis must always run fully.

Max-length analysis may be skipped under controlled circumstances, but the full
exact state/vocab equivalence pass is mandatory and must remain in the build
path.

## Why This Note Exists

On 2026-05-09, `Github_easy---o76439` regressed catastrophically because the
`fast_sound_id_map` shortcut in
`src/compiler/stages/id_map_and_terminal_dwa/l2p/mod.rs` defaulted on and kept
the L2P id map effectively identity-like for partition `p2`.

Observed profile before removal:

- `original_states=3465`
- `max_length_skipped=true`
- `max_length_reps=3465`
- `exact_reps=3465`
- `fast_sound_id_map_used=true`
- `p2 L2P total ~= 49069ms`
- `determinize_ms ~= 37825ms`

With the shortcut disabled, the same schema dropped to roughly:

- compile `~= 696ms`
- `p2 L2P total ~= 226ms`

Correctness stayed clean.

The failure mode was not a benign speed/precision tradeoff. It removed the
exact reduction step that keeps the terminal NWA and determinization tractable.

## Policy

- Do not make this bypass the default again.
- Do not document it as an acceptable production optimization.
- If a debug-only experiment ever reintroduces similar logic, it must stay out
  of the normal code path and must not change production defaults.