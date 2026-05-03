---
name: glrmask-mre-minimization
description: Build and recursively minimize glrmask2 MREs for mask/commit-token mismatches, schema-to-grammar bugs, GLRM grammar witnesses, and schema/prefix/token/vocab repros. Use when GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE fails, when a token is committable but absent from the mask or present in the mask but not committable, when glrmask accepts or rejects a token incorrectly, or when reducing a JSON-schema/GLRM repro to a minimal inline Rust test.
---

# GLRMask MRE Minimization

## Scope

Use this for `glrmask` bugs that need a minimal Rust repro: mask/commit mismatches, schema-to-grammar bugs, GLRM witnesses, and schema/prefix/token/vocab reductions.

## Runtime Mask/Commit Bugs

For mask/commit behavior, use `GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE=1`. The oracle is exact: a token should be in the mask iff it can be committed from the same prefix.

## Repro Workflow

1. Start with the exact schema/grammar, prefix, disputed token, and vocab. Inline the active artifact in the test body while minimizing.
2. Use `commit_bytes` for prefix setup. Compare mask, `commit_token`, and `commit_bytes` on the disputed token when investigating equivalence.
3. Minimize recursively in this priority order: vocab, schema/grammar, prefix, token bytes, then vocab again after every major grammar/schema cut.
4. Aggressively de-semantify survivors: shorten names, literals, bounds, regexes, object wrappers, punctuation, duplicate tokens, and token ordering. Keep a survivor only after an explicit failed deletion/weakening/renaming attempt.
5. If the generated grammar matters, dump it with `GLRMASK_PRINT_GRAMMAR_GLRM=1`, duplicate the test with inline GLRM, then minimize the GLRM directly.
6. Document only genuinely surprising load-bearing artifacts in the test code, especially duplicate vocab bytes, token ordering, nullable branches, or source-looking strings that resisted renaming.

Use these commands when they help locate the source artifact:

```bash
make show-schema PROBLEM=$SCHEMA
make show-example PROBLEM=$SCHEMA INDEX=N
```

## Fast Reduction

For slow repros, compile once and read mutable candidates from stable `/tmp` files, with inline originals as fallbacks. Reducers can then overwrite only `/tmp` schema/grammar/prefix/token/vocab candidates before rerunning the already-built release test. After reduction, inline the minimized artifacts and remove the temporary hooks.

Use deletion-ddmin over large artifacts. Treat only the exact failing oracle as interesting; parser errors, compile errors, and unrelated panics are not interesting.

For detailed standards from prior reductions, read `references/recursive-minimization.md`. For the `/tmp` candidate-file and ddmin pattern, read `references/tmp-candidate-ddmin.md`.
