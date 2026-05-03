---
name: glrmask-debugging
description: Debug glrmask2 runtime mask/commit-token behavior, schema-to-grammar bugs, GLRM grammar witnesses, and recursive minimization of schema/prefix/token/vocab repros. Use when GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE fails, when a token is committable but absent from the mask or present in the mask but not committable, when glrmask accepts or rejects a token incorrectly, or when converting a JSON-schema repro into a minimal inline Rust test and then a minimal GLRM grammar.
---

# GLRMask Debugging

## Scope

Use this skill for `glrmask` bugs and repro minimization. For CFA sweep discrepancy triage and `llguidance_native` discrepancy handlers, use `$cfa-discrepancy-handling` first.

Treat the ground-truth checker as a signal, not gospel. Decide what is actually allowed by the JSON schema or GLRM grammar before choosing a fix.

## Runtime Mask/Commit Bugs

Distinguish these runtime failures:

1. `glrmask` fails to commit and mask a token that it should have committed and masked, or commits/masks one it should not have.
2. Commit and mask disagree: a token is committable but absent from the mask, or present in the mask but not committable.

Use `GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE=1` when investigating mask/commit behavior. Treat any equivalence failure as serious: a token should be in the mask if and only if it can be committed.

Debug straightforward breakages normally first, especially when recent local changes make the likely cause obvious. If the bug remains elusive, build and recursively minimize a repro.

## Repro Workflow

Create a minimal reproducible Rust test, commonly in `integration.rs` or the relevant existing test module.

Start with the exact schema, prefix, token, and vocab involved in the failure. Inline the schema or GLRM grammar in the test body. Do not hide the active artifact in helpers while minimizing.

Use these commands when they help locate the source artifact:

```bash
make show-schema PROBLEM=$SCHEMA
make show-example PROBLEM=$SCHEMA INDEX=N
```

Commit the prefix with `commit_bytes`. Then commit the token with `commit_token` or `commit_bytes`, choosing whichever best exposes the failure. For mask/commit equivalence issues, compare both paths explicitly.

Minimize recursively:

1. Minimize the vocab first. Try a vocab containing only the problematic token; that is often enough.
2. Minimize the schema and prefix together. Removing schema branches often requires removing corresponding input bytes.
3. Minimize field names, object wrappers, pattern properties, bounds, branch counts, prefixes, token bytes, and vocabulary entries.
4. Keep reducing until every remaining piece has survived an explicit attempt to delete it, inline it, weaken it, literalize it, or scale it down.

After the schema-based repro is truly minimal, commit that checkpoint if useful. Then replace the schema with a direct GLRM grammar.

Use this environment variable to print the generated grammar when needed:

```bash
GLRMASK_PRINT_GRAMMAR_GLRM=1
```

Minimize the GLRM grammar again. Do not stop just because the grammar is smaller or cleaner; stop only when further deletion or weakening no longer preserves the failure.

For detailed minimization standards and examples from prior `glrmask2` work, read `references/recursive-minimization.md`.
