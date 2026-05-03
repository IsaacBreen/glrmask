---
name: glrmask-mre-minimization
description: Build and recursively minimize glrmask2 MREs for mask/commit-token mismatches, schema-to-grammar bugs, GLRM grammar witnesses, and schema/prefix/token/vocab repros. Use when GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE fails, when a token is committable but absent from the mask or present in the mask but not committable, when glrmask accepts or rejects a token incorrectly, or when reducing a JSON-schema/GLRM repro to a minimal inline Rust test.
---

# GLRMask MRE Minimization

## Scope

Use this skill for `glrmask` bugs that need a minimal reproducible example, especially mask/commit mismatches. For CFA sweep discrepancy triage and `llguidance_native` discrepancy handlers, use `$cfa-discrepancy-handling` first.

Treat the ground-truth checker as a signal, not gospel. Decide what is actually allowed by the JSON schema or GLRM grammar before choosing a fix.

## Runtime Mask/Commit Bugs

Distinguish these runtime failures:

1. `glrmask` fails to commit and mask a token that it should have committed and masked, or commits/masks one it should not have.
2. Commit and mask disagree: a token is committable but absent from the mask, or present in the mask but not committable.

Use `GLRMASK_ASSERT_COMMIT_TOKEN_MASK_EQUIVALENCE=1` when investigating mask/commit behavior. Treat any equivalence failure as serious: a token should be in the mask if and only if it can be committed.

Debug straightforward breakages normally first, especially when recent local changes make the likely cause obvious. If the bug remains elusive, build and recursively minimize a repro.

## Repro Workflow

Create minimal reproducible Rust tests in `tests/mre.rs`. Do not put MREs in `integration.rs` or another existing test module.

Start with the exact schema, prefix, token, and vocab involved in the failure. Inline the schema or GLRM grammar in the test body. Do not hide the active artifact in helpers while minimizing.

Use these commands when they help locate the source artifact:

```bash
make show-schema PROBLEM=$SCHEMA
make show-example PROBLEM=$SCHEMA INDEX=N
```

Commit the prefix with `commit_bytes`. Then commit the token with `commit_token` or `commit_bytes`, choosing whichever best exposes the failure. For mask/commit equivalence issues, compare both paths explicitly.

Minimize recursively:

1. Minimize the vocab first. A small vocab makes all subsequent constraint builds much faster. Try a vocab containing only the problematic token; if that does not reproduce, reduce by compiler vocab partition, token class, and ddmin deletion before spending time on schema or prefix cuts.
2. Minimize the schema and prefix together. Removing schema branches often requires removing corresponding input bytes.
3. Minimize field names, object wrappers, pattern properties, bounds, branch counts, prefixes, token bytes, and vocabulary entries.
4. Aggressively de-semantify the witness. Try replacing property names, enum literals, token bytes, and fixed text with single letters or repeated dummy bytes. Try deleting punctuation such as quotes, braces, colons, commas, lookahead separators, and object wrappers with synchronized prefix edits. Try replacing meaningful strings like UUIDs, names, and labels with all-zero/all-`a` forms, shorter shapes, or literals.
5. Aggressively weaken terminals and regexes. Try smaller repetition bounds, removing lower or upper bounds, replacing broad classes with literals and literals with small classes, factoring repeated subpatterns into named terminals, and making long generated regexes readable. A simplification that preserves the exact oracle is worth keeping even when it is "only" readability.
6. Do not trust zero-consumption intuition. Optional stars, lookahead separators, nullable branches, and terminals that match no bytes in the current prefix can still be load-bearing for mask construction. Explicitly test deleting or weakening them.
7. Keep reducing until every remaining piece has survived an explicit attempt to delete it, inline it, weaken it, literalize it, rename it, shorten it, or scale it down.

For slow Rust repros, avoid recompiling the test for every minimization attempt. Temporarily make the test read the active schema, grammar, prefix, disputed token bytes, or vocab candidate from stable files under `/tmp`, with the inline original as the fallback. Compile once in release mode, then have the reducer overwrite only the `/tmp` candidate files before rerunning the same already-built test command. After reduction, inline the minimized artifact back into the Rust test and remove the `/tmp` override.

Use ddmin-style deletion passes instead of ad hoc editing when the artifact is large. Split properties, grammar alternatives, regex alternatives, prefix fields, literal characters, token bytes, token entries, or JSON object keys into chunks; try deleting a chunk; keep the deletion only if the exact failing oracle still reproduces; then recurse from the smaller artifact. When chunk deletion stops working, reduce chunk size down to single elements, then repeat the whole process on nested survivors. Log both accepted and rejected cuts so the remaining pieces have evidence behind them.

After the schema-based repro is truly minimal, commit that checkpoint if useful. Then add a new duplicate test that uses the generated GLRM grammar directly instead of the schema. Keep the schema-based test; do not replace it.

Use this environment variable to print the generated grammar when needed:

```bash
GLRMASK_PRINT_GRAMMAR_GLRM=1
```

Minimize the GLRM grammar again. Do not stop just because the grammar is smaller or cleaner; stop only when further deletion or weakening no longer preserves the failure.

For detailed minimization standards and examples from prior `glrmask2` work, read `references/recursive-minimization.md`. For the concrete `/tmp` candidate-file and ddmin reducer pattern, read `references/tmp-candidate-ddmin.md`.
