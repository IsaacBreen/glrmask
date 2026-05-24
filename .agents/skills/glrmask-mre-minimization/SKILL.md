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

Before changing grammar/importer/lowering code, run the same prefix and token
through `commit_bytes(token_bytes)` and classify the bug:

- `commit_bytes` accepts but `mask` or `commit_token` rejects: the language is
  correct and the bug is below the grammar. Do not repair it by changing
  equivalent grammar structure, terminal grouping, helper nonterminals,
  factoring, chunking, or byte fusion. Minimize the lower-layer mismatch.
- `commit_bytes` rejects and the token should be valid: the source/importer or
  generated grammar may have wrong language semantics. A grammar/importer fix is
  allowed only after this is established.

For mask mismatches, the MRE must preserve this classification. A smaller repro
that makes the symptom disappear by changing only parse structure is a
workaround, not evidence of a fix.

Debug straightforward breakages normally first, especially when recent local changes make the likely cause obvious. If the bug remains elusive, build and recursively minimize a repro.

## Repro Workflow

Create minimal reproducible Rust tests in `tests/mre.rs`. Do not put MREs in `integration.rs` or another existing test module.

Start with the exact schema, prefix, token, and vocab involved in the failure. Inline the schema or GLRM grammar in the test body. Do not hide the active artifact in helpers while minimizing.

Document the weird load-bearing behavior directly in the MRE code. If a survivor looks obviously accidental, such as duplicate vocab bytes, order-dependent vocab grouping, nullable branches that consume nothing, strange regex bounds, or source-looking strings that could not be renamed, add a short comment beside that artifact explaining what simplifications were tried and failed.

Use these commands when they help locate the source artifact:

```bash
make show-schema PROBLEM=$SCHEMA
make show-example PROBLEM=$SCHEMA INDEX=N
```

Commit the prefix with `commit_bytes`. Then check the same token three ways:
`mask()`, `commit_token(token_id)`, and `commit_bytes(token_bytes)`. For
mask/commit equivalence issues, compare all three paths explicitly and record the
truth table in the MRE notes or test comments.

Minimize recursively:

1. Minimize the vocab first. A small vocab makes all subsequent constraint builds much faster. Try a vocab containing only the problematic token; if that does not reproduce, reduce by compiler vocab partition, token class, and ddmin deletion before spending time on schema or prefix cuts.
2. Minimize the schema and prefix together. Removing schema branches often requires removing corresponding input bytes.
3. Minimize field names, object wrappers, pattern properties, bounds, branch counts, prefixes, token bytes, and vocabulary entries.
4. Aggressively de-semantify the witness. Try replacing property names, enum literals, token bytes, and fixed text with single letters or repeated dummy bytes. Try deleting punctuation such as quotes, braces, colons, commas, lookahead separators, and object wrappers with synchronized prefix edits. Try replacing meaningful strings like UUIDs, names, and labels with all-zero/all-`a` forms, shorter shapes, or literals.
For parser-shape and path-count oracles, do not stop when the witness no longer resembles the source language, but do not accept a smaller witness that changes the causal class. If the surprising case is a completed string followed by separator prefix, a numeric regex-continuation ambiguity is not an equivalent MRE even if it preserves `parser_path_count`. Keep trying to delete fixed prefixes, separators, field/key literals, wrappers, optional branches, and helper nonterminals, but preserve the disputed terminal/follow relationship unless you have proved it is irrelevant to the bug.

When the observed split is between a schema-fixed object key path and a generic `JSON_STRING`/additional-property key path, preserve that class explicitly. First identify the live parser states in the actual replay and map them to fixed-key versus generic-key table actions. A valid minimized witness may be a direct GLRM grammar, but it must still have one branch that recognizes the key as fixed text and another branch that recognizes the same bytes as `JSON_STRING`; a generic regex ambiguity or unrelated separator-follow/table shape is not equivalent just because the final `parser_path_count` is the same. Assert survival at the point that matters, such as the next-key quote after a value comma, not merely at the first transient split.
5. Aggressively weaken terminals and regexes. Try smaller repetition bounds, removing lower or upper bounds, replacing broad classes with literals and literals with small classes, factoring repeated subpatterns into named terminals, and making long generated regexes readable. A simplification that preserves the exact oracle is worth keeping even when it is "only" readability.
Always test literal-vs-regex substitutions explicitly in both directions. Regex terminals can be load-bearing for tokenizer/parser state splitting even when they accept the same current bytes as a literal, so record rejected literalization attempts instead of assuming equivalence.
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
