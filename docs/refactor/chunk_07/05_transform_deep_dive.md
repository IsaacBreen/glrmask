# Transform deep dive

A transform is a morphism:

```text
NamedGrammar -> NamedGrammar
```

It must preserve the denoted language modulo intentionally documented approximations. It must not allocate flat compiler ids. It must not depend on GLR tables, parser DWA construction, Terminal DWA construction, or runtime constraints.

## Factoring

Factoring extracts common structure to reduce downstream grammar size and parser/DWA state counts. It is a shape optimization, not a semantic extension.

Review questions:

1. Does the transform preserve terminal vs nonterminal status?
2. Are generated helper names deterministic?
3. Are internal-only terminals preserved as internal-only?
4. Does factoring avoid changing regex-like terminal expressions?

## Simplification

Simplification removes syntactic wrappers and inlines safe single-use rules. It should be conservative. Its job is not to prove arbitrary grammar equivalences.

Review questions:

1. Does simplifying `Sequence([])` preserve epsilon behavior?
2. Does simplifying singleton choices preserve display/debug expectations?
3. Does inlining respect terminal/nonterminal boundaries?
4. Does repeated simplification converge?

## Exact-subtraction transform

The exact-subtraction transform rewrites set-like grammar syntax into helper named rules. It is separated from local lowering-time exact subtraction because it may introduce multiple helpers and inspect multiple named rules.

Review questions:

1. Are generated helper names stable?
2. Does the partitioning preserve every non-excluded segment?
3. Do error messages distinguish unknown names from impossible exact alternatives?
4. Does it avoid exponential helper growth on common JSON Schema object patterns?

## Terminal choice promotion

Terminal choice promotion turns some literal-ish alternatives into terminal-level choices when that is exact and beneficial. It is a shape optimization that can influence downstream lexer/DWA structure.

Review questions:

1. Is promotion exact, not approximate?
2. Does promotion respect ignore terminals?
3. Does promotion preserve display names sufficiently for diagnostics?
4. Does promotion interact safely with exact subtraction?
