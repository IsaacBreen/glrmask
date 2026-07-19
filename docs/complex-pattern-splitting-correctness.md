# Correctness of importer-level complex pattern splitting

## Scope

The optimization is part of the JSON Schema importer. It applies only to
explicit string `pattern` constraints. It does not inspect or rewrite arbitrary
compiler terminals, GLRM grammars, EBNF grammars, recognized formats, plain
bounded strings, literals, or special-token terminals.

The importer parses the pattern to `regex-syntax` HIR and considers only fully
anchored alternatives. A branch is eligible only when it contains one
structurally unique bounded repetition

```text
P A{0,N} S
```

where `N >= 24`, `A` is non-nullable, and a conservative HIR complexity gate
classifies the repeated body as expensive. Simple patterns such as
`^[a-z]{0,100}$` remain one terminal.

If a sibling `minLength` or a max-length envelope that the importer would
actually preserve is not already implied by the pattern, the optimization is
not applied. Patterns combined with a recognized `format`, unanchored patterns,
llguidance-compat lowering, nested ambiguous candidate repeats, and unsupported
HIR forms also retain the existing monolithic path.

## Transformation

Let `K` be the importer-selected block size, and write

```text
N = qK + r,  0 <= r < K.
```

For one eligible anchored branch, define

```text
C = A^K
D = A{0,K-1} S
E = A{0,r} S.
```

The importer creates wide pattern terminals for the opening quote plus `P`, for
`C`, for `D` plus the closing quote, and for `E` plus the closing quote. It then
emits nonterminal alternatives corresponding to

```text
P A{0,K-1} S
P C A{0,K-1} S
P C^2 A{0,K-1} S
...
P C^(q-1) A{0,K-1} S
P C^q A{0,r} S.
```

Other fully anchored pattern alternatives are retained as whole quoted pattern
terminals in the same nonterminal choice.

All generated terminals from one schema pattern receive the same importer
pattern-family partition key. The lowered grammar carries an explicit
`requires_global_terminal_observation` correctness flag because parser-visible
residual terminals can overlap terminals assigned to another construction
family. The static token-equivalence pass consumes that flag by observing every
terminal residual instead of using its usual family-local projection. This is a
safety requirement, not a second splitting mechanism: there is no compiler-wide
terminal-splitting pass.

## Count theorem

The alternatives above cover the integer intervals

```text
[0,K-1], [K,2K-1], ..., [(q-1)K,qK-1], [qK,qK+r].
```

They are contiguous, pairwise disjoint, start at zero, and end at `N`. Their
union is therefore exactly `[0,N]`. Consequently, for every language `A`,
including ambiguous languages,

```text
A{0,N}
=
A{0,K-1} union C A{0,K-1} union ... union C^q A{0,r}.
```

Concatenating the same `P` and `S` on both sides preserves equality. Adding the
JSON quotes on both sides also preserves equality. Taking the union with the
unchanged anchored alternatives preserves equality of the complete pattern
language.

The implementation checks the interval cover using checked, constant-space
arithmetic before emitting a split.

## Sibling length constraints

The existing importer does not preserve every expensive `pattern`/`maxLength`
product. The splitting analysis is parameterized by the sibling bounds that the
importer would actually enforce.

It computes exact decoded-character lower and upper bounds over the HIR:

- empty and anchors have length zero;
- a Unicode class has length one;
- a UTF-8 literal contributes its Unicode scalar count;
- concatenation adds bounds;
- alternation takes the minimum lower and maximum upper bound;
- repetition multiplies finite bounds, with an unbounded nonempty repetition
  producing no finite upper bound.

A split is admitted only when the pattern language implies every effective
sibling bound. Thus bypassing the old terminal intersection cannot broaden or
narrow the importer's language.

## Grammar substitution theorem

Let the original importer output contain one terminal `T` for an eligible
pattern, with byte language `L`. Let the optimized importer replace that use by
a nonterminal `X` whose productions are the alternatives above. By the count
theorem and unchanged passthrough branches,

```text
L(X) = L(T) = L.
```

Replacing a grammar symbol by a nonterminal denoting exactly the same language
preserves every surrounding derivation. Hence the complete imported grammar
accepts exactly the same byte language before and after the optimization.

## Mask corollary

For a grammar language `L`, committed byte prefix `u`, vocabulary token bytes
`beta(v)`, and arbitrary continuation `w`, the next-token mask is

```text
M_L(u) = { v : there exists w with u beta(v) w in L }.
```

Because the imported byte language is unchanged, membership of every token in
every reachable mask is unchanged. EOS admissibility is likewise unchanged.

This is a relative pass proof: it proves that the importer transformation
preserves `GrammarDef` language. It assumes the pre-existing grammar, lexer and
runtime compiler correctly implement that unchanged grammar semantics.
