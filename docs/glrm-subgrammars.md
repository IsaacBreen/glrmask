# GLRM subgrammars and ignore semantics

GLRM supports named, closed subgrammars:

```glrm
start document;
ignore WS;
t WS ::= " "+;

g json ::= {
    start value;
    ignore JSON_WS;
    t JSON_WS ::= [ \n\r\t]+;

    nt value ::= object | array | STRING;
    nt object ::= "{" members? "}";
    nt members ::= "," ~+ ( member+ );
    nt member ::= STRING ":" value;
    nt array ::= "[" "," ~ ( value* ) "]";
    t STRING ::= /"([^"\\]|\\.)*"/;
};

nt document ::= "BEGIN" json "END";
```

`g json ::= { ... };` is analogous to a named nonterminal declaration: `json` can be referenced in expressions in the enclosing grammar. `subgrammar json ::= { ... };` is accepted as a verbose alias, but `g` is the canonical GLRM spelling.

## Closed scopes

Every subgrammar body is a complete grammar scope and must declare its own `start` rule.

Definitions do not cross scope boundaries:

- A subgrammar cannot reference terminals, nonterminals, or sibling subgrammars from its parent scope.
- The parent cannot reference private definitions inside the subgrammar.
- Only the declared subgrammar name is exported to the enclosing scope.
- The same local definition names may be reused in different scopes.
- Nested subgrammars follow the same rules.

This is deliberately lexical rather than import-like. Copying a standalone grammar into a `g name ::= { ... };` block does not silently bind any names from its new surroundings.

## Ignore semantics

For a grammar scope with ignore terminal `I`, ignore behaves as implicit `I*`:

1. before the first lexical atom in the scope,
2. between lexical atoms in the scope,
3. after the final lexical atom before leaving the scope.

Ignore is not inserted inside a terminal match. For example:

```glrm
start value;
ignore WS;
t WS ::= " "+;
t WORD ::= "ab";
nt value ::= WORD;
```

accepts `"  ab  "` but not `"a b"`.

The ignore terminal must be a local, emitting, non-nullable terminal. It is implicit and may not also be referenced explicitly from parser rules.

## Ignore and subgrammar boundaries

Ignore never inherits into a subgrammar.

Suppose an outer grammar ignores `WS`, and a referenced subgrammar ignores `NL`. At entry, the language around the boundary is:

```text
outer WS* ; inner NL* ; first inner lexical atom
```

At exit it is:

```text
last inner lexical atom ; inner NL* ; outer WS*
```

So `WS` is not legal between two inner lexical atoms, and `NL` is not legal between two outer lexical atoms. The scopes switch at the subgrammar boundary.

A subgrammar with no `ignore` declaration has no local ignore at all. The enclosing grammar may still consume its own ignore before entering or after returning from the subgrammar, because those bytes belong to the enclosing grammar's side of the boundary. That is not inheritance.

The tokenizer may cross these boundaries inside a single vocabulary token. The parser-side lowering preserves the same semantics for a token containing, for example, outer ignore bytes followed by inner ignore bytes and then the first inner terminal.

## Implementation

The current optimized runtime has a single global `ignore_terminal` and special-cases it in commit, masking, dynamic masking, parser templates, follow handling, L1/L2P construction, terminal interchangeability, NWA construction, and serialization. Making that field dynamically scoped would be invasive and easy to get wrong at ambiguous GLR boundaries.

GLRM subgrammars are therefore flattened before ordinary AST lowering:

- every grammar scope is alpha-renamed into a private namespace,
- references are validated and rewritten only against definitions from that scope,
- each subgrammar name becomes a nonterminal alias to the child's private entry rule,
- lexer partition labels and catch-all `*` assignments are localized to their grammar scope,
- a scope-local ignore is lowered to an ordinary parser-visible skip language.

For a local lexer catch-all, anonymous lexical atoms are promoted to private named terminals and the catch-all is materialized as explicit assignments before flattening. This prevents an outer `*` from accidentally partitioning a child's terminals. If equal terminal languages from independent scopes are deduplicated by ordinary terminal lowering, the corresponding partition groups are coalesced because one deduplicated terminal cannot belong to two groups.

For ignore `I`, one nullable skip rule is generated:

```glrm
nt __skip ::= eps | __skip I;
```

Each distinct lexical atom in the scope gets a reusable non-null wrapper:

```glrm
nt __ignored_A ::= __skip A;
nt __ignored_lbrace ::= __skip "{";
```

Parser rules use those wrappers. The private scope entry receives one trailing `__skip`. This avoids placing a nullable `skip` nonterminal between every symbol, which causes null-production power-set expansion and severe GLR table growth.

The core compiler and runtime consequently see one ordinary flat grammar with `ignore_terminal = None` whenever subgrammar flattening is used.

## Nullable roots

Grammar preparation records whether the start language admits epsilon after nullable terminals have been expanded, but before parser epsilon productions are eliminated. Runtime completion uses that semantic bit only when the parser GSS is exactly the untouched initial stack.

Consequently, `nt empty ::= eps;` is complete before any commit, legacy global ignore accepts ignore-only input around a nullable root, and a nullable subgrammar with a local ignore accepts both empty and local-ignore-only input. A partial nonempty branch does not inherit this completion: for `eps | "a" "b"`, the state after only `"a"` is not complete and EOS is not admitted.
