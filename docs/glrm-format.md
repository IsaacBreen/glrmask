# Native GLRM format

GLRM describes recognition components, not model-specific tokenizer configuration
or sampling policy. Source is UTF-8. The examples below use the canonical syntax;
no version header or semicolons are required. Semicolons may end declarations,
but line breaks and indentation have no semantic role.

## Declarations and names

```glrm
start value
internal t DIGIT = /[0-9]/
t INTEGER = DIGIT+
nt value = INTEGER | "(" value ")"
```

`start` chooses one explicit entry. `nt` creates context-free parser structure;
`t` creates an emitting lexical terminal. `internal t` defines a reusable regular
helper which cannot be used as an emitting parser symbol or an ignore target.
An ordinary rule whose language happens to be regular is not automatically a
single emitting terminal. Anonymous literals and regexes can occur in `nt` bodies.

Declaration introducers delimit complete expressions even on one line:

```glrm
start root nt root = "a" nt other = "b"
```

Use backticks when a name would otherwise be a reserved word:

```glrm
start `start`
nt `start` = "a" | "(" `start` ")"
```

Use `eps` for an explicit empty match. An unfinished alternative or declaration
body is an error, not implicit epsilon. Ordinary parser alternatives are unordered;
valid ambiguity is retained. Completing one derivation does not automatically
remove other accepting or continuing derivations.

## Regular expressions and byte strings

| Form | Meaning |
|---|---|
| `"text"` or `'text'` | Unicode text encoded as UTF-8, without normalization |
| `b"\xFF"` or `b'\xFF'` | Explicit byte string |
| `/pattern/` | Regular language over Unicode scalar values, encoded as UTF-8 |
| `b/pattern/` | Regular language over bytes |

String escapes include newline, tab, carriage return, quote, backslash, `\xHH`,
and Unicode scalar escapes such as `\u{1F600}`. In text strings, `\xE9` means the
scalar U+00E9 and therefore encodes to two UTF-8 bytes. In byte strings it means
one byte, E9. Unknown escapes, invalid scalar values and unpaired surrogates are
errors. Byte source forms reject raw non-ASCII characters; use byte escapes.

The checked regex dialect supports useful regular syntax and Unicode classes,
including `\p{L}`. Supported external flags are `i` (case insensitive), `s`
(dot includes newline), `x` (extended whitespace/comments), and explicit Unicode
`u` on a Unicode regex. Duplicated or unsupported flags fail. Inline flags cannot
change the byte/Unicode alphabet chosen by the source form. Unicode properties
and case folding use the tables in the locked `regex-syntax` dependency.

Regexes describe whole lexemes, not searches. `/./` excludes newline; `/./s`
matches one Unicode scalar and `b/./s` matches one byte. A scalar may be split
across model tokens. Malformed UTF-8 cannot match a Unicode scalar but can match
explicit byte expressions. Bare `.` and bare bracket classes are not native atoms.

Backreferences, lookaround, anchors, locale/line-anchor modes and internal
reluctant quantifiers (`*?`, `+?`, `??`, `{m,n}?`) are rejected. Regular-language
recognition has no implicit Python-style ordered-regex match-selection policy.

## Prefix-minimal terminal language

Use regular-language subtraction to retain only words with no strictly shorter
accepted prefix:

```glrm
internal t THROUGH_END_BODY = /.*END/s
t THROUGH_END = THROUGH_END_BODY - (THROUGH_END_BODY b/./s+)
nt response = THROUGH_END "!"
start response
```

For a regular byte language `L`, its prefix-minimal language is
`L - (L b/./s+)`. The suffix must match one or more arbitrary bytes, including
newline and malformed UTF-8; Unicode `/./s+` is not an equivalent substitute for
mixed byte/Unicode operands. Parenthesize alternatives and the right-hand
concatenation. For example, `(/a|ab/) - ((/a|ab/) b/./s+)` accepts `a`, not `ab`.
There is no native `shortest { ... }` expression form.

This subtraction acts on one regular terminal expression before unrelated
terminals are combined. It does not globally change lexer policy or commit a
context-free rule to its first completion. Operands may use regular terminal
helpers, literals, regexes and regular algebra, but not parser-rule references,
grammar calls or token-set events. If `L` accepts epsilon, its prefix-minimal
language is exactly epsilon; that remains invalid as an ignored terminal. Other
terminals retain maximal-munch semantics.

This is not a translation for an internal lazy quantifier. In an ordered regex
matcher, `a+?b.*` can retain a long final suffix; prefix-minimal acceptance of the
entire language is a different operation. Lark's LLM-profile `[lazy]` annotation
continues to provide that whole-terminal operation.

## Algebra and repetitions

Regular terminal expressions support concatenation, unordered alternatives,
`?`, `*`, `+`, `{n}`, `{m,n}` and `{m,}`. Ranges are validated, not wrapped on overflow.
`&` is regular-language intersection and `-` is regular-language subtraction in
regular contexts:

```glrm
t IDENTIFIER = /[a-z]+/ - /(if|else|while)/
start root
nt root = IDENTIFIER
```

The established **structural nonterminal subtraction** form is distinct:

```glrm
nt value = "(" value ")" | "x" | "y"
nt without_direct_x = value - "x"
start without_direct_x
```

It removes matching top-level alternatives under the source structural comparison;
it does not remove `x` from recursive uses of `value` or subtract arbitrary CFG
languages. Optimization must not change which subtraction contract applies.

Separator sequences use `separator ~ (...)` and `separator ~+ (...)`. Each item's
outer optional/repetition marker controls presence and separator placement. The
`~` form permits the empty case under the existing item-presence rules; `~+`
requires a present result. A required item deriving epsilon is still present and
can require a separator. An omitted optional item does not. Parenthesize inner
quantification separately from item quantification, for example:

```glrm
start row
nt row = "," ~ ("a"? "b"? "c"?)
```

This is not a homogeneous list macro. Preserve required/optional and
present-but-empty distinctions when writing or transforming a grammar.

## Isolated inline grammars and host binding

```glrm
grammar child {
    start value
    nt value = "item"
}
extern grammar PAYLOAD
start response
nt response = child PAYLOAD child
```

Inline grammars own their names, start and ignore settings. Parent declarations
do not enter a child's lexical scope.

`extern token MARK` consumes one original model-token ID selected by a host-bound
`TokenSet` value. `extern grammar PAYLOAD` invokes an independent
recognition fragment. Bind externs through a `Module`:

```python
parent = glrmask.Module(glrmask.Grammar.from_glrm(source))
parent.bind("PAYLOAD", glrmask.Grammar.from_ebnf('start ::= "data"'))
constraint = parent.compile(vocab)
```

Bindings accept `Grammar`, `Module`, `Expr`, `TokenSet`, or a single token ID, subject to
the extern's kind. A child must have all its own externs bound before attachment.
Each extern binds once; qualified descendant paths such as `child::MARK` are
unsupported. Attaching a module captures its bindings and settings, so later
changes to the original module do not affect that occurrence. `module.externs`
lists declared externs and `module.externs.unbound` lists those still unresolved.
See [the guide](../GUIDE.md) for the host lifecycle.

Numeric token literals (`<[123]>`), named special-token literals (`<|name|>`) and
inline JSON Schema (`%json`) are deliberately absent from GLRM. Bind exact events
or a separately constructed JSON Schema grammar from the host instead.

## Ignore policy

`ignore NAME` names a local, nonnullable emitting terminal. Multiple declarations
retain distinct emitting identities rather than replacing their languages by one
union terminal. That distinction matters under per-terminal maximal munch.

```glrm
t SPACE = /[ \t]+/
t COMMENT = /#[^\n]*/
ignore SPACE
ignore COMMENT
start root
nt root = "hello"
```

A parent's ignore does not act inside a child component. At a boundary, child
trailing ignore and a valid return to the parent's suffix remain alternatives;
ignore must not make an irreversible early choice that changes valid output.

## Finite-automaton bodies

```glrm
start framed
nt framed = fa {
    start q0
    accept q2
    edge q0 -> q1: "("
    edge q1 -> q1: "a" | "b"
    edge q1 -> q2: ")"
}
```

There is one start state. Multiple `accept` declarations are allowed. States may
be names or integers; `eps` is an epsilon edge. Explicit `edge` introducers allow
multiline labels without significant newlines or required semicolons. Labels in
terminal FAs must be regular expressions; parser FAs can use parser expressions.
A parser FA with nonterminal labels is not asserted to be a regular byte language.

## Optimization hints and diagnostics

`pragma namespace { ... }` blocks contain optimization hints, never semantic
features. An unknown namespace is skipped only as a lexically balanced block;
quoted braces and regex contents are not structural delimiters. Unknown directives
inside a recognized namespace fail, so misspelled supported hints are not ignored.
Removing or rearranging lexer-partition hints must not change valid token traces.

Source errors retain original UTF-8 byte spans and one-based line/column positions
where a precise location is available. Columns count Unicode scalars. Requirement
locations refer to the declaring component's document, not a flattened generated
grammar. Programmatically constructed nodes may have no source span.
