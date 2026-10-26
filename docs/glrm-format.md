# GLRM

GLRM is GLRMask's own grammar format. Source is UTF-8. Line breaks and indentation don't matter, and semicolons after
declarations are optional.

```glrm
start value
internal t DIGIT ::= /[0-9]/
t INTEGER ::= DIGIT+
nt value ::= INTEGER | "(" value ")"
```

## Declarations

- `start NAME` names the rule the grammar matches. Every grammar has exactly one.
- `t NAME ::= ...` declares a terminal: a regular language the lexer matches as one unit, taking the longest match.
- `internal t NAME ::= ...` declares a regular helper for use inside other terminals. It isn't a terminal itself, so rules
  and `ignore` can't use it.
- `nt NAME ::= ...` declares a rule. Rule bodies can use terminals, other rules, and literals and regular expressions
  directly.
- `ignore NAME` skips a terminal wherever it can appear between others.
- `extern token NAME` and `extern grammar NAME` declare externs, bound later from your code.
- `grammar NAME { ... }` declares an inline grammar.

A rule whose language happens to be regular is still a rule, not a terminal.

Each declaration keyword starts a new declaration, so several fit on one line: `start root nt root ::= "a" nt other ::= "b"`.
Put a name in backticks to use a keyword as a name: ``nt `start` ::= "a"``.

`eps` matches the empty string. An empty alternative or body is an error, so write `eps` where you mean it.

When several parses match, GLRMask keeps them all. A complete parse doesn't rule out others that continue.

## Strings and regular expressions

| Form | Matches |
|---|---|
| `"text"` or `'text'` | The text, encoded as UTF-8 |
| `b"\xFF"` or `b'\xFF'` | The bytes |
| `/pattern/` | A regular expression over Unicode characters, encoded as UTF-8 |
| `b/pattern/` | A regular expression over bytes |

Escapes are `\n`, `\t`, `\r`, `\"`, `\'`, `\\`, `\xHH` and `\u{HHHH}`. In text, `\xE9` is the character U+00E9, two
bytes in UTF-8; in a byte string or byte expression, it's the single byte E9. Byte forms can't contain non-ASCII
characters except as escapes. Unknown escapes and invalid characters are errors.

Regular expressions support the usual syntax, including Unicode classes such as `\p{L}`, and the flags `i` (ignore
case), `s` (`.` matches newline), `x` (whitespace and comments allowed) and `u`. A regular expression describes a whole
terminal: it is matched, not searched for. `/./` excludes newline; `/./s` matches any one character and `b/./s` any one
byte. A model token can end in the middle of a character.

Backreferences, lookaround, anchors and lazy quantifiers such as `*?` are errors.

## Operators

| Syntax | Meaning |
|---|---|
| `a b` | `a` followed by `b` |
| `a \| b` | `a` or `b` |
| `a?`, `a*`, `a+` | Optional, zero or more, one or more |
| `a{n}`, `a{m,n}`, `a{m,}` | Exactly `n`, from `m` to `n`, at least `m` |
| `a & b` | Both: regular intersection |
| `a - b` | `a` but not `b`: regular subtraction |
| `sep ~ (items)` | Items separated by `sep` |
| `sep ~+ (items)` | The same, with at least one item |

`&` and `-` work on regular languages: terminals, helpers, literals and regular expressions, but not rules or externs.

```glrm
start root
t IDENTIFIER ::= /[a-z]+/ - /(if|else|while)/
nt root ::= IDENTIFIER
```

On a rule, `-` removes matching top-level alternatives instead: in `nt without_x ::= value - "x"`, the alternative `"x"` of
`value` is removed, but `x` can still appear inside a nested `value`.

In `sep ~ (...)`, each item's own `?`, `*` or `+` says whether it must appear, and a separator goes between items that
do. `"," ~ ("a"? "b"? "c"?)` matches `a,b,c`, `a,c`, `b` and the empty string.

### The shortest match

To make a terminal end as soon as it can, subtract every longer match: for a regular language `L`, use
`L - (L b/./s+)`.

```glrm
start response
internal t THROUGH_END_BODY ::= /.*END/s
t THROUGH_END ::= THROUGH_END_BODY - (THROUGH_END_BODY b/./s+)
nt response ::= THROUGH_END "!"
```

`THROUGH_END` stops at the first `END`. Use `b/./s+`, not `/./s+`, so that the suffix can be any bytes. This changes only
that terminal; every other terminal still takes the longest match. Lark's `[lazy]` does the same.

## Externs and inline grammars

```glrm
grammar item {
    start value
    nt value ::= "item"
}
extern grammar PAYLOAD
extern token END_CALL
start response
nt response ::= item PAYLOAD item END_CALL
```

An inline grammar has its own names, start rule and ignored terminals, and can't refer to names outside it. Externs are
bound from your code; see [Composing grammars](../GUIDE.md#composing-grammars). `extern` is allowed only at the top
level.

GLRM has no syntax for token IDs or inline JSON Schema. Bind a token set or a JSON Schema grammar to an extern instead.

## Ignored terminals

```glrm
start root
t SPACE ::= /[ \t]+/
t COMMENT ::= /#[^\n]*/
ignore SPACE
ignore COMMENT
nt root ::= "hello" "world"
```

An ignored terminal can't match the empty string. Each `ignore` stays a separate terminal, which matters because each
takes its longest match on its own. A grammar's ignored terminals don't apply inside a bound or inline grammar.

## Finite automata

A rule or terminal can be written as a finite automaton:

```glrm
start framed
nt framed ::= fa {
    start q0
    accept q2
    edge q0 -> q1: "("
    edge q1 -> q1: "a" | "b"
    edge q1 -> q2: ")"
}
```

There is one `start` state and any number of `accept` states. States are names or numbers, and an `eps` label is an
empty edge. In a terminal, labels are regular expressions; in a rule, they can use rules too.

## Hints

`pragma glrmask { ... }` holds hints that can make compiling faster. They never change what the grammar matches. Other
namespaces, such as `pragma other { ... }`, are skipped, but an unknown hint inside `pragma glrmask` is an error.

## Errors

Errors give the line and column in the source, counting columns in characters. `use`, `export`, `import` and `super` are
reserved.
