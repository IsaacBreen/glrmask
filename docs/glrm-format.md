# GLRM

GLRM is GLRMask's own grammar format. Source is UTF-8. Line breaks, indentation and semicolons don't matter, and no
version header is needed.

```glrm
start value
internal t DIGIT ::= /[0-9]/
t INTEGER ::= DIGIT+
nt value ::= INTEGER | "(" value ")"
```

## Declarations

| Example | Declares |
|---|---|
| `start value` | The rule the grammar matches. Every grammar has exactly one. |
| `t INTEGER ::= /[0-9]+/` | A terminal: a regular language the lexer matches as one unit, taking the longest match. |
| `internal t DIGIT ::= /[0-9]/` | A regular helper for other terminals. Rules and `ignore` can't use it. |
| `nt value ::= INTEGER \| "(" value ")"` | A rule. Its body can use terminals, rules, literals and regular expressions. |
| `ignore SPACE` | A terminal skipped wherever it can appear between others. |
| `extern token END_CALL` | A token set, bound from your code. |
| `extern grammar PAYLOAD` | A grammar, bound from your code. |
| `grammar item { start v nt v ::= "item" }` | An inline grammar, with its own names, start rule and ignored terminals. |

Every definition uses `::=`.

A rule whose language happens to be regular is still a rule, not a terminal.

Names match `[A-Za-z_][A-Za-z0-9_]*`. Put a keyword in backticks to use it as a name: ``nt `start` ::= "a"``. `use`,
`export`, `import` and `super` are reserved.

Comments are `// ...` and `/* ... */`.

## Strings and regular expressions

| Example | Matches |
|---|---|
| `"text"` or `'text'` | The text, encoded as UTF-8 |
| `b"\xFF"` or `b'\xFF'` | The bytes |
| `/[a-z]+/i` | A regular expression over Unicode characters, encoded as UTF-8 |
| `b/[\x00-\x7F]+/` | A regular expression over bytes |
| `eps` | The empty string |

Escapes are `\n`, `\t`, `\r`, `\0`, `\"`, `\'`, `\\`, `\xHH`, `\uHHHH` and `\u{H...}`. In text, `\xE9` is the
character U+00E9; in a byte form, it's the byte E9. Byte forms can't contain non-ASCII characters except as escapes.

Regular expressions are matched against the whole terminal, not searched for. They support classes such as `[a-z]`,
`\d` and `\p{L}`, groups, alternation, greedy quantifiers and the flags `i`, `s`, `x` and `u`. `/./` excludes newline,
`/./s` matches any character and `b/./s` any byte. Backreferences, lookaround, anchors and lazy quantifiers are errors.
Classes and `.` only work inside `/.../`.

## Operators

| Example | Matches |
|---|---|
| `a b` | `a` then `b` |
| `a \| b` | `a` or `b` |
| `(a \| b) c` | Grouping |
| `a?`, `a*`, `a+` | Optional, zero or more, one or more |
| `a{3}`, `a{2,5}`, `a{2,}` | Exactly 3, from 2 to 5, at least 2 |
| `/[a-z]+/ & /.{1,8}/` | Both |
| `/[a-z]+/ - ("if" \| "else")` | The first but not the second |
| `"," ~ ("a"? "b"? "c"?)` | `a,b,c`, `a,c`, `b` or the empty string: the items in order, with `,` between those present |
| `"," ~+ ("a"? "b"?)` | The same, with at least one item |

`&` and `-` work on regular languages: terminals, helpers, literals and regular expressions. On a rule, `-` removes
matching top-level alternatives instead: `nt without_x ::= value - "x"` drops the alternative `"x"` of `value`, but `x`
can still appear inside a nested `value`.

An empty alternative or body is an error; write `eps` where you mean it.

When several parses match, GLRMask keeps them all. A complete parse doesn't rule out others that continue.

## The shortest match

A terminal takes the longest match. To make one end as soon as it can, subtract every longer match:

```glrm
start response
internal t BODY ::= /.*END/s
t THROUGH_END ::= BODY - (BODY b/./s+)
nt response ::= THROUGH_END "!"
```

`THROUGH_END` stops at the first `END`. Use `b/./s+`, not `/./s+`, so the suffix can be any bytes. Lark's `[lazy]` does
the same.

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

Externs are declared at the top level and bound from your code; see [Composing grammars](../GUIDE.md#composing-grammars).
An inline grammar can't refer to names outside it. GLRM has no syntax for token IDs: bind a token set to an
`extern token` instead.

## Ignored terminals

```glrm
start root
t SPACE ::= /[ \t]+/
ignore SPACE
nt root ::= "hello" "world"
```

An ignored terminal can't match the empty string or be used in a rule. A grammar's ignored terminals don't apply inside
a bound or inline grammar.

## Finite automata

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

One `start` state, any number of `accept` states. States are names or numbers; an `eps` label is an empty edge. In a
terminal, labels are regular expressions; in a rule, they can use rules too.

## Hints

`pragma glrmask { ... }` holds hints that can make compiling faster without changing what the grammar matches. Other
namespaces are skipped; an unknown hint inside `pragma glrmask` is an error.

## Errors

Errors give the line and column, counting columns in characters.
