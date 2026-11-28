# GLRM

GLRM is GLRMask's own grammar format.

```glrm
start value
// Regular helper
internal t DIGIT = /[0-9]/
t INTEGER = DIGIT+
/* Recursive rule */
nt value = INTEGER | "(" value ")"
```

## Declarations

- `start value`: start rule (one per grammar).
- `t INTEGER = /[0-9]+/`: terminal, matched by the lexer.
- `internal t DIGIT = /[0-9]/`: regular helper for terminal definitions.
- `nt value = INTEGER`: rule, which can reference terminals and other rules.

Names use ASCII letters, digits and underscores, with a letter or underscore first. Use backticks for reserved names. Semicolons are optional; indentation and line breaks are insignificant.

## Literals and regular expressions

- `"text"` or `'text'`: UTF-8 text.
- `b"\xFF"`: raw bytes.
- `/[a-z]+/i`: Unicode regular expression.
- `b/[\x00-\x7F]+/`: byte regular expression.
- `eps`: empty string.

Strings support escapes such as `\n`, `\t`, `\r`, `\0`, `\\`, `\xHH` and, for text, `\uHHHH` and `\u{H...}`. In a text literal, `\xE9` encodes U+00E9 as UTF-8; in a byte literal it produces one byte. Write non-ASCII bytes with escapes.

Regular expressions support character classes, groups, alternation, greedy quantifiers and the flags `i`, `s`, `x` and `u`. Use `/./s` for any Unicode character or `b/./s` for any byte. Backreferences, lookaround, anchors and lazy quantifiers are unsupported. Escape a literal slash outside a character class as `\/`.

## Operators

- `a b`: sequence.
- `a | b`: alternatives.
- `(a | b) c`: grouping.
- `a?`, `a*`, `a+`: optional, zero or more, one or more.
- `a{3}`, `a{2,5}`, `a{2,}`: exactly 3, from 2 to 5, at least 2.
- `a & b`: intersection of regular languages.
- `a - b`: difference of regular languages. With a rule on the left, removes matching top-level alternatives.
- `"," ~ ("a"? "b"? "c"?)`: items in order, separated by commas if present.
- `"," ~+ ("a"? "b"?)`: same, requiring at least one item.

Repetition binds before sequence, sequence before intersection and difference, and alternation last. Parenthesize a compound subtraction operand. Use `eps` for an empty alternative.

## Terminal matching

Terminals take the longest match. For a shortest match, subtract every longer match:

```glrm
start response
internal t BODY = /.*END/s
t THROUGH_END = BODY - (BODY b/./s+)
nt response = THROUGH_END "!"
```

`THROUGH_END` stops at the first `END`. The byte regex allows any suffix, including invalid UTF-8.

## Externs and inline grammars

```glrm
grammar item {
    start value
    nt value = "item"
}
extern grammar PAYLOAD
extern token END_CALL
start response
nt response = item PAYLOAD item END_CALL
```

Bind `PAYLOAD` to a grammar and `END_CALL` to a token set in your code. Each occurrence of `END_CALL` matches one token from that set. Extern declarations belong at the top level. Inline grammars have their own scope and start rule. See [Composing grammars](../GUIDE.md#composing-grammars).

## Ignored terminals

```glrm
start root
t SPACE = /[ \t]+/
ignore SPACE
nt root = "hello" "world"
```

Ignored terminals can occur between lexical atoms and at grammar boundaries. They must consume input and cannot appear explicitly in rules. Each inline or bound grammar has its own ignored terminals.

## Finite automata

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

Declare one start state and any number of accepting states. Edge labels can be expressions; use `eps` for an empty edge.
