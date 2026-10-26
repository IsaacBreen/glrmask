# Lark

`Grammar.from_lark` reads [Lark](https://github.com/lark-parser/lark) grammars, with a few additions for language
models. A standard Lark grammar means the same with or without them: each addition is syntax that standard Lark rejects.

```python
import glrmask

grammar = glrmask.Grammar.from_lark('''
%import common (INT, WS)
%ignore WS
start: "[" INT ("," INT)* "]"
''')
```

## Standard Lark

Supported:

- Rules, terminals, literals, regular expressions, grouping and repetition. `x~3` repeats exactly three times, and
  `x~2..5` from two to five.
- Case-insensitive literals such as `"yes"i`, the regular-expression flags `i`, `s`, `x` and `u`, and Unicode literal
  ranges such as `"α".."ω"`.
- `%import common.INT`, `%import common.INT -> NUMBER` and `%import common (INT, WS)`, from a built-in copy of Lark
  1.3.1's `common` library: numbers, names, whitespace and comments. Imports never read files or the network.
- `%ignore`.

The grammar matches the rule named `start`. `?rule`, `!rule` and `-> alias` only shape Lark's parse trees, which GLRMask
doesn't build, so they have no effect.

Not supported, and reported as errors: priorities, templates, `%declare`, `%extend`, `%override`, imports from files,
other directives, and bare `.` (write `/./`, or `/./s` to include newline). Regular expressions have the same limits as
in [GLRM](glrm-format.md#strings-and-regular-expressions).

Each terminal takes its longest match. Python's `re` and Lark's lexers can choose differently, so the same grammar may
split text differently in Lark.

## Additions

| Syntax | Meaning |
|---|---|
| `@name` | A grammar extern |
| `<\|name\|>` | A token extern |
| `<[5]>`, `<[1,3-8]>` | Token 5; tokens 1 and 3 to 8 |
| `<[^1,3-8]>`, `<[*]>` | Any token except 1 and 3 to 8; any token |
| `%json { ... }` | A JSON Schema, inline |
| `name[lazy]: ...` | A terminal that ends as soon as it can |

### Externs

```python
call = glrmask.Grammar.from_lark('start: <|begin_call|> "<" @payload ">"')
call.bind("begin_call", begin_call_id)  # your tokenizer's ID for <|begin_call|>
call.bind("payload", glrmask.Grammar.from_json_schema({"type": "null"}))
```

`@payload` is an extern even if the grammar also has a rule called `payload`. Externs bind the same way as in any other
grammar; see [Composing grammars](../GUIDE.md#composing-grammars). GLRMask never guesses a token ID from a name such as
`<|begin_call|>`: you bind it.

### Token IDs

`<[...]>` matches one token by ID. Ranges include both ends and are ranges of IDs, not of characters. `^` and `*` are
relative to the vocabulary's tokens with bytes: they never match a special token. Every ID must be a token with bytes
in the vocabulary, so none can be a special token; to allow one, declare it with `<|name|>` and bind it.
Token IDs can't appear inside a terminal or a `[lazy]` body. See [token sets](token-sets.md).

### Inline JSON Schema

```lark
start: "<" %json {"type": "string", "const": "a b"} ">"
```

The schema is compiled as its own grammar, with its own whitespace rules: `%ignore` doesn't apply inside it.

### `[lazy]`

```lark
start: text "!"
text[lazy]: /.*END/s
```

A `[lazy]` terminal stops at its shortest match: here, the first `END`. Its body must be regular. Every other terminal
still takes the longest match. Lazy quantifiers such as `/a+?/` are errors, not a way to get this; they mean something
else.
