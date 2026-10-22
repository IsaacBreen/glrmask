# Lark recognition and LLM compatibility profiles

The Lark frontend lowers into GLRMask's shared component, lexical and exact-event
representation. It does not have a separate masking engine or translate through
GLRM text. It is a documented recognition dialect, not complete emulation of
Python Lark, every lexer mode, or every llguidance/XGrammar extension.

## Profiles

The public default is `recognition`:

```python
import glrmask

grammar = glrmask.Grammar.from_lark('''
%import common (INT, WS)
%ignore WS
start: "[" INT ("," INT)* "]"
''')
```

Enable selected LLM extensions explicitly with `profile="llm"`. In Rust, use
`Grammar::from_lark_with(source, LarkOptions::llm())`; a caller-supplied named-token
map can be added with `LarkOptions::token(name, ids)`.

| Feature | Recognition | LLM profile |
|---|---|---|
| Rules, terminals, literals, grouping, regular expressions, repetition | Yes | Yes |
| Curated `%import common`, aliases and grouped imports | Yes | Yes |
| Local `%ignore` with separate ignored-terminal identities | Yes | Yes |
| Regular-body `[lazy]` | No | Yes |
| Exact ID `<[123]>`, sets/ranges `<[1,3-8]>` | No | Yes |
| Complement `<[^1,3-8]>`, wildcard `<[*]>` | No | Yes |
| Explicitly resolved named special token `<|name|>` | No | Yes |
| Host-bound grammar `@name` | No | Yes |
| Isolated inline JSON Schema `%json { ... }` | No | Yes |

## Ordinary Lark support and explicit limits

Rules have an explicit default entry named `start`; no first-declaration fallback
is used. Uppercase terminal definitions remain lexical definitions; helper-only
terminals are not accidentally emitted just because they appear in an import.
Parse-tree-only `?rule`, `!rule`, and `-> alias` syntax may be accepted and ignored,
because this API returns masks and acceptance, not syntax trees or captures.

`x~N` means exactly N repetitions. `x~M..N` uses inclusive bounds and rejects an
inverted or overflowing bound. Unicode literal ranges such as `"α".."ω"` require
single scalar endpoints, not one-byte endpoints. Case-insensitive `"yes"i` and
checked regex flags `i`, `s`, `x`, `u` use the shared regular frontend. Regex syntax
is not silently converted to a literal when parsing fails.

`%import common.INT`, `%import common.INT -> NUMBER`, and
`%import common (INT, WS)` resolve from a pinned built-in set based on Lark 1.3.1.
They do not read arbitrary files or access the network. Numeric, name, whitespace
and comment terminals are provided; unsupported names and import paths fail.
Backtracking/reluctant source patterns in the built-in library use explicitly
implemented regular-language equivalents rather than a Python regex runtime.

Unknown directives, `%declare`, `%extend`, `%override`, general file imports and
templates are not silently skipped. Rule and terminal priorities are unsupported,
including explicit zero priorities. Bare `.` is not a grammar atom; write `/./`
or `/./s` for a Unicode scalar with the intended newline behavior.

GLRMask preserves its per-terminal maximal-munch lexical contract. Python regex
ordered alternatives and different Lark lexer configurations can select different
endpoints; identical source notation is not a promise of identical segmentation.

## Whole-terminal shortest matching

Select `profile="llm"` to use the nonstandard `[lazy]` annotation:

```lark
start: text "!"
text[lazy]: /.*END/s
```

The body must be representable as a regular terminal expression. `[lazy]` means
prefix-minimal acceptance of the whole expression: it retains words with no
strictly shorter accepted prefix. Native GLRM expresses the same regular byte
language as `L - (L b/./s+)`, with parentheses around compound operands; it has no
`shortest { ... }` expression form. `[lazy]` does not apply to arbitrary CFG rules,
grammar calls, exact events, or a regularization that erases implicit ignore
boundaries.

Internal reluctant quantifiers such as `/a+?b.*/` are rejected, not silently treated
as ordinary quantifiers or translated into a different whole-terminal language.
Anchors, lookaround, backreferences and unsupported flags also fail explicitly.

## Original model-token events

```python
vocab = glrmask.Vocab.from_id_to_bytes({0: b"x", 1: b"x", 4: b"", 5: b""})
grammar = glrmask.Grammar.from_lark(
    'start: <|begin|> <[^0-1]>',
    profile="llm",
    token_names={"begin": 1},
)
constraint = grammar.compile(vocab)
state = constraint.start()
assert not state.mask()[0] and state.mask()[1]
state.commit_token(1)
assert state.mask()[4] and state.mask()[5]
```

Each event atom consumes one original token ID. Ranges are inclusive numeric ID
ranges, never spelling ranges. Wildcards and complements are relative to the
vocabulary's finite grammar-consumable event domain, including spelling-less IDs,
not just IDs with nonempty text. Unknown positive IDs and ranges crossing undefined
holes are errors. An empty resulting set is the empty language, not epsilon.
The final root's separate end-token policy can reserve IDs from ordinary content.

Named specials use only the caller's explicit map; no model-name heuristics,
metadata guessing, decoded-spelling fallback or arbitrary alias choice occurs.
Different frameworks can use the same notation but different name-resolution rules.

Exact events are parser atoms, not regex characters. They cannot occur inside a
byte terminal, ignored expression or `[lazy]` operand. Equal spellings never erase
ID identity; an ID can retain both ordinary-byte and exact-event interpretations
when the grammar permits both.

## Grammar and schema composition

```python
grammar = glrmask.Grammar.from_lark(
    'start: "<" @payload ">"', profile="llm",
)
child = glrmask.Grammar.from_json_schema({"type": "null"})
composed = glrmask.Module(grammar)
composed.bind("payload", child)
```

`@payload` declares a grammar-valued host requirement. A local rule named
`payload` is a different symbol, so the two can coexist without an accidental
collision. Bind it through a `Module`, as with a GLRM extern. A child must be
fully bound before attachment, and qualified descendant binding is unsupported.

Inline JSON Schema is an isolated child, not production text pasted into the
parent's lexer scope:

```lark
start: "<" %json {"type":"string","const":"a b"} ">"
```

It uses the existing JSON Schema compiler, its supported schema domain and its own
whitespace policy. Parent `%ignore` does not change JSON string contents. Tokens may
cross the parent/child boundary mid-spelling under the usual composition semantics.
Native GLRM intentionally has no `%json`, numeric-event or named-special literals.

## Deliberately excluded features

Neither profile supports capture annotations, parse-tree production, substring
matching/generation, temperature, token or character/byte budgets, priorities, or
special stop/suffix backtracking and capture-hiding controls. Unsupported semantic
attributes are errors. A selected compatibility profile is not blanket permission
to accept and ignore the rest of a foreign grammar dialect.

