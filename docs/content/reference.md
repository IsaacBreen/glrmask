## Generating

Each sequence has its own `ConstraintState`. At each step, mask the model's scores, choose a token and commit it. Stop
when nothing more can be committed:

```python
import numpy as np
import glrmask

chars = b"0123456789+*()"
eos = len(chars)
vocab = glrmask.Vocab.from_id_to_bytes(
    {i: bytes([c]) for i, c in enumerate(chars)}, special_token_ids=[eos])

arithmetic = glrmask.Grammar.from_glrm(r'''
    start sum
    t NUMBER = /[0-9]+/
    nt sum = product | sum "+" product
    nt product = atom | product "*" atom
    nt atom = NUMBER | "(" sum ")"
''')
grammar = glrmask.Grammar.from_expr(glrmask.Expr.sequence(arithmetic, glrmask.TokenSet([eos])))
constraint = grammar.compile(vocab)

rng = np.random.default_rng(0)
state = constraint.start()
text = b""
while not state.is_terminated():
    logits = rng.standard_normal(eos + 1)  # Your model's scores go here.
    logits[~state.mask(size=len(logits))] = -np.inf
    token = int(logits.argmax())
    assert state.commit_token(token)
    text += chars[token:token + 1]
print(text.decode())
```

`mask()` returns a Boolean array indexed by token ID; `size` makes it as long as your logits. `fill_mask` is faster: it
writes a packed bitmask into a buffer you reuse: bit `id % 32` of word `id / 32` is set when token `id` is allowed.
`Constraint.mask_len()` gives the buffer's length in words. In Rust, Node.js, C and C++, masks are always packed.

`commit_token` returns whether the token was allowed. `commit_tokens` commits several in order, and `commit_bytes`
commits text, such as a prefix you supply. Text never matches a token set. A failed commit ends the state, so to try a
token without committing to it, copy the state first.

### Ending

`is_accepting()` says whether the text so far is a complete match. More may still be allowed: `1+2` is complete, but
`1+23` is too. `is_terminated()` says whether nothing more can be committed.

GLRMask has no end-token setting. To require an end-of-sequence token, put it after the grammar, as above. It's allowed
once the grammar is complete, and nothing is allowed after it. If the token has no bytes, add it as a special token:
`special_token_ids` in Python, `with_special_tokens` in Rust, `specialTokenIds` in Node.js.

### Forced tokens

`forced_tokens()` returns the tokens that must come next, without committing them, so you can commit them without
running the model. When the forced text could be tokenized in several ways, it takes the longest token at each step.
See [forced tokens](docs/forcing.md).

## Composing grammars

A grammar can declare externs: named slots that you bind later.

- `extern grammar NAME` takes a grammar, module, expression or token set.
- `extern token NAME` takes a token ID or token set, and matches exactly one of those tokens. It never matches their text.

```python
import glrmask

weather = glrmask.Grammar.from_json_schema(
    '{"type": "object", "properties": {"city": {"type": "string"}},'
    ' "required": ["city"], "additionalProperties": false}')
search = glrmask.Grammar.from_json_schema(
    '{"type": "object", "properties": {"query": {"type": "string"}},'
    ' "required": ["query"], "additionalProperties": false}')
call_format = glrmask.Grammar.from_glrm('''
    extern grammar WEATHER_ARGS
    extern grammar SEARCH_ARGS
    start call
    nt call = "<weather>" WEATHER_ARGS "</weather>" | "<search>" SEARCH_ARGS "</search>"
''')

call = call_format.copy()
call.bind(WEATHER_ARGS=weather, SEARCH_ARGS=search)

vocab = glrmask.Vocab.from_id_to_bytes({i: bytes([i]) for i in range(256)})
state = call.compile(vocab).start()
assert state.commit_bytes(b'<weather>{"city": "Perth"}</weather>')
assert state.is_accepting()
```

Each grammar keeps its own rules, including the whitespace it ignores, and one model token can span the boundary
between parts.

`bind` changes the grammar or module in place; copy it first to bind the same one in different ways. Each extern is
bound once, on the grammar or module that declares it. A value can't be bound while it has unbound externs of its own.
`externs` lists the externs in declaration order, and `externs.unbound` the ones not yet bound (in Rust, `externs()`
and `externs().unbound()`).

### Modules

A `Module` holds a grammar, its bindings and any precompiled work. `precompile(vocab)` compiles as much as it can now,
even with externs unbound, and `compile` reuses that work. Use modules when the same parts appear in many grammars. Tool
calls are the typical case: precompile each tool's schema once, and any combination of tools then compiles quickly.

```python
weather_args = glrmask.Module(weather)
weather_args.precompile(vocab)
search_args = glrmask.Module(search)
search_args.precompile(vocab)

call = glrmask.Module(call_format)
call.precompile(vocab)
call.bind(WEATHER_ARGS=weather_args, SEARCH_ARGS=search_args)
constraint = call.compile(vocab)
```

`grammar.precompile(vocab)` is short for creating a module from the grammar and precompiling it.

`module.save()` returns bytes, and `Module.load(data)` restores the module with its bindings and precompiled work. A
module can be saved with externs unbound. Precompiled work is for one vocabulary; compiling it with another is an error.

`constraint.save()` and `Constraint.load(data, vocab)` save and load a compiled constraint. The vocabulary isn't saved,
and `load` checks that you pass the one it was compiled for. Bytes saved by one GLRMask version may not load in another.

### Expressions

`Expr.choice(*parts)` matches any one of its parts, `Expr.sequence(*parts)` matches them one after another, and
`Expr.literal(text)` matches exactly `text`, with no whitespace skipped. Parts can be grammars, modules, expressions and
token sets; in Rust and Node.js, token IDs too. `Grammar.from_expr` makes a grammar from an expression, and an
expression can be bound to a grammar extern directly.

## Token sets

A `TokenSet` is a set of token IDs. In a grammar it matches any one of its tokens. Create one from IDs,
`TokenSet([1, 2, 3])`, or from ranges, `TokenSet.from_ranges([range(0, 10)])`. Ranges exclude their end in Python and
Node.js, and include it in C and C++; Rust takes `0..10` or `0..=9`. Every ID must be in the vocabulary when you compile.

Tokens with no bytes, such as end-of-sequence or tool-call tokens, are special tokens. Add them to the vocabulary so
that token sets can use them; they never match text. Two tokens with the same bytes are still different tokens to a
token set. See [token sets](docs/token-sets.md).

## Mask strategies

`MaskStrategy` chooses how masks are computed:

- `ParserDwa` builds a parser DWA for the grammar. Slowest to compile, with the fastest masks.
- `TokenTrie` computes each mask by walking the vocabulary. Fastest to compile, with slower masks. It is heavily based on [llguidance](https://github.com/guidance-ai/llguidance).
- `Auto`, the default, reuses any precompiled work; otherwise it uses `ParserDwa`, or `TokenTrie` when that would be too large.

`FAST_MASKS` and `FAST_COMPILE` are aliases for `ParserDwa` and `TokenTrie`.

`TokenTrie` has two settings:

- `parser_backend` chooses how the parser is represented: `LrTable`, the default, or `TemplateDfa`, which is slower to compile, with faster masks.
- `merge_equivalent_tokens` merges tokens that the grammar treats the same, so masks walk fewer tokens. It is off by default. Tokens are not merged in a composition that includes precompiled parts.

```python
constraint = grammar.compile(vocab, mask_strategy=glrmask.TokenTrie(
    parser_backend=glrmask.ParserBackend.TEMPLATE_DFA, merge_equivalent_tokens=True))
```

`BoundaryMaskStrategy` chooses how masks are computed at grammar boundaries, where a grammar meets a bound extern:

- `ParserDwa` builds a parser DWA across each boundary. Slower to compile, with faster masks.
- `TokenTrie` walks the vocabulary at each boundary as needed. Faster to compile, with slower masks.
- `Auto`, the default, uses `ParserDwa`, or `TokenTrie` when that would be too large.

Every strategy produces the same masks; they differ only in speed. A strategy you choose explicitly reports an error if
it can't be used, for example because its automaton would be too large. Only `Auto` falls back.

Pass strategies to `compile` and `precompile` (in Python, as keyword arguments). They apply to the whole compile,
including bound grammars. Precompiled work is never thrown away. Each precompiled part keeps its masks unless you
request a faster strategy for it, and only the other parts are compiled now. Parts compiled now that link with
precompiled parts use `TemplateDfa`.

## Markers

`@mark(id)` in a GLRM rule marks a point in the grammar. After each commit, `state.marker_events()` lists the markers
the commit passed, with their byte offsets. `constraint.marker_ids()` lists the grammar's marker IDs. A marker means
some parse reached that point; that parse may still fail later. Markers never change what's allowed. See
[markers](docs/markers.md).

## Errors

Creating, binding, compiling and loading grammars can fail; generating can't, because commits return `false` instead.
Each error message says what went wrong, with the line and column in the grammar source when that's the cause.
Python raises `ValueError`, Node.js throws a `GlrMaskError`, C returns a status and sets `glrmask_error_message()`, and
C++ throws `glrmask::Error`. In Rust, `Error::kind()` gives the kind of error and `source_span()` its place in the source.

## Grammar formats

**JSON Schema.** Drafts 4, 6, 7, 2019-09 and 2020-12, chosen by `$schema`; without one, 2020-12. `format` is treated as
an annotation, and `oneOf` requires exactly one match. A schema feature GLRMask can't enforce is an error. The JSON
has one layout, the one Python's `json.dumps` writes by default: no whitespace outside strings except a space after each
`,` and `:`, as in `{"a": [1, 2]}`.

**GLRM** is GLRMask's own format. It has terminals (`t`) and nonterminals (`nt`), externs, markers, scoped grammars, and
regular-expression intersection and subtraction. See [GLRM](docs/glrm-format.md).

```glrm
start value
t INTEGER = /-?(0|[1-9][0-9]*)/
nt value = INTEGER | "null"
```

**Lark**, with GLRMask's additions: `<|name|>` for a token extern, `@name` for a grammar extern, `%json` for an inline
JSON Schema and `[lazy]` for a terminal that ends as soon as it can. See [Lark](docs/lark-dialect.md).

Features GLRMask doesn't support, such as Lark priorities, are errors. They are never silently ignored.
