## Generating

Each sequence has its own `ConstraintState`. At each step, mask the model's scores, allow EOS when the state is accepting, and choose a token. Stop if EOS is chosen; otherwise commit the token:

```python
import numpy as np
import glrmask

chars = b"0123456789+*()"
eos = len(chars)
vocab = glrmask.Vocab({i: bytes([c]) for i, c in enumerate(chars)})

arithmetic = glrmask.Grammar.from_glrm(r'''
    start sum
    t NUMBER = /[0-9]+/
    nt sum = product | sum "+" product
    nt product = atom | product "*" atom
    nt atom = NUMBER | "(" sum ")"
''')
constraint = arithmetic.compile(vocab)

rng = np.random.default_rng(0)
state = constraint.start()
text = b""
while True:
    logits = rng.standard_normal(eos + 1)  # Your model's scores go here.
    allowed = np.zeros(len(logits), dtype=bool)
    allowed[:state.mask_bit_len()] = state.mask(dtype=bool)
    allowed[eos] = state.is_accepting()
    assert allowed.any()
    logits[~allowed] = -np.inf
    token = int(logits.argmax())
    if token == eos:
        break
    assert state.commit_token(token)
    text += chars[token:token + 1]
print(text.decode())
```

Packed masks use 32-bit words: bit `id % 32` of word `id // 32` is set when token `id` is allowed.
`mask_bit_len()` returns the number of token slots (largest domain ID plus one, or zero if empty), including gaps.
Allocate `(mask_bit_len() + 31) // 32` words for `fill_mask`, which writes into reusable storage and zeros surplus words.
Python `mask(*, dtype=np.uint32)` defaults to packed uint32; `mask(dtype=bool)` returns exactly
one Boolean per token slot. Use `mask().view(np.int32)` for signed packed words without copying.
To pad to wider logits, allocate a zeroed Boolean array and copy the logical mask into its prefix as above.
For a wider packed mask, allocate the required words and use `fill_mask`. It accepts writable contiguous
1-D NumPy int32 or uint32 arrays (not bool); surplus words are zeroed.

For PyTorch serving with V equal to the logits width (at least every state mask bit length):

```python
# CPU staging buffer, with each row reused for one sequence.
bitmask = torch.empty((len(states), (V + 31) // 32), dtype=torch.int32, pin_memory=True)
for i, state in enumerate(states):
    state.fill_mask(bitmask[i].numpy())  # zero-copy NumPy view of CPU row
mask_on_device = bitmask.to(device, non_blocking=True)
# Apply mask_on_device with your packed-mask kernel. Do not reuse the CPU
# buffer until the asynchronous transfer completes.
```

`commit_token` returns whether the token was allowed. `commit_tokens` commits several in order and returns the number committed. `commit_bytes`
commits text, such as a prefix you supply. Text never matches a token set. A failed commit ends the state, so to try a
token without committing to it, copy the state first.

`is_accepting()` says whether the text so far is a complete match. More may still be allowed: `1+2` is complete, but
`1+23` is too. `is_terminated()` says whether nothing more can be committed.

### Forced tokens

`commit_forced_tokens()` commits the tokens the grammar forces next and returns them; call it on a copy to look without committing. When the forced text could be tokenized in several ways, it takes the longest token at each step, at most 1,000 tokens per call.
See [forced tokens](https://github.com/IsaacBreen/glrmask/blob/main/docs/forcing.md).

## Composing grammars

A grammar can declare externs: named slots that you bind later.

- `extern grammar NAME` takes a portable grammar or expression on a Grammar; on a Module it can also take a module or token set.
- `extern token NAME` can be declared on a Grammar but must be filled on a Module with a token set. It matches exactly one of those tokens, never their text.

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

vocab = glrmask.Vocab({i: bytes([i]) for i in range(256)})
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

A `Grammar` holds portable source and bindings. A `Module` can also hold model-specific token IDs and precompiled work. `precompile(vocab)` compiles as much as it can now,
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
token sets. `Grammar.from_expr` makes a grammar from an expression that contains neither modules nor token sets, including nested parts. Expressions containing either can be bound to a `Module`.

## Token sets

A `TokenSet` is a set of token IDs. In a grammar it matches any one of its tokens. Create one from IDs,
`TokenSet([1, 2, 3])`, or from ranges, `TokenSet.from_ranges([range(0, 10)])`. Ranges in C and C++ use `{start, end}` with `end` excluded, as in every language. Rust takes `0..10` or `0..=9`. An ID doesn't need to be in the vocabulary.

Token IDs go from 0 to 16,777,215 (2^24 - 1, `MAX_TOKEN_ID` in Rust). A larger ID in a vocabulary, token set or binding
is an error.

Tokens with no bytes, such as tool-call markers, are special tokens. They never match text, so only a token set can allow one, and they don't need to be in the vocabulary. Two tokens with the same bytes are still different tokens to a token set. See [token sets](https://github.com/IsaacBreen/glrmask/blob/main/docs/token-sets.md).

## Mask strategies

`MaskStrategy` expresses a cost preference:

- `Auto`: let the library choose. The default.
- `FastCompile`: prefer lower compilation cost, even if later masks cost more.
- `FastMasks`: prefer lower mask cost, even if compilation costs more.

These are distinct preferences, not concrete engine selections or speed guarantees. If a preferred plan declines
on applicability or deterministic size grounds, another supported plan may be used. Invalid input, corrupt
artifacts and operational resource failures are not hidden by fallback. All supported plans preserve mask semantics.
Retained compatible work may be reused in any mode.

`BoundaryMaskStrategy` expresses the same three preferences for applicable grammar boundary work, independently
of overall strategy. Pass the preferences to `compile` and `precompile` on either Grammar or Module.

```python
constraint = grammar.compile(vocab, mask_strategy=glrmask.MaskStrategy.FAST_MASKS,
                             boundary_mask_strategy=glrmask.BoundaryMaskStrategy.FAST_COMPILE)
```

## Errors

Creating, binding, compiling and loading grammars can fail; generating can't, because commits return `false` instead.
Each error message says what went wrong, with the line and column in the grammar source when that's the cause.
In Rust, `Error::kind()` gives the kind of error: `Syntax`, `InvalidInput`, `Compilation`, `ResourceLimit`, `StrategyUnsupported`, `Serialization` or `Internal`. `source_span()` gives its place in the source.
Python raises `GLRMaskError` (a `ValueError`) with a `kind` string. Node.js throws a `GLRMaskError` with a `code`, C returns a status and sets `glrmask_error_message()`, and C++ throws `glrmask::Error`. New kinds may be added in a minor release.

## Grammar formats

**JSON Schema.** Drafts 4, 6, 7, 2019-09 and 2020-12, chosen by `$schema`; without one, 2020-12. `format` is treated as
an annotation, and `oneOf` requires exactly one match. A schema feature GLRMask can't enforce is an error. The JSON
has one layout, the one Python's `json.dumps` writes by default: no whitespace outside strings except a space after each
`,` and `:`, as in `{"a": [1, 2]}`.

**GLRM** is GLRMask's own format. It has terminals (`t`) and nonterminals (`nt`), externs, scoped grammars, and
regular-expression intersection and subtraction. See [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md).

```glrm
start value
t INTEGER = /-?(0|[1-9][0-9]*)/
nt value = INTEGER | "null"
```

**Lark**, with GLRMask's additions: `<|name|>` for a token extern, `@name` for a grammar extern, `%json` for an inline
JSON Schema and `[lazy]` for a terminal that ends as soon as it can. See [Lark](https://github.com/IsaacBreen/glrmask/blob/main/docs/lark-dialect.md).

Features GLRMask doesn't support, such as Lark priorities, are errors. They are never silently ignored.
