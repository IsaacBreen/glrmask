## Generating

Each sequence has its own `ConstraintState`. At each step, mask the model's scores, allow EOS when the state is accepting, and choose a token. Stop if EOS is chosen; otherwise commit the token:

See the quickstart above for the sampling loop and unpacking the packed mask.


Masks indicate which tokens are allowed next, as a bitmask: token `i` is bit `i % 32` of word `i // 32`.
Python `mask_bit_len` (a method in Rust and C++, `maskBitLength` in Node) is the mask's length in bits (the largest token ID in the constraint, including special and end tokens, plus one, or zero if there are no token IDs), including gaps.
Allocate `(c.mask_bit_len + 31) // 32` words for `fill_mask`, which writes into reusable storage and zeros surplus words.
Python `mask()` returns packed uint32 words. Use `mask().view(np.int32)` for signed packed words without copying.
The constraint and each state have the same constant mask bit length.
For a wider packed mask, allocate the required words and use `fill_mask`. It accepts writable contiguous
1-D aligned, native-endian NumPy uint32 arrays (including rows of C-contiguous matrices); for an int32 buffer, pass `buf.view(np.uint32)`; all words are written, and surplus words are zeroed on success. Buffer validation errors leave it unchanged; an unexpected internal failure may have partially written it. Mask computation may release the GIL. Compile and precompile run native compilation without the GIL. Concurrent calls must use different states and nonoverlapping buffers; do not access or resize the buffer (including aliases) during the call.

For PyTorch serving with V equal to the logits width (at least every state mask bit length):

```python
# CPU staging buffer, with each row reused for one sequence.
bitmask = torch.empty((len(states), (V + 31) // 32), dtype=torch.int32, pin_memory=True)
for i, state in enumerate(states):
    state.fill_mask(bitmask[i].numpy().view(np.uint32))  # zero-copy view of the CPU row
mask_on_device = bitmask.to(device, non_blocking=True)
# Apply mask_on_device with your packed-mask kernel. Do not reuse the CPU
# buffer until the asynchronous transfer completes.
```

`commit_token` and `commit_bytes` consume the input. Text never matches a token set. If the input isn't allowed, they return false and terminate the state.

`is_accepting()` says whether the output so far could end here. More may still be allowed: `1+2` is complete, but
`1+23` is too. `is_terminated()` says whether nothing more can be committed. Together they distinguish in progress (false, false), may stop (true, false), complete (true, true), and dead (false, true). Treat EOS as a caller-side stop decision; committing an unknown EOS token rejects even an accepting state.

### Forced bytes

`forced_bytes()` returns the bytes that must come next. Commit them with `commit_bytes()`. If feeding tokens to a model, tokenize the bytes and commit those tokens instead. The query stops before a choice, an accepting position or a token-set item. Handle special tokens yourself. Long forced runs may be returned in parts; call again after committing.


## Composing grammars

A grammar can declare externs: named slots that you bind later.

- `extern grammar NAME` takes a grammar, module, expression or token set.
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

call = glrmask.Module(call_format)
call.bind("WEATHER_ARGS", weather)
call.bind("SEARCH_ARGS", search)

vocab = glrmask.Vocab({i: bytes([i]) for i in range(256)})
state = call.compile(vocab).start()
assert state.commit_bytes(b'<weather>{"city": "Perth"}</weather>')
assert state.is_accepting()
```

Each grammar keeps its own rules, including the whitespace it ignores, and one model token can span the boundary
between parts.

`bind` changes the module in place; copy it first to bind the same one in different ways. Each extern is
bound once, on the module that declares it. An unknown name is an error; an unbound name accepts a value of the appropriate kind; an already bound name rejects rebinding. Bindings capture values, not later mutations. A value can't be bound while it has unbound externs of its own.
`externs` is a read-only mapping of extern names to `Extern`, and `externs.unbound` lists the ones not yet bound (in Rust, `externs()`
and `externs().unbound()`). This is a shallow view of the direct declarations, not a recursive search. Qualified names use `::` for scoped declarations.

### Modules

A `Grammar` describes a language. A `Module` holds a grammar, its bindings and precompiled work. `precompile(vocab)` compiles as much as it can now,
even with externs unbound, and `compile` finishes compiling the module into a constraint. Use modules when the same parts appear in many grammars. Tool
calls are the typical case: precompile each tool's schema once, and any combination of tools then compiles quickly.

```python
weather_args = glrmask.Module(weather)
weather_args.precompile(vocab)
search_args = glrmask.Module(search)
search_args.precompile(vocab)

call = glrmask.Module(call_format)
call.precompile(vocab)
call.bind("WEATHER_ARGS", weather_args)
call.bind("SEARCH_ARGS", search_args)
constraint = call.compile(vocab)
```


`Module.bind(name, value)` accepts a grammar, module, constraint, expression or token set.

`module.save()` returns bytes, and `Module.load(data)` restores the module with its bindings and precompiled work. A
module can be saved with externs unbound. Precompiled work is for one vocabulary; compiling or precompiling it with another is an error.

`constraint.save()` and `Constraint.load(data, vocab)` save and load a compiled constraint. The vocabulary isn't saved,
and `load` checks that you pass the one it was compiled for. Bytes saved by one GLRMask version may not load in another. Load artifacts only from trusted sources; malformed input is rejected where detected, but this is not a hardened untrusted-data parser.

### Expressions

`Expr.choice(*parts)` matches any one of its parts, `Expr.sequence(*parts)` matches them one after another, and
`Expr.literal(text)` matches exactly `text`, with no whitespace skipped. Parts can be grammars, modules, constraints, expressions and
token sets. `Grammar.from_expr` makes a grammar from an expression that contains no modules, constraints or token sets, including nested parts. Expressions containing any of these can be bound to a `Module`.

## Token sets

A `TokenSet` is a set of token IDs. In a grammar it matches any one of its tokens. Create one from IDs,
`TokenSet([1, 2, 3])`, or from ranges, `TokenSet([range(0, 10)])`. Ranges in C and C++ use `{start, end}` with `end` excluded, as in every language. Rust takes only half-open ranges such as `0..10`. Empty ranges contribute no tokens; reversed C/C++/Node ranges are invalid. Sets are normalized, merging overlapping and adjacent ranges. Python ranges may have any step. An ID doesn't need to be in the vocabulary.

Tokens with no bytes, such as tool-call markers, are special tokens. They never match text, so only a token set can allow one in a mask, and they don't need to be in the vocabulary. Two tokens with the same bytes are still different tokens to a token set. See [token sets](https://github.com/IsaacBreen/glrmask/blob/main/docs/token-sets.md).

## Mask strategies

`MaskStrategy` controls how the constraint is compiled:

- `AUTO` (Rust/Python), `Auto` (Node/C++): The default; currently `FAST_MASKS` (`FastMasks` in Node/C++).
- `FAST_COMPILE` (Rust/Python), `FastCompile` (Node/C++): Faster compilation, slower masks.
- `FAST_MASKS` (Rust/Python), `FastMasks` (Node/C++): Faster masks, slower compilation.

These are preferences, not speed guarantees. GLRMask may use another strategy; masks have the same semantics. AUTO does not reject a grammar merely because a particular strategy is unsupported.

Pass the preference to `compile` on a Grammar or Module, or to `precompile` on a Module.

```python
constraint = grammar.compile(vocab, mask_strategy=glrmask.MaskStrategy.FAST_MASKS)
```

## Errors

Creating, binding, compiling and loading grammars can fail. Commits return `false` only on rejection; internal failures panic in Rust and surface as errors in foreign bindings. Discard a state after an internal failure.
Each error message says what went wrong. It may also include a location in the grammar source.
In Rust, `Error::kind()` gives the kind of error: `Syntax`, `InvalidInput`, `Compilation`, `ResourceLimit`, `StrategyUnsupported`, `Serialization`, `VocabMismatch`, `BufferTooSmall` or `Internal`.
Python raises `GLRMaskError` (an `Exception`) with a `kind` string: one of `"syntax"`, `"invalid_input"`, `"compilation"`, `"resource_limit"`, `"strategy_unsupported"`, `"serialization"`, `"vocab_mismatch"`, `"buffer_too_small"`, `"internal"`. Node.js throws a `GLRMaskError` with a `code`, C returns a status and sets `glrmask_error_message()`, and C++ throws `glrmask::Error`.

## Grammar formats

**JSON Schema.** Drafts 4, 6, 7, 2019-09 and 2020-12, chosen by `$schema`; without one, 2020-12. `format` is treated as
an annotation, and `oneOf` requires exactly one match. A schema feature GLRMask can't enforce is an error. The JSON
has one layout, the one Python's `json.dumps` writes by default: no whitespace outside strings except a space after each
`,` and `:`, as in `{"a": [1, 2]}`.

**[GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md)** has terminals (`t`) and nonterminals (`nt`), externs, scoped grammars, and
regular-expression intersection and subtraction.

```glrm
start value
t INTEGER = /-?(0|[1-9][0-9]*)/
nt value = INTEGER | "null"
```

**Lark**, with GLRMask's additions: `<|name|>` for a token extern, `@name` for a grammar extern, `%json` for an inline
JSON Schema and `[lazy]` for a terminal that ends as soon as it can. See [Lark](https://github.com/IsaacBreen/glrmask/blob/main/docs/lark-dialect.md).

Features GLRMask doesn't support, such as Lark priorities, are errors. They are never silently ignored.
