## Vocabularies and token sets

`Vocab(entries)` takes `{id, bytes}` entries, with each token's exact bytes as `Bytes` or a
`std::string_view`. Use the bytes the tokenizer produces, not its display form: a token may be part of a UTF-8 character
or contain NUL. IDs need not be contiguous, and two tokens may have the same bytes. An ID can't appear twice. Leave out
tokens with no bytes, such as tool-call tokens; a token set can still allow them.

```cpp
glrmask::Vocab vocab({
    {0, glrmask::Bytes{0xc3}},
    {1, glrmask::Bytes{0xa9}},
    {2, "é"},
});
```

`TokenSet(ids)` is a set of token IDs, and
`TokenSet::from_ranges({{start, end}})` builds one from ranges that include `start` but not `end`. See [token
sets](https://github.com/IsaacBreen/glrmask/blob/main/docs/token-sets.md).

## Grammars

| Factory | Input |
| --- | --- |
| `Grammar::from_glrm(source)` | [GLRM](https://github.com/IsaacBreen/glrmask/blob/main/docs/glrm-format.md) |
| `Grammar::from_json_schema(source)` | JSON Schema, as JSON text |
| `Grammar::from_lark(source)` | [Lark](https://github.com/IsaacBreen/glrmask/blob/main/docs/lark-dialect.md) |

## Composing grammars

```cpp
std::vector<glrmask::VocabEntry> entries;
for (glrmask::TokenId id = 0; id < 256; ++id) entries.emplace_back(id, glrmask::Bytes{static_cast<std::uint8_t>(id)});
glrmask::Vocab vocab(entries);

glrmask::Module weather(glrmask::Grammar::from_json_schema(R"({
    "type": "object", "properties": {"city": {"type": "string"}},
    "required": ["city"], "additionalProperties": false})"));
weather.precompile(vocab);

glrmask::Module call(glrmask::Grammar::from_glrm(R"(
    extern grammar ARGS
    start call
    nt call = "<weather>" ARGS "</weather>")"));
call.bind("ARGS", weather);
auto constraint = call.compile(vocab);
```

On a Module, `bind(name, value)` takes a `Grammar`, `Module`, `Constraint`, `Expr`, or `TokenSet`. It changes the module in
place; if it fails, nothing changes. Copy a module first to bind the same one in different ways. `externs().unbound()` lists the unbound names.

`Expr::choice`, `Expr::sequence` and `Expr::literal` combine grammars without writing a new one:

```cpp
auto reply = glrmask::Expr::choice({
    glrmask::Grammar::from_lark(R"(start: "yes")"), glrmask::Grammar::from_lark(R"(start: "no")")});
auto grammar = glrmask::Grammar::from_expr(glrmask::Expr::sequence({glrmask::Expr::literal("Answer: "), reply}));
```

Grammar expressions must not contain modules, constraints or token sets, even indirectly. Bind model-specific values on a Module.

## Mask strategies

`compile` and `precompile` take a `CompileOptions` with `mask_strategy`.

| `MaskStrategy` | Preference |
| --- | --- |
| `Auto` | The default; currently `FastMasks`. |
| `FastCompile` | Faster compilation, slower masks. |
| `FastMasks` | Faster masks, slower compilation. |

These are preferences, not speed guarantees.
See [mask strategies](https://github.com/IsaacBreen/glrmask/blob/main/GUIDE.md#mask-strategies).

```cpp
glrmask::CompileOptions options;
options.mask_strategy = glrmask::MaskStrategy::FastMasks;
auto constraint = grammar.compile(vocab, options);
```

## Generating

```cpp
auto state = constraint.start();
std::vector<std::uint32_t> mask((constraint.mask_bit_len() + 31) / 32);
state.fill_mask(mask);
glrmask::TokenId id = 42;
bool allowed = (mask[id / 32] >> (id % 32)) & 1;
```

A mask contains packed uint32 words: token i is bit i % 32 of word i / 32. `fill_mask` writes it
into a buffer you reuse; `mask()` returns a new one.

`commit_token` and `commit_bytes` consume the input. If the input isn't allowed, they return false and terminate the state. Commit special tokens and token sets with `commit_token`;
bytes never match them.

`is_accepting()` says whether the output so far could end here, and `is_terminated()` whether nothing more can be
committed. `forced_bytes()` returns the bytes that must come next. Tokenize and commit them yourself; exact token-set items are not included.

[examples/generate.cpp](https://github.com/IsaacBreen/glrmask/blob/main/bindings/cpp/examples/generate.cpp) is a
complete generation loop.

## Saving

```cpp
auto bytes = constraint.save();
auto restored = glrmask::Constraint::load(bytes, vocab);

auto module_bytes = call.save();
auto module = glrmask::Module::load(module_bytes);
```

Saved bytes are the same in every binding, so a constraint saved from Python loads in C++. A saved constraint doesn't
include its vocabulary: pass the same vocabulary to `load`, which checks that it matches exactly. A saved module records
the vocabulary its precompiled parts were built for; `compile` and `precompile` check it. Saved bytes are only guaranteed to load in
the same GLRMask version on the same architecture, so keep the grammar source to compile again after an upgrade. Don't
load saved bytes from untrusted sources.

Empty ranges are no-ops; reversed ranges are invalid. Bindings capture their values. Stop decisions belong to the caller: committing an unknown EOS rejects the state even if it was accepting.

## Errors

Operations that can fail throw `glrmask::Error`, a `std::runtime_error`. `what()` is the message. `status()` is one of:

| Status | |
| --- | --- |
| `GLRMASK_SYNTAX` | Invalid grammar syntax. |
| `GLRMASK_INVALID_INPUT` | A handle or argument is invalid. |
| `GLRMASK_COMPILATION` | Compilation failed. |
| `GLRMASK_RESOURCE_LIMIT` | A resource limit was exceeded. |
| `GLRMASK_STRATEGY_UNSUPPORTED` | The chosen strategy does not support the grammar. |
| `GLRMASK_SERIALIZATION` | Saved bytes could not be loaded. |
| `GLRMASK_VOCAB_MISMATCH` | Compiled for a different vocabulary. |
| `GLRMASK_INTERNAL` | |
| `GLRMASK_BUFFER_TOO_SMALL` | The buffer passed to `fill_mask` is shorter than `(mask_bit_len() + 31) / 32`. |
| `GLRMASK_CLOSED` | The object was moved from. |

## Objects and threads

Every object owns its native handle and frees it when destroyed. Copying a `Module`,
`Grammar` or state gives an independent copy. A state keeps its constraint alive.

A `Constraint` can be shared between threads, each with its own states. Don't use one grammar, module or state from two
threads at once.

## C

`glrmask.h` is the C interface; the C++ header is built on it. Each function returns a `glrmask_status`, and on failure
`glrmask_error_message()` gives the message, valid until the next GLRMask call on the same thread (except the message accessor); empty when no error is recorded. C commits return a status and write 0 to the out-parameter if the input is not allowed, terminating the state. Free each object once
with its `_free` function. Initialize a `glrmask_compile_options options` with `glrmask_compile_options_init(&options, sizeof(options))`, or pass a null pointer for the
defaults. The header documents ownership in full.

## Building from source

You need a stable Rust toolchain, CMake 3.20 or newer, and a C/C++ compiler. On Windows, use a Visual Studio Developer
PowerShell with the MSVC build tools.

```sh
cmake -S bindings/cpp -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
cmake --install build/native --config Release --prefix /absolute/path/glrmask-install
```

This builds a static library; add `-DGLRMASK_SHARED=ON` for a shared one. Then use it as in
[Installation](#installation), with `-DCMAKE_PREFIX_PATH=/absolute/path/glrmask-install`. A shared build on Windows
needs `glrmask_c.dll` next to the executable or on `PATH`.
