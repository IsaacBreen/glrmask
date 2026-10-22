## Vocabulary and exact tokens

`Vocab` maps original `uint32_t` token IDs to the exact bytes produced by the tokenizer. Bytes may be partial UTF-8 or contain NUL; do not normalize them or reconstruct them from a display spelling. The string-view constructor copies exactly the supplied range, so use `Bytes` for embedded NUL. Different IDs may have identical bytes. Duplicate byte-backed IDs, duplicate exact-only IDs, and overlap between those domains are rejected. `size()` counts byte-backed entries only.

```cpp
glrmask::Vocab vocab({
    {0, glrmask::Bytes{0xc3}},
    {1, glrmask::Bytes{0xa9}},
    {31, glrmask::Bytes{0xc3, 0xa9}},
}, {99});  // Exact-only control without byte spelling.
```

`Token(id)` is a raw original-ID event. `vocab.tokens(ids)` produces vocabulary-qualified `ExactTokens`; `vocab.token_ranges({{first, last}, ...})` uses inclusive endpoints and normalizes overlapping or adjacent ranges without expanding every ID. `ExactTokens::ranges()` stays compact and `ids()` materializes the IDs. An empty set denotes an empty event language, not epsilon. Each match consumes one chosen token event. Exact and end-token events must be committed with `commit_token`, not simulated by `commit_bytes`.

## Grammars

| Factory | Input |
| --- | --- |
| `Grammar::from_ebnf(source)` | EBNF text |
| `Grammar::from_glrm(source)` | Native GLRM text |
| `Grammar::from_json_schema(source)` | JSON Schema encoded as JSON text |
| `Grammar::from_lark(source, options)` | Lark text and an optional profile |
| `Grammar::exact(tokens)` | One vocabulary-qualified exact-token event |

The constructors own their source text. Import semantics and limitations match the Rust core. See the [GLRM reference](https://github.com/IsaacBreen/glrmask/blob/v0.3.1/docs/glrm-format.md) and [Lark dialect](https://github.com/IsaacBreen/glrmask/blob/v0.3.1/docs/lark-dialect.md). The binding does not expose parse trees, and the JSON Schema importer is not a complete validator.

Recognition is the default Lark profile. LLM extensions and named token atoms require an explicit map:

```cpp
glrmask::LarkOptions lark;
lark.profile = glrmask::LarkProfile::Llm;
lark.token_names["begin"] = {99};
auto grammar = glrmask::Grammar::from_lark("start: <|begin|>", lark);
```

## Preparation and build choices

`Grammar` holds reusable source and mutable bindings. `bind` is transactional: a successful call returns `void`, while an invalid binding leaves the receiver unchanged. Clones and already attached children keep independent snapshots. Source bindings accept `Grammar`, `Token`, or `ExactTokens`. Qualified descendant names use `::`, as in `payload::TOOL`; inspect `requirements()` instead of guessing paths.

```cpp
glrmask::Vocab composed_vocab({{0, "["}, {1, "hello"}, {2, "]"}, {3, "[hello]"}});
auto parent = glrmask::Grammar::from_glrm(
    R"(glrm 1; start root; extern grammar payload;
        nt root = "[" payload "]";)");
auto child = glrmask::Grammar::from_ebnf(R"(start ::= "hello")");
parent.bind("payload", child);
auto constraint = parent.compile(composed_vocab);
```

`Grammar::prepare(vocab, boundary?)` returns a `PreparedGrammar`, reusable vocabulary-specific machinery that may still contain open slots. Prepared bindings accept `PreparedGrammar`, `Token`, or `ExactTokens`; a runnable `Constraint` is not a preparation-stage operand. `requirements()` reports `name`, `path`, `kind`, and optional `source_span`; spans use a half-open byte range and one-based line and Unicode-scalar column.

Preparation retains only an optional boundary-metadata request. Final optimization and root end tokens belong to `compile` or `link`:

```cpp
glrmask::Vocab prepared_vocab(
    {{0, "["}, {1, "hello"}, {2, "]"}, {3, "[hello]"}}, {99});
auto open_parent = glrmask::Grammar::from_glrm(
    R"(glrm 1; start root; extern grammar payload;
        nt root = "[" payload "]";)");
auto prepared_child = glrmask::Grammar::from_ebnf(R"(start ::= "hello")");
auto prepared = open_parent.prepare(prepared_vocab);
prepared.bind("payload", prepared_child.prepare(prepared_vocab));
glrmask::BuildOptions options;
options.optimization = glrmask::Optimization::FastRuntime;
options.end_tokens = {99};
auto runnable = prepared.link(options);
```

| `Optimization` | Intent |
| --- | --- |
| `Auto` | Core default trade-off. |
| `FastBuild` | Shorter compilation. |
| `Balanced` | Balance compilation and masking cost. |
| `FastRuntime` | Lower per-token masking cost, accepting more compilation work. |

`BoundaryTriggerDetail::None`, `Tokens`, and `Exact` select optional boundary metadata. An omitted request preserves a prepared request; an explicit `None` clears it at final link. Unsupported combinations return the core's error rather than selecting a hidden fallback.

## Decoding

A `Constraint` is immutable and reusable. `start()` creates an independent state; `ConstraintState::clone()` copies the parser snapshot without sharing mutable sequence state. A state keeps its native constraint alive after its C++ owner is destroyed, moved from, or disposed.

```cpp
auto constraint = grammar.compile(vocab);
auto state = constraint.start();
std::vector<glrmask::TokenId> mask(constraint.mask_len());
state.fill_mask(mask);
if ((mask[0] & 1) != 0) state.commit_token(0);
```

Masks are packed original-ID bitsets. Bit `id % 32` of word `id / 32` is set when token `id` is allowed. The word count is `constraint.mask_len()`, not `vocab.size()`; sparse or very large IDs still require a dense output buffer. `fill_mask` zeros surplus words. An undersized buffer throws `glrmask::Error` with `status() == GLRMASK_BUFFER_TOO_SMALL` and is left unchanged. `state.mask()` allocates a new vector.

`is_rejected()` reports an irrecoverably rejected prefix. A failed grammar commit follows the core's state semantics rather than acting as a general rollback; clone before speculative commits. `forced_tokens()` performs read-only bounded lookahead. `marker_ids()` and `marker_events()` expose possible recognition observations from the initial or latest commit. A marker carries an `id` and a `uint64_t` byte offset; it is not a final parse capture.

## End tokens

`is_accepting()` means the current prefix is a complete match, not that generation must stop. To stop on an explicit event, set root `end_tokens` in `BuildOptions` and stop when one is sampled and committed.

An allowed end token leaves the state accepting and non-rejected, empties the next mask, and prevents further commits. End-token policy belongs to the root and is not inherited by attached children.

## Persistence

```cpp
auto bytes = constraint.save();
auto restored = glrmask::Constraint::load(bytes);

auto external = constraint.save(true);       // Omit vocabulary storage.
auto rebound = glrmask::Constraint::load(external, &vocab);

auto prepared_bytes = prepared.save();
auto restored_prepared = glrmask::PreparedGrammar::load(prepared_bytes, &vocab);
```

These are the Rust core's artifact formats, not C++ object serialization. Compatible builds of Rust, Python, C++, and Node can exchange them. Exact vocabulary identity is checked when a vocabulary is supplied; an external-vocabulary artifact cannot load without it, and equal vocabulary size is insufficient. Compatibility is not promised across releases, architectures, or unrelated builds. Keep source and configuration for long-term regeneration, and treat untrusted artifacts according to the core loader's limitations.

## Errors and ownership

Fallible C++ operations throw `glrmask::Error`; `status()` returns the underlying C status and `what()` owns the diagnostic text. Copies clone a handle and moves transfer it. Using a moved-from or disposed object throws `GLRMASK_CLOSED`. `dispose()` and normal RAII destruction release ownership, and repeated C++ disposal is harmless.

A Rust unwinding panic is contained at the boundary and reported as `GLRMASK_PANIC`; discard any state it affected. Process aborts, allocation failure, stack overflow, and invalid foreign memory are not recoverable this way. Immutable constraints may be shared between threads with independent states. Do not concurrently mutate a grammar, prepared grammar, or state, and do not destroy a handle while another thread uses it.

## C interface

`glrmask.h` is the complete C ABI and is valid C. Handles are opaque and owning, text and bytes are length-delimited, and result views are borrowed until their result handle is freed. Free each owning handle exactly once with its matching `_free`, and do not use a C++ allocator on Rust-owned memory.

| Status | Meaning |
| --- | --- |
| `GLRMASK_OK` | Success. |
| `GLRMASK_INVALID_ARGUMENT` | Invalid pointer, length, UTF-8, option, or vocabulary input. |
| `GLRMASK_ENGINE_ERROR` | Grammar, state, vocabulary-identity, or artifact error from the core. |
| `GLRMASK_PANIC` | A contained unwinding panic; discard affected state. |
| `GLRMASK_BUFFER_TOO_SMALL` | Caller mask capacity is insufficient. |
| `GLRMASK_CLOSED` | A disposed or moved-from high-level object was used. |

Output handles are null on failure. `glrmask_error_message()` is thread-local and valid only until the next GLRMask call on that thread. Vocabulary and committed bytes are arbitrary binary data; all other lengths are element counts. Option structs must carry the `struct_size` returned by their default constructors, and a null build-options pointer selects defaults. A state handle independently retains its constraint. C++ exceptions never cross the C ABI.

## Building from source

Building from a checkout requires a current stable Rust toolchain, CMake 3.20 or newer, and the platform C/C++ compiler; C++ consumers need C++17. On Windows, use the MSVC toolchain from a Visual Studio Developer PowerShell or Developer Command Prompt.

```sh
cmake -S bindings/cpp -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native --config Release
cmake --install build/native --config Release --prefix /absolute/path/glrmask-install
```

The default is static; add `-DGLRMASK_SHARED=ON` for a shared build. The installed package exports `glrmask::glrmask` for C++17 and `glrmask::c` for C. Configure consumers with `-DCMAKE_PREFIX_PATH=/absolute/path/glrmask-install`. A Windows shared deployment needs `glrmask_c.dll` beside the executable or on `PATH`; Unix shared deployments use the normal runtime library path. Use matching headers and binaries from the same build.
