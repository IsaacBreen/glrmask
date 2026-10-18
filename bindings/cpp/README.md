# C++ and C

GLRMask's native C++ interface calls the same Rust compiler and decoding engine as the Rust and Python APIs. It does not embed Python, launch a subprocess, or reimplement the parser. The public C++17 facade is `glrmask.hpp`; the underlying, length-delimited C interface is `glrmask.h`.

Bring the exact token-ID-to-byte mapping from your inference backend. GLRMask compiles a grammar for that vocabulary, returns the allowed-token mask, and advances when you commit a sampled token. It does not run the language model or choose a sampling policy.

## Build and install

Building from source requires a current stable Rust toolchain, CMake 3.20 or newer, and the platform C/C++ compiler. On Windows, use the MSVC toolchain from a Visual Studio Developer PowerShell or Developer Command Prompt. C++ consumers require C++17. Binary consumers of an installed package do not need Rust.

Run from a complete GLRMask checkout:

```sh
cmake -S bindings/cpp -B build/native -DCMAKE_BUILD_TYPE=Release -DGLRMASK_BUILD_TESTS=ON
cmake --build build/native --config Release
ctest --test-dir build/native -C Release --output-on-failure
cmake --install build/native --config Release --prefix /absolute/path/glrmask-install
```

The default is a static library. Add `-DGLRMASK_SHARED=ON` to build against the shared library instead. The CMake build invokes Cargo with the locked dependencies and the `release` profile. It does not change the core library's normal compilation settings.

The native CI matrix builds Windows x64 and arm64/MSVC, macOS x64 and arm64, and Linux x64 and arm64 on Ubuntu 22.04 (glibc 2.35). macOS artifacts target macOS 11 or newer. Use the matching or newer runtime, or build from source for another deployment. CI produces platform-specific CMake-installable archives; these are build artifacts, not a promise that a new registry or GitHub release has been published. Use matching headers and binaries from the same build. Other toolchains and 32-bit targets are outside that matrix.

### Use the installed package

```cmake
cmake_minimum_required(VERSION 3.20)
project(my_decoder LANGUAGES CXX)
find_package(glrmask 0.3 CONFIG REQUIRED)
add_executable(my_decoder main.cpp)
target_link_libraries(my_decoder PRIVATE glrmask::glrmask)
```

Configure the application with `-DCMAKE_PREFIX_PATH=/absolute/path/glrmask-install`. The exported `glrmask::glrmask` target includes the C++17 facade and native library. A pure C consumer can link `glrmask::c` and include only `glrmask.h`.

For a Windows shared-library deployment, place `glrmask_c.dll` beside the application or put the installation's `bin` directory on `PATH`. For a Unix shared-library deployment, configure the application's runtime library search path as usual. The macOS library uses `@rpath/libglrmask_c.dylib`, not an absolute source-build path. The static package avoids runtime DLL/shared-library deployment.

### Development builds and existing artifacts

`GLRMASK_CARGO_TARGET_DIR` selects an owned Cargo cache. Its default honors `CARGO_TARGET_DIR`, or uses the CMake build directory. `GLRMASK_CARGO_PROFILE=bindings-dev` selects the repository's incremental development profile. `GLRMASK_RUST_TARGET` passes an explicit Rust target triple; a compatible CMake toolchain is still required for cross-compiling C/C++ consumers.

To use a library you already built, set `GLRMASK_LIBRARY_DIR` to the directory containing `glrmask_c.lib` on Windows, or `libglrmask_c.a` on Unix. For a shared build, supply the matching DLL and import library, or `.so`/`.dylib`. This bypasses Cargo without silently searching other worktrees or caches.

## A decoding loop

```cpp
#include <glrmask.hpp>
#include <vector>

int main() {
    glrmask::Vocab vocab({{0, "hello"}, {1, " "}, {2, "world"}});
    auto grammar = glrmask::Grammar::from_ebnf(
        R"(start ::= "hello" " " "world")");
    auto constraint = grammar.compile(vocab);
    auto state = constraint.start();
    std::vector<glrmask::TokenId> mask(constraint.mask_len());

    // Substitute your model's sampler for these three predetermined tokens.
    for (glrmask::TokenId id : {0u, 1u, 2u}) {
        state.fill_mask(mask);
        if ((mask[id / 32] & (glrmask::TokenId{1} << (id % 32))) == 0)
            return 1;
        state.commit_token(id);
    }
    return state.is_accepting() ? 0 : 2;
}
```

The complete runnable example is [examples/decode.cpp](examples/decode.cpp). In a serving loop, allocate the packed buffer once and reuse `fill_mask`. `state.mask()` is a convenient allocating alternative.

## Vocabulary and packed masks

`Vocab` accepts `VocabEntry` values containing an original `uint32_t` token ID and arbitrary bytes. Its string-view constructor copies exactly the supplied byte range; use `Bytes` or an explicitly sized `std::string_view` for embedded NUL. Partial UTF-8 token pieces are valid. Do not normalize, decode and re-encode, or merge token IDs with identical byte spellings.

```cpp
glrmask::Vocab vocab({
    {0, glrmask::Bytes{0xc3}},
    {1, glrmask::Bytes{0xa9}},
    {31, glrmask::Bytes{0xc3, 0xa9}},
}, {99});  // 99 is an exact-only control, without ordinary token bytes.
```

Duplicate byte-backed IDs, duplicate exact-only IDs, and overlap between those two domains are rejected. Different IDs may have identical bytes. `vocab.size()` counts byte-backed entries, not exact-only controls.

Masks are packed **original-ID** bitsets. Bit `id % 32` in word `id / 32` is one when that token is allowed. The word count is `constraint.mask_len()`, not `vocab.size()`. Sparse IDs and configured end tokens can increase the output size; very large original IDs can therefore require a large dense output buffer even when the vocabulary or token set is compact.

`fill_mask(pointer, word_count)` and `fill_mask(vector)` require at least that many words. An undersized buffer raises `GLRMASK_BUFFER_TOO_SMALL` without changing the buffer. Extra words are zeroed. No temporary language-bridge mask is allocated by the fill operation; allocations inside the engine itself are not covered by that statement.

## Grammar sources and build choices

| Factory | Input |
| --- | --- |
| `Grammar::from_ebnf(source)` | EBNF text |
| `Grammar::from_glrm(source)` | Native GLRM text |
| `Grammar::from_json_schema(source)` | JSON Schema encoded as JSON text |
| `Grammar::from_lark(source, options)` | Lark text and an explicit optional profile |
| `Grammar::exact(tokens)` | One exact-token event from a vocabulary-qualified set |

The constructors own their source text; temporary C++ strings need not outlive the returned grammar. Importer capabilities, supported JSON Schema keywords, Lark profiles and language limitations are the same as the core library. See the [GLRM reference](../../docs/glrm-format.md) and [Lark dialect](../../docs/lark-dialect.md). These bindings do not add a full JSON Schema validator or parse-tree/capture API.

```cpp
glrmask::BuildOptions options;
options.optimization = glrmask::Optimization::FastRuntime;
options.end_tokens = {99};
options.boundary_trigger = glrmask::BoundaryTriggerDetail::Tokens;
auto constraint = grammar.compile(vocab, options);
```

| `Optimization` | Intent |
| --- | --- |
| `Auto` | Let the core choose its default trade-off. |
| `FastBuild` | Use the ordinary Dynamic compiler/runtime. |
| `Balanced` | Use the core's balanced/O2 path. |
| `FastRuntime` | Prefer low per-token runtime cost, accepting more build work. |

`BuildOptions` applies to the final `compile` or `link`. Preparation accepts only the optional boundary-metadata request: `grammar.prepare(vocab, BoundaryTriggerDetail::Tokens)`. End tokens and final optimization are not silently retained at preparation. An omitted boundary request differs from explicitly selecting `None`, which clears a retained request at final link. Unsupported combinations produce the same error as the Rust core; the binding does not substitute a hidden fallback.

Recognition is the default Lark profile. Selected LLM extensions and named exact-token resolution must be enabled explicitly:

```cpp
glrmask::LarkOptions lark;
lark.profile = glrmask::LarkProfile::Llm;
lark.token_names["begin"] = {99};
auto event_grammar = glrmask::Grammar::from_lark("start: <|begin|>", lark);
```

## Source composition and reusable preparation

Bindings mutate the receiving `Grammar` or `PreparedGrammar` **transactionally**. Successful `bind` returns `void`; invalid bindings leave the receiver unchanged. `clone()` and ordinary C++ copy construction create independent binding snapshots. Binding a child also snapshots it: later edits to the original child do not change an already attached occurrence.

```cpp
glrmask::Vocab composed_vocab({{0, "["}, {1, "hello"}, {2, "]"}, {3, "[hello]"}});
auto parent = glrmask::Grammar::from_glrm(
    R"(glrm 1; start root; extern grammar payload;
        nt root = "[" payload "]";)");
auto child = glrmask::Grammar::from_ebnf(R"(start ::= "hello")");
parent.bind("payload", child);
auto runnable = parent.compile(composed_vocab);
```

For reusable compiled machinery, prepare the unbound parent and child separately, bind the prepared child, and link a runnable root:

```cpp
auto parent_module = glrmask::Grammar::from_glrm(
    R"(glrm 1; start root; extern grammar payload;
        nt root = "[" payload "]";)").prepare(composed_vocab);
auto child_module = child.prepare(composed_vocab);
parent_module.bind("payload", child_module);
auto runnable = parent_module.link();
```

Preparation can retain open slots, whereas compilation and final linking require a closed runnable root. Binding does not itself perform a hidden final compilation/link.

`requirements()` returns owned `Requirement` records with `name`, component `path`, `kind`, and an optional `source_span`. Qualified descendant names use `::`, for example `payload::TOOL`. A span has a half-open source byte range plus one-based line and Unicode-scalar column. Inspect actual requirements instead of inventing generated paths.

Source-stage bindings accept `Grammar`, `Token`, or `ExactTokens`. Prepared-stage bindings accept `PreparedGrammar`, `Token`, or `ExactTokens`. A runnable `Constraint` is not a public prepared-stage binding argument in this interface.

## Exact tokens and ranges

`Token(id)` is a raw original-ID event; the grammar's eventual vocabulary resolves it. `vocab.tokens(ids)` produces vocabulary-qualified `ExactTokens`, preserving identity even when other tokens have the same bytes.

```cpp
auto selected = vocab.tokens({31});
auto event = glrmask::Grammar::exact(selected);
auto control = glrmask::Grammar::from_glrm(
    R"(glrm 1; start root; extern token CONTROL; nt root = CONTROL;)");
control.bind("CONTROL", glrmask::Token(99));
```

`vocab.token_ranges({{first, last}, ...})` uses **inclusive endpoints** in C and C++. Overlapping or adjacent ranges normalize without expanding all IDs. A reversed interval is rejected. `ExactTokens::ranges()` stays compact; `ids()` intentionally materializes IDs. `size()`, `empty()` and `clone()` are also available. An empty exact-token set denotes an empty event language, not epsilon. Each match consumes one chosen token event, not a sequence of the set's IDs.

Exact-token events cannot be simulated by committing their decoded bytes. Use `commit_token` for exact controls and end tokens. `commit_bytes` advances ordinary byte input only.

## State, termination and markers

`Constraint` is immutable and reusable. `start()` creates a new sequence state; `ConstraintState::clone()` takes an independent parser snapshot. A state keeps the native constraint alive after its C++ owner is destroyed, moved from or disposed. Each sequence requires its own state.

`is_accepting()` means the current prefix is a complete match. It does **not** necessarily mean generation must stop: the grammar may permit continuation. Callers can stop at acceptance, or configure root `end_tokens` and stop when one is sampled and committed. An allowed end token leaves the state accepting and non-rejected, empties its next mask, and prevents further commits. End-token policy belongs to the root, not embedded children.

`is_rejected()` reports an irrecoverably rejected prefix. A failed grammar commit follows the core's state semantics; do not assume it is a general rollback operation. For speculative branches, retain a clone before committing. Argument validation and undersized output-buffer errors do not perform a commit.

`forced_tokens()` performs read-only, bounded lookahead and returns a forced token prefix when the core can determine one. It does not advance the state. The core can return a forcing error rather than loop indefinitely.

`Constraint::marker_ids()` returns the marker inventory. `ConstraintState::marker_events()` returns an owned copy of the initial or latest commit's marker observations. A `MarkerEvent` contains `id` and a `uint64_t byte_offset`. These are possible recognition observations, including those from ambiguous branches; they are not final parse captures or a parse-tree API. Cloning preserves the current batch.

## Persistence

```cpp
auto bytes = constraint.save();
auto restored = glrmask::Constraint::load(bytes);

auto external = constraint.save(true);  // Omit vocabulary storage.
auto rebound = glrmask::Constraint::load(external, &vocab);

auto module_bytes = parent_module.save();
auto module = glrmask::PreparedGrammar::load(module_bytes, &vocab);
```

These use the Rust core's artifact formats, not C++ object serialization. With the same compatible GLRMask build, artifacts can be exchanged with Rust, Python and the Node binding. Exact vocabulary identity is validated when a vocabulary is supplied; an external-vocabulary artifact cannot load without it. Equal vocabulary size alone is insufficient.

Artifacts remain versioned core implementation formats. Do not assume compatibility across GLRMask releases, architectures or unrelated builds. Persist source/configuration as well when long-term regeneration matters. Treat untrusted artifacts according to the core's loader limitations; language-boundary checks do not turn them into a portable, security-hardened file format.

## Errors and ownership

Fallible C++ operations throw `glrmask::Error`, whose `status()` is the underlying C status. `what()` owns the diagnostic text. C++ exceptions never cross the C ABI. Ordinary copies clone a handle; moves transfer it. Access to a moved-from or disposed C++ object throws `GLRMASK_CLOSED`. `dispose()` and normal RAII destruction release ownership; repeated C++ disposal is harmless.

A Rust unwinding panic is contained at the boundary and reported as `GLRMASK_PANIC`. A state affected by such a panic is unusable and must be discarded. A process abort, allocation failure, stack overflow or invalid foreign memory is not recoverable by this mechanism.

Immutable compiled constraints can be shared between threads, with independent states. Do not concurrently mutate a grammar, prepared module or sequence state, or destroy a handle while another thread uses it. The C++ wrapper does not add synchronization to mutable handles.

## C interface reference

The complete signatures and ABI memory contract are in [include/glrmask.h](include/glrmask.h). The header is valid C and does not depend on C++ types. Its opaque handles and owning result objects avoid exposing Rust layouts, Rust strings, C++ exceptions or allocator-specific containers.

| Family | Operations |
| --- | --- |
| Library | `glrmask_abi_version`, `glrmask_version`, `glrmask_error_message`, default build/Lark options |
| Vocabulary | `glrmask_vocab_new`, clone, length, tokens, token ranges, free |
| Exact token sets | Clone, length, materialized IDs, compact inclusive ranges, free |
| Source grammars | Create from source/exact set, clone, bind grammar/token/set, requirements, prepare, compile, free |
| Prepared grammars | Clone, bind prepared/token/set, requirements, link, save/load, free |
| Constraints | Clone, start, mask length, marker inventory, save/load, free |
| Sequence states | Clone, mask length, fill mask, commit token/bytes, acceptance/rejection, forcing, marker events, free |
| Result objects | Bytes/u32/event data views; requirement count/indexed views; matching frees |

All text is UTF-8 with an explicit byte length and does not need a terminating NUL. Vocabulary and committed bytes are arbitrary binary data. Lengths are element counts unless named as byte lengths. Input pointers are borrowed for the duration of a call; returned handles own their data. State handles independently retain their constraints.

Free each owning C handle exactly once using its matching `_free` function; freeing `NULL` is allowed. A result's `_data`/`_get` view is borrowed until that result handle is freed. Do not free the view, and do not use a C++ allocator on Rust-owned memory. Build and Lark option structs must have the `struct_size` returned by their default constructors. A null build-options pointer selects defaults.

| Status | Meaning |
| --- | --- |
| `GLRMASK_OK` (0) | Success. |
| `GLRMASK_INVALID_ARGUMENT` (1) | Invalid pointer/length, UTF-8, option or vocabulary input. |
| `GLRMASK_ENGINE_ERROR` (2) | A grammar, state, vocabulary-identity or artifact error from the core. |
| `GLRMASK_PANIC` (3) | An unwinding internal panic was contained. Discard affected state. |
| `GLRMASK_BUFFER_TOO_SMALL` (4) | Caller output capacity is insufficient. |
| `GLRMASK_CLOSED` (5) | A disposed/moved-from high-level object was used. Raw freed pointers are invalid, not safely detectable closed handles. |

Output handles are set to `NULL` before a fallible construction. On failure, immediately copy `glrmask_error_message()` if needed: its NUL-terminated, thread-local text is borrowed only until the next GLRMask call on that thread. Successful calls clear it. Optionally querying an exact-token range count with null output and zero capacity does not expand IDs.

The boundary checks nulls, alignment and overflowing lengths, but cannot validate arbitrary addresses or detect use-after-free. C callers must supply live, correctly typed handles and valid, non-overlapping memory. Do not cast unrelated addresses to handles or overlap mutable output with input/handle storage.

## Validation and scope

The repository contains direct Rust-versus-C mask/commit tests, pure C ABI checks, C++ state/composition/persistence/marker/error/thread tests, and C++/Node artifact exchange tests. CMake installation and an external relocated consumer are part of the native validation workflow. The same C++ tests run for static and shared builds in the platform matrix.

This interface covers the current public grammar/constraint lifecycle. It does not expose private parser tables, benchmark internals, a Python-only tokenizer adapter, model inference, or browser WebAssembly. The C ABI revision is currently 1; use the matching package's headers and library rather than assuming that future pre-1.0 interfaces or artifact formats are frozen.
