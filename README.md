# GLRMask

**GLRMask: grammar-constrained token masking for context-free languages.**

GLRMask compiles a grammar together with an LLM vocabulary into an immutable `Constraint`, then produces the next-token mask for each decoding step and advances the constraint state as sampled tokens are committed. The Rust crate, PyPI distribution, and Python import are all named **`glrmask`**:

```text
Rust crate:    glrmask
PyPI package:  glrmask
Python import: glrmask
```

The core contract is tokenization-complete existential admissibility. For a current byte prefix `u`, a vocabulary token `v` with byte spelling `β(v)`, and compiled language `L`, the mask admits `v` exactly when some continuation `w` exists such that `u β(v) w ∈ L`. A token may therefore cross lexer-token or grammar-terminal boundaries and still be admissible.

GLRMask supports general context-free grammars, not only regular languages, and moves much of the stack-dependent token-admissibility work out of the per-token mask query and into compilation.

## What it supports

- **Tokenization-complete grammar-constrained masking.** For the compiled grammar and byte-backed vocabulary, a token is admitted exactly when appending its bytes leaves at least one completion in the compiled language.
- **General context-free constraints.** Recursive and ambiguous grammar structure is supported through the parser-based compilation and runtime.
- **Grammar inputs.** EBNF and Lark grammars are supported directly.
- **JSON Schema.** A pragmatic JSON Schema subset is supported, with documented semantic deviations. Full JSON Schema conformance is **not** claimed.
- **Compile once, run incrementally.** A compiled `Constraint` is immutable; `constraint.start()` creates mutable state for one generation stream.
- **Serializable constraints.** Compiled constraints can be saved and loaded without recompiling the source grammar.
- **Python and Rust APIs.** The same core compiler and runtime are exposed through the `glrmask` Rust crate and Python package.

## Installation

### Python

Install the v0.1 release from PyPI:

```bash
python -m pip install glrmask==0.1.0
```

On platforms with a published wheel, `pip` installs the native extension directly. Building from source requires Python, a Rust toolchain, and the platform's native linker/build tools. From a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./python
```

`pip` installs the Python runtime dependency (`numpy`) and uses Maturin in an isolated build environment when a source build is required.

### Rust

Add the v0.1 crate from crates.io:

```bash
cargo add glrmask@0.1.0
```

or add it directly to `Cargo.toml`:

```toml
[dependencies]
glrmask = "0.1.0"
```

## Minimal Python example

A vocabulary maps the exact token bytes used by the model to token IDs. Compile a grammar against that vocabulary, create a state, ask for the next-token mask, and commit the sampled token:

```python
import glrmask

vocab = glrmask.Vocab.from_dict({
    b"hello": 0,
    b" ": 1,
    b"world": 2,
})

constraint = glrmask.Constraint.from_ebnf(
    'start ::= "hello" " " "world"',
    vocab,
)
state = constraint.start()

assert state.mask().tolist() == [True, False, False]
state.commit_token(0)

assert state.mask().tolist() == [False, True, False]
state.commit_token(1)

assert state.mask().tolist() == [False, False, True]
state.commit_token(2)

assert state.is_finished()
```

The Python `mask()` convenience method returns a Boolean NumPy array in original token-ID space. For serving code that already owns a mask buffer, `fill_mask(...)` is also available.

The same example is kept as an executable file:

```bash
python examples/python_quickstart.py
```

## Rust quickstart

The Rust API returns a packed `u32` bitmask. Bit `token_id % 32` in word `token_id / 32` indicates whether that token is currently admissible.

```rust
use glrmask::{Constraint, Vocab};

fn token_allowed(mask: &[u32], token_id: usize) -> bool {
    let word = token_id / 32;
    word < mask.len() && ((mask[word] >> (token_id % 32)) & 1) != 0
}

fn main() {
    let vocab = Vocab::new(
        vec![
            (0, b"hello".to_vec()),
            (1, b" ".to_vec()),
            (2, b"world".to_vec()),
        ],
        None,
    );

    let constraint = Constraint::from_ebnf(
        r#"start ::= "hello" " " "world""#,
        &vocab,
    )
    .unwrap();

    let mut state = constraint.start();
    assert!(token_allowed(&state.mask(), 0));
    state.commit_token(0).unwrap();
    assert!(token_allowed(&state.mask(), 1));
    state.commit_token(1).unwrap();
    assert!(token_allowed(&state.mask(), 2));
    state.commit_token(2).unwrap();
    assert!(state.is_finished());
}
```

The repository version can be run with:

```bash
cargo run --example ebnf
```

## Structured output with JSON Schema

JSON Schema input is useful for tool arguments and other structured outputs. This small example constrains one object field to an enum and shows the token mask changing as JSON fragments are committed:

```python
import glrmask

vocab = glrmask.Vocab.from_dict({
    b'{"action": "': 0,
    b"search": 1,
    b"answer": 2,
    b'"}': 3,
})

schema = r'''
{
  "type": "object",
  "properties": {
    "action": {"enum": ["search", "answer"]}
  },
  "required": ["action"],
  "additionalProperties": false
}
'''

constraint = glrmask.Constraint.from_json_schema(schema, vocab)
state = constraint.start()

assert state.mask().tolist() == [True, False, False, False]
state.commit_token(0)
assert state.mask().tolist() == [False, True, True, False]
state.commit_token(1)
assert state.mask().tolist() == [False, False, False, True]
state.commit_token(3)
assert state.is_finished()
```

**JSON Schema support is a pragmatic subset, not full specification conformance.** Unsupported constructs may be rejected, and some documented cases deliberately broaden or restrict the accepted instances for tractability. In particular, some expensive patterned-string cases may drop an upper `maxLength` bound, and open-object lowering can impose ordering restrictions on additional or pattern-matched properties. Read [JSON Schema semantic deviations](docs/json-schema-semantic-deviations.md) before relying on exact equivalence to a source schema.

A small Rust JSON Schema example is also available:

```bash
cargo run --example json_schema
```

## A genuinely context-free example

The classic language `a^n b^n` for `n >= 1` is not regular. It requires matching an unbounded number of `a` symbols with the same number of following `b` symbols. A finite-state constraint cannot represent the language exactly for unbounded `n`; a context-free grammar can:

```text
start ::= "a" start "b" | "a" "b"
```

With vocabulary tokens `a` and `b`, after the first `b` is committed the grammar no longer admits another `a`; it must close exactly the number of recursive levels that were opened.

```bash
cargo run --example context_free
```

See [`examples/context_free.rs`](examples/context_free.rs) for the complete executable example.

## Precompiled constraints and runtime state

Compilation is vocabulary-specific: the compiler reasons about the actual token byte strings that may be sampled. A constraint built for one tokenizer vocabulary must not be reused with a different vocabulary.

For repeated use, compile once and serialize the result:

```rust
let bytes = constraint.save();
let restored = Constraint::load(&bytes).unwrap();
let mut state = restored.start();
```

The repository also contains an execution-only runtime crate for loading versioned, vocabulary-specific runtime artifacts without carrying the grammar import and compilation pipeline into the serving process.

## How it works

At a high level, token validity depends on two interacting questions:

1. **Lexical:** from the current lexer state, what grammar-terminal sequences can a candidate LLM token produce, including tokens that cross terminal boundaries?
2. **Syntactic:** given the current parser stack, which of those terminal sequences can be accepted while keeping the grammar completable?

`glrmask` resolves much of that interaction ahead of time:

1. The input grammar is parsed and lowered, and the compiler builds lexical automata plus a GLR parser table.
2. For the concrete LLM vocabulary, token bytes are related to the terminal sequences they can produce from relevant lexer states.
3. The compiler combines that lexical information with the parser's per-terminal stack effects and builds a deterministic weighted automaton over parser-stack symbols (internally, the **Parser DWA**). Its Boolean weights encode the lexer-state/token pairs compatible with each path.
4. At runtime, constraint state tracks the active lexer state or states together with reachable parser stacks. A mask query evaluates the compiled automaton against those stacks and materializes the surviving vocabulary tokens. Committing a token advances the lexical and parser state for the next step.

The key design tradeoff is therefore deliberate: spend more work when a grammar and vocabulary are compiled so that repeated online mask queries can reuse a precomputed representation instead of reconstructing the same stack-dependent candidate behavior from scratch.

## Performance and benchmarks

Compilation cost and online decoding cost should be measured separately. The design is aimed at workloads where a compiled grammar is reused across many mask queries or generation streams; a one-shot grammar may value compilation latency differently from a long-running serving workload.

The v0.1 public benchmark is a bounded `make example-slow-all` comparison of the benchmark harness's `llguidance`, `glrmask`, `glrmask-native`, and `xgrammar` backends on one controlled machine. It is intentionally **not** the full benchmark corpus. Backend versions, machine details, exact methodology, failures or timeouts, and the measured results are recorded in [the v0.1 benchmark report](docs/benchmark-0.1.md).

A separate [10,263-problem CFA full-corpus engineering report](docs/benchmark-full-corpus-2026-07-16.md) records a later coverage-aware run. It uses different framework coverage and multiple `glrmask-main` revisions, so it must not be read as a replacement for the bounded release-tag benchmark.

The comparison is a performance measurement, not a declaration that one backend is semantic ground truth. Different constrained-decoding systems can intentionally expose different token-admissibility policies, so raw mask disagreements require separate correctness analysis.

## Other API features

### Lark input

```python
constraint = glrmask.Constraint.from_lark(lark_source, vocab)
```

The Rust API provides the corresponding `Constraint::from_lark(...)` constructor.

### Exact LLM token IDs

EBNF, Lark, and GLRM grammars can match an exact LLM token ID with `@token(<id>)`:

```text
start ::= "hello" @token(128009)
```

A special-token atom is matched only by committing that exact token ID. Its byte spelling, when present in the vocabulary, does not implicitly match the atom as ordinary bytes.

### State helpers

Rust `ConstraintState` includes:

- `mask()` / `fill_mask(...)`
- `commit_token(token_id)`
- `commit_tokens(&token_ids)`
- `commit_bytes(bytes)`
- `force()`
- `is_complete()` / `is_finished()`

Python exposes the corresponding incremental operations, with `mask()` returning a Boolean NumPy array.

## Limitations

- **JSON Schema is not fully conformant.** It is a pragmatic subset with [documented semantic deviations](docs/json-schema-semantic-deviations.md); some unsupported constructs error, while some documented cases broaden or restrict semantics.
- **Compiled constraints are vocabulary-specific.** Recompile when the tokenizer vocabulary or token-byte mapping changes.
- **Benchmark results are environment-specific.** The [v0.1 benchmark](docs/benchmark-0.1.md) records one bounded benchmark target on one machine; it is not the full corpus or a hardware-independent guarantee.
- **Serving-framework integrations are not part of v0.1.** GLRMask v0.1 ships the compiler/runtime library and public Rust/Python APIs. Direct integrations with serving systems such as vLLM are follow-up work.

## Examples

```bash
python examples/python_quickstart.py
cargo run --example ebnf
cargo run --example context_free
cargo run --example json_schema
```

## License

Licensed under either the MIT License or the Apache License, Version 2.0, at your option.
