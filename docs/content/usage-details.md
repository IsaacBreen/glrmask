### Bindings

GLRM uses `extern NAME;` for every external reference. Replace the former `extern token NAME;` and `extern grammar NAME;` declarations with this spelling; typed v1 declarations are rejected with migration guidance. Recompile saved unlinked modules from source: older typed-slot module formats are rejected. Compiled children still require an exactly compatible vocabulary.

GLRM declares external references with `extern NAME;`. The value supplied to `Grammar::bind` determines whether a reference consumes exact token IDs or matches a source grammar. Attach compiled children with `UnlinkedConstraint::bind`:

```rust
use glrmask::Grammar;

let parent = Grammar::from_glrm(
    r#"
    glrm 1;
    start document;
    extern payload;
    nt document = payload;
    "#,
);
let source_child = Grammar::from_json_schema(r#"{"type":"null"}"#);
let composed = parent.bind("payload", &source_child)?;
let _a = composed.compile(vocab)?;
```

Use `Token(id)` in Python or `Token::new(id)` in Rust to consume exactly one ordinary or special token ID, regardless of its text. The value stores no vocabulary. Compilation validates membership in the actual vocabulary; binding to an unlinked constraint validates against its compiled vocabulary. Sparse gaps and unknown IDs are rejected. Equal IDs across tokenizers carry no identity guarantee.

```rust
use glrmask::Grammar;

let grammar = Grammar::from_glrm(
    r#"
    glrm 1;
    start message;
    extern TOOL_CALL;
    nt message = TOOL_CALL;
    "#,
);
let token = glrmask::Token::new(tool_call_id);
let grammar = grammar.bind("TOOL_CALL", token)?;
let _constraint = grammar.compile(vocab)?;
```

To select one token from many IDs, bind `vocab.tokens(range(vocab_size))` in Python, or `vocab.token_ranges([0..=last_id])` in Rust. For disjoint sets, Python accepts `vocab.token_ranges([range(0, 1000), range(2000, 3000)])`; Rust uses inclusive ranges such as `[0..=999, 2000..=2999]`. Ranges stay compact through compilation, linking and masking rather than becoming one grammar alternative per ID. Overlaps are merged, and ranges containing undefined IDs are rejected. A set consumes one token, not a sequence. Inspect `.ranges` in Python or `.ranges()` in Rust without materializing all IDs.

Bindings are immutable. Calling `bind(...)` returns a new `Grammar` or `UnlinkedConstraint`; the original remains reusable.

Compilation fails if any required bindings are unresolved.

```rust
let parent = Grammar::from_glrm(
    r#"
    glrm 1;
    start document;
    extern payload;
    nt document = payload;
    "#,
);
let result = parent.compile(vocab);
if let Err(error) = result {
    println!("{error}");
}
```

```text
Compilation error: external "payload" is unbound; compile_unlinked() if a reusable pre-link artifact is intended
```

Composition is useful when your set of tools changes. You can compile the surrounding grammar and each tool schema once, then reuse them as tools are added or removed. Only the connections between them need to be rebuilt.

### Cached parents with `UnlinkedConstraint`

An `UnlinkedConstraint` lets you compile the surrounding grammar before filling its slots. A `UnlinkedConstraint` may remain open, can be saved and loaded, and is deliberately not runnable.

```rust
use glrmask::{BuildOptions, Grammar, Optimization};

let parent = Grammar::from_glrm(
    r#"
    glrm 1;
    start document;
    extern payload;
    nt document = payload;
    "#,
);
let host = parent.compile_unlinked(vocab)?;
let child_a = Grammar::from_ebnf(r#"start ::= "a""#).compile(vocab)?;
let child_b = Grammar::from_ebnf(r#"start ::= "b""#).compile(vocab)?;

let a = host.bind("payload", &child_a)?;
let b = host.bind("payload", &child_b)?;
let _constraint_a = a.link_with(
    BuildOptions::default().optimization(Optimization::FastRuntime),
)?;
let _constraint_b = b.link()?;
```

`UnlinkedConstraint::bind` is compiled-only: it accepts a compiled `Constraint` or an exact-token value. It does not accept an unresolved `UnlinkedConstraint`, and it does not parse or compile source children. Link a child first before binding it to the parent. Composition stays deferred until `link`/`link_with`, so the final optimization preference can choose the boundary construction strategy.

Python follows the same basic idea.

### Choosing an optimization mode

GLRMask provides three options with different trade-offs between compilation speed and mask generation. The best choice depends on how often you reuse a constraint.

- `FastBuild`: prioritize fast compilation. Useful when constraints change frequently or are used for short generations. This is the [Dynamic mode in the benchmarks](https://glrmask.com/#benchmarks).
- `Balanced`: a middle ground between compilation time and mask-generation speed.
- `FastRuntime`: do more work during compilation to generate masks faster. Useful when you reuse the same constraint across many generations. This is the [Static mode in the benchmarks](https://glrmask.com/#benchmarks).

`Auto` is the default: let GLRMask choose.

```rust
use glrmask::{BuildOptions, Optimization};

let constraint = grammar.compile_with(
    vocab,
    BuildOptions::default().optimization(Optimization::FastBuild),
)?;
```

### Ending generation

Pass the model’s end-token IDs when you compile a constraint. GLRMask allows those tokens only when the generated text satisfies the grammar. Configure them when you want to continue until EOS; stop when the sampled token is a configured end ID, committing that token first.

```rust
use glrmask::BuildOptions;

let options = BuildOptions::default().end_tokens([eos_id]);
let constraint = grammar.compile_with(vocab, options)?;
let mut state = constraint.start();

loop {
    let mask = state.mask();
    // Use your model's sampler.
    let token_id = sample(&mask);
    state.commit_token(token_id)?;
    if token_id == eos_id {
        break;
    }
}
```

`is_accepting()` tells you whether the text generated so far is a complete valid match, which may still permit continuation. To stop at the first complete match, check acceptance before each iteration:

```rust
let mut state = constraint.start();

while !state.is_accepting() {
    let mask = state.mask();
    let token_id = sample(&mask);
    state.commit_token(token_id)?;
}
```

An initially accepting grammar stops immediately, without generating a token. `is_rejected()` means the generated prefix cannot be completed to match the grammar.

When composing constraints, set `end_tokens` on the final compile or link call; a child constraint’s end tokens do not end the enclosing generation.

### Persistence

Both compiled object types are serializable:

```rust
use glrmask::{Constraint, UnlinkedConstraint};

let module_bytes = host.save();
let host = UnlinkedConstraint::load(module_bytes)?;

let constraint_bytes = constraint.save();
let constraint = Constraint::load(constraint_bytes.as_slice())?;
```

## Grammar formats

GLRMask accepts JSON Schema, GLRM, Lark, and EBNF. GLRM is the native composition format. A grammar begins with `glrm 1;` and a `start` declaration:

```glrm
glrm 1;
start value;

t NUMBER = /-?(0|[1-9][0-9]*)/;
nt value = NUMBER | "null";
```

Declarations use `=`, epsilon is written as `eps`, and regexes use full-match semantics. Inline `g name = { ... };` grammars and externally bound `extern name;` slots have the same language semantics, including scope-local ignores.

An external reference can consume one ordinary or special model-token ID when bound to `glrmask.Token(id)`:

```rust
let grammar = Grammar::from_glrm(
    r#"
    glrm 1;
    start message;
    extern TOOL_CALL;
    nt message = TOOL_CALL "lookup()";
    "#,
);
let token = glrmask::Token::new(tool_call_token_id);
let constraint = grammar.bind("TOOL_CALL", token)?.compile(vocab)?;
```

Lark and EBNF also support explicit `@token(<id>)` atoms when the numeric ID is deliberately part of the grammar source.
