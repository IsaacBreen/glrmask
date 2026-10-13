### Source, compiled, and exact-token bindings

GLRM declares external references with `extern NAME;`. The value supplied to `Grammar::bind` determines whether a reference consumes exact token IDs or matches a source grammar. Attach compiled children with `UnlinkedConstraint::bind`:

```rust
use glrmask::{Grammar, Result, Vocab};

# fn demo(vocab: &Vocab) -> Result<()> {
let parent = Grammar::from_glrm(
    "glrm 1; start document; extern payload; nt document = payload;",
);
let source_child = Grammar::from_json_schema(r#"{"type":"null"}"#);
let source_composition = parent.bind("payload", &source_child)?;

let _a = source_composition.compile(vocab)?;
# Ok(())
# }
```

Use `Token::new(id)` to bind one exact token ID. Compilation checks it against the actual vocabulary:

```rust
# use glrmask::{Grammar, Result, Vocab};
# fn demo(vocab: &Vocab, tool_call_id: u32) -> Result<()> {
let grammar = Grammar::from_glrm(
    "glrm 1; start message; extern TOOL_CALL; nt message = TOOL_CALL;",
);
let grammar = grammar.bind("TOOL_CALL", glrmask::Token::new(tool_call_id))?;
let _constraint = grammar.compile(vocab)?;
# Ok(())
# }
```

For a set of possible IDs, use `Vocab::tokens(ids)` or compact inclusive Rust ranges:

```rust
# use glrmask::{Grammar, Result, Vocab};
# fn demo(vocab: &Vocab) -> Result<()> {
let selection = vocab.token_ranges([0..=999, 2000..=2999])?;
let grammar = Grammar::from_glrm("glrm 1; start message; extern T; nt message = T;")
    .bind("T", selection)?;
let _constraint = grammar.compile(vocab)?;
# Ok(())
# }
```

Python retains contiguous `range` inputs without expanding their IDs. Python stops are exclusive:

```python
all_tokens = vocab.tokens(range(vocab_size))
selection = vocab.token_ranges([range(0, 1000), range(2000, 3000)])
constraint = grammar.bind("T", selection).compile(vocab)
```

A set consumes **one** token selected from the set, not a token sequence. Overlapping and adjacent ranges are normalized; undefined IDs, including gaps inside a range, are rejected. Set bindings retain their vocabulary identity. A set from an incompatible vocabulary is rejected even if its numeric IDs happen to exist in both vocabularies. `Token::new(id)` itself remains an unqualified single ID.

`ExactTokens::ranges()` (Python: `.ranges`) and `.len()` (Python: `len(...)`) inspect the compact set. `.ids()` (Python: `.ids`) deliberately materializes the IDs. Grammar transitions scale with the number of ranges, not the number of selected IDs; vocabulary preparation and writing a dense output mask still have their usual costs. The final root's end-token policy still applies, even when an end-token ID lies inside a selected range.

Model control/tool tokens do not need fake byte spellings. When constructing a
Rust vocabulary manually, use `Vocab::new_with_exact_token_ids(byte_entries,
exact_only_ids)` for IDs that are valid exact-token bindings but must never
enter the byte language. Python's `Vocab.from_llama_cpp()` records omitted
control/EOG/empty-piece IDs this way automatically.

### Cached parents with `UnlinkedConstraint`

Use `compile_unlinked` when a compiled parent will be reused with request-specific children. A `UnlinkedConstraint` may remain open, can be saved and loaded, and is deliberately not runnable.

```rust
# use glrmask::{BuildOptions, Grammar, Optimization, Result, Vocab};
# fn demo(vocab: &Vocab) -> Result<()> {
let parent = Grammar::from_glrm(
    "glrm 1; start document; extern payload; nt document = payload;",
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
# Ok(())
# }
```

`UnlinkedConstraint::bind` is compiled-only: it accepts a compiled `Constraint` or an exact-token value. It does not accept an unresolved `UnlinkedConstraint`, and it does not parse or compile source children. Link a child first before binding it to the parent. Composition stays deferred until `link`/`link_with`, so the final optimization preference can choose the boundary construction strategy.

Python uses the same lifecycle:

```python
parent = glrmask.Grammar.from_glrm(
    'glrm 1; start document; extern payload; nt document = payload;'
)
host = parent.compile_unlinked(vocab)
child = glrmask.Grammar.from_json_schema(payload_schema).compile(vocab)
constraint = host.bind("payload", child).link(
    optimization=glrmask.Optimization.FAST_RUNTIME,
)
```

### Build/runtime trade-off

{{dynamic}}

`Optimization` expresses intent rather than exposing internal engine names:

- `AUTO` / `Auto`: let GLRMask choose.
- `FAST_BUILD` / `FastBuild`: ordinary Dynamic compilation and masking (O1).
- `BALANCED` / `Balanced`: bounded native parser with vocabulary partitioning (O2).
- `FAST_RUNTIME` / `FastRuntime`: spend more build work to favor lower mask latency where supported.

All four modes preserve accepted-language semantics and produce the same public `Constraint` type.

### Native template parser backend

Public final-constraint APIs select their parser representation internally.
Callers choose `Optimization`; there is no public parser backend selector or
accessor. Parser-provider construction remains internal repository tooling.

The native runtime and its artifact contain the template relations, not an
LR table. Mask generation and token commitment use the existing shared engines;
parser advancement and admissibility use those relations. Built-in grammar
compilation can use temporary LR analysis to derive the program, then discards
the table before constructing a `Constraint`. Native runtime table access and implicit LR fallback panic. Ordinary Dynamic
(O1), selected by FastBuild, retains and executes its LR table.
Data-only parser providers remain available through Rust's `internal-api`
feature and Python's `_internal` namespace.

Built-in compiled components with finite embedding transfers support nested
and nullable [template composition](docs/template-parser-composition.md).
Compiled components link directly through shared immutable template graphs and
scoped views. Unsupported native requests return errors. Performance and load-time tradeoffs depend on the
grammar and mode, so template selection is not an automatic speed guarantee.
See [the template parser contract](docs/template-parser.md) for provider
examples, exact POP/READ/PUSH semantics, validation, and persistence.
The [September 30 validation report](docs/template-parser-validation-2026-09-30.md)
records historical measurements at its pinned revision. Those measurements do
not qualify the current native replacement.

### Ordinary Dynamic development backend

Ordinary Dynamic (O1) defaults to the retained LR runtime. Its grammar, lexer,
vocabulary, masking and commit code share the existing compiler/runtime pipeline;
it skips native parser characterization, template construction and native runtime
metadata. O2/Balanced continues to build templates, and Static is unchanged.

For internal development only, `GLRMASK_DYNAMIC_TEMPLATE_DFA=1` selects native
parser templates for an ordinary Dynamic compilation (`true`, `yes` and `on`
also enable it). An absent/disabled variable selects LR. This choice is resolved
before compiler worker pools and does not affect O2, Static, existing objects or
artifact loading. There is no new public backend configuration. Serialized
artifacts preserve their built representation and external-vocabulary artifacts
require the exact original mapping.

### End tokens are final-root policy

End-token policy belongs to the final `compile`/`compile_with` or `link`/`link_with` operation. It is not inherited when a completed `Constraint` is embedded as a child.

```python
constraint = grammar.compile(vocab, end_tokens=[eos_id])
state = constraint.start()
# eos_id is allowed only when the grammar body is accepting.
```

A state reports `is_accepting()` for grammar-body acceptance and `is_rejected()` for an irrecoverable invalid prefix. Callers choose between stopping at the first complete match with `is_accepting()` or permitting continuation until a configured end-token ID is sampled and committed. Always commit the chosen token before checking for an end-token break; acceptance alone does not mean an end token was consumed. Committing an allowed end token empties the next-token mask, keeps the state accepting and non-rejected, and rejects further commits.

### Persistence

Both compiled object types are serializable:

```python
module_bytes = host.save()
host = glrmask.UnlinkedConstraint.load(module_bytes)

constraint_bytes = constraint.save()
constraint = glrmask.Constraint.load(constraint_bytes)
```

A loaded `Constraint` retains its selected backend. Built-in template-backed children retain their finite embedding transfers and can be linked with the template backend; arbitrary data-only and older artifacts may lack those transfers. A child's standalone end-token policy is stripped when embedded, while its compiled grammar body is retained.

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

```python
grammar = glrmask.Grammar.from_glrm('''
glrm 1;
start message;
extern TOOL_CALL;
nt message = TOOL_CALL "lookup()";
''')
constraint = grammar.bind("TOOL_CALL", glrmask.Token(tool_call_token_id)).compile(vocab)
```

Lark and EBNF also support explicit `@token(<id>)` atoms when the numeric ID is deliberately part of the grammar source.
