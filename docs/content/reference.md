### Isolated components and typed bindings

GLRM distinguishes two external contracts. `extern token NAME` consumes one exact
model-token event from a bound ID set. `extern grammar NAME` invokes a separately
scoped grammar, which can contain bytes, exact events, or epsilon. Binding a value
of the wrong kind is an error; an exact ID is never replaced by its decoded text.

```rust
use glrmask::{Grammar, Module, Result, Vocab};

# fn demo(vocab: &Vocab, end_id: u32) -> Result<()> {
let grammar = Grammar::from_glrm(r#"
    extern grammar PAYLOAD
    extern token END
    start response
    nt response = PAYLOAD END
"#);
let mut parent = Module::new(&grammar);
let child = Grammar::from_json_schema(r#"{"type":"null"}"#);
parent.bind("PAYLOAD", &child)?;
parent.bind("END", glrmask::Token::new(end_id))?;
let constraint = parent.compile(vocab)?;
# let _ = constraint;
# Ok(())
# }
```

Use `vocab.tokens(ids)` or `vocab.token_ranges(ranges)` for exact sets. Rust ranges
are inclusive; Python `range` stops are exclusive. The set consumes **one** ID,
not a sequence. Sparse vocabulary holes and incompatible vocabulary identities
are errors. IDs with the same spelling remain distinct, and spelling-less IDs
are not epsilon. See [exact-token ranges](docs/exact-token-ranges.md).

A grammar-valued extern also accepts `Token` or `ExactTokens` directly: this
widens an exact event into a general fragment without changing its meaning.
Neither conversion uses decoded token spelling.
An arbitrary grammar cannot bind a token-valued extern.

### Configure, precompile, attach and execute

`Module` is a reusable, independently configured grammar snapshot. Constructing
one from a `Grammar` needs no vocabulary and does no grammar compilation.
Compilation settings belong to the module, never to the source `Grammar`.
The parent may have unresolved externs while precompiling.
`bind(name, value)` mutates the receiving handle and returns `Result<()>` in Rust
or `None` in Python. In Python, `bind` also supports atomic keyword arguments
(`m.bind(**bindings)`).
Each successful bind publishes a new shared snapshot; an error leaves the previous
configuration intact. Attaching a child captures its current configuration, not
a live reference to later edits. Every child must have all of its externs bound
before it is attached. In Rust, use `clone()` to specialize an independent
snapshot; in Python, use `copy()`.

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({
    0: b"x", 1: b"|", 2: b"x", 4: b"", 5: b"",
})
leaf1 = glrmask.Module(glrmask.Grammar.from_glrm('''
    extern token MARK
    start body
    nt body = "x" MARK
'''))
leaf1.bind("MARK", vocab.tokens([4]))
leaf1.precompile(vocab)

leaf2 = glrmask.Module(glrmask.Grammar.from_glrm('''
    extern token MARK
    start body
    nt body = "x" MARK
'''))
leaf2.bind("MARK", vocab.tokens([5]))
leaf2.precompile(vocab)

parent = glrmask.Module(glrmask.Grammar.from_glrm('''
    extern grammar LEFT
    extern grammar RIGHT
    start pair
    nt pair = LEFT "|" RIGHT
'''))
parent.precompile(vocab)

package = parent.copy()
assert list(package.externs.unbound) == ["LEFT", "RIGHT"]
package.bind(LEFT=leaf1, RIGHT=leaf2)
assert len(package.externs.unbound) == 0

constraint = package.compile(vocab)
state = constraint.start()
for token_id in [0, 4, 1, 2, 5]:
    assert state.mask()[token_id]
    state.commit_token(token_id)
assert state.is_accepting()
```

Rust uses `Module::new(&grammar)`, `precompile(&vocab)?`,
`bind(name, &child)?`, and `compile(&vocab)?` or `compile_with(&vocab, options)?`.
Module settings are configured via `set_mask_strategy` and
`set_boundary_mask_strategy` (in Python: constructor keywords or properties
`m.mask_strategy = ...`). Each explicitly configured child keeps its own
settings; a raw grammar child receives default local settings, not an inherited
parent override. Every child must be closed (have no unbound externs) before
attachment. Each extern is bound once on the module that declared it; qualified
`::` paths are rejected.

Inline grammars (`g name = { ... }`) cannot refer to anything outside
themselves, including outer rules, terminals, or externs. An outer reference
is an undefined-name error. The keywords `use`, `export`, `import`, and `super`
are reserved.

Rust `m.externs().unbound()` (Python: `m.externs.unbound`) reports unresolved
externs. Rust `m.externs()` (Python: `m.externs`) reports all declared externs
in declaration order. Introspection uses the `Extern` type (`name()`, `kind()`,
`span()`). Precompilation is allowed while externs remain open; final
`compile()` requires every declared extern to be bound.

### Persistence and root policy

Modules can be saved while incomplete, including mixed raw and precompiled
children. Retained work, settings and source identities survive the round-trip:

```python
saved = package.save()
restored = glrmask.Module.load(saved, vocab=vocab)
assert len(restored.externs.unbound) == 0
```

Runnable constraints separately support `save()` and `Constraint.load(...)`.
Artifacts preserve original token coordinates and vocabulary identity, not merely
a set of decoded spellings. Unsupported older pre-release formats are rejected.
Retained compiled work is vocabulary-specific, not portable across arbitrary models.
The vocabulary is supplied once to each root `precompile` or `compile` operation,
not once per child. Final compilation handles outstanding work, so attaching a
cold child never requires a separate manual precompilation step. Incompatible
retained representations or targets produce errors, not implicit loss of work.

Configure `end_tokens` on the final compile operation. Acceptance means a
complete grammar match exists; it does not require generation to stop when longer
matches remain possible. A configured end ID is admitted only at acceptance. Commit
the sampled end ID before ending the generation loop. Child components do not
supply the enclosing generation's root termination policy.

### Lightweight expressions

`Expr.choice` is unordered context-free alternation, and `Expr.sequence` is
language concatenation. In Python, both accept positional parts:
`Expr.choice(*parts)` and `Expr.sequence(*parts)`. `Expr.literal` matches exact
bytes without whitespace or ignore inserted between parts. Leaves can be
`Grammar`, `Module`, `Expr`, `Token`, or `ExactTokens`. In Python, passing a
plain string raises `TypeError` (use `Expr.literal`). Parts with open externs
raise an error immediately. An expression can be bound directly into a
module extern (`m.bind("name", expr)`). Whole grammar and module leaves keep
their own lexical scopes, bindings and settings. Intermediate expression
operators add neither scope nor public path segments.

### Determined output

`state.forced_tokens()` returns a read-only library-canonical prefix for genuinely
forced ordinary text. It greedily chooses the longest admitted nonempty spelling,
then the lowest original-ID alias. Exact-token events retain their IDs, and the
query stops before root acceptance or an intermediate event opportunity that a
greedy text token would cross. The correctness-first implementation may construct
a full vocabulary mask. Rust returns `Result<Vec<u32>>`; Python returns `list[int]`
and raises errors for typed proof/resource failures. The result is one canonical
sequence, not an exhaustive tree of equivalent token histories. See
[forcing contracts](docs/forcing.md).

### Streaming markers

Place `@mark(id)` in a parser expression to observe possible input boundaries.
`constraint.marker_ids()` lists the numeric IDs. `state.marker_events()` returns
the events from `start()` or the most recent commit, including markers inside a
model token. Copy a batch before the next commit when retaining history.

Markers only read the parsing frontier; they never consume a model token or
select a derivation. An event can come from an alternative that later fails.
There is no automatic pairing, capture nesting, or ambiguity resolution. The
[marker reference](docs/markers.md) defines ignored-input behavior, byte offsets,
composition, persistence, and the Rust/Python APIs.

### Build/runtime trade-off

{{dynamic}}

`MaskStrategy` trades off compilation time against per-token masking latency:

- `ParserDwa` (alias `FAST_RUNTIME`): static parser-DWA masks (slowest build, fastest runtime masks).
- `TokenTrie` (alias `FAST_BUILD`): dynamic token-trie walk (fastest build).
- `TokenClassTrie`: token-trie walk over token equivalence classes.
- `Auto`: library chooses the strategy (default).

`BoundaryMaskStrategy` sets the minimum metadata detail to build for composition boundaries:

- `FastBuild`: conservative token-set boundary trigger (faster build, default).
- `FastRuntime`: full exact proper-prefix boundary trigger (lower latency).

All modes preserve accepted-language semantics and produce the same public `Constraint` type.

### Native template parser backend

Public final-constraint APIs select their parser representation internally.
Callers choose `MaskStrategy`; there is no public parser backend selector or
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

### TokenTrie backend

`MaskStrategy::TokenTrie` uses the retained LR runtime. Its grammar, lexer,
vocabulary, masking and commit code share the existing compiler/runtime pipeline;
it skips native parser characterization, template construction and native runtime
metadata. `TokenClassTrie` continues to build templates, and `ParserDwa` is unchanged.

For internal development only, `GLRMASK_DYNAMIC_TEMPLATE_DFA=1` selects native
parser templates for an ordinary Dynamic compilation (`true`, `yes` and `on`
also enable it). An absent/disabled variable selects LR. This choice is resolved
before compiler worker pools and does not affect O2, Static, existing objects or
artifact loading. There is no new public backend configuration. Serialized
artifacts preserve their built representation and external-vocabulary artifacts
require the exact original mapping.


## Grammar formats

**JSON Schema acceptance policy.**
Compilation enforces supported assertions or reports an unsupported-feature error.
An absent `$schema` selects Draft 2020-12; explicit Draft 4, 6, 7, 2019-09, and
2020-12 declarations select their dialect semantics. `format` is annotation-only;
required format-assertion and unknown required vocabularies are unsupported.
`oneOf` remains exclusive, with no approximation mode. Existing canonical
serialization limits remain documented in the [acceptance policy](https://github.com/IsaacBreen/glrmask/pull/13).

GLRM is the native recognition language. A version header and statement semicolons
are not required; canonical source uses explicit declarations and insignificant
line breaks:

```glrm
start value
internal t DIGIT = /[0-9]/
t INTEGER = /-?(0|[1-9][0-9]*)/
nt value = INTEGER | "null"
```

It retains `internal t`, unordered context-free alternatives, scoped grammars,
regular intersection/subtraction, structural nonterminal-alternative subtraction,
separated sequences, explicit finite automata and Unicode/byte expressions.
Native `shortest { ... }` notation is removed; the internal prefix-minimal
regular-language operation remains available to the Lark importer and can be
expressed with native regular subtraction. See [the native format](docs/glrm-format.md).

Lark is a recognition-focused frontend with an explicit optional LLM compatibility
profile. That profile adds selected exact-token syntax, named token resolution,
`@name` grammar bindings and isolated inline `%json` schemas. Those literal families
are **not native GLRM syntax**; native callers provide equivalent components through
typed host bindings. The regular-body `[lazy]` annotation uses the same shortest
terminal operation. See [the Lark dialect](docs/lark-dialect.md).

Regex-internal reluctant quantifiers, priorities, captures, substring generation,
sampling annotations and grammar token/character budgets are not supported.
Unsupported semantic features produce errors rather than being silently discarded.
The Lark frontend does not claim Python regex endpoint ordering or compatibility
with every Python Lark lexer mode.
