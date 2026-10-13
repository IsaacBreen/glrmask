### Isolated components and typed bindings

GLRM distinguishes two external contracts. `extern token NAME` consumes one exact
model-token event from a bound ID set. `extern grammar NAME` invokes a separately
scoped grammar, which can contain bytes, exact events, or epsilon. Binding a value
of the wrong kind is an error; an exact ID is never replaced by its decoded text.

```rust
use glrmask::{Grammar, Result, Vocab};

# fn demo(vocab: &Vocab, end_id: u32) -> Result<()> {
let parent = Grammar::from_glrm(r#"
    extern grammar PAYLOAD
    extern token END
    start response
    nt response = PAYLOAD END
"#);
let child = Grammar::from_json_schema(r#"{"type":"null"}"#);
let complete = parent
    .bind("PAYLOAD", &child)?
    .bind("END", glrmask::Token::new(end_id))?;
let constraint = complete.compile(vocab)?;
# let _ = constraint;
# Ok(())
# }
```

Use `vocab.tokens(ids)` or `vocab.token_ranges(ranges)` for exact sets. Rust ranges
are inclusive; Python `range` stops are exclusive. The set consumes **one** ID,
not a sequence. Sparse vocabulary holes and incompatible vocabulary identities
are errors. IDs with the same spelling remain distinct, and spelling-less IDs
are not epsilon. See [exact-token ranges](docs/exact-token-ranges.md).

For a deliberately general grammar parameter, adapt exact events explicitly:
`Grammar::exact(vocab.tokens([id])?)` in Rust or
`glrmask.Grammar.exact(vocab.tokens([id]))` in Python. This is an event-aware grammar
fragment, not a regex over token spellings.

### Prepare, attach, specialize, link

`Grammar.prepare(vocab)` creates a non-runnable `PreparedGrammar`. The parent and
its children may both have unresolved parameters. Binding is immutable and does
not link a child prematurely. Reuse one prepared definition with separate binding
environments:

```python
import glrmask

vocab = glrmask.Vocab.from_id_to_bytes({
    0: b"x", 1: b"|", 2: b"x", 4: b"", 5: b"",
})
leaf = glrmask.Grammar.from_glrm('''
    extern token MARK
    start body
    nt body = "x" MARK
''').prepare(vocab)
parent = glrmask.Grammar.from_glrm('''
    extern grammar LEFT
    extern grammar RIGHT
    start pair
    nt pair = LEFT "|" RIGHT
''').prepare(vocab)

package = parent.bind("LEFT", leaf).bind("RIGHT", leaf)
assert [r.name for r in package.requirements] == ["LEFT::MARK", "RIGHT::MARK"]
complete = package.bind("LEFT::MARK", vocab.tokens([4])).bind(
    "RIGHT::MARK", vocab.tokens([5]),
)
constraint = complete.link()
state = constraint.start()
for token_id in [0, 4, 1, 2, 5]:
    assert state.mask()[token_id]
    state.commit_token(token_id)
assert state.is_accepting()
assert [r.name for r in leaf.requirements] == ["MARK"]
```

Rust uses the same operations: `prepare(&vocab)?`, `bind(name, &child)?`, and
`link()?` or `link_with(options)?`. Source `Grammar` objects also support open
attachment and later `CHILD::PARAM` binding. A descendant can be bound only after
its attachment exists. Unknown names and attempts to rebind a filled parameter
are rejected.

Each inline `grammar NAME { ... }` has an isolated interface, start rule, terminals,
and ignore policy. It does not inherit enclosing externs. Two references to the
same named inline grammar share that definition's bindings. Attaching the same
prepared child at two different names creates independently bindable occurrences.

`requirements()` in Rust, or `.requirements` in Python, reports structured paths,
expected kinds, and available original source spans. Preparation is allowed while
requirements remain; final `compile()` or `link()` requires every declared
parameter to be bound, including declarations whose uses are unreachable.
Introspection has finite traversal limits rather than expanding arbitrary shared
graphs without a bound.

### Persistence and root policy

Prepared components can be saved while incomplete:

```python
saved = package.save()
restored = glrmask.PreparedGrammar.load(saved, vocab=vocab)
assert [r.name for r in restored.requirements] == ["LEFT::MARK", "RIGHT::MARK"]
```

Runnable constraints separately support `save()` and `Constraint.load(...)`.
Artifacts preserve original token coordinates and vocabulary identity, not merely
a set of decoded spellings. Unsupported older pre-release formats are rejected.
Prepared artifacts are vocabulary-specific, not portable across arbitrary models.

Configure `end_tokens` on the final compile or link operation. Acceptance means a
complete grammar match exists; it does not require generation to stop when longer
matches remain possible. A configured end ID is admitted only at acceptance. Commit
the sampled end ID before ending the generation loop. Child components do not
supply the enclosing generation's root termination policy.

### Determined output

`state.forced()` selects a declared canonical, admitted encoding of proved common
bytes; `state.forced_token_ids()` returns locally inevitable original IDs without
choosing between encodings. Both are read-only and conservative. Canonical encoding
can change subsequent model logits and is not a probability-preserving optimization.
See [forcing contracts](docs/forcing.md) for acceptance barriers, exact events,
aliases, proof limits, and replay requirements.

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


## Grammar formats

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
separated sequences, explicit finite automata, Unicode/byte expressions and
terminal-only `shortest(...)`. See [the native format](docs/glrm-format.md).

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
