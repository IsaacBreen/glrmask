# Mathematical contracts for the JSON Schema importer

## 1. Domains

Let:

- `Σ` be the byte alphabet.
- `Σ*` be all finite byte strings.
- `J` be the set of JSON values.
- `Text(v) ⊆ Σ*` be the set of JSON texts accepted by this crate's JSON lexical
  policy that decode to value `v`.
- `Schema` be the subset of JSON Schema supported by the importer.

For a supported JSON Schema `S`, JSON Schema semantics gives a denotation
`⟦S⟧ ⊆ J`.

The exact grammar language for `S` is:

```text
G(S) = ⋃ { Text(v) | v ∈ ⟦S⟧ }.
```

The importer emits some grammar language `Emit(S) ⊆ Σ*`.

## 2. Exactness and broadening

Every importer rule must be classified as one of three classes.

### Exact

```text
Emit(S) = G(S)
```

Examples intended to be exact:

- `type: "null"`
- `type: "boolean"`
- finite `enum`/`const` literals
- closed objects with fixed required/optional properties when no unsupported
  keyword intervenes
- bounded arrays when all item schemas are exact
- string `format` patterns for known formats when represented as regular
  languages over JSON string bodies

### Safe over-approximation

```text
G(S) ⊆ Emit(S)
```

Examples currently present in the codebase include broadening for some unsafe
`allOf` intersections over parser-shaped object/array grammar expressions.  Such
broadening is acceptable only when the source code names the fallback and docs
record which schema family loses precision.

### Rejection

The importer rejects `S` with a `SchemaImportError` when neither exactness nor a
safe over-approximation is acceptable for publication.  A rejection is preferable
to silent unsound narrowing.

## 3. Soundness direction

For constrained decoding, unsound **narrowing** is more dangerous than broadening:
rejecting valid continuations makes a model unable to produce values permitted by
the schema.  Unsound broadening can allow invalid values, which is also harmful,
but it is at least observable by downstream validation.  Therefore the importer
must never implement an undocumented narrowing.  Each approximation site must
state whether it is exact, broadening, or rejected.

## 4. Object denotation

An object schema with literal properties `P`, required set `R`, pattern
properties `Q`, and additional policy `A` denotes finite maps `m` satisfying:

```text
R ⊆ dom(m)
for each k in dom(m):
  if k ∈ P: m[k] ∈ ⟦P[k]⟧
  for each (pattern, schema) in Q where pattern matches k:
      m[k] ∈ ⟦schema⟧
  if k is not covered by P or Q:
      A permits k and m[k] ∈ ⟦A_schema⟧ when A is schema-valued
minProperties ≤ |dom(m)| ≤ maxProperties if bounded
```

The lowering must then account for arbitrary key order unless it intentionally
constructs a deterministic canonical order.  The current implementation uses
permutation/separated-sequence machinery and factorization tricks for performance.

## 5. Array denotation

An array schema with prefix items `p_0..p_{n-1}`, tail item schema `t`, and bounds
`minItems/maxItems` denotes arrays `a` such that:

```text
minItems ≤ len(a) ≤ maxItems, if maxItems exists
for i < n: a[i] ∈ ⟦p_i⟧
for i ≥ n: a[i] ∈ ⟦t⟧
```

Tuple-form legacy `items: [...]` is loaded into the same representation.

## 6. String denotation

JSON Schema string constraints apply to decoded strings, while emitted grammars
accept encoded JSON strings.  Thus every string lowerer has two levels:

```text
decoded regex / length / format constraint
  -> JSON string body byte regex
  -> quoted JSON string terminal
```

This distinction is why regex lowering lives in `lower/string.rs` rather than
`schema/scalar.rs`.

## 7. Number denotation

The current numeric representation uses `f64` for bounds and `multipleOf`.  This
is inherited behavior and not publication-ideal.  Publication-quality numeric
semantics should eventually use exact decimal rationals:

```text
minimum, maximum, exclusiveMinimum, exclusiveMaximum ∈ DecimalRational
multipleOf ∈ PositiveDecimalRational
```

This chunk records that target without changing numeric behavior.
