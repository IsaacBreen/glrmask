# JSON lexical language specification

The lowerer emits grammars over JSON text, not abstract JSON values.  This file
defines the intended lexical language used by the importer.

## 1. Whitespace policy

Current separator terminals are canonicalized around commas and colons.  The
existing behavior should be preserved until a deliberate whitespace policy change
is made.  If whitespace support is broadened, it must happen through named JSON
lexical terminals, not by embedding ad hoc spaces into object or array code.

## 2. Strings

A JSON Schema `type: string` describes decoded Unicode strings.  The grammar must
recognize encoded JSON string texts.  Therefore every pattern has to pass through
this semantic map:

```text
decoded regex over Unicode scalar values
  -> JSON string body regex over encoded bytes/escapes
  -> quoted JSON string terminal or grammar expression
```

Important distinction:

- decoded `.` means any decoded character accepted by the regex engine,
- encoded `.` in a grammar regex would mean a byte-level regex wildcard and is
  usually wrong.

## 3. Numbers

JSON numbers are lexical forms.  Numeric constraints are semantic.  For example,
`multipleOf: 0.1` is a decimal-rational statement, not a floating-point statement.
The current code inherits `f64` storage.  Publication target: replace numeric
bounds with decimal rational values and make regex generation state exactness.

## 4. Objects

JSON object values are unordered maps.  JSON object texts are ordered key-value
lists.  Object lowering must therefore account for all legal orders unless a
canonical-order policy is explicitly documented.  The current lowerer emits
permutation/pair-list languages for many fixed-object shapes.

## 5. Arrays

Arrays are ordered both as values and as text.  Array lowering is therefore much
simpler than object lowering: tuple/prefix items are sequence constraints, and
homogeneous `items` constraints are repetition constraints.

## 6. Literal values

`const` and `enum` lowering uses `serde_json::to_string` for literals.  This is a
canonicalization choice.  It means the grammar for an exact literal accepts the
canonical text emitted by serde, not every whitespace-preserving original text.
If broader literal whitespace is desired, implement it through lexical terminals
for separators and containers.
