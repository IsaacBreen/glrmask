# Target JSON Schema source tree blueprint

The Chunk 08 tree is a major boundary improvement, but not the final ideal.  The
ideal publication tree should eventually be:

```text
src/import/json_schema/
  mod.rs
  schema/
    mod.rs
    document.rs
    node.rs
    assertions.rs
    object.rs
    array.rs
    string.rs
    number.rs
    references.rs
  load/
    mod.rs
    pointer.rs
    raw.rs
    references.rs
    object_keywords.rs
    array_keywords.rs
    string_keywords.rs
    number_keywords.rs
    combinator_keywords.rs
    unsupported.rs
  normalize/
    mod.rs
    all_of.rs
    any_of.rs
    one_of.rs
    not.rs
    merge.rs
    shape.rs
    factor_required.rs
    factor_variants.rs
    subsumption.rs
  lower/
    mod.rs
    context.rs
    names.rs
    builtins.rs
    refs.rs
    literals.rs
    dispatch.rs
    object/
      mod.rs
      fixed.rs
      open.rs
      pattern.rs
      additional.rs
      any_of.rs
      shadow.rs
      trie.rs
    array/
      mod.rs
      homogeneous.rs
      tuple.rs
      terminal_fast_path.rs
    string/
      mod.rs
      length.rs
      pattern.rs
      format.rs
      utf8.rs
      value_filter.rs
    number/
      mod.rs
      integer.rs
      decimal.rs
      range.rs
      multiple.rs
  tests/
    mod.rs
    refs.rs
    objects.rs
    arrays.rs
    strings.rs
    numbers.rs
    combinators.rs
    regressions.rs
```

## Sorting principle

A function belongs in:

- `schema/` if it only defines loaded data shapes.
- `load/` if it reads arbitrary JSON Schema JSON values.
- `normalize/` if it transforms or compares schema meanings.
- `lower/` if it emits `GrammarExpr` or rule names.
- `tests/` if it verifies a semantic family.
- `options.rs` if it is importer policy, especially env-var-backed policy.
- `diagnostics.rs` if it constructs errors or reports support boundaries.

## Anti-patterns to remove

- A file named for a JSON type but containing schema algebra and grammar rule
  allocation together.
- Functions that inspect raw `serde_json::Value` after the loading phase.
- Functions that emit `GrammarExpr` outside `lower/`.
- Support decisions hidden inside regex/helper code without a coverage-matrix
  entry.
- Comments that describe heuristics but not their semantic inclusion relation.
