# Review checklist for Chunk 08

## Tree-level review

- [ ] `src/import/json_schema/mod.rs` reads as a facade, not an implementation dump.
- [ ] Typed schema syntax lives under `schema/`.
- [ ] Raw `serde_json::Value` loading lives under `load/`.
- [ ] Combinator algebra lives under `normalize/`.
- [ ] Grammar expression construction lives under `lower/`.
- [ ] Options/env vars live in `options.rs`.
- [ ] Error construction lives in `diagnostics.rs`.
- [ ] Tests are ready to split by semantic family.

## Semantic review

- [ ] Every `allOf` fallback has an exact/broadening/rejection classification.
- [ ] Every `anyOf` object factoring optimization has a language-equivalence argument.
- [ ] Every string regex conversion distinguishes decoded characters from encoded JSON bytes.
- [ ] Every object additional-property exclusion uses the same key-language convention.
- [ ] Every numeric multiple/range lowering states whether it is exact under current numeric representation.

## Publication review

- [ ] Comments avoid temporary language such as "simple importer" unless referring to a historical note in docs.
- [ ] Schema-specific heuristics are not hidden as generic algorithms.
- [ ] Env vars are documented and namespaced.
- [ ] Public docs say JSON Schema is value-level and grammar lowering is a separate interpretation.
- [ ] Unsupported keywords have a clear table entry.
