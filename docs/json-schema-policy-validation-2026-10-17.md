# JSON Schema acceptance policy validation (2026-10-17)

**Unreleased:** this policy is implemented in draft PR #13 and is not a claim about currently published packages.

Source baseline: `385888ea316dd219d7cb7c73acf21d137a392a8b`. This change is
limited to the importer, its tests, probe tooling, and documentation. It does not
change llguidance, frozen benchmark inputs/binaries, parser backends, bindings,
services, or release settings.

## Policy and intentional breaking changes

Compilation preserves supported assertions or refuses unsupported combinations.
The former default `oneOf` approximation is removed. The old coercion option,
including its environment variable, errors if explicitly enabled; no new global
strictness switch or approximation mode is introduced. Existing per-import
configuration remains otherwise unchanged.

An undeclared dialect is now Draft 2020-12. Declare an older `$schema` for legacy
`items` tuples, `id`, `dependencies`, and old `$ref` sibling semantics. Formats
are annotations in the supported standard dialects; required format assertion
and unknown required vocabularies are unsupported. Unknown annotations and
literal payloads do not become schemas during scanning.

Exact restricted cases include disjoint `oneOf`, boolean/type complements,
constant/vacuous conditionals, trivially unique arrays, supported property
counts/names, fixed closed-object dependencies, and validated finite literals.
Other cases refuse explicitly, including unsafe recursive intersections,
multiple sibling combinators, and dynamic map minima above one without distinct
required names proving the minimum. Unsupported assertions never justify an
empty language. See [the policy and remaining serialization limits](json-schema-semantic-deviations.md).

Notable implementation repairs include previously skipped untyped object
assertions, lossy `allOf` union fallback and object union merging, overwritten
numeric bounds, skipped string length checks during finite-literal filtering,
numeric equality, recursive literal guesses, and accidental object `items`
interpretation. Pattern/maxLength resource refusal already existed; its stale
documentation has been corrected, without reducing its guard.

## Verification

All native checks use an owned incremental development target and the moved
isolated worktree `/private/tmp/glrmask-schema-policy-20261017`. `RUSTC_WRAPPER`
is disabled only for this development harness. No production latency claim is
made. A warm small-change probe rebuild took 13.25 seconds.

| Check | Result |
|---|---|
| Importer crate unit suite | 82 passed |
| Facade JSON Schema unit suite | 339 passed; 1 existing pathological dump ignored |
| New public policy integration suite (FastBuild and FastRuntime) | 6 passed |
| Existing public API suite | 22 passed |
| Existing public API review suite | Initially 21 passed; later 20 passed / 1 artifact-byte-equality failure; isolated retry passed |
| Python policy + existing public API tests | 14 passed |
| Canonical guide synchronization and diff whitespace | Passed |

The intermittent Rust review failure is in a GLRM-only deferred-boundary artifact
byte-equality assertion, outside this importer change. No API/runtime code was
modified to hide it. The Python extension required macOS dynamic lookup and
leaf-only `-C strip=none` to avoid the known Rust/LLVM LINKEDIT stripping bug;
these were build-command flags, not repository configuration changes.

The focused official differential covers 206 schemas / 1,005 fixture cases from
Draft 7 and Draft 2020-12 (plus annotation fixtures). Python jsonschema 4.24.0
checks schema validity and independently agrees with every official label.
Each native mode builds 150 schemas, refuses 56 explicitly as unsupported, and
judges 749 instances using token masks, commits, and final acceptance. Both
modes have **zero invalid acceptances, zero runtime errors**, and 12 valid
rejections from existing numeric spelling/object literal order restrictions.
The differential command intentionally exits nonzero for those raw rejections;
it is not an all-conformance-tests-passed claim. Unsupported/resource refusals,
malformed-schema claims, and unsupported empty-language claims have separate
reporting; erroneous empty/malformed claims with valid witnesses fail the harness.

On the audit's fixed 25 discriminating schemas, prior Dynamic/Static build
coverage was 19/25 with 14 invalid acceptances. Current FastBuild/FastRuntime
coverage is 9/25, with all 23 executed instance judgments correct in each mode.
This inexpensive build-coverage check uses a byte vocabulary and is not a
matched performance benchmark. A full benchmark belongs to the separately
assigned benchmark task; it must use the immutable implementation commit.

Evidence receipts, exact commands/logs, fixture hashes, raw differences, and
witness outcomes are retained in
`/Users/isaacbreen/Documents/Codex/2026-10-17/task-13/evidence/`.

## Relationship to llguidance

The pinned comparison is llguidance 1.9.1,
`f0971424ec072d3e4d4196bcc7f31a2f60527df9`; the existing audit and upstream
[v1.9.1 compiler](https://github.com/guidance-ai/llguidance/blob/v1.9.1/parser/src/json/compiler.rs)
and [current compiler](https://github.com/guidance-ai/llguidance/blob/main/parser/src/json/compiler.rs)
were consulted. Like its strict default, GLRMask refuses unproved exclusive
alternatives. Differences are intentional: GLRMask offers no coercion option,
uses annotation-only formats rather than enforcing known formats/rejecting
unknown ones, retains its exact fixed-property dependency subset, and refuses
map minima whose distinct-key count it cannot prove. Additional GLRMask
serialization limits remain explicit. Neither upstream implementation bugs nor
its feature limits are treated as a correctness specification.

The [official validation specification](https://json-schema.org/draft/2020-12/json-schema-validation)
and [official test suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite)
provide semantic references. The closed LLG expansion experiment was not rerun
or changed.
