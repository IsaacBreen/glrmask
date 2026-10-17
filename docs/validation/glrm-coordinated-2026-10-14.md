# Coordinated GLRM/Lark validation — 14 October 2026

This record covers the coordinated component API, native GLRM, selected Lark
compatibility features and exact-event forcing changes. Counts refer to the
latest successful run of each suite, not the sum of repeated attempts. The
machine was an Apple M1 Pro running macOS 27, with native Rust 1.95.0 and Python
3.12.4. Tests used the canonical workspace/test declarations and an owned
incremental development target; performance qualification is a separate gate.

## Correctness coverage

| Gate | Passed | Existing ignored |
|---|---:|---:|
| Root library unit tests | 1,186 | 51 |
| Thirteen component library suites | 1,264 | 7 |
| All 44 configured root integration targets | 410 | 0 |
| Python suite, installed from a real wheel | 191 | 0 |
| Workspace executable Rust doctest | 1 | 0 |
| Rust examples extracted from five Markdown documents | 9 | 0 |

The Rust non-doctest total is **2,860 passing tests**, with 58 pre-existing ignored
tests. No failing test was converted into an ignore to obtain these results.
An interrupted workspace run exhausted disk space; its complete per-crate
results were retained and the four remaining crates were rerun successfully.
The integration matrix ran in bounded batches. Five historical syntax fixtures
were migrated to the canonical API and all affected targets were rerun. These
are per-suite receipts, not a claim of one uninterrupted monolithic command.

Independent finite byte/event interpreters check 2,880 reachable original-ID
states across source/prepared/loaded paths and the four optimization modes,
including duplicate spellings, spelling-less events, partial UTF-8, accepting
alternatives and cross-component token boundaries. Additional tests cover empty
event languages, high vocabulary IDs, separate ignored-terminal identities,
component-local names, immutable independent bindings, bounded shared graphs,
and malformed prepared-artifact manifests. Finite exhaustive probes complement
the broader regression suite; they are not a proof for every possible grammar.

The external default-feature Rust API was checked separately from workspace
feature unification: canonical APIs compile; the removed public unlinked type,
old preparation spelling and implicit runnable-child binding do not. Python
stubs were compared with the actual native class methods and packed-mask types.

## Packages and documentation

The source distribution contains the new grammar/component, parser-source,
checked-regex and prepared-DAG modules, plus Python stubs and `py.typed`.
The wheel is built and installed into a separate directory, and the test process
verifies that imports come from that installation rather than the editable source.
Both archives select their intended README. Interpreter caches are excluded.

Seven ordinary Python documentation blocks execute against the installed wheel.
Twelve distinct native/Lark blocks compile and compute a mask, and one open
component example prepares its declared requirements. The optional model-backed
`llama_cpp`/Torch demonstration is syntax-checked only: it needs external packages
and a caller-supplied `model.gguf`; no model inference run is claimed.

Documentation generation and the unchanged benchmark-asset synchronization checks
pass. A `[lazy]` profile table inconsistency found by the executable examples was
corrected to require the Lark LLM profile; runtime recognition was not relaxed.

## Reproduction and evidence

The adjacent JSON indexes each configured integration target and its successful
receipt. Detailed logs, source fingerprints, executable/archive hashes and prior
failed attempts are retained in the task's `glrm-coordinated-930283-work`
directory, with the authoritative Obsidian handoff naming each gate. The ordinary
commands are workspace `cargo test --lib`, root configured integration tests,
workspace `cargo test --doc`, and `pytest -q python/tests`. Use the supported
native toolchain and preserve exact feature/profile choices when comparing runs.

This record does not claim Windows qualification of the final Mac-only changes,
a full 9,558-schema performance sweep, unchanged model sampling probabilities
under canonical forcing, or package publication. Release-profile timing evidence
must be interpreted using its own corpus, baseline, binaries and measurements.

## Matched release measurements

The separately recorded release-profile check is in
`glrm-coordinated-performance-2026-10-14.md`, with its full trial summaries and
measured driver alongside it. Twelve alternating process trials produced
3,456,000 timed mask computations across fifteen matched case/mode combinations.
All checked masks and token traces agree with the pre-redesign baseline. The
report retains small compile/link overheads and explicitly limits its conclusions
to the synthetic corpus; it does not replace a full schema-corpus performance run.
