# Test layout

- `integration.rs`, `end_tokens.rs`, and `rollback.rs` cover the public API.
- `json_schema/` contains focused cross-cutting JSON Schema suites.
- `regressions/` contains isolated reproductions grouped by subsystem. Corpus IDs remain in filenames where they are useful for traceability.
- `fixtures/` contains only data used by a current test.

Each regression remains a separate Cargo test target, declared in `Cargo.toml`. This preserves process isolation for tests that modify environment variables.
